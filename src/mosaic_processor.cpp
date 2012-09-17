#include "mosaic_cam_pose/mosaic_processor.h"
#include <nav_msgs/Odometry.h>
#include <ros/package.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/camera_subscriber.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>

#define DRAW_RICH_KEYPOINTS_MODE     0
#define DRAW_OUTLIERS_MODE           0
#define DRAW_OPENCV_WINDOW           1

namespace enc = sensor_msgs::image_encodings;

const std::string winName = "Correspondences";
enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1, DISTANCE_FILTER = 2};

/** @function MosaicProcessor */
MosaicProcessor::MosaicProcessor(Parameters p, std::string transport){

  p.matcherFilterType = getMatcherFilterType( p.matcherFilterName );

  std::memcpy(&this->parameters,&p,sizeof(Parameters));

  ROS_INFO("Creating detector, descriptor extractor and descriptor matcher ...");

  detector_ = cv::FeatureDetector::create( p.featureDetectorType );
  descriptorExtractor_ = cv::DescriptorExtractor::create( p.descriptorExtractorType );
  descriptorMatcher_ = cv::DescriptorMatcher::create( p.descriptorMatcherType );

  first_run_ = true;

  if( detector_.empty() || descriptorExtractor_.empty() || descriptorMatcher_.empty()  )
  {
    ROS_ERROR("Can not create detector or descriptor extractor or descriptor matcher of given types");

  }


  ROS_INFO_STREAM("Reading the mosaic image " << p.mosaicImgName);
  mosaicImg = cv::imread(parameters.mosaicImgName);

  if( mosaicImg.empty())
  {
    ROS_ERROR("Mosaic image is empty");
  }

  ROS_INFO("Extracting keypoints from first image...");
  detector_->detect( mosaicImg, keypointsMosaic_ );
  ROS_INFO_STREAM(keypointsMosaic_.size() << " points");

  ROS_INFO("Computing descriptors for keypoints from first image...");
  descriptorExtractor_->compute( mosaicImg, keypointsMosaic_, descriptorsMosaic_ );

  ROS_INFO("Obtaining mosaic points...");
  cv::KeyPoint::convert(keypointsMosaic_, pointsMosaic_);
  pointsMosaic3D_.resize(pointsMosaic_.size());
  for (size_t i_mos=0;i_mos<pointsMosaic_.size();i_mos++){
      pointsMosaic3D_[i_mos].x = pointsMosaic_[i_mos].x/MOSAIC_PX_METER;
      pointsMosaic3D_[i_mos].y = pointsMosaic_[i_mos].y/MOSAIC_PX_METER;
      pointsMosaic3D_[i_mos].z = 0;
  }

  ros::NodeHandle nh;

  std::string cam_ns = nh.resolveName("stereo");
  std::string image_topic = ros::names::clean(cam_ns + "/left/" + nh.resolveName("image"));
  std::string info_topic = cam_ns + "/left/camera_info";

  // Subscribe to input topics.
  ROS_INFO("Subscribing to:\n\t* %s \n\t* %s", 
      image_topic.c_str(),
      info_topic.c_str());

  image_transport::ImageTransport it(nh);
  //Subscribe to image
  //image_transport::Subscriber image_sub;
  cam_sub_ = it.subscribeCamera(image_topic, 1, &MosaicProcessor::cameraCallback, this, transport);
  posePub_ = nh.advertise<geometry_msgs::PoseStamped>("pose", 1);
  odomPub_ = nh.advertise<nav_msgs::Odometry>("odom_gt", 1);

#if DRAW_OPENCV_WINDOW
  cv::namedWindow(winName,CV_WINDOW_NORMAL);
#endif
}

/** @function ~MosaicProcessor */
MosaicProcessor::~MosaicProcessor(){
}

/** @function getMatcherFilterType */
int MosaicProcessor::getMatcherFilterType( const std::string& str )
{
  if( str == "NoneFilter" )
    return NONE_FILTER;
  if( str == "CrossCheckFilter" )
    return CROSS_CHECK_FILTER;
  if( str == "DistanceFilter")
    return DISTANCE_FILTER;
  CV_Error(CV_StsBadArg, "Invalid filter name");
  return -1;
}

/** @function simpleMatching */
void MosaicProcessor::simpleMatching( cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches12 )
{
  std::vector<cv::DMatch> matches;
  descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

/** @function crossCheckMatching */
void MosaicProcessor::crossCheckMatching( cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& filteredMatches12, int knn=1 )
{
  filteredMatches12.clear();
  std::vector<std::vector<cv::DMatch> > matches12, matches21;
  descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
  descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
  for( size_t m = 0; m < matches12.size(); m++ )
  {
    bool findCrossCheck = false;
    for( size_t fk = 0; fk < matches12[m].size(); fk++ )
    {
      cv::DMatch forward = matches12[m][fk];

      for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
      {
        cv::DMatch backward = matches21[forward.trainIdx][bk];
        if( backward.trainIdx == forward.queryIdx )
        {
          filteredMatches12.push_back(forward);
          findCrossCheck = true;
          break;
        }
      }
      if( findCrossCheck ) break;
    }
  }
}

void MosaicProcessor::thresholdMatching( cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& filteredMatches12, double matching_threshold)
{
  filteredMatches12.clear();
  std::vector<std::vector<cv::DMatch> > matches12;
  int knn = 2;
  descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
  for( size_t m = 0; m < matches12.size(); m++ )
  {
    if (matches12[m].size() == 1)
    {
      filteredMatches12.push_back(matches12[m][0]);
    }
    else if (matches12[m].size() == 2) // normal case
    {
      if (matches12[m][0].distance / matches12[m][1].distance < matching_threshold)
      {
        filteredMatches12.push_back(matches12[m][0]);
      }
    }
  }
}



/** @function imageCallback */
void MosaicProcessor::cameraCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info){

  cv_bridge::CvImagePtr cv_ptr;

  try
  {
    if (sensor_msgs::image_encodings::isColor(msg->encoding))
      //ROS_INFO("\t* It would be slightly faster with MONO images");
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  frameImg = cv_ptr->image;

  assert( !mosaicImg.empty() );
  assert( !frameImg.empty());

  ROS_INFO("Extracting keypoints from second image...");
  detector_->detect( frameImg, keypointsFrame_ );
  ROS_INFO_STREAM(keypointsFrame_.size() << " points");

  ROS_INFO("Computing descriptors for keypoints from second image...");
  descriptorExtractor_->compute( frameImg, keypointsFrame_, descriptorsFrame_ );

  ROS_INFO("Matching descriptors...");
  switch( parameters.matcherFilterType )
  {
    case CROSS_CHECK_FILTER :
      crossCheckMatching( descriptorMatcher_, descriptorsFrame_, descriptorsMosaic_, filteredMatches_, 1 );
      break;
    case DISTANCE_FILTER:
      //thresholdMatching( descriptorMatcher_, descriptorsMosaic_, descriptorsFrame_, filteredMatches_, parameters.matching_threshold);
      //changed order for train and query
      thresholdMatching( descriptorMatcher_, descriptorsFrame_, descriptorsMosaic_, filteredMatches_, parameters.matching_threshold);
      break;
    default :
      simpleMatching( descriptorMatcher_, descriptorsFrame_, descriptorsMosaic_, filteredMatches_ );
      break;
  }
  ROS_INFO_STREAM(filteredMatches_.size() << " points");

  std::vector<int> queryIdxs( filteredMatches_.size() ), trainIdxs( filteredMatches_.size() );
  std::vector<cv::Point2f> image_points(filteredMatches_.size());
  std::vector<cv::Point3f> world_points(filteredMatches_.size());
  for( size_t i = 0; i < filteredMatches_.size(); i++ )
  {
    queryIdxs[i] = filteredMatches_[i].queryIdx;
    trainIdxs[i] = filteredMatches_[i].trainIdx;
    image_points[i] = keypointsFrame_[queryIdxs[i]].pt;
    world_points[i] = pointsMosaic3D_[trainIdxs[i]];
  }
#if DRAW_OPENCV_WINDOW
  cv::Mat H12,H21;
  if( parameters.ransacReprojThreshold >= 0 )
  {

    ROS_INFO("Computing homography (RANSAC)...");
    //same change
    //cv::KeyPoint::convert(keypointsMosaic_, pointsMosaic_, queryIdxs);
    //cv::KeyPoint::convert(keypointsFrame_, pointsFrame_, trainIdxs);
    cv::KeyPoint::convert(keypointsMosaic_, pointsMosaic_, trainIdxs);
    cv::KeyPoint::convert(keypointsFrame_, pointsFrame_, queryIdxs);
    H12 = cv::findHomography( cv::Mat(pointsMosaic_), cv::Mat(pointsFrame_), CV_RANSAC, parameters.ransacReprojThreshold );
    H21 = cv::findHomography( cv::Mat(pointsFrame_), cv::Mat(pointsMosaic_), CV_RANSAC, parameters.ransacReprojThreshold );
  }

  cv::Mat drawImg;
  if( !H12.empty() ) // filter outliers
  {
    int count=0;
    std::vector<char> matchesMask( filteredMatches_.size(), 0 );
    //cv::KeyPoint::convert(keypointsMosaic_, pointsMosaic_, queryIdxs);
    //cv::KeyPoint::convert(keypointsFrame_, pointsFrame_, trainIdxs);
    cv::KeyPoint::convert(keypointsMosaic_, pointsMosaic_, trainIdxs);
    cv::KeyPoint::convert(keypointsFrame_, pointsFrame_, queryIdxs);
    cv::Mat pointsMosaicT; 
    cv::perspectiveTransform(cv::Mat(pointsMosaic_), pointsMosaicT, H12);
    double ransacReprojThreshold = parameters.ransacReprojThreshold;
    double maxInlierDist = ransacReprojThreshold < 0 ? 3 : ransacReprojThreshold;
    for( size_t i1 = 0; i1 < pointsMosaic_.size(); i1++ )
    {
      if( cv::norm(pointsFrame_[i1] - pointsMosaicT.at<cv::Point2f>((int)i1,0)) <= maxInlierDist ){ // inlier
        matchesMask[i1] = 1;
        count++;
      }
    }
    ROS_INFO_STREAM(count << " inliers");
    // draw inliers
/*    cv::drawMatches( mosaicImg, keypointsMosaic_, 
        frameImg, keypointsFrame_, 
        filteredMatches_, drawImg, 
        CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
*/    cv::drawMatches(  
        frameImg, keypointsFrame_,
        mosaicImg, keypointsMosaic_,
        filteredMatches_, drawImg, 
        CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
#if DRAW_RICH_KEYPOINTS_MODE
        , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
        );

#if DRAW_OUTLIERS_MODE
    // draw outliers
    for( size_t i1 = 0; i1 < matchesMask.size(); i1++ )
      matchesMask[i1] = !matchesMask[i1];
    cv::drawMatches( mosaicImg, keypointsMosaic_, 
        frameImg, keypointsFrame_, 
        filteredMatches_, drawImg, 
        CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), matchesMask,
        cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
#endif
  }
  else
    drawMatches( mosaicImg, keypointsMosaic_, frameImg, keypointsFrame_, filteredMatches_, drawImg );


#endif

  ROS_INFO("Trying to find the camera pose...");
  
  //cuando se hace la rectificacion también se modifican las distancias focales.
  //los parametros originales estan en la K y los rectificados en la P
  
  const cv::Mat P(3,4, CV_64FC1, const_cast<double*>(cam_info->P.data()));
  // We have to take K' here extracted from P to take the R|t into account
  // that was performed during rectification.
  // This way we obtain the pattern pose with respect to the same frame that
  // is used in stereo depth calculation.
  const cv::Mat K_prime = P.colRange(cv::Range(0,3));


  if(first_run_){
    rvec_ = cv::Mat(3, 1, CV_64FC1);
    tvec_ = cv::Mat(3, 1, CV_64FC1);
    useExtrinsicGuess_ = false;
  }
  int numIterations = 100;
  float allowedReprojectionError = parameters.ransacReprojThreshold;//8.0; // used by ransac to classify inliers
  int maxInliers = 100; // stop iteration if more inliers than this are found
  cv::Mat inliers;
  cv::solvePnPRansac(world_points, image_points, K_prime, 
      cv::Mat(), rvec_, tvec_, useExtrinsicGuess_, numIterations,
      allowedReprojectionError, maxInliers, inliers);
  int numInliers = cv::countNonZero(inliers);
  int minInliers = 8;
  if (numInliers >= minInliers)
  {
    ROS_INFO_STREAM("Found transform with " 
      << numInliers << " inliers from " 
      << pointsMosaic3D_.size() << " matches:\n"
      << "  rvec: " << rvec_ << "\n"
      << "  tvec: " << tvec_ );

    //TODO: draw the real inliers for solvePnPRansac and publish that as an ros image topic
    //that is only processed if there are any subscribers.
    //
    //filteredMatches?
    //const vector<vector<DMatch>>& matches1to2
    //the output from the filtered matches is valid?? Its not used in solvePnP
    //
    //matchesMask? crec que puc posar els inliers de la funcio distància
    //const vector<vector<char>>& matchesMask
    //
/*    cv::drawMatches( mosaicImg, keypointsMosaic_, 
        frameImg, keypointsFrame_, 
        filteredMatches_, drawImg, 
        CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), inliers);
 */   
    // publish result
    ros::Time stamp = msg->header.stamp;
    if (stamp.toSec()==0.0)
      stamp = ros::Time::now();
    publishTransform(tvec_, rvec_, stamp, msg->header.frame_id);
    if(first_run_)
      useExtrinsicGuess_ = true;
  }
  else
  {
    ROS_INFO("Not enough inliers (%i) in %zu matches. Minimum is %i.", 
        numInliers, pointsMosaic3D_.size(), minInliers);
  }

#if DRAW_OPENCV_WINDOW 
  //-- Get the corners from the frame image ( the object to be "detected" )
  std::vector<cv::Point2f> frame_corners(4);
  frame_corners[0] = cvPoint(0,0); frame_corners[1] = cvPoint( frameImg.cols, 0 );
  frame_corners[2] = cvPoint( frameImg.cols, frameImg.rows ); frame_corners[3] = cvPoint( 0, frameImg.rows );
  std::vector<cv::Point2f> scene_corners(4);

  perspectiveTransform( frame_corners, scene_corners, H21);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( drawImg, scene_corners[0]+frame_corners[1], scene_corners[1]+frame_corners[1], cv::Scalar(0, 255, 0), 4 );
  line( drawImg, scene_corners[1]+frame_corners[1], scene_corners[2]+frame_corners[1], cv::Scalar( 0, 255, 0), 4 );
  line( drawImg, scene_corners[2]+frame_corners[1], scene_corners[3]+frame_corners[1], cv::Scalar( 0, 255, 0), 4 );
  line( drawImg, scene_corners[3]+frame_corners[1], scene_corners[0]+frame_corners[1], cv::Scalar( 0, 255, 0), 4 );
  cv::imshow( winName, drawImg );
  cv::waitKey(5);
#endif
}


void MosaicProcessor::publishTransform(const cv::Mat& tvec, const cv::Mat& rvec, 
    const ros::Time& stamp, const std::string& camera_frame_id)
{
  tf::Vector3 axis(
      rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0));
  double angle = cv::norm(rvec);
  tf::Quaternion quaternion(axis, angle);

  tf::Vector3 translation(
      tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
  
  tf::Transform transform(quaternion, translation);
  tf::StampedTransform stampedTransform(
      transform, stamp, camera_frame_id, "/mosaic");
  tfBroadcaster_.sendTransform(stampedTransform);

  geometry_msgs::PoseStamped pose_msg;
  pose_msg.header.stamp = stamp;
  pose_msg.header.frame_id = camera_frame_id;
  tf::poseTFToMsg(transform.inverse(), pose_msg.pose);

  //publish the odometry msg
  nav_msgs::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = "odom_gt";

  //set the position
  odom.pose.pose = pose_msg.pose;

  odomPub_.publish(odom);

  posePub_.publish(pose_msg);
}

std::ostream& operator<<(std::ostream& out, const MosaicProcessorHeader::Parameters& params)
{
  out << "\t* Mosaic image name           = " << params.mosaicImgName             << std::endl;
  out << "\t* Feature detector type       = " << params.featureDetectorType       << std::endl;
  out << "\t* Descriptor extractor type   = " << params.descriptorExtractorType   << std::endl;
  out << "\t* Descriptor matcher type     = " << params.descriptorMatcherType     << std::endl;
  out << "\t* Matcher filter name         = " << params.matcherFilterName         << std::endl;
  out << "\t* Matcher filter threshold    = " << params.matching_threshold        << std::endl;
  out << "\t* RANSAC reproj. threshold    = " << params.ransacReprojThreshold;//     << std::endl;
  return out;
}

/** @function main */
int main(int argc, char** argv)
{

  cv::initModule_nonfree();

  //ROS part
  ros::init(argc, argv, "mosaic_processor");
  ros::NodeHandle nh;
  if (ros::names::remap("stereo") == "stereo") {
    ROS_WARN("'stereo' has not been remapped! Example command-line usage:\n"
        "\t$ rosrun mosaic_cam_pose mosaic_processor stereo:=/stereo_down/left image:=image_rect");
  }

  if (ros::names::remap("image").find("rect") == std::string::npos) {
    ROS_WARN("mosaic_processor needs rectified input images. The used image "
        "topic is '%s'. Are you sure the images are rectified?",
        ros::names::remap("image").c_str());
  }

  std::string transport = argc > 1 ? argv[1] : "raw";

  MosaicProcessor::Parameters p;

  std::string path = ros::package::getPath("mosaic_cam_pose");

  nh.setParam("mosaicImage", path+"/src/mosaic2-20.png");
  nh.setParam("featureDetectorType", "SIFT");
  nh.setParam("descriptorExtractorType", "SIFT");
  nh.setParam("descriptorMatcherType", "FlannBased");
  nh.setParam("matcherFilterName", "DistanceFilter");
  nh.setParam("matching_threshold",0.8);
  nh.setParam("ransacReprojThreshold", 5.0);

  nh.getParam("mosaicImage", p.mosaicImgName);
  nh.getParam("featureDetectorType", p.featureDetectorType);
  nh.getParam("descriptorExtractorType", p.descriptorExtractorType);
  nh.getParam("descriptorMatcherType", p.descriptorMatcherType);
  nh.getParam("matcherFilterName", p.matcherFilterName);
  nh.getParam("matching_threshold",p.matching_threshold);
  nh.getParam("ransacReprojThreshold", p.ransacReprojThreshold);

  ROS_INFO_STREAM("The parameters set are: \n" << p);

  MosaicProcessor processor(p,transport);

  ros::spin();

  return 0;
}


