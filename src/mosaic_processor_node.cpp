
#include <opencv2/nonfree/nonfree.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/camera_subscriber.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include "mosaic_pose_estimation/mosaic_processor.h"


namespace enc = sensor_msgs::image_encodings;

class MosaicProcessorNode
{
public:
  MosaicProcessorNode(const std::string transport)
  {
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    MosaicProcessor::Parameters p;
    std::string path = ros::package::getPath("mosaic_pose_extractor");
    nh.getParam("mosaic_image", p.mosaicImgName);
    nh.getParam("px_per_meter", p.pxPerMeter);
    nh.param("feature_detector_type", p.featureDetectorType, std::string("SIFT"));
    nh.param("descriptor_extractor_type", p.descriptorExtractorType, std::string("SIFT"));
    nh.param("descriptor_matcher_type", p.descriptorMatcherType, std::string("FlannBased"));
    nh.param("matcher_filter_name", p.matcherFilterName, std::string("DistanceFilter"));
    nh.param("matching_threshold",p.matching_threshold, 0.8);
    nh.param("ransac_reprojection_threshold", p.ransacReprojThreshold, 5.0);

    ROS_INFO_STREAM("The parameters set are: \n" << p);

    ROS_INFO_STREAM("Instantiating the mosaic processor.");
    mosaicProcessor_ = boost::shared_ptr<MosaicProcessor>(
        new MosaicProcessor(p));

    std::string cam_ns = nh.resolveName("stereo");
    std::string image_topic = 
      ros::names::clean(cam_ns + "/left/" + nh.resolveName("image"));
    std::string info_topic = cam_ns + "/left/camera_info";

    // Subscribe to input topics.
    ROS_INFO("Subscribing to:\n\t* %s \n\t* %s", 
        image_topic.c_str(),
        info_topic.c_str());

    image_transport::ImageTransport it(nh);
    image_transport::ImageTransport it_private(nh_private);
    cam_sub_ = it.subscribeCamera(image_topic, 1, 
        &MosaicProcessorNode::cameraCallback, this, transport);
    posePub_ = nh_private.advertise<geometry_msgs::PoseStamped>("pose", 1);
    odomPub_ = nh_private.advertise<nav_msgs::Odometry>("odom_gt", 1);
    matchesImgPub_ = it_private.advertise("matches_image",1);
  }

private:

  void cameraCallback(
      const sensor_msgs::ImageConstPtr& msg, 
      const sensor_msgs::CameraInfoConstPtr& cam_info)
  {
    cv_bridge::CvImagePtr cv_ptr;

    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, 
          sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    mosaicProcessor_->setCameraInfo(cam_info);
    mosaicProcessor_->process(cv_ptr->image);
    ROS_INFO_STREAM(mosaicProcessor_->getNumInliers() << " inliers.");

    // publish result
    tf::Transform transform = mosaicProcessor_->getTransformation();
    ros::Time stamp = msg->header.stamp;
    if (stamp.toSec()==0.0)
      stamp = ros::Time::now();

    tf::StampedTransform stampedTransform(
      transform, stamp, msg->header.frame_id, "/mosaic");

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = msg->header.frame_id;
    tf::poseTFToMsg(transform, pose_msg.pose);

    //publish the odometry msg
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "odom_gt";

    //set the position
    odom.pose.pose = pose_msg.pose;

    odomPub_.publish(odom);
    posePub_.publish(pose_msg);
    tfBroadcaster_.sendTransform(stampedTransform);

    if (matchesImgPub_.getNumSubscribers() > 0)
    {
      cv::Mat drawImg = mosaicProcessor_->drawMatches();
      cv_bridge::CvImagePtr drawImgPtr(new cv_bridge::CvImage);
      drawImgPtr->encoding = cv_ptr->encoding;
      drawImgPtr->image = drawImg;
      matchesImgPub_.publish(drawImgPtr->toImageMsg());
      ROS_DEBUG("Image published.");
    }
  }

  ros::Publisher posePub_;
  ros::Publisher odomPub_;
  image_transport::Publisher matchesImgPub_;
  image_transport::CameraSubscriber cam_sub_;
  tf::TransformBroadcaster tfBroadcaster_;

  boost::shared_ptr<MosaicProcessor> mosaicProcessor_;
 
};

int main(int argc, char** argv)
{
  cv::initModule_nonfree();

  ros::init(argc, argv, "mosaic_processor");
  ros::NodeHandle nh("~");
  if (ros::names::remap("stereo") == "stereo") 
  {
    ROS_WARN("'stereo' has not been remapped! "
             "Example command-line usage:\n"
             "\t$ rosrun mosaic_cam_pose mosaic_processor "
             "stereo:=/stereo_down/left image:=image_rect");
  }

  if (ros::names::remap("image").find("rect") == std::string::npos) {
    ROS_WARN("mosaic_processor needs rectified input images. "
             "The used image topic is '%s'. Are you sure the images are "
             "rectified?",
        ros::names::remap("image").c_str());
  }

  std::string transport = argc > 1 ? argv[1] : "raw";
  MosaicProcessorNode processor_node(transport);
  ros::spin();
  return 0;
}


