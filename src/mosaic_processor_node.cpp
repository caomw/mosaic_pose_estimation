
#include <opencv2/nonfree/nonfree.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
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
  MosaicProcessorNode(const std::string transport): first_tf_(true)
  {
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    MosaicProcessor::Parameters p;
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    nh_private.getParam("mosaic_image", p.mosaicImgName);

    // Chech if image path is relative or absolute
    if (p.mosaicImgName.substr(0,1) != "/")
      p.mosaicImgName = path + "/" + p.mosaicImgName;

    nh_private.getParam("px_per_meter", p.pxPerMeter);
    nh_private.param("feature_detector_type", p.featureDetectorType, std::string("SIFT"));
    nh_private.param("descriptor_extractor_type", p.descriptorExtractorType, std::string("SIFT"));
    nh_private.param("descriptor_matcher_type", p.descriptorMatcherType, std::string("FlannBased"));
    nh_private.param("matcher_filter_type", p.matcherFilterName, std::string("DistanceFilter"));
    nh_private.param("matching_threshold",p.matching_threshold, 0.8);
    nh_private.param("ransac_reprojection_threshold", p.ransacReprojThreshold, 5.0);
    nh_private.param("min_num_inliers", p.minNumInliers, 10);
    nh_private.param("reset_origin", reset_origin_, true);
    nh_private.param("max_tf_diff", max_tf_diff_, 1.5);
    nh_private.param("base_frame_id", base_frame_id_, std::string(""));

    ROS_INFO_STREAM("The parameters set are: \n" << p);
    ROS_INFO_STREAM("reset_origin = " << (reset_origin_ ? "true" : "false"));

    ROS_INFO_STREAM("Instantiating the mosaic processor.");
    mosaicProcessor_ = boost::shared_ptr<MosaicProcessor>(
        new MosaicProcessor(p));

    std::string cam_ns = nh.resolveName("stereo");
    std::string image_topic =
      ros::names::clean(cam_ns + "/left/" + nh.resolveName("image"));
    std::string info_topic = cam_ns + "/left/camera_info";

    // Init
    prev_transform_.setIdentity();

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
    // Wait for tf between camera and base link
    if (!is_tf_init_)
    {
      // If no base link frame specified...
      if (base_frame_id_ == "")
      {
        camera2base_.setIdentity();
        is_tf_init_ = true;
      }
      else
      {
        try
        {
          // Extract the transform
          tf_listener_.lookupTransform(msg->header.frame_id,
              base_frame_id_,
              ros::Time(0),
              camera2base_);
          is_tf_init_ = true;
        }
        catch (tf::TransformException ex)
        {
          ROS_WARN("%s", ex.what());
          return;
        }
      }
    }

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
    if (!mosaicProcessor_->process(cv_ptr->image))
    {
      first_tf_ = true;
      ROS_ERROR("Cannot find pose. Skipping!");
      return;
    }
    ROS_INFO_STREAM("Found pose with " << mosaicProcessor_->getNumInliers() << " inliers.");

    // publish result
    tf::Transform transform = mosaicProcessor_->getTransformation();

    // Convert to base link
    transform = camera2base_.inverse() * transform;

    // Check the distance between previous transform
    double diff = tfDiff(transform, prev_transform_);
    if (diff > max_tf_diff_ && !first_tf_)
    {
      first_tf_ = false;
      ROS_WARN("TF between current and previous image to large. Skipping...");
      return;
    }
    first_tf_ = false;
    prev_transform_ = transform;

    static bool first_run(true);
    if (first_run)
    {
      first_run = false;
      if (reset_origin_)
      {
        initial_pose_ = transform;
      }
    }
    transform = initial_pose_.inverse() * transform;
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

  double tfDiff(tf::Transform a, tf::Transform b)
  {
    return sqrt( (a.getOrigin().x() - b.getOrigin().x())*(a.getOrigin().x() - b.getOrigin().x()) +
                 (a.getOrigin().y() - b.getOrigin().y())*(a.getOrigin().y() - b.getOrigin().y()) +
                 (a.getOrigin().z() - b.getOrigin().z())*(a.getOrigin().z() - b.getOrigin().z()) );
  }

  bool reset_origin_;
  bool is_tf_init_;
  bool first_tf_;
  double max_tf_diff_;
  tf::StampedTransform camera2base_;
  std::string base_frame_id_;
  tf::Transform initial_pose_;
  ros::Publisher posePub_;
  ros::Publisher odomPub_;
  image_transport::Publisher matchesImgPub_;
  image_transport::CameraSubscriber cam_sub_;
  tf::TransformBroadcaster tfBroadcaster_;
  tf::TransformListener tf_listener_;
  tf::Transform prev_transform_;

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


