
#include <fstream>
#include <stdexcept>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/camera_subscriber.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>
#include <bag_tools/camera_bag_processor.h>
#include <image_proc/processor.h>
#include "mosaic_pose_estimation/mosaic_processor.h"


namespace enc = sensor_msgs::image_encodings;

class MosaicPoseEstimator
{
public:
  MosaicPoseEstimator(const std::string& config_file, std::ostream& output):
    output_stream_(output)
  {
    MosaicProcessor::Parameters p;

    std::ifstream fin(config_file.c_str());
    YAML::Parser parser(fin);
    YAML::Node doc;
    parser.GetNextDocument(doc);
    doc["mosaic_image"] >> p.mosaicImgName;
    doc["px_per_meter"] >> p.pxPerMeter;
    doc["feature_detector_type"] >> p.featureDetectorType;
    doc["descriptor_extractor_type"] >> p.descriptorExtractorType;
    doc["descriptor_matcher_type"] >> p.descriptorMatcherType;
    doc["matcher_filter_type"] >> p.matcherFilterName;
    doc["matching_threshold"] >> p.matching_threshold;
    doc["ransac_reprojection_threshold"] >> p.ransacReprojThreshold;
    doc["min_num_inliers"] >> p.minNumInliers;
    doc["show_image"] >> show_image_;
    doc["reset_origin"] >> reset_origin_;

    std::cout << "Parameters:" << std::endl << p << std::endl;
    mosaic_processor_ = boost::shared_ptr<MosaicProcessor>(
        new MosaicProcessor(p));

    window_name_ = "Matches";

    output_stream_ << "# timestamp x y z qx qy qz qw num_features num_matches num_inliers"  << std::endl;

    if (show_image_)
    {
      cv::namedWindow(window_name_, CV_WINDOW_NORMAL);
    }
  }

  ~MosaicPoseEstimator()
  {
    if (show_image_)
      cv::destroyWindow(window_name_);
  }

  void process(const sensor_msgs::ImageConstPtr& img, 
      const sensor_msgs::CameraInfoConstPtr& info)
  {
    image_proc::ImageSet preprocessed;
    image_geometry::PinholeCameraModel camera_model;
    camera_model.fromCameraInfo(info);
    if (!processor_.process(img, camera_model, preprocessed, 
          image_proc::Processor::RECT_COLOR))
    {
      std::cerr << "ERROR Processing image" << std::endl;
      return;
    }

    mosaic_processor_->setCameraInfo(info);
    bool found_pose = mosaic_processor_->process(preprocessed.rect_color);
    if (!found_pose)
    {
      std::cerr << "ERROR finding pose, found " << mosaic_processor_->getNumInliers() << " inliers. Skipping." << std::endl;
    }
    else
    {
      tf::Transform transform = mosaic_processor_->getTransformation();

      static bool first_run = true;
      if (first_run)
      {
        first_run = false;
        if (reset_origin_)
        {
          initial_pose_ = transform;
        }
      }

      // re-base transform to initial pose
      transform = initial_pose_.inverse() * transform;
      geometry_msgs::Pose pose_msg;
      tf::poseTFToMsg(transform, pose_msg);
      std::ostringstream ostr;
      output_stream_ << img->header.stamp << " " 
        << pose_msg.position.x << " "
        << pose_msg.position.y << " " 
        << pose_msg.position.z << " "
        << pose_msg.orientation.x << " " 
        << pose_msg.orientation.y << " "
        << pose_msg.orientation.z << " " 
        << pose_msg.orientation.w << " "
        << mosaic_processor_->getNumFeatures() << " "
        << mosaic_processor_->getNumMatches() << " "
        << mosaic_processor_->getNumInliers() << std::endl;
    }

    if (show_image_)
    {
      cv::Mat draw_img = mosaic_processor_->drawMatches();
      cv::resize(draw_img, draw_img, cv::Size(draw_img.cols * 1000 / draw_img.rows,1000));
      cv::imshow(window_name_, draw_img);
      cv::waitKey(5);
    }
  }

private:

  image_proc::Processor processor_;
  boost::shared_ptr<MosaicProcessor> mosaic_processor_;
  bool show_image_;
  bool reset_origin_;
  std::string window_name_;
  std::ostream& output_stream_;
  tf::Transform initial_pose_;

};

int main(int argc, char** argv)
{

  if (argc < 4)
  {
    std::cout << "Usage: " << argv[0] << " CONFIG_FILE CAMERA_TOPIC OUTPUT_FILE BAGFILE [BAGFILE...]" << std::endl;
    std::cout << "  Example: " << argv[0] << " params.yaml /camera data.txt bag1.bag bag2.bag" << std::endl;
    return 0;
  }

  std::string config_file(argv[1]);
  std::string camera_topic(argv[2]);

  std::ofstream output(argv[3]);
  if (!output.is_open())
  {
    std::cerr << "ERROR: cannot open " << argv[3] << " for writing." << std::endl;
    return -1;
  }
  std::cout << "Writing data to " << argv[3] << std::endl;
  
  cv::initModule_nonfree();
  ros::Time::init();

  MosaicPoseEstimator estimator(config_file, output);
  bag_tools::CameraBagProcessor bag_processor(camera_topic);
  
  bag_processor.registerCallback(
      boost::bind(&MosaicPoseEstimator::process, estimator, _1, _2));

  for (int i = 4; i < argc; ++i)
  {
    std::cout << "Processing bagfile: " << argv[i] << std::endl;
    bag_processor.processBag(argv[i]);
  }

  output.close();

  return 0;
}


