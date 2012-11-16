
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
    outputStream_(output)
  {
    MosaicProcessor::Parameters p;

    std::ifstream fin(config_file.c_str());
    YAML::Parser parser(fin);
    YAML::Node doc;
    parser.GetNextDocument(doc);
    doc["mosaicImgName"] >> p.mosaicImgName;
    doc["pxPerMeter"] >> p.pxPerMeter;
    doc["featureDetectorType"] >> p.featureDetectorType;
    doc["descriptorExtractorType"] >> p.descriptorExtractorType;
    doc["descriptorMatcherType"] >> p.descriptorMatcherType;
    doc["matcherFilterName"] >> p.matcherFilterName;
    doc["matching_threshold"] >> p.matching_threshold;
    doc["ransacReprojThreshold"] >> p.ransacReprojThreshold;
    doc["showImage"] >> showImage_;

    std::cout << "Parameters:" << std::endl << p << std::endl;
    mosaicProcessor_ = boost::shared_ptr<MosaicProcessor>(
        new MosaicProcessor(p));

    windowName_ = "Matches";

    outputStream_ << "# timestamp x y z qx qy qz qw num_features num_matches num_inliers"  << std::endl;

    if (showImage_)
    {
      cv::namedWindow(windowName_, CV_WINDOW_NORMAL);
    }
  }

  ~MosaicPoseEstimator()
  {
    if (showImage_)
      cv::destroyWindow(windowName_);
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

    mosaicProcessor_->setCameraInfo(info);
    mosaicProcessor_->process(preprocessed.rect_color);

    tf::Transform transform = mosaicProcessor_->getTransformation();

    geometry_msgs::Pose pose_msg;
    tf::poseTFToMsg(transform.inverse(), pose_msg);
    std::ostringstream ostr;
    outputStream_ << img->header.stamp << " " 
      << pose_msg.position.x << " "
      << pose_msg.position.y << " " 
      << pose_msg.position.z << " "
      << pose_msg.orientation.x << " " 
      << pose_msg.orientation.y << " "
      << pose_msg.orientation.z << " " 
      << pose_msg.orientation.w << " "
      << mosaicProcessor_->getNumFeatures() << " "
      << mosaicProcessor_->getNumMatches() << " "
      << mosaicProcessor_->getNumInliers() << std::endl;

    if (showImage_)
    {
      cv::Mat drawImg = mosaicProcessor_->drawMatches();
      cv::resize(drawImg, drawImg, cv::Size(drawImg.cols * 1000 / drawImg.rows,1000));
      cv::imshow(windowName_, drawImg);
      cv::waitKey(5);
    }
  }

private:

  image_proc::Processor processor_;
  boost::shared_ptr<MosaicProcessor> mosaicProcessor_;
  bool showImage_;
  std::string windowName_;

  std::ostream& outputStream_;

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


