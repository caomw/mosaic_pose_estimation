#ifndef MOSAIC_PROCESSOR_H
#define MOSAIC_PROCESSOR_H
#endif

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <tf/transform_broadcaster.h>

#define MOSAIC_PX_METER 2000.0 //number of pixels in one meter

class MosaicProcessorHeader {
  public:
    struct Parameters{
      std::string mosaicImgName;
      std::string featureDetectorType;
      std::string descriptorExtractorType;
      std::string descriptorMatcherType;
      std::string matcherFilterName;
      int matcherFilterType;
      double ransacReprojThreshold;
    }parameters;

};

class MosaicProcessor : public MosaicProcessorHeader {
  public:
    cv::Mat mosaicImg;
    cv::Mat frameImg;   
    MosaicProcessor(Parameters param, std::string transport);   

    ~MosaicProcessor();    
    void cameraCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info);

  protected:

  private:
    ros::Publisher posePub_;
    tf::TransformBroadcaster tfBroadcaster_;
    image_transport::CameraSubscriber cam_sub_;
    std::vector<cv::DMatch> filteredMatches_;
    int matcherFilterType_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoefficients_;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher_;
    std::vector<cv::KeyPoint> keypointsMosaic_;
    std::vector<cv::KeyPoint> keypointsFrame_;
    std::vector<cv::Point3f> pointsMosaic3D_;
    std::vector<cv::Point2f> pointsFrame2D_;
    std::vector<cv::Point2f> pointsMosaic_;
    std::vector<cv::Point2f> pointsFrame_;;
    cv::Mat descriptorsMosaic_;
    cv::Mat descriptorsFrame_;
    cv::Mat H12_,H21_;
    int getMatcherFilterType(const std::string& str);
    void simpleMatching(cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
        const cv::Mat& descriptors1, const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& matches12);
    void crossCheckMatching(cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
        const cv::Mat& descriptors1, const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& filteredMatches12, int knn);
    void publishTransform(const cv::Mat& tvec, const cv::Mat& rvec, const ros::Time& stamp, const std::string& camera_frame_id);
};

