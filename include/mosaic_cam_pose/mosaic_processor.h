#ifndef MOSAIC_PROCESSOR_H
#define MOSAIC_PROCESSOR_H
#endif

#include <image_transport/image_transport.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <tf/transform_broadcaster.h>
#include <ros/ros.h>


#define MOSAIC_PX_METER 3657.6 //number of pixels in one meter

class MosaicProcessorHeader {
  public:
    struct Parameters{
      std::string mosaicImgName;
      std::string featureDetectorType;
      std::string descriptorExtractorType;
      std::string descriptorMatcherType;
      std::string matcherFilterName;
      int matcherFilterType;
      double matching_threshold;
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
    bool first_run_;
    bool useExtrinsicGuess_;
    int matcherFilterType_;
    ros::Publisher posePub_;
    ros::Publisher odomPub_;
    image_transport::CameraSubscriber cam_sub_;
    tf::TransformBroadcaster tfBroadcaster_;
    std::vector<cv::DMatch> filteredMatches_;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptorExtractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptorMatcher_;
    std::vector<cv::KeyPoint> keypointsMosaic_;
    std::vector<cv::KeyPoint> keypointsFrame_;
    std::vector<cv::Point3f> pointsMosaic3D_;
    std::vector<cv::Point2f> pointsMosaic_;
    std::vector<cv::Point2f> pointsFrame_;;
    cv::Mat descriptorsMosaic_;
    cv::Mat descriptorsFrame_;
    cv::Mat rvec_;
    cv::Mat tvec_;
    int getMatcherFilterType(const std::string& str);
    void simpleMatching(cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
        const cv::Mat& descriptors1, const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& matches12);
    void crossCheckMatching(cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
        const cv::Mat& descriptors1, const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& filteredMatches12, int knn);
    void thresholdMatching( cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
        const cv::Mat& descriptors1, const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& filteredMatches12, double matching_threshold);
    void publishTransform(const cv::Mat& tvec, const cv::Mat& rvec, const ros::Time& stamp, const std::string& camera_frame_id);
};

