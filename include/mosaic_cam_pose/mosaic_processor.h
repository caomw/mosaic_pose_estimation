/// Copyright (c) 2012,
/// Systems, Robotics and Vision Group
/// University of the Balearican Islands
/// All rights reserved.
/// 
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
///     * Neither the name of Systems, Robotics and Vision Group, University of 
///       the Balearican Islands nor the names of its contributors may be used 
///       to endorse or promote products derived from this software without 
///       specific prior written permission.
/// 
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
/// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
/// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
/// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
/// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
/// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
/// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef MOSAIC_PROCESSOR_H
#define MOSAIC_PROCESSOR_H

#include <image_transport/image_transport.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <tf/transform_broadcaster.h>
#include <ros/ros.h>


#define MOSAIC_PX_METER     3657.6 //number of pixels in one meter girona
//#define MOSAIC_PX_METER 508.54 //number of pixels in one meter uib

#define MAX_SPEED             0.2

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
    MosaicProcessor(Parameters& param, std::string& transport);   

    ~MosaicProcessor(){};    
    void cameraCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info);

  protected:

  private:
    bool first_run_;
    bool useExtrinsicGuess_;
    int matcherFilterType_;
    ros::Publisher posePub_;
    ros::Publisher odomPub_;
    image_transport::Publisher matchesImgPub_;
    image_transport::CameraSubscriber cam_sub_;
    tf::TransformBroadcaster tfBroadcaster_;
    tf::StampedTransform previousPose_;
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

#endif

