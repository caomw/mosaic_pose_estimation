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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <tf/transform_datatypes.h>
#include <sensor_msgs/CameraInfo.h>


#define MOSAIC_PX_METER     3657.6 //number of pixels in one meter girona
//#define MOSAIC_PX_METER 508.54 //number of pixels in one meter uib

#define MAX_SPEED             0.2

class MosaicProcessor
{

public:
  struct Parameters 
  {
    std::string mosaicImgName;
    std::string featureDetectorType;
    std::string descriptorExtractorType;
    std::string descriptorMatcherType;
    std::string matcherFilterName;
    int matcherFilterType;
    double matching_threshold;
    double ransacReprojThreshold;
    double pxPerMeter;
  };

  MosaicProcessor(const Parameters& params);

  /**
   * Updates the camera info used for PnP solving
   */
  void setCameraInfo(const sensor_msgs::CameraInfoConstPtr& cam_info);

  /**
   * Processes one image, use the getters to retrieve information
   * about the result.
   * @return true if solution found, false otherwise (e.g. too few matches)
   */
  bool process(const cv::Mat& image);

  /**
   * @return the current transformation from mosaic origin to camera
   */
  tf::Transform getTransformation();

  inline int getNumFeatures() const
  {
    return keypointsFrame_.size();
  }

  inline int getNumMatches() const
  {
    return filteredMatches_.size();
  }

  inline int getNumInliers() const
  {
    return inliers_.size();
  }


  /**
   * Draws all matches with current and reference image in a big
   * canvas.
   * @return Nice image with rendered matches
   */
  cv::Mat drawMatches();

private:

  cv::Mat currentImage_;
  cv::Mat mosaicImage_;
  cv::Mat cameraMatrix_;
  std::vector<int> inliers_;
  std::vector<char> matchesMask_;
  bool first_run_;
  bool useExtrinsicGuess_;
  int matcherFilterType_;
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
  Parameters params_;
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
};


std::ostream& operator<<(std::ostream& ostr,
    const MosaicProcessor::Parameters& params);

#endif

