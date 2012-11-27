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

#include "mosaic_pose_estimation/mosaic_processor.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>


//const std::string winName = "Correspondences";
enum { NONE_FILTER = 0, CROSS_CHECK_FILTER = 1, DISTANCE_FILTER = 2};

/** @function MosaicProcessor */
MosaicProcessor::MosaicProcessor(const Parameters& p) : params_(p)
{
  params_.matcherFilterType = getMatcherFilterType(params_.matcherFilterName );
  detector_ = cv::FeatureDetector::create(params_.featureDetectorType);
  descriptorExtractor_ = 
    cv::DescriptorExtractor::create(params_.descriptorExtractorType );
  descriptorMatcher_ = cv::DescriptorMatcher::create( params_.descriptorMatcherType );

  if(detector_.empty() || descriptorExtractor_.empty() || 
      descriptorMatcher_.empty()  )
  {
    ROS_ERROR("Can not create detector or descriptor extractor or "
              "descriptor matcher of given types");
  }

  ROS_INFO_STREAM("Reading the mosaic image: " << params_.mosaicImgName);
  mosaicImage_ = cv::imread(params_.mosaicImgName);

  if(mosaicImage_.empty())
  {
    ROS_ERROR("Mosaic image is empty");
  }

  ROS_INFO("Extracting keypoints from mosaic image...");
  detector_->detect(mosaicImage_, keypointsMosaic_ );

  ROS_INFO("Computing descriptors for keypoints from mosaic...");
  descriptorExtractor_->compute(mosaicImage_, keypointsMosaic_, 
      descriptorsMosaic_ );
  ROS_INFO_STREAM(keypointsMosaic_.size() << " features.");

  ROS_INFO("Computing world coordinates...");
  cv::KeyPoint::convert(keypointsMosaic_, pointsMosaic_);
  pointsMosaic3D_.resize(pointsMosaic_.size());
  for (size_t i_mos=0;i_mos<pointsMosaic_.size();i_mos++) {
    pointsMosaic3D_[i_mos].x = 
      pointsMosaic_[i_mos].x/params_.pxPerMeter;
    pointsMosaic3D_[i_mos].y = 
      pointsMosaic_[i_mos].y/params_.pxPerMeter;
    pointsMosaic3D_[i_mos].z = 0;
  }
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
void MosaicProcessor::simpleMatching(
    cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches12 )
{
  std::vector<cv::DMatch> matches;
  descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

/** @function crossCheckMatching */
void MosaicProcessor::crossCheckMatching(
    cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& filteredMatches12, int knn=1 )
{
  filteredMatches12.clear();
  std::vector<std::vector<cv::DMatch> > matches12, matches21;
  descriptorMatcher->knnMatch(descriptors1, descriptors2, matches12, knn);
  descriptorMatcher->knnMatch(descriptors2, descriptors1, matches21, knn);
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

void MosaicProcessor::thresholdMatching(
    cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& filteredMatches12, double matching_threshold)
{
  filteredMatches12.clear();
  std::vector<std::vector<cv::DMatch> > matches12;
  int knn = 2;
  descriptorMatcher->knnMatch(descriptors1, descriptors2, matches12, knn);
  for( size_t m = 0; m < matches12.size(); m++ )
  {
    if (matches12[m].size() == 1)
    {
      filteredMatches12.push_back(matches12[m][0]);
    }
    else if (matches12[m].size() == 2) // normal case
    {
      if (matches12[m][0].distance / matches12[m][1].distance 
          < matching_threshold)
      {
        filteredMatches12.push_back(matches12[m][0]);
      }
    }
  }
}

void MosaicProcessor::setCameraInfo(
    const sensor_msgs::CameraInfoConstPtr& cam_info)
{
  const cv::Mat P(3,4, CV_64FC1, const_cast<double*>(cam_info->P.data()));
  // We have to take K' here extracted from P to take the R|t into i
  // account // that was performed during rectification.
  // This way we obtain the pattern pose with respect to the same 
  // frame that is used in stereo depth calculation.
  cameraMatrix_ = P.colRange(cv::Range(0,3)).clone();
}

bool MosaicProcessor::process(const cv::Mat& image)
{
  currentImage_ = image;
  assert(!mosaicImage_.empty());
  assert(!currentImage_.empty());

  //equalize histogram:
  /*
  assert(currentImage_.type() == CV_8UC3);
  std::vector<cv::Mat> in(3),out(3);
  cv::split(currentImage_,in);
  for(int i=0;i<3;i++)
    cv::equalizeHist(in[i],out[i]);
  cv::merge(out,currentImage_);
  */

  detector_->detect(currentImage_, keypointsFrame_ );
  descriptorExtractor_->compute(currentImage_, keypointsFrame_, 
      descriptorsFrame_ );

  ROS_DEBUG("Matching descriptors...");
  switch( params_.matcherFilterType )
  {
    case CROSS_CHECK_FILTER :
      crossCheckMatching(descriptorMatcher_, descriptorsFrame_, 
          descriptorsMosaic_, filteredMatches_, 1);
      break;
    case DISTANCE_FILTER:
      thresholdMatching(descriptorMatcher_, descriptorsFrame_, 
          descriptorsMosaic_, filteredMatches_, 
          params_.matching_threshold);
      break;
    default :
      simpleMatching(descriptorMatcher_, descriptorsFrame_, 
          descriptorsMosaic_, filteredMatches_ );
      break;
  }

  // we need at least 5 matches for solvePnPRansac
  if (filteredMatches_.size() < 5)
    return false;

  std::vector<int> queryIdxs(filteredMatches_.size()), 
    trainIdxs(filteredMatches_.size());
  std::vector<cv::Point2f> image_points(filteredMatches_.size());
  std::vector<cv::Point3f> world_points(filteredMatches_.size());
  for( size_t i = 0; i < filteredMatches_.size(); i++ )
  {
    queryIdxs[i] = filteredMatches_[i].queryIdx;
    trainIdxs[i] = filteredMatches_[i].trainIdx;
    image_points[i] = keypointsFrame_[queryIdxs[i]].pt;
    world_points[i] = pointsMosaic3D_[trainIdxs[i]];
  }

  ROS_DEBUG("Trying to find the camera pose...");

  bool useExtrinsicGuess = true;
  if (rvec_.empty() || tvec_.empty())
    useExtrinsicGuess = false;

  int numIterations = 1000;
  float allowedReprojectionError = params_.ransacReprojThreshold;//8.0
  int maxInliers = 10000; // stop if more inliers than this are found
  cv::solvePnPRansac(world_points, image_points, cameraMatrix_, 
                     cv::Mat(), rvec_, tvec_, useExtrinsicGuess, 
                     numIterations, allowedReprojectionError, 
                     maxInliers, inliers_);
  matchesMask_.resize(filteredMatches_.size());
  std::fill(matchesMask_.begin(), matchesMask_.end(), 0);
  for (size_t i = 0; i < inliers_.size(); ++i)
  {
    matchesMask_[inliers_[i]] = 1;
  }

  if (static_cast<int>(inliers_.size()) >= params_.minNumInliers)
  {
    return true;
  }
  else
  {
    rvec_ = cv::Mat();
    tvec_ = cv::Mat();
    return false;
  }
}

cv::Mat MosaicProcessor::drawMatches()
{
  cv::Mat drawImg;
  cv::drawMatches(currentImage_, keypointsFrame_,
      mosaicImage_, keypointsMosaic_, filteredMatches_, drawImg,
      CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask_);

  std::vector<char> outliers(matchesMask_.size());
  for (size_t i = 0; i < outliers.size(); ++i)
    outliers[i] = !matchesMask_[i];

  cv::drawMatches(currentImage_, keypointsFrame_,
      mosaicImage_, keypointsMosaic_, filteredMatches_, drawImg,
      CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), outliers,
      cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | 
      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  return drawImg;
}

tf::Transform MosaicProcessor::getTransformation()
{
  if (rvec_.empty() || tvec_.empty())
    return tf::Transform();
  tf::Vector3 axis(rvec_.at<double>(0, 0), rvec_.at<double>(1, 0), 
      rvec_.at<double>(2, 0));
  double angle = cv::norm(rvec_);
  tf::Quaternion quaternion(axis, angle);

  tf::Vector3 translation(tvec_.at<double>(0, 0), tvec_.at<double>(1, 0), 
      tvec_.at<double>(2, 0));
  // solvePnPRansac calculates object pose, we want camera pose
  // for fixed object so we have to invert here
  return tf::Transform(quaternion, translation).inverse();
}


std::ostream& operator<<(std::ostream& out, 
    const MosaicProcessor::Parameters& params)
{
  out << "\t* Mosaic image name         = " 
    << params.mosaicImgName << std::endl;
  out << "\t* Pixels per meter          = " 
    << params.pxPerMeter << std::endl;
  out << "\t* Feature detector type     = " 
    << params.featureDetectorType << std::endl;
  out << "\t* Descriptor extractor type = " 
    << params.descriptorExtractorType << std::endl;
  out << "\t* Descriptor matcher type   = " 
    << params.descriptorMatcherType << std::endl;
  out << "\t* Matcher filter name       = " 
    << params.matcherFilterName << std::endl;
  out << "\t* Matcher filter threshold  = " 
    << params.matching_threshold << std::endl;
  out << "\t* Mininum number of inliers = " 
    << params.minNumInliers << std::endl;
  out << "\t* RANSAC reproj. threshold  = " 
    << params.ransacReprojThreshold;
  return out;
}


