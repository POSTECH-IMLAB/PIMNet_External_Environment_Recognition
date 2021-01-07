#ifndef _BGM_TL_CLASSIFIER_HPP_
#define _BGM_TL_CLASSIFIER_HPP_

#include "extern/GMR_saliency/Saliency/GMRsaliency.h"

#include "tl_define.hpp"

#include <opencv2/core/core.hpp>

#include <vector>

namespace bgm
{


class TLClassifier
{
  enum {MIN_WIDTH = 200, MAX_WIDTH = 100};

  public:
    void GetColHistFeature(const cv::Mat& img,
                           std::vector<float>& feature);

    TLStatus Classify(const cv::Mat& tl_patch);

  private:
    TLStatus ClassifyRYGL(const cv::Mat& tl_patch) const;
    TLStatus ClassifyRYG2(const cv::Mat& tl_patch) const;
    TLStatus ClassifyRRL(const cv::Mat& tl_patch,
                         const cv::Rect& sal_roi) const;
    TLStatus ClassifyGRL(const cv::Mat& tl_patch,
                         const cv::Rect& sal_roi) const;
    bool IsRL(const cv::Mat& tl_patch) const;
    //bool IsLG(const cv::Mat& tl_patch) const;

    cv::Rect GetSalROI(const cv::Mat& sal_map);
    void GetSalMap(cv::Mat& tl_patch, cv::Mat& sal_map);


    void MultiThresh2(const cv::Mat& gray,
                      std::vector<int>& thresholds) const;
    void MultiThresh3(const cv::Mat& gray,
                      std::vector<int>& thresholds) const;
    cv::Rect GetBoundingBox(const cv::Mat& bin_img) const;
    
    void GetRGBMax(const cv::Mat& rgb_img, 
                   cv::Mat& max_img) const;
    void GetHistogram(const cv::Mat& gray_img,
                      int hist_size, const float* hist_range,
                      cv::Mat& histogram) const;
    void ExtractUV(const cv::Mat& float_img,
                   cv::Mat& u, cv::Mat& v) const;

    void GetColorHistogram(const cv::Mat& src,
                           std::vector<float>& histogram) const;


    GMRsaliency saliency_;
};

}
#endif // !_BGM_TL_CLASSIFIER_HPP_

