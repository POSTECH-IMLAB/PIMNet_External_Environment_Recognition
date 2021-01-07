#ifndef _BGM_TL_DETECTOR_EM_HPP_
#define _BGM_TL_DETECTOR_EM_HPP_

#include <opencv2/core/core.hpp>

#include <vector>

namespace bgm
{

class TLDetectorEM
{
  public:
    void Detect(const cv::Mat& img,
                std::vector<cv::Rect>& result);
};

}
#endif // _BGM_TL_DETECTOR_EM_HPP_