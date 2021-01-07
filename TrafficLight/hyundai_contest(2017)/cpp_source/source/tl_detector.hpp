#ifndef _BGM_TL_DETECTOR_HPP_
#define _BGM_TL_DETECTOR_HPP_

#include <opencv2/core/core.hpp>

#include <vector>

namespace bgm
{

class TLDetector {
  public:
    TLDetector();
    ~TLDetector();
    void Detect(const cv::Mat& img,
                std::vector<cv::Rect>& detection);

  private:
    void FitImage(const cv::Mat& img,
                  int& fit_width, int &fit_height);
    //void GetRawVector(const cv::Mat& img, int *w, int *h,
    //                  unsigned char *raw_vector) const;

    int buffer_size_;
    unsigned char *raw_vec_buffer_;
}; // class TLDetector

} // namespace bgm

#endif // !_BGM_TL_DETECTOR_HPP_
