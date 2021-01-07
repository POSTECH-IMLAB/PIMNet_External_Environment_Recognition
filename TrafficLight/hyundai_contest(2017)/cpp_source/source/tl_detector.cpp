#include "tl_detector.hpp"

#include "extern/ACF/ParamTL2.h"
#include "extern/ACF/ACFNative.h"

namespace bgm
{

TLDetector::TLDetector()
  : buffer_size_(0), raw_vec_buffer_(NULL)
{ }

TLDetector::~TLDetector() {
  if (raw_vec_buffer_)
    delete raw_vec_buffer_;
}

void TLDetector::Detect(const cv::Mat& img,
                        std::vector<cv::Rect>& detection) {
  detection.clear();

  int fit_width, fit_height;
  FitImage(img, fit_width, fit_height);

  ACFNative_Boxes* bbs = ACFNative_Run(raw_vec_buffer_, 
                                       fit_width, fit_height);
  if (bbs->nbrOfBoxes > 0) {
    detection.resize(bbs->nbrOfBoxes);
    for (int i = 0; i < bbs->nbrOfBoxes; i++) {
      const ACFNative_Box& bbox = bbs->boxes[i];
      detection[i] = cv::Rect(bbox.x, bbox.y,
                              bbox.width, bbox.height);
    }
  }
}

void TLDetector::FitImage(const cv::Mat& img,
                          int& fit_width, int &fit_height) {
  fit_width = img.cols - (img.rows % (int)shrink);
  fit_height = img.rows - (img.rows % (int)shrink);
  int fit_size = fit_width * fit_height * 3;

  if (buffer_size_ < fit_size) {
    if (raw_vec_buffer_ == NULL)
      delete raw_vec_buffer_;
    raw_vec_buffer_ = new unsigned char[fit_size];
    buffer_size_ = fit_size;
  }

  cv::Mat fit_mat(fit_height, fit_width,
                  CV_8UC3, raw_vec_buffer_);
  cv::Rect fit_rect(0, 0, fit_width, fit_height);
  img(fit_rect).copyTo(fit_mat);
  assert(fit_mat.data == raw_vec_buffer_);  
}


} // namespace bgm