#include "tl_classifier.hpp"
//#include "tl_colorhist_svm.h"
#include "tl_colorhist_svm3.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace bgm
{

void TLClassifier::GetColHistFeature(const cv::Mat& img,
                                     std::vector<float>& feature) {
  cv::Mat src = img;
  cv::Rect sal_roi = GetSalROI(src);
  feature.clear();
  GetColorHistogram(img(sal_roi), feature);
}

TLStatus TLClassifier::Classify(const cv::Mat& tl_patch) {
  cv::Mat src = tl_patch;
  cv::Rect sal_rect = GetSalROI(src);

  if (sal_rect.height < 5)
    return TLStatus::NONE;

  cv::Mat sal_roi = tl_patch(sal_rect);


  TLStatus tl_status = ClassifyRYGL(sal_roi);

  //if (tl_status != TLStatus::GREEN) {
  //  std::cout << "width : " << sal_roi.cols;
  //  std::cout << " height : " << sal_roi.rows << std::endl << std::endl;
  //}
  
  //if (tl_status == TLStatus::RED)
  //  return ClassifyRYG2(sal_roi);
  //else
  //  return tl_status;

  if (tl_status == TLStatus::RED)
    return ClassifyRRL(tl_patch, sal_rect);
  //else if (tl_status == TLStatus::GREEN)
  //  return ClassifyGRL(tl_patch, sal_rect);
  else
    return tl_status;
}

TLStatus TLClassifier::ClassifyRYGL(const cv::Mat& tl_patch) const {
  std::vector<float> color_hist;
  GetColorHistogram(tl_patch, color_hist);
  int HIST_SIZE = color_hist.size();

  std::vector<float>::const_iterator itr;
  int size;

  // g_svm
  itr = color_hist.cbegin();
  float g_conf = 0;
  float *g_w = g_weights;
  size = HIST_SIZE;
  while (size--)
    g_conf += (*g_w++) * (*itr++);
  g_conf += g_bias;

  //// l_svm
  //itr = color_hist.cbegin();
  //float l_conf = 0;
  //float *l_w = l_weights;
  //size = HIST_SIZE;
  //while (size--)
  //  l_conf += (*l_w++) * (*itr++);
  //l_conf += l_bias;

  // y_svm
  itr = color_hist.cbegin();
  float y_conf = 0;
  float *y_w = y_weights;
  size = HIST_SIZE;
  while (size--)
    y_conf += (*y_w++) * (*itr++);
  y_conf += y_bias;

  // r_svm
  itr = color_hist.cbegin();
  float r_conf = 0;
  float *r_w = r_weights;
  size = HIST_SIZE;
  while (size--)
    r_conf += (*r_w++) * (*itr++);
  r_conf += r_bias;
  
  //float conf[4] = {
  //  g_conf>0.1 ? g_conf:0,
  //  l_conf>0.1 ? l_conf:0,
  //  y_conf>0.1 ? l_conf:0,
  //  r_conf>0.1 ? r_conf:0};
  float conf[4] = {g_conf, /*l_conf*/ -2.0f, y_conf, 
                   r_conf > 0.1 ? r_conf:-2.0f};
  int max_idx = std::max_element(conf, conf + 4) - conf;
  TLStatus tl_status;
  switch (max_idx) {
    case 0:
      return TLStatus::GREEN;
    case 1:
      return TLStatus::RED_LEFT;
    case 2:
      return TLStatus::YELLOW;
    case 3:
      //return TLStatus::RED;
      return ClassifyRYG2(tl_patch);
    default:
      return TLStatus::NONE;
  }
}

TLStatus TLClassifier::ClassifyRYG2(const cv::Mat& tl_patch) const {
  int h_unit = tl_patch.rows / 3;
  cv::Mat mid_roi;
  if (h_unit != 0) {
    cv::Rect mid_rect(0, h_unit, tl_patch.cols, h_unit);
    mid_roi = tl_patch(mid_rect);
  }
  else
    mid_roi = tl_patch;


  cv::Mat rgb_max, bright;
  GetRGBMax(mid_roi, rgb_max);
  cv::threshold(rgb_max, bright, 0, 255, 
                CV_THRESH_BINARY | CV_THRESH_OTSU);

  cv::Mat hsv;
  cv::cvtColor(mid_roi, hsv, CV_BGR2HSV);
  std::vector<cv::Mat> hsv_plains;
  cv::split(hsv, hsv_plains);
  cv::Mat h = hsv_plains[0];
  //hsv_plains[0].convertTo(h, CV_32F);
  //h /= 255.0;

  int r = 0, y = 0;
  int size = h.rows * h.cols;
  unsigned char *h_data = h.data;
  unsigned char *bright_data = bright.data;
  int value, b;
  while (size--) {
    value = *h_data++;
    b = *bright_data++;
    if ((value < 15 || value > 162) && b)
      r++;
    else if ((value >= 15 && value < 33) && b)
      y++;
  }

  if (r + y < h.rows * h.rows * 0.8)
    return TLStatus::GREEN;
  else if (r > y)
    return TLStatus::RED;
  else
    return TLStatus::YELLOW;
}

TLStatus TLClassifier::ClassifyRRL(const cv::Mat& tl_patch,
                                   const cv::Rect& sal_roi) const {
  int width = sal_roi.height * 3;
  if (tl_patch.cols - sal_roi.x > width) {
    cv::Rect roi(sal_roi.x, sal_roi.y, width, sal_roi.height);
    if (IsRL(tl_patch(roi)))
      return TLStatus::RED_LEFT;
  }
  return TLStatus::RED;
}
TLStatus TLClassifier::ClassifyGRL(const cv::Mat& tl_patch,
                                   const cv::Rect& sal_roi) const {
  int width = sal_roi.height * 3;
  if (sal_roi.x + sal_roi.width - width > 0) {
    cv::Rect roi(sal_roi.x + sal_roi.width - width,
                 sal_roi.y, width, sal_roi.height);
    if (IsRL(tl_patch(roi)))
      return TLStatus::RED_LEFT;
  }
  return TLStatus::GREEN;
}

bool TLClassifier::IsRL(const cv::Mat& tl_patch) const {
  int width_unit = tl_patch.cols / 3;
  cv::Rect left_rect(0, 0, width_unit, tl_patch.rows);
  cv::Rect right_rect(width_unit * 2, 0, 
                      tl_patch.cols - width_unit * 2, 
                      tl_patch.rows);
  TLStatus left_status = ClassifyRYG2(tl_patch(left_rect));
  TLStatus right_status = ClassifyRYG2(tl_patch(right_rect));

  if (left_status == TLStatus::RED &&
      right_status == TLStatus::GREEN)
    return true;
  else
    return false;
}

cv::Rect TLClassifier::GetSalROI(const cv::Mat& tl_patch) {
  cv::Mat resized, blurred, sharpen;
  cv::resize(tl_patch, resized, cv::Size(200, 100));
  cv::GaussianBlur(resized, blurred, cv::Size(0, 0), 3);
  cv::addWeighted(resized, 1.5, blurred, -0.5, 0, sharpen);
  cv::Mat sal_map;
  GetSalMap(sharpen, sal_map);


  if (sharpen.cols == 200 && sharpen.rows == 100) {
    std::vector<int> optimal_threshs;
    MultiThresh3(sal_map, optimal_threshs);

    cv::Mat most_saliency;
    cv::threshold(sal_map, most_saliency, optimal_threshs[0],
                  255, CV_THRESH_BINARY);
    //cv::threshold(sal_map, most_saliency, 0, 255,
    //              CV_THRESH_BINARY | CV_THRESH_OTSU);
    //cv::threshold(sal_map, most_saliency, 10, 255,
    //              CV_THRESH_BINARY);
    cv::Mat bgr_max;
    GetRGBMax(sharpen, bgr_max);
    cv::Mat most_bgr_max;
    cv::threshold(bgr_max, most_bgr_max, 0, 255,
                  CV_THRESH_BINARY | CV_THRESH_OTSU);
    most_saliency &= most_bgr_max;

    //cv::resize(most_saliency, most_saliency,
    //           cv::Size(tl_patch.cols, tl_patch.rows));
    cv::Rect bounding_box = GetBoundingBox(most_saliency);
    bounding_box.x *= tl_patch.cols / 200.0f;
    bounding_box.width *= tl_patch.cols / 200.0f;
    bounding_box.y *= tl_patch.rows / 100.0f;
    bounding_box.height *= tl_patch.rows / 100.0f;
    if (bounding_box.width > 0 && bounding_box.height > 0)
      return bounding_box;
    else
      return cv::Rect(0, 0, tl_patch.cols, tl_patch.rows);
  }
  else
    return cv::Rect(0, 0, tl_patch.cols, tl_patch.rows);


}

void TLClassifier::GetSalMap(cv::Mat& src, cv::Mat& sal_map) {
  cv::Mat float_map = saliency_.GetSal(src);
  float_map *= 255;
  float_map.convertTo(sal_map, CV_8U);
}

void TLClassifier::MultiThresh2(const cv::Mat& gray, 
                                std::vector<int>& thresholds) const {
  cv::Mat hist;
  const float HIST_RANGE[2] = {0, 256};
  GetHistogram(gray, 256, HIST_RANGE, hist);
  const unsigned char *histogram = hist.data;
  //const int N = gray.rows * gray.cols - histogram[0];
  const int N = gray.rows * gray.cols;

  double W0K, W1K, W2K, M0, M1, M2, currVarB, optimalThresh1, optimalThresh2, maxBetweenVar, M0K, M1K, M2K, MT;

  optimalThresh1 = 0;
  optimalThresh2 = 0;

  W0K = 0;
  W1K = 0;

  M0K = 0;
  M1K = 0;

  MT = 0;
  maxBetweenVar = 0;
  for (int k = 0; k <= 255; k++) {
    MT += k * (histogram[k] / (double)N);
  }


  for (int t1 = 0; t1 <= 255; t1++) {
    W0K += histogram[t1] / (double)N; //Pi
    M0K += t1 * (histogram[t1] / (double)N); //i * Pi
    M0 = M0K / W0K; //(i * Pi)/Pi

    W1K = 0;
    M1K = 0;

    for (int t2 = t1 + 1; t2 <= 255; t2++) {
      W1K += histogram[t2] / (double)N; //Pi
      M1K += t2 * (histogram[t2] / (double)N); //i * Pi
      M1 = M1K / W1K; //(i * Pi)/Pi

      W2K = 1 - (W0K + W1K);
      M2K = MT - (M0K + M1K);

      if (W2K <= 0) break;

      M2 = M2K / W2K;

      currVarB = W0K * (M0 - MT) * (M0 - MT) + W1K * (M1 - MT) * (M1 - MT) + W2K * (M2 - MT) * (M2 - MT);

      if (maxBetweenVar < currVarB) {
        maxBetweenVar = currVarB;
        optimalThresh1 = t1;
        optimalThresh2 = t2;
      }
    }
  }

  thresholds.resize(2);
  thresholds[0] = optimalThresh1;
  thresholds[1] = optimalThresh2;
}

void TLClassifier::MultiThresh3(const cv::Mat& gray,
                                std::vector<int>& thresholds) const {
  cv::Mat hist;
  const float HIST_RANGE[2] = {0, 256};
  GetHistogram(gray, 256, HIST_RANGE, hist);
  const unsigned char *histogram = hist.data;
  //const int N = gray.rows * gray.cols - histogram[0];
  int N = gray.rows * gray.cols;

  double W0K, W1K, W2K, W3K, M0, M1, M2, M3, currVarB, maxBetweenVar, M0K, M1K, M2K, M3K, MT;
  double optimalThresh1, optimalThresh2, optimalThresh3;

  W0K = 0;
  W1K = 0;
  M0K = 0;
  M1K = 0;
  MT = 0;
  maxBetweenVar = 0;

  for (int k = 0; k <= 255; k++) {
    MT += k * (histogram[k] / (double)N);
  }

  for (int t1 = 0; t1 <= 255; t1++) {
    W0K += histogram[t1] / (double)N; //Pi
    M0K += t1 * (histogram[t1] / (double)N); //i * Pi
    M0 = M0K / W0K; //(i * Pi)/Pi

    W1K = 0;
    M1K = 0;

    for (int t2 = t1 + 1; t2 <= 255; t2++) {
      W1K += histogram[t2] / (double)N; //Pi
      M1K += t2 * (histogram[t2] / (double)N); //i * Pi
      M1 = M1K / W1K; //(i * Pi)/Pi
      W2K = 1 - (W0K + W1K);
      M2K = MT - (M0K + M1K);

      if (W2K <= 0) break;

      M2 = M2K / W2K;

      W2K = 0;
      M2K = 0;

      for (int t3 = t2 + 1; t3 <= 255; t3++) {
        W2K += histogram[t3] / (double)N; //Pi
        M2K += t3 * (histogram[t3] / (double)N); // i*Pi
        M2 = M2K / W2K; //(i*Pi)/Pi
        W3K = 1 - (W0K + W1K + W2K);
        M3K = MT - (M0K + M1K + M2K);

        M3 = M3K / W3K;
        currVarB = W0K * (M0 - MT) * (M0 - MT) + W1K * (M1 - MT) * (M1 - MT) + W2K * (M2 - MT) * (M2 - MT) + W3K * (M3 - MT) * (M3 - MT);

        if (maxBetweenVar < currVarB) {
          maxBetweenVar = currVarB;
          optimalThresh1 = t1;
          optimalThresh2 = t2;
          optimalThresh3 = t3;
        }
      }
    }
  }

  thresholds.resize(3);
  thresholds[0] = optimalThresh1;
  thresholds[1] = optimalThresh2;
  thresholds[2] = optimalThresh3;
}

cv::Rect TLClassifier::GetBoundingBox(const cv::Mat& bin_img) const {
  int left = bin_img.cols - 1, top = bin_img.rows - 1;
  int right = 0, bot = 0;
  
  for (int i = 0; i < bin_img.rows; i++) {
    const unsigned char *row_ptr = bin_img.ptr<unsigned char>(i);
    for (int j = 0; j < bin_img.cols; j++) {
      if (row_ptr[j]) {
        if (i < top) top = i;
        if (i > bot) bot = i;
        if (j < left) left = j;
        if (j > right) right = j;
      }
    }
  }

  cv::Rect bounding_box(left, top, right - left, bot - top);
  return bounding_box;
}

void TLClassifier::GetRGBMax(const cv::Mat& bgr_img, 
                             cv::Mat& max_img) const {
  max_img = cv::Mat(bgr_img.rows, bgr_img.cols, CV_8UC1);
  int size = bgr_img.rows * bgr_img.cols;

  std::vector<cv::Mat> bgr_plains;
  cv::split(bgr_img, bgr_plains);

  //unsigned char *bgr_data = bgr_img.data;
  unsigned char *b_data = bgr_plains[0].data;
  unsigned char *g_data = bgr_plains[1].data;
  unsigned char *r_data = bgr_plains[2].data;
  unsigned char *max_data = max_img.data;

  while (size--) {
    //unsigned char b = *bgr_data++;
    //unsigned char g = *bgr_data++;
    //unsigned char r = *bgr_data++;
    unsigned char b = *b_data++;
    unsigned char g = *g_data++;
    unsigned char r = *r_data++;

    *max_data++ = std::max(std::max(b, g), r);
  }
}

void TLClassifier::GetHistogram(const cv::Mat& gray_img,
                                int hist_size, 
                                const float* hist_range,
                                cv::Mat& histogram) const {
  const int size[1] = {hist_size};
  const float *range = {hist_range};
  const int channels[1] = {0};
  cv::calcHist(&gray_img, 1, channels, cv::Mat(),
               histogram, 1, size, &range, true, false);
}

void TLClassifier::ExtractUV(const cv::Mat& float_img,
                             cv::Mat& u, cv::Mat& v) const {
  std::vector<cv::Mat> bgr;
  cv::split(float_img, bgr);
  const cv::Mat& b_mat = bgr[0];
  const cv::Mat& g_mat = bgr[1];
  const cv::Mat& r_mat = bgr[2];

  u = cv::Mat(float_img.rows, float_img.cols, CV_32FC1);
  v = cv::Mat(float_img.rows, float_img.cols, CV_32FC1);

  int size = float_img.rows * float_img.cols;
  float *b_data = (float*)(b_mat.data);
  float *g_data = (float*)(g_mat.data);
  float *r_data = (float*)(r_mat.data);
  float *u_data = (float*)(u.data);
  float *v_data = (float*)(v.data);
  float b, g, r;
  while (size--) {
    b = *b_data++;
    g = *g_data++;
    r = *r_data++;

    *u_data++ = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    *v_data++ = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  }
}

void TLClassifier::GetColorHistogram(const cv::Mat& src,
                                     std::vector<float>& histogram) const {
  cv::Mat float_img;
  src.convertTo(float_img, CV_32F);
  float_img /= 255.0f;

  cv::Mat yuv, hsv;
  cv::cvtColor(float_img, yuv, CV_BGR2YUV);
  cv::cvtColor(float_img, hsv, CV_BGR2HSV);

  std::vector<cv::Mat> yuv_plains;
  cv::split(yuv, yuv_plains);
  cv::Mat u, v;
  //yuv_plains[1].convertTo(u, CV_32F);
  //yuv_plains[2].convertTo(v, CV_32F);
  ExtractUV(float_img, u, v);

  std::vector<cv::Mat> hsv_plains;
  cv::split(hsv, hsv_plains);
  cv::Mat h;
  hsv_plains[0].convertTo(h, CV_32F);
  h /= 360.0f;

  float range[2] = {0.0f, 1.0f};
  int hist_size = 72;
  cv::Mat u_hist, v_hist, h_hist;
  GetHistogram(u, hist_size, range, u_hist);
  GetHistogram(v, hist_size, range, v_hist);
  GetHistogram(h, hist_size, range, h_hist);
  //double min_val, max_val;
  //float range[2];
  //int hist_size = 72;
  //cv::Mat u_hist, v_hist, h_hist;
  //cv::minMaxLoc(u, &min_val, &max_val);
  //range[0] = min_val, range[1] = max_val;
  //GetHistogram(u, hist_size, range, u_hist);
  //cv::minMaxLoc(v, &min_val, &max_val);
  //range[0] = min_val, range[1] = max_val;
  //GetHistogram(v, hist_size, range, v_hist);
  //cv::minMaxLoc(h, &min_val, &max_val);
  //range[0] = min_val, range[1] = max_val;
  //GetHistogram(h, hist_size, range, h_hist);

  int size = src.cols * src.rows;
  u_hist /= size;
  v_hist /= size;
  h_hist /= size;

  histogram.clear();
  //histogram.insert(histogram.end(),
  //                 (float*)v_hist.data,
  //                 (float*)v_hist.data + hist_size);
  //histogram.insert(histogram.end(),
  //                 (float*)u_hist.data, 
  //                 (float*)u_hist.data + hist_size);
  histogram.insert(histogram.end(),
                   (float*)u_hist.data,
                   (float*)u_hist.data + hist_size);
  histogram.insert(histogram.end(),
                   (float*)v_hist.data,
                   (float*)v_hist.data + hist_size);
  histogram.insert(histogram.end(),
                   (float*)h_hist.data, 
                   (float*)h_hist.data + hist_size);
}

} // namespace bgm