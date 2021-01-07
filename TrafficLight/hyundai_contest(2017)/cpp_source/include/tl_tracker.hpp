#ifndef _BGM_TL_TRACKER_HPP_
#define _BGM_TL_TRACKER_HPP_

#include "tl_define.hpp"
#include "tl_detector.hpp"
#include "tl_classifier.hpp"
#include "tl_detector_em.hpp"

#include <opencv2/core/core.hpp>

#include <vector>
#include <list>

namespace bgm
{

class TLTracker
{
  enum Status {NO_DETECTION, TRACKING};
  struct TL
  {
    cv::Rect area;
    std::vector<TLStatus> history;
  };

  public:
    TLTracker(int detection_period);
    TLStatus Track(const cv::Mat& img);

  protected:
    void Detect(const cv::Mat& img, std::vector<cv::Rect>& locations);
  private:
    
    cv::Rect GetNextSearchingArea(const cv::Rect& tl_area,
                                  int img_width, int img_height) const;
    cv::Rect GetNearestDetection(const cv::Rect& prev_location,
                                 const std::vector<cv::Rect>& detection) const;
    TLStatus GetSceneStatus();
    TLStatus GetTLStatus(const TL& tl) const;
    bool IsOut(const cv::Rect& prev_location,
               int img_width, int img_height) const;

    TLDetector detector_;
    //TLDetectorEM detector_;
    TLClassifier classifier_;
    int detection_period_;
    int detection_cnt_;
    Status status_;
    std::list<TL> tl_;
    TLStatus prev_scene_status_;
    std::list<TLStatus> history_;
    //TLStatus prev_sequence_status_;
    bool miss_;

};

} // namespace bgm
#endif // !_BGM_TL_TRACKER_HPP_
