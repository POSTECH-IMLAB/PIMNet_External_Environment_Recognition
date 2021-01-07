#include "tl_tracker.hpp"

namespace bgm
{

TLTracker::TLTracker(int detection_period) 
  : detection_period_(detection_period),
    detection_cnt_(0), prev_scene_status_(TLStatus::NONE),
    miss_(false) { }

TLStatus TLTracker::Track(const cv::Mat& img) {
  //std::cout << "cnt : " << detection_cnt_ << std::endl;
  int prev_size = tl_.size();
  if (detection_cnt_++ == 0) {
    //std::cout << "reset" << std::endl;
    tl_.clear();

    cv::Rect searching_area(0, 0, img.cols, img.rows * 2 / 5);
    std::vector<cv::Rect> detected;
    Detect(img(searching_area), detected);

    for (int i = 0; i < detected.size(); i++) {
      cv::Mat tl_patch(img(detected[i]));
      TLStatus color = classifier_.Classify(tl_patch);

      TL tl;
      tl.area = detected[i];
      tl.history.push_back(color);
      tl_.push_back(tl);
    }

    if (prev_size > 0 && tl_.size() == 0)
      miss_ = true;
    else
      miss_ = false;

    //if (prev_size > 0 && tl_.size() == 0)
    //  return prev_scene_status_;
  }
  else {
    detection_cnt_ %= detection_period_;

    std::list<TL>::iterator tl_itr = tl_.begin();
    while (tl_itr != tl_.end()) {
      cv::Rect prev_location = tl_itr->area;
      cv::Rect roi = GetNextSearchingArea(prev_location,
                                          img.cols, img.rows);
      cv::Mat patch = img(roi);
      //cv::resize(img(roi), patch,
      //           cv::Size(roi.width * 5, roi.height * 5));
      std::vector<cv::Rect> detected;
      Detect(patch, detected);

      if (detected.size() == 0 && 
          IsOut(prev_location, img.cols, img.rows)) {
        tl_.erase(tl_itr++);
      }
      else {
        cv::Rect next_location;
        if (detected.size() > 1) {
          next_location = GetNearestDetection(prev_location,
                                              detected);
          next_location.x += roi.x;
          next_location.y += roi.y;
        }
        else if (detected.size() == 1) {
          next_location = detected[0];
          next_location.x += roi.x;
          next_location.y += roi.y;
        }
        else
          next_location = prev_location;


        TLStatus color = classifier_.Classify(img(next_location));
        
        tl_itr->area = next_location;
        if (color == TLStatus::NONE && tl_itr->history.size() > 0)
          tl_itr->history.push_back(tl_itr->history.back());
        else
          tl_itr->history.push_back(color);

        tl_itr++;
      }

      //if (tl_.size() == 0)
      //  detection_cnt_ = 0;
    }
    //if (tl_.size() == 0 && detection_cnt_ > detection_period_)
    //  detection_cnt_ = 0;
    //else if (tl_.size() == 0)
    //  detection_cnt_ %= detection_period_;
  }

  cv::Mat show_img = img;
  std::list<TL>::iterator itr = tl_.begin();
  while (itr != tl_.end()) {
    cv::Scalar color(255, 255, 255);
    //cv::Scalar color;
    //if (itr->history.back() == TLStatus::GREEN)
    //  color = cv::Scalar(0, 255, 0);
    //else if (itr->history.back() == TLStatus::YELLOW)
    //  color = cv::Scalar(0, 255, 255);
    //else if (itr->history.back() == TLStatus::RED)
    //  color = cv::Scalar(0, 0, 255);

    cv::rectangle(show_img, itr->area, color, 2);
    itr++;
  }

  return GetSceneStatus();
}

void TLTracker::Detect(const cv::Mat& img,
                       std::vector<cv::Rect>& locations) {
  detector_.Detect(img, locations);
}

cv::Rect TLTracker::GetNextSearchingArea(const cv::Rect& tl_area,
                                         int img_width, int img_height) const {
  //int border_h = tl_area.height * 1.5;
  //int border_w = tl_area.width;

  //int x = std::max(0, tl_area.x - border_w);
  //int y = std::max(0, tl_area.y - border_h);
  //int width = tl_area.width + border_w * 2;
  //if (x + width - 1 > img_width)
  //  width = img_width - x;
  //int height = tl_area.height + border_h * 2;
  //if (y + height - 1 > img_height)
  //  height = img_height - y;

  //cv::Rect next(x, y, width, height);
  const int W = 200, H = 100;
  int x = std::max(0, tl_area.x + (tl_area.width / 2) - (W / 2));
  int y = std::max(0, tl_area.y + (tl_area.height / 2) - (H / 2));
  int width = std::min(img_width - x, W);
  int height = std::min(img_height - y, H);
  return cv::Rect(x, y, width, height);
}

cv::Rect TLTracker::GetNearestDetection(
  const cv::Rect& prev_location,
  const std::vector<cv::Rect>& detection) const {

  cv::Point pt1 = prev_location.tl();
  int nearest_index = 0;
  int nearest_distance = INT_MAX;
  for (int i = 0; i < detection.size(); i++) {
    cv::Point pt2 = detection[i].tl();
    int distance = std::pow(pt1.x - pt2.x, 2) + std::pow(pt1.y - pt2.y, 2);
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest_index = i;
    }
  }

  return detection[nearest_index];
}

bool TLTracker::IsOut(const cv::Rect& prev_location,
                      int img_width, int img_height) const {
  int col_border = img_width / 20;
  int row_border = img_height / 20;

  if (prev_location.x < col_border ||
      prev_location.x + prev_location.width > img_width ||
      prev_location.y < row_border ||
      prev_location.y + prev_location.height > img_height)
    return true;
  else
    return false;
}

TLStatus TLTracker::GetSceneStatus() {
  std::vector<int> vote(6, 0);

  if (tl_.size() > 0) {
    std::list<TL>::const_iterator itr = tl_.cbegin();
    while (itr != tl_.cend()) {
      vote[GetTLStatus(*itr)]++;
      itr++;
    }

    TLStatus scene_status;
    int max = (std::max_element(vote.begin(), vote.end())) - vote.cbegin();
    switch (max) {
      case TLStatus::RED:
        scene_status = TLStatus::RED;
        break;
      case TLStatus::YELLOW:
        scene_status = TLStatus::YELLOW;
        break;
      case TLStatus::GREEN:
        scene_status = TLStatus::GREEN;
        break;
      case TLStatus::RED_LEFT:
        scene_status = TLStatus::RED_LEFT;
        break;
      case TLStatus::GREEN_LEFT:
        scene_status = TLStatus::GREEN_LEFT;
        break;
      default:
        scene_status = TLStatus::NONE;
        break;
    }

    if (prev_scene_status_ == TLStatus::RED &&
        scene_status != TLStatus::GREEN &&
        scene_status != TLStatus::NONE)
      scene_status = TLStatus::RED;
    else if (prev_scene_status_ == TLStatus::YELLOW &&
        scene_status != TLStatus::RED &&
        scene_status != TLStatus::NONE)
      scene_status = TLStatus::YELLOW;
    else if (prev_scene_status_ == TLStatus::GREEN &&
        scene_status != TLStatus::YELLOW &&
        scene_status != TLStatus::NONE)
      scene_status = TLStatus::GREEN;

    prev_scene_status_ = scene_status;
  }
  else if(miss_ == false)
    prev_scene_status_ = TLStatus::NONE;

  

  return prev_scene_status_;
}

TLStatus TLTracker::GetTLStatus(const TL& tl) const {
  //int r_cnt = 0, y_cnt = 0, g_cnt = 0, rl_cnt = 0, lg_cnt = 0;
  std::vector<int> vote(6, 0);
  int start_idx = std::max((int)tl.history.size() - 3, 0);
  for (int i = start_idx; i < tl.history.size(); i++)
    vote[tl.history[i]]++;
  std::vector<int>::iterator max_iter =
    std::max_element(vote.begin(), vote.end());

  int max = max_iter - vote.begin();
  switch (max) {
    case TLStatus::RED:
      return TLStatus::RED;
    case TLStatus::YELLOW:
      return TLStatus::YELLOW;
    case TLStatus::GREEN:
      return TLStatus::GREEN;
    case TLStatus::RED_LEFT:
      return TLStatus::RED_LEFT;
    case TLStatus::GREEN_LEFT:
      return TLStatus::GREEN_LEFT;
    default:
      return TLStatus::NONE;
  }
}

} // namespace bgm