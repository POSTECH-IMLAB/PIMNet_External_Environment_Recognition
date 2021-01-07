//#include <opencv_2.1/include/cv.h>
//#include <opencv_2.1/include/cv.hpp>
//#include <opencv_2.1/include/highgui.h>
//#include <opencv_2.1/include/highgui.hpp>

#include "tl_tracker.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>
#include <Windows.h>
#include <time.h>
#include <iostream>
#include <fstream>

#include "extern/opencv_210/CvImage.h"

using std::vector;
using std::string;

bgm::TLTracker tl_tracker(1);
double total_proc_time = 0.0;
unsigned long long total_clocks = 0;
unsigned long long process_clocks = 0;
unsigned long long frame_idx = 0;
std::string result = "NONE";
bgm::TLStatus prev_status1 = bgm::TLStatus::NONE;
bgm::TLStatus prev_status2 = bgm::TLStatus::NONE;

vector<string> listFilesInDirectory(string directoryName) {
  WIN32_FIND_DATA FindFileData;
  const char* FileName = directoryName.c_str();
  HANDLE hFind = FindFirstFile(FileName, &FindFileData);

  vector<string> listFileNames;
  listFileNames.push_back(FindFileData.cFileName);

  while (FindNextFile(hFind, &FindFileData))
    listFileNames.push_back(FindFileData.cFileName);

  return listFileNames;
}

void Process(const cv::Mat& frame) {
  clock_t before_proc = clock();
  bgm::TLStatus status = tl_tracker.Track(frame);
  clock_t after_proc = clock();
  total_proc_time += (after_proc - before_proc) / (double)CLOCKS_PER_SEC;
  process_clocks = after_proc - before_proc;
  total_clocks += process_clocks;
  cv::Scalar color;
  std::string text = "NONE";
  switch (status) {
    case bgm::TLStatus::RED:
      color = cv::Scalar(0, 0, 255);
      text = "RED";
      break;
    case bgm::TLStatus::YELLOW:
      color = cv::Scalar(0, 255, 255);
      text = "YELLOW";
      break;
    case bgm::TLStatus::GREEN:
      color = cv::Scalar(0, 255, 0);
      text = "GREEN";
      break;
    case bgm::TLStatus::RED_LEFT:
      color = cv::Scalar(255, 255, 0);
      text = "LEFT";
      break;
    case bgm::TLStatus::GREEN_LEFT:
      color = cv::Scalar(255, 255, 0);
      text = "LEFT";
      break;
  }

  cv::Mat show_img = frame;
  cv::putText(show_img, text, cv::Point(10, frame.rows - 10), 1, 4, color, 3);
  //cv::putText(show_img, "YELLOW", cv::Point(10, frame.rows - 10), 1, 4, 
  //            cv::Scalar(0, 255, 255), 3);
  double time = total_clocks / (double)CLOCKS_PER_SEC;
  double fps = (frame_idx + 1) / time;
  
  //cv::putText(show_img, "fps : " + std::to_string(fps), 
  //            cv::Point(10, frame.rows - 55), 1, 2, 
  //            cv::Scalar(255, 255, 255), 2);
  cv::imshow("result", show_img);
  //if (prev_status1 == bgm::TLStatus::NONE 
  //    && prev_status2 == bgm::TLStatus::NONE
  //    && status != bgm::TLStatus::NONE)
  //  cv::waitKey(0);
  //else
  //  cv::waitKey(1);
  //int key = cv::waitKey(1);
  //if(key != -1)
  //  cv::waitKey(0);

  //std::cout << text << std::endl;
  result = text;

  if (prev_status1 == bgm::TLStatus::NONE 
      && prev_status2 == bgm::TLStatus::NONE
      && status != bgm::TLStatus::NONE)
    cv::waitKey(0);
  else if(cv::waitKey(33) != -1)
    cv::waitKey(0);

  prev_status1 = prev_status2;
  prev_status2 = status;
  
}

int main(int argc, char **argv) {
  if (argc != 4)
    return -1;
  std::string img_path = std::string(argv[1]) + "/";
  //std::string img_ext = "*.bmp";
  std::string img_ext = "*." + std::string(argv[2]);
  std::string out_path = argv[3] + std::string("/");
  //std::string img_path = "D:/DB/roadImg/";
  

  std::vector<std::string> file_list = listFilesInDirectory(img_path + img_ext);

  std::ofstream out(out_path + "result.txt");

  for (int fr_idx = 0; fr_idx < file_list.size(); fr_idx++) {
    std::cout << fr_idx << ")) " << file_list[fr_idx] << std::endl;
    out << fr_idx << ")) " << file_list[fr_idx] << std::endl;
    if (fr_idx < 431)
      continue;
    cv::Mat frame = cv::imread((img_path + file_list[fr_idx]).c_str(), 1);
    //cv::resize(frame, frame, cv::Size(frame.cols / 1.5, frame.rows / 1.5));
    //cv::resize(frame, frame, cv::Size(frame.cols * 2, frame.rows * 2));
    
    Process(frame);
    cv::imwrite(out_path + file_list[fr_idx] + ".png", frame);
    frame_idx++;    
    
    std::cout << "\ttotal process time : " << total_proc_time << " sec, ";
    out << "\tresult : " << result << ", total process time : " << total_proc_time << " sec, ";
    std::cout << "avg. fps : " << (fr_idx + 1) / total_proc_time << std::endl;
    out << "avg. fps : " << (fr_idx + 1) / total_proc_time << std::endl;
  }

  
  return 0;
}