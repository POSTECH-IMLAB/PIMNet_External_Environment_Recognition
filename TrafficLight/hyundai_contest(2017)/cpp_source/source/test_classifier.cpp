//#include "tl_classifier.hpp"
//
//#include <opencv2\core\core.hpp>
//#include <opencv2\highgui\highgui.hpp>
//
//#include <io.h>
//#include <conio.h>
//
//#include <vector>
//
//void GetFiles(const std::string& path,
//              std::vector<std::string>& files) {
//  files.clear();
//
//  _finddata_t fd;
//  intptr_t handle;
//  int result = 1;
//  handle = _findfirst((path + "/*.png").c_str(), &fd);
//
//  if (handle == -1)
//    return;
//
//  while (result != -1) {
//    files.push_back(path + "/" + std::string(fd.name));
//    result = _findnext(handle, &fd);
//  }
//
//  _findclose(handle);
//}
//
//int main() {
//  //cv::Mat src = cv::imread("D:/DB/traffic_light3/test/L/757.png");
//  bgm::TLClassifier classifier;
//  //classifier.Classify(src);
//
//  std::vector<std::string> files;
//  GetFiles("D:/DB/traffic_light3/test/L", files);
//
//  int correct = 0;
//  int total = files.size();
//
//  for (int i = 0; i < files.size(); i++) {
//    //if (i < 444)
//    //  continue;
//    cv::Mat img = cv::imread(files[i]);
//    bgm::TLStatus color = classifier.Classify(img);
//
//    std::cout << i << ')' << files[i] << " : ";
//    switch (color) {
//      case bgm::TLStatus::GREEN:
//        std::cout << "GREEN" << std::endl;
//        //correct++;
//        break;
//      case bgm::TLStatus::RED_LEFT:
//        std::cout << "RED_LEFT" << std::endl;
//        correct++;
//        break;
//      case bgm::TLStatus::YELLOW:
//        std::cout << "YELLOW" << std::endl;
//        //correct++;
//        break;
//      case bgm::TLStatus::RED:
//        std::cout << "RED" << std::endl;
//        //correct++;
//        break;
//      case bgm::TLStatus::NONE:
//        std::cout << "NONE" << std::endl;
//        total--;
//        //correct++;
//        break;
//
//    }
//  }
//
//  std::cout << "accuracy : " << correct << '/' << total;
//  std::cout << " = " << correct / (float)total << std::endl;
//  return 0;
//}