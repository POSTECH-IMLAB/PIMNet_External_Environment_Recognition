#include "tl_detector_em.hpp"

#include <stdlib.h>
#include <stdio.h>
#include "extern/VJ/imgInfo.h"
#include "extern/VJ/CascadeClassifier.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

namespace bgm
{

float* integral_image(float *img, int width, int height) {
  float* ii = (float *)malloc(width * height * sizeof(float));
  float* s = (float *)malloc(width * height * sizeof(float));
  int x, y;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      if (x == 0) s[(y*width) + x] = img[(y*width) + x];
      else s[(y*width) + x] = s[(y*width) + x - 1] + img[(y*width) + x];
      if (y == 0) ii[(y*width) + x] = s[(y*width) + x];
      else ii[(y*width) + x] = ii[((y - 1)*width) + x] + s[(y*width) + x];
    }
  }

  free(s);
  return ii;
}

float* squared_integral_image(float *img, int width, int height) {
  float* ii = (float *)malloc(width * height * sizeof(float));
  float* s = (float *)malloc(width * height * sizeof(float));
  int x, y;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      if (x == 0) s[(y*width) + x] = pow(img[(y*width) + x], 2);
      else s[(y*width) + x] = s[(y*width) + x - 1] + pow(img[(y*width) + x], 2);
      if (y == 0) ii[(y*width) + x] = s[(y*width) + x];
      else ii[(y*width) + x] = ii[((y - 1)*width) + x] + s[(y*width) + x];
    }
  }
  free(s);
  return ii;
}

float evaluate_integral_rectangle(float *ii, int iiwidth, int x, int y, int w, int h) {
  float value = ii[((y + h - 1)*iiwidth) + (x + w - 1)];
  if (x > 0) value -= ii[((y + h - 1)*iiwidth) + (x - 1)];
  if (y > 0) value -= ii[(y - 1)*iiwidth + (x + w - 1)];
  if (x > 0 && y > 0) value += ii[(y - 1)*iiwidth + (x - 1)];
  return value;
}
int detection[1000][3];
int detection_cnt = 0;
void detect_objects(CascadeClassifier *cc, float fscale, float fincrement, CvMat * image) {
  float *iimg, *siimg;
  int i, j, increment;
  int fnotfound = 0;
  float mean, stdev;
  int index;
  float *buffer;


  detection_cnt = 0;


  img_width = cvGetSize(image).width;
  img_height = cvGetSize(image).height;
  printf("%d %d", img_width, img_height);
  buffer = (float*)malloc(sizeof(float)*img_width*img_height);


  /// Image To array!!
  for (j = 0; j < img_height; j++)
    for (i = 0; i < img_width; i++) {
      index = j*img_width + i;
      //ptr[5] = 255;
      buffer[index] = cvGetReal2D(image, j, i);
      //			cvSetReal2D(image, j, i, (uchar)gsimg[index]);
      //			myimage->data = 255;// cvmSet(myimage, i, j, 255);//(uchar)gsimg[index]);
    }


  // Calculate integral image and squared integral image
  iimg = integral_image(buffer, img_width, img_height);
  siimg = squared_integral_image(buffer, img_width, img_height);

  // Run face detection on multiple scales
  int base_resolution = cc->baseres;
  while (base_resolution <= img_width && base_resolution <= img_height) {
    increment = base_resolution * fincrement;
    if (increment < 1) increment = 1;

    // Slide window over image
    for (i = 0; (i + base_resolution) <= img_width; i += increment) {
      for (j = 0; (j + base_resolution) <= img_height; j += increment) {
        // Calculate mean and std. deviation for current window
        mean = evaluate_integral_rectangle(iimg, img_width, i, j, base_resolution, base_resolution) / pow(base_resolution, 2);
        stdev = sqrt((evaluate_integral_rectangle(siimg, img_width, i, j, base_resolution, base_resolution) / pow(base_resolution, 2)) - pow(mean, 2));

        if (classify(cc, iimg, img_width, i, j, mean, stdev) == true) {
          if (detection_cnt > 1000) {
            printf("Found over 50 objects \n");
            continue;
          }
          detection[detection_cnt][0] = i;
          detection[detection_cnt][1] = j;
          detection[detection_cnt][2] = base_resolution;
          detection_cnt++;
        }
        else
          fnotfound++;
      }
    }

    scale(cc, fscale);
    base_resolution = cc->baseres;
  }
  // Merge overlapping detections
  // merge_detections(detections);

  //printf(" %d objects found ( %d total subwindows checked) ", detection_cnt, detection_cnt + fnotfound);
  free(iimg);
  free(siimg);
  free(buffer);
}


void merge_detections() {
  int x1, y1, x2, y2, s1, s2;
  int minx, miny, maxx, maxy;
  int i, j, k;

  for (i = 0; i<detection_cnt; i++) {
    x1 = detection[i][0]; y1 = detection[i][1]; s1 = detection[i][2];

    for (j = i + 1; j < detection_cnt; j++) {
      x2 = detection[j][0]; y2 = detection[j][1]; s2 = detection[j][2];

      if (j != i && ((x1 < x2 + s2) && (x2 < x1 + s1) && (y1 < y2 + s2) && (y2 < y1 + s1))) {
        // There's overlapping between detections
        if (x1 > x2) {
          minx = x2;
          maxx = x1 + s1;
        }
        else {
          minx = x1;
          maxx = x2 + s2;
        }
        if (y1 > y2) {
          miny = y2;
          maxy = y1 + s1;
        }
        else {
          miny = y1;
          maxy = y2 + s2;
        }

        detection[i][0] = minx; detection[i][1] = miny; detection[i][2] = std::max(detection[i][2], std::max(maxx - minx, maxy - miny));

        // 배열의 값을 한 칸 씩 당기고, detection_cnt를 감소 시켜주자
        for (k = j; k < detection_cnt - 1; k++) {
          detection[k][0] = detection[k + 1][0];
          detection[k][1] = detection[k + 1][1];
          detection[k][2] = detection[k + 1][2];
        }
        detection_cnt--;
      }
    }
  }
}

void TLDetectorEM::Detect(const cv::Mat& img,
                          std::vector<cv::Rect>& result) {
  float strictness = 1;
  float scalefstep = 1.25;
  float slidefstep = 0.1;
  int i;
  int j;
  int f;
  int index;
  //CvMat* img;
  CvMat* img_gray;

  //CvMat* img_resize;
  // Load cascade classififer model
  CascadeClassifier c;
  uchar* ptr;
  CvPoint p1;
  CvPoint p2;
  CvScalar sc;
  int width;
  int height;

  // for Pedestrian
  float scalex = 1 / 5.0;
  float scaley = 1 / 2.5;

  cv::Mat gray;
  cv::cvtColor(img, gray, CV_BGR2GRAY);
  cv::resize(gray, gray,
             cv::Size(gray.cols * scalex, gray.rows * scaley),
             CV_8UC1);

  initClassifier(&c, "./tl_all.model");
  fnc_strictness(&c, strictness);

  CvMat img_resize = gray;
  // Load PNG image
  // image 는 우선 imgInfo.h 에 존재
  initClassifier_partially(&c);	// detect_objects 에서 c의 값듯을 바꾸는 것이 있어서 다시 모델을 읽지 않고, 그 값들만 초기화
  detect_objects(&c, scalefstep, slidefstep, &img_resize);

  //for (i = 0; i < detection_cnt; i++){
  //	// For pedestrian.. resize first
  //	float resize = 0.5;
  //	
  //	detection[i][0] += detection[i][2] * ((1.0 - resize)/2.0);
  //	detection[i][1] += detection[i][2] * ((1.0 - resize)/2.0);
  //	detection[i][2] *= resize;
  //}

  merge_detections();
  merge_detections();
  merge_detections();

  if (detection_cnt > 0) {
    result.resize(detection_cnt);
    for (i = 0; i < detection_cnt; i++) {
      result[i].x = detection[i][0] / scalex;
      result[i].y = detection[i][1] / scaley;
      result[i].width = detection[i][2] / scalex;
      result[i].height = detection[i][2] / scaley;
    }
  }
  else
    result.clear();
}


}