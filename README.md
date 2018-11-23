# PIMNet_AnimalDetector

Created by EunSeop and [Daijin Kim](http://imlab.postech.ac.kr/members_d.htm) at [POSTECH IM Lab](http://imlab.postech.ac.kr)

### Overview
We propose a new animal detection method for ADAS based on [YOLOv2](https://github.com/pjreddie/darknet) that extracts features through base network and a hourglass method robust to small object detection. To improve the performance of Recall, we use anchor boxes to detect object. 

### How to use
You can easily train/test your images (or videos) using shell scripts (test_animals.sh, train_animals.sh).
For training, you should download the pre-trained weight file from https://pjreddie.com/media/files/yolov2.weights

### Acknowledgements

This research was supported by the MSIT (Ministry of Science, ICT), Korea, under the SW Starlab support program (IITP-2017-0-00897) supervised by the IITP (Institute for Information & communications Technology Promotion) and also supported by the MSIT, Korea, under the ICT Consilience Creative program (IITP-2017-R0346-16-1007) supervised by the IITP.

