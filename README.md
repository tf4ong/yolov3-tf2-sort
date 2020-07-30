# Current work in progress
# Yolov3-tf2-SORT
- A integration of Yolov3-tf2, SORT, RFID detections for mice tracking in home-cage environment
- Original inspiration and prototype by [Braeden Jury](https://github.com/ubcbraincircuits/NaturalMouseTracker)
- A raspberry-pi based system for offline tracking of mice 
## Related hardware
- Code for data collection written by [Alice Xiong](https://github.com/ubcbraincircuits/NaturalMouseTracker_rpi)
##### RFID System
- 5 ID-20LA RFID tag readers setup at the custome locations at the bottom of the cage
- [RFID reader details](https://www.sparkfun.com/products/11828)
- [RFID reader base details](https://www.sparkfun.com/products/9963)
- RFID tag reader module developed by [Jamie Boyd](https://github.com/jamieboyd/RFIDTagReader)
- Locations of the readers must be sparse enough to minimize electromagnetic interference
- User to identify and validate mouse 
##### Overhead Camera 
- Records videos at 10.0 fps
- [Camera Details](https://www.buyapi.ca/product/raspberry-pi-camera-g-with-fisheye-lens/)
## Analytical Codes and Modules
##### Yolov3-tensorflow2 
yolov3-tf2:forked from [zzh8829 GitHub Page](https://github.com/zzh8829/yolov3-tf2)
- A tensorflow2 implementation of yolov3-tf2 which is easy to install and use
- Can be cloned and directly used on Anaconda or on Google Colab with GPU turned on 
- Simple object recognition module for detecting mice
- Orginal implmentation in darknet can be found at [AlexeyAB GitHub Page](https://github.com/AlexeyAB/darknet)
##### SORT 
SORT: forcked from [abewly GitHub Page](https://github.com/abewley/sort)
- A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
- Implements a visual multiple object tracking framework based on 
rudimentary data association and state estimation techniques. 
- Produces object identities on the fly and tracks them 
- Initiatially described in [this paper](https://arxiv.org/abs/1602.00763)
- Greatly depends on detection quality
- Handles detection lost during occlusions and Yolov3 failures
##### RFID_Matching 
Customed writted script for track reassocation and RFID assignment
###### SORT Track Identity Reassociation: A Euclidean Distance Method
SORT was orignally designed for tracking objects moving in and out of frames at relatively uniform speed.
Mice movements are often fast changing, therefore SORT often produces new identities for the same mouse.
Taking advantage of known number of mice detected in the previous frames and that a new mouse can only enter at the 
designated location, we can therfore reassign new false positive identities to real identities generated. Here, 
a centroid tracking algorithm based on Euclidean distances is employed. A tutorial of centroid tracking can be found 
[here](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)
###### RFID Identification and Verification
- Current work in progress
- RFID detections at the frame of interest will be associated with the SORT track that is in highest proximation 
- The SORT ID will then be matched with the read tag at that frame
### Preliminary results of Yolov3-SORT (Work in Progress) 
- Currently >80% of the Yolov3 detections are matched with RFID readings
- Accuaray of the RFID tags to detection matched is >85%
- Results based on a 3 minute video


![](Sample_RFID.gif)
