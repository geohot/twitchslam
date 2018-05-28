# twitchslam

A toy implementation of SLAM written while livestreaming.

<img width=600px src="https://raw.githubusercontent.com/geohot/twitchslam/master/example.png" />

Stream
-----

https://www.twitch.tv/tomcr00s3

Will be streaming again tomorrow. May 28, 2018

By the end of the day, twitchslam will be decent.

Usage
-----

```
export REVERSE=1   # Hack for reverse video
export F=500       # Focal length (in px)

./slam.py <video.mp4>

# an example from the repo
REVERSE=1 F=650 ./slam.py videos/test_countryroad_reverse.mp4 

# kitti video test
REVERSE=1 F=984 ./slam.py videos/test_kitti984_reverse.mp4
```

Libraries Used
-----

* SDL2 for 2-D display
* cv2 for feature extraction
* pangolin for 3-D display
* g2opy for optimization (soon!)

DONE
-----

* BUGFIX: Why is window initting small? (fixed? ish, zoom is broken sometimes)
* BUGFIX: Turning doesn't work well (fixed?)
* Stop using essential matrix for pose estimation once you have a track (done)
 * Add kinematic model (done)
 * Run g2o to only optimize the latest pose (done)
* Add search by projection to refind old map points (done)
 * Check if points are in the field of view of the camera (done)

TODO
-----

* BUGFIX: Fix moving up! (y axis flipped?)
* Improve init to not need REVERSE environment variable
* Add optimizer for F
* Add multiscale feature extractor
* Add linux libraries and OS check
* Check accuracy with ground truth
* Profile and speed up
* Add loading and saving of map support

