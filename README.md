# twitchslam

A toy implementation of monocular SLAM written while livestreaming.

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

# good example
F=525 ./slam.py videos/test_freiburgxyz525.mp4

# ground truth
F=525 ./slam.py videos/test_freiburgrpy525.mp4 videos/test_freiburgrpy525.npz

# kitti example
REVERSE=1 F=984 ./slam.py videos/test_kitti984_reverse.mp4

# extract ground truth
tools/parse_ground_truth.py videos/groundtruth/freiburgrpy.txt videos/test_freiburgrpy525.npz 
```

Classes
-----

* Frame -- An image with extracted features
* Point -- A 3-D point in the Map and it's 2-D Frame correspondences
* Map -- A collection of points and frames
* Display2D -- SDL2 display of the current image
* Display3D -- Pangolin display of the current map

Libraries Used
-----

* SDL2 for 2-D display
* cv2 for feature extraction
* pangolin for 3-D display
* g2opy for optimization (soon!)

Rendered Scene Test
-----

```
./fakeslam.py
```

NOTE: The test currently doesn't work reliably. It seems adding a small amount of Gaussian noise to the point positions can cause the optimizer to fall into really bad local minima. This may just be caused by poor initialization, as I'm not sure how stable Essential matricies are.

TODO: Investigate the stability of Essential matrix recovery.

DONE
-----

* BUGFIX: Why is window initting small? (fixed? ish, zoom is broken sometimes)
* BUGFIX: Turning doesn't work well (fixed?)
* Stop using essential matrix for pose estimation once you have a track (done)
 * Add kinematic model (done)
 * Run g2o to only optimize the latest pose (done)
* Add search by projection to refind old map points (done)
 * Check if points are in the field of view of the camera (done)
* BUGFIX: Fix moving up! (y axis flipped? nah, it's okay)
* Add loading and saving of map support

TODO
-----

* Investigate if we need KeyFrames!
* BUGFIX: Improve lockups to happen less
* Improve init to not need REVERSE environment variable
* Add optimizer for F
* Add multiscale feature extractor
* Add Linux libraries and OS check
* Profile and speed up more (tomorrow 6/1/18)
 * Profile with flame!
 * Search by projection less stupidly
 * Don't add all points to the optimization graph for pose at least
* Add automated test for freiburg running on commit
 * Check accuracy with ground truth

LICENSE
-----

All my code is MIT licensed. Videos and libraries follow their respective licenses.

