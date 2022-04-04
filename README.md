# twitchslam

A toy implementation of monocular SLAM written while livestreaming.

<img width=600px src="https://raw.githubusercontent.com/geohot/twitchslam/master/example.png" />

# Docker
```bash
docker build -t twitchslam .

docker run --runtime=nvidia --gpus all  --net=host -e DISPLAY --rm -v /tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=all --env DISPLAY_COOKIE="DISPLAY_COOKIE" -it twitchslam
```

You’ll need to replace DISPLAY_COOKIE with your your display cookie which you can get from xauth, it should look like the following
```bash
$ xauth list
username/machine:0 MIT-MAGIC-COOKIE-1 [32 character string]
```
The final command should look something like one of the following
```bash
docker run --runtime=nvidia --gpus all  --net=host \
    -e DISPLAY --rm -v /tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=all \
    --mount type=bind,source=/mnt/HDD/home/aditya/elab_visualiser/elab/1644664247513,target=/videos \
    --env DISPLAY_COOKIE="sped-machine/unix:1  MIT-MAGIC-COOKIE-1  a1654666c7a832aa599d9cf267033379" \
    --env VIDEO_PATH="1644664247513.mp4" \
    -e SEEK=100 -e  FSKIP=5 -e F=1000 \
    -it twitchslam

docker run --runtime=nvidia --gpus all  --net=host \
    -e DISPLAY --rm -v /tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=all \
    --mount type=bind,source="$(pwd)"/videos,target=/videos \
    --env DISPLAY_COOKIE="sped-machine/unix:1  MIT-MAGIC-COOKIE-1  a1654666c7a832aa599d9cf267033379" \
    --env VIDEO_PATH="test_freiburgrpy525.mp4" \
    -e F=525 \
    -it twitchslam

docker run --runtime=nvidia --gpus all  --net=host \
    -e DISPLAY --rm -v /tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=all \
    --mount type=bind,source="$(pwd)"/videos,target=/videos \
    --env DISPLAY_COOKIE="sped-machine/unix:1  MIT-MAGIC-COOKIE-1  a1654666c7a832aa599d9cf267033379" \
    --env VIDEO_PATH="test_freiburgxyz525.mp4" \
    -e F=525 \
    -it twitchslam

docker run --runtime=nvidia --gpus all  --net=host -e DISPLAY --rm -v /tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=all --mount type=bind,source="$(pwd)"/videos,target=/videos --env DISPLAY_COOKIE="sped-machine/unix:1  MIT-MAGIC-COOKIE-1  a1654666c7a832aa599d9cf267033379" --env VIDEO_PATH="test_freiburgrpy525.mp4"   -it twitchslam

# To get shell access
docker run --runtime=nvidia --gpus all  --net=host -e DISPLAY --rm -v /tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=all --mount type=bind,source="$(pwd)"/videos,target=/videos --env DISPLAY_COOKIE="sped-machine/unix:1  MIT-MAGIC-COOKIE-1  a1654666c7a832aa599d9cf267033379" -it twitchslam /bin/bash
```

## Install nvidia-docker2

```bash
xhost +local:docker

sudo apt install nvidia-docker2

sudo systemctl daemon-reload
sudo systemctl restart docker
```

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

F=525 python3 slam.py videos/test_freiburgxyz525.mp4

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

