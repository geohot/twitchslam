#!/usr/bin/env python3

# ffmpeg -pattern_type glob -i "*.png" ~/twitchslam/videos/test_freiburgroom525.mp4

import sys
import numpy as np

# q = [qx,qy,qz,qw]
# NOT [qw,qx,qy,qz]
def _quaternion_matrix(q):
  n = np.dot(q, q)
  q *= np.sqrt(2.0 / n)
  q = np.outer(q, q)
  return np.array([
      [1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]],
      [    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]],
      [    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1]]],
      dtype=np.float64)

# TODO: Write this not dumb
#quaternion_matrix = np.vectorize(_quaternion_matrix)
def quaternion_matrix(qq):
  ret = np.zeros((qq.shape[0], 3, 3))
  for i in range(qq.shape[0]):
    ret[i] = _quaternion_matrix(qq[i])
  return ret


# handle ground truth
def read_lines(x):
  return filter(lambda x: x[0] != "#", open(x).read().strip().split("\n"))

if __name__ == "__main__":
  # read in and interpolate to frame time
  timing = read_lines(sys.argv[1].replace(".txt", "_timings.txt"))
  timing = np.array(list(map(lambda x: float(x.split(" ")[0]), timing)))
  gtt = np.array(list(map(lambda x: list(map(float, x.split(" "))), read_lines(sys.argv[1]))))
  gt = np.zeros((timing.shape[0], 1+3+4), dtype=np.float64)
  for i in range(1+3+4):
    gt[:, i] = np.interp(timing, gtt[:, 0], gtt[:, i])
  gt[:, 4:] /= np.linalg.norm(gt[:, 4:], axis=1)[:, None]

  # extract rotation and translation
  gt_R = quaternion_matrix(gt[:, 4:])
  gt_t = gt[:, 1:4]

  # 4x4 pose matrix
  gt_pose = np.zeros((gt.shape[0], 4, 4), dtype=np.float64)
  gt_pose[:, :3, :3] = gt_R
  gt_pose[:, :3, 3] = gt_t
  gt_pose[:, 3, 3] = 1.0

  # bring pose into first frame
  gt_pose = np.matmul(np.linalg.inv(gt_pose[0]), gt_pose)
  np.savez(sys.argv[2], pose=gt_pose)

