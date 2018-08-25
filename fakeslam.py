#!/usr/bin/env python3
import os
import sys

sys.path.append("lib/macosx")
sys.path.append("lib/linux")

import numpy as np
from slam import SLAM
from renderer import Renderer
from display import Display2D, Display3D

if __name__ == "__main__":
  W,H=640,480
  F=(45*np.pi/180)*H
  print(F)
  
  K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
  Kinv = np.linalg.inv(K)

  slam = SLAM(W, H, K)
  r = Renderer(W, H)

  disp3d = Display3D()
  disp2d = Display2D(W, H)

  pos_x = 0
  while 1:
    frame = r.draw([pos_x,0,-10])
    slam.process_frame(frame, None)
    disp3d.paint(slam.mapp)
    img = slam.mapp.frames[-1].annotate(frame)
    disp2d.paint(img)

    pos_x += 0.01

