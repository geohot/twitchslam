#!/usr/bin/env python3
import sys
import time
sys.path.append("lib/macosx")
sys.path.append("lib/linux")

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class Renderer(object):
  def __init__(self, W, H):
    self.W, self.H = W, H
    self.vertices = (
      (1, -1, -1),
      (1, 1, -1),
      (-1, 1, -1),
      (-1, -1, -1),
      (1, -1, 1),
      (1, 1, 1),
      (-1, -1, 1),
      (-1, 1, 1)
    )

    self.edges = (
      (0,1), (0,3), (0,4),
      (2,1), (2,3), (2,7),
      (6,3), (6,4), (6,7),
      (5,1), (5,4), (5,7)
    )

    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(self.W, self.H)
    glutCreateWindow(b"OpenGL Offscreen")
    glutHideWindow()

  def draw(self):
    glClearColor(0.5, 0.5, 0.5, 1.0)
    glColor(0.0, 1.0, 0.0)
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
    glViewport(0, 0, self.W, self.H)

    glBegin(GL_LINES)
    for edge in self.edges:
      for vertex in edge:
        glVertex3fv(self.vertices[vertex])
    glEnd()

    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    ret = glReadPixels(0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE)
    ret = np.fromstring(ret, np.uint8).reshape((self.W, self.H, 4))
    return ret[:, :, 0:3]

if __name__ == "__main__":
  W,H = 640,480

  r = Renderer(W, H)
  draw = r.draw()
  print(draw.shape)

  draw = np.zeros((480, 640, 3))

  from display import Display2D
  disp2d = Display2D(W, H)
  while 1:
    disp2d.paint(draw)
    time.sleep(1)


