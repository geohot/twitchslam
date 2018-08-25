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

    # not used
    glutDisplayFunc(self.fakedraw)
    #glutMainLoop()

  def fakedraw(self):
    pass

  def draw(self):
    # set up 2d screen
    glClearColor(0.5, 0.5, 0.5, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor(0.0, 1.0, 0.0)

    # set up camera
    glPushMatrix()
    glLoadIdentity()
    gluPerspective(45, self.W/self.H, 0.1, 50.0)
    glTranslatef(0,0,-10)
    glRotatef(1, 3, 1, 1)

    # draw stuff
    glBegin(GL_LINES)
    for edge in self.edges:
      for vertex in edge:
        glVertex3fv(self.vertices[vertex])
    glEnd()

    # render to numpy buffer and return
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    ret = glReadPixels(0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE)
    ret = np.fromstring(ret, np.uint8).reshape((self.H, self.W, 4))
    glutSwapBuffers()

    glPopMatrix()
    return ret[:, :, 0:3]

if __name__ == "__main__":
  W,H = 640,480

  r = Renderer(W, H)

  from display import Display2D
  disp2d = Display2D(W, H)
  while 1:
    draw = r.draw()
    disp2d.paint(draw)
    time.sleep(0.5)

