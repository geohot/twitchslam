#!/usr/bin/env python3
import sys
import time
sys.path.append("lib/macosx")
sys.path.append("lib/linux")

from helpers import add_ones
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

np.set_printoptions(suppress=True)

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

  def draw(self, pos):
    # set up 2d screen
    glClearColor(0.3, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # set up camera
    glPushMatrix()
    glLoadIdentity()
    gluPerspective(45, self.W/self.H, 0.1, 100.0)
    glTranslatef(pos[0], pos[1], pos[2])

    # draw stuff and track verts
    verts = []
    def draw_cube(offset):
      glBegin(GL_LINES)
      for edge in self.edges:
        for vertex in edge:
          vv = self.vertices[vertex]
          vv = (vv[0]+offset[0], vv[1]+offset[1], vv[2]+offset[2])
          verts.append(vv)
          glVertex3fv(vv)
      glEnd()

    glColor(0.0, 1.0, 0.0)
    draw_cube([0,0,-50])

    glColor(1.0, 0.0, 0.0)
    draw_cube([5,2,-30])

    glColor(0.0, 0.0, 1.0)
    draw_cube([-10,2,-30])

    glColor(0.0, 1.0, 1.0)
    draw_cube([5,5,-25])

    glColor(1.0, 1.0, 0.0)
    draw_cube([-5,-5,-35])

    glColor(1.0, 1.0, 1.0)
    draw_cube([-5,5,-45])

    # extract "map"
    projected = []
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    for v in verts:
      cc = gluProject(v[0], v[1], v[2], modelview, projection)
      cc = np.array(cc)
      cc /= cc[2]
      projected.append(cc[0:2])

    # render to numpy buffer and return
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    ret = glReadPixels(0, 0, self.W, self.H, GL_RGBA, GL_UNSIGNED_BYTE)
    ret = np.fromstring(ret, np.uint8).reshape((self.H, self.W, 4))
    glutSwapBuffers()

    glPopMatrix()
    return np.copy(ret[:, :, 0:3]), np.array(projected)

if __name__ == "__main__":
  W,H = 640,480

  r = Renderer(W, H)

  from display import Display2D
  disp2d = Display2D(W, H)
  i = 0
  while 1:
    draw, _ = r.draw([i,0,-10])
    disp2d.paint(draw)
    time.sleep(0.05)
    i += 0.02

