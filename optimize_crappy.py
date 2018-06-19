from helpers import add_ones
#import autograd.numpy as np
from scipy.optimize import least_squares, leastsq
import cv2

import numpy as np
import sympy as sp
from sympy.utilities.autowrap import autowrap, ufuncify, binary_function
from sympy.printing.ccode import ccode

EPS = 1e-10

def rotation_from_matrix(R):
  return cv2.Rodrigues(R)[0].flatten()

def rotation_to_matrix(w):
  wx,wy,wz = w
  theta = sp.sqrt(wx**2 + wy**2 + wz**2 + wy**2 + wz**2) + EPS
  omega = sp.Matrix([[0,-wz,wy],
                     [wz,0,-wx],
                     [-wy,wx,0]])
  R = sp.eye(3) +\
    omega*(sp.sin(theta)/theta) +\
    (omega*omega)*((1-sp.cos(theta))/(theta*theta))
  return R

"""
# test these
assert(np.allclose(np.eye(3), rotation_to_matrix(np.array([0,0,0]))))
for i in range(20):
  w = np.random.randn(3)
  what = rotation_from_matrix(rotation_to_matrix(w))
  assert(np.allclose(w, what))
"""

def optimize(frames, points, *args):
  # get point location guesses + camera poses (initial parameter vector)
  x0 = []
  for p in points:
    x0.append(p.pt)
  for f in frames:
    t = f.pose[:3, 3]
    R = f.pose[:3, :3]
    w = rotation_from_matrix(R)
    x0.append(t)
    x0.append(w)
  x0 = np.array(x0).flatten()

  # get target residuals (measurement vector)
  uvs = []
  for p in points:
    for f, idx in zip(p.frames, p.idxs):
      uv = f.kps[idx]
      uvs.append(uv)
  b = np.array(uvs).flatten()

  # f(ptw(9)) -> uv(2)
  def proj(p, t, w):
    R = rotation_to_matrix(w)
    proj = (R * p)+t
    return (proj[0] / proj[2], proj[1] / proj[2])

  def get_symbolic_jacobians():
    p = sp.Matrix(sp.symbols("px py pz"))
    t = sp.Matrix(sp.symbols("tx ty tz"))
    w = sp.Matrix(sp.symbols("wx wy wz"))
    uv = sp.Matrix(proj(p, t, w))
    fuv = autowrap(uv)
    fjp = autowrap(uv.jacobian(p))
    fjt = autowrap(uv.jacobian(t))
    fjw = autowrap(uv.jacobian(w))
    return fuv,fjp,fjt,fjw
  fuv,fjp,fjt,fjw = get_symbolic_jacobians()

  # compute residuals f(x) = b'
  def res(x):
    J = np.zeros((b.shape[0], x0.shape[0]))
    ret = []
    j = 0
    for i, p in enumerate(points):
      for f, idx in zip(p.frames, p.idxs):
        pt = x[i*3:(i+1)*3]
        fidx = len(points)*3 + f.id*6
        tw = x[fidx:fidx+6]
        ptw = np.concatenate([pt, tw], axis=0).tolist()

        uv = fuv(*ptw)
        J[j*2:(j+1)*2, i*3:(i+1)*3] = fjp(*ptw)
        J[j*2:(j+1)*2, fidx:fidx+3] = fjt(*ptw)
        J[j*2:(j+1)*2, fidx+3:fidx+6] = fjw(*ptw)

        j += 1
        ret.append(uv)
    return np.array(ret).flatten(), J

  bhat, J = res(x0)
  print(J)
  print(J.shape)
  print(np.sum((bhat-b)**2))
  
  exit(0)

  # TODO: actually do this
  # http://www.cs.technion.ac.il/users/wwwb/cgi-bin/tr-get.cgi/2014/MSC/MSC-2014-16.pdf page 17
  # http://www.telesens.co/2016/10/13/bundle-adjustment-part-1-jacobians/ defines 2x3 jacobian

  # define function fun(parameter) = measurement

  def fun(x):
    return np.sum((res(x)-b)**2)

  # stack poses
  grad_fun = grad(fun)
  print("computing at x0")

  # gradient descent
  for i in range(20):
    loss = fun(x0)
    d = grad_fun(x0)
    print(loss, d)
    x0 -= d

  """
  poses = []
  for i, p in enumerate(self.points):
    for f, idx in zip(p.frames, p.idxs):
      poses.append(f.pose[:3])
  poses = np.concatenate(poses, axis=1)
  print(poses.shape)

  loss = np.dot(poses, x0)
  print(loss)
  """

  print("running least squares with %d params" % len(x0))
  
