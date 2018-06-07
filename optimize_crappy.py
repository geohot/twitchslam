from helpers import add_ones
import autograd.numpy as np
from autograd import grad
from scipy.optimize import least_squares, leastsq

def optimize(points):
  # get point location guesses (parameter vector)
  x0 = []
  for p in points:
    x0.append(p.pt)
  x0 = np.array(x0).flatten()

  # get target residuals (measurement vector)
  uvs = []
  for p in points:
    for f, idx in zip(p.frames, p.idxs):
      uv = f.kps[idx]
      uvs.append(uv)
  b = np.array(uvs).flatten()

  # define function fun(parameter) = measurement

  # compute residuals
  def fun(x):
    print(x[0:10])
    ret = []
    for i, p in enumerate(points):
      for f, idx in zip(p.frames, p.idxs):
        proj = np.dot(f.pose[:3], add_ones(x[i*3:(i+1)*3]))
        ret.append(proj)
    ret = np.array(ret)
    ret = ret[:, 0:2] / ret[:, 2:]
    return ret.flatten() - b

  # stack poses
  grad_fun = grad(fun)
  print(grad_fun(x0))

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
  
