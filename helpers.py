import os
import numpy as np

# colors shamelessly stolen from
# https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

def hamming_distance(a, b):
  r = (1 << np.arange(8))[:,None]
  return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

def triangulate(pose1, pose2, pts1, pts2):
  ret = np.zeros((pts1.shape[0], 4))
  for i, p in enumerate(zip(pts1, pts2)):
    A = np.zeros((4,4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][1] * pose1[2] - pose1[1]
    A[2] = p[1][0] * pose2[2] - pose2[0]
    A[3] = p[1][1] * pose2[2] - pose2[1]
    _, _, vt = np.linalg.svd(A)
    ret[i] = vt[3]
  return ret

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  if len(x.shape) == 1:
    return np.concatenate([x,np.array([1.0])], axis=0)
  else:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

# pose
def fundamentalToRt(F):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,d,Vt = np.linalg.svd(F)
  if np.linalg.det(U) < 0:
    U *= -1.0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]

  # TODO: Resolve ambiguities in better ways. This is wrong.
  if t[2] < 0:
    t *= -1
  
  # TODO: UGLY!
  if os.getenv("REVERSE") is not None:
    t *= -1
  return np.linalg.inv(poseRt(R, t))

def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

# from https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
class EssentialMatrixTransform(object):
  def __init__(self):
    self.params = np.eye(3)

  def __call__(self, coords):
    coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
    return coords_homogeneous @ self.params.T

  def estimate(self, src, dst):
    assert src.shape == dst.shape
    assert src.shape[0] >= 8

    # Setup homogeneous linear equation as dst' * F * src = 0.
    A = np.ones((src.shape[0], 9))
    A[:, :2] = src
    A[:, :3] *= dst[:, 0, np.newaxis]
    A[:, 3:5] = src
    A[:, 3:6] *= dst[:, 1, np.newaxis]
    A[:, 6:8] = src

    # Solve for the nullspace of the constraint matrix.
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)

    # Enforcing the internal constraint that two singular values must be
    # non-zero and one must be zero.
    U, S, V = np.linalg.svd(F)
    S[0] = S[1] = (S[0] + S[1]) / 2.0
    S[2] = 0
    self.params = U @ np.diag(S) @ V

    return True
    
  def residuals(self, src, dst):
    # Compute the Sampson distance.
    src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
    dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])

    F_src = self.params @ src_homogeneous.T
    Ft_dst = self.params.T @ dst_homogeneous.T

    dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

    return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2
                                       + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)

