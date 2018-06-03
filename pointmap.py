from helpers import poseRt, hamming_distance
from frame import Frame
import time
import numpy as np
import g2o
import json

#LOCAL_WINDOW = 20
LOCAL_WINDOW = None

class Point(object):
  # A Point is a 3-D point in the world
  # Each Point is observed in multiple Frames

  def __init__(self, mapp, loc, color, tid=None):
    self.pt = np.array(loc)
    self.frames = []
    self.idxs = []
    self.color = np.copy(color)
    self.id = tid if tid is not None else mapp.add_point(self)

  def homogeneous(self):
    return np.array([self.pt[0], self.pt[1], self.pt[2], 1.0])

  def orb(self):
    return [f.des[idx] for f,idx in zip(self.frames, self.idxs)]

  def orb_distance(self, des):
    return min([hamming_distance(o, des) for o in self.orb()])
  
  def delete(self):
    for f,idx in zip(self.frames, self.idxs):
      f.pts[idx] = None
    del self

  def add_observation(self, frame, idx):
    assert frame.pts[idx] is None
    assert frame not in self.frames
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)

class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []
    self.max_frame = 0
    self.max_point = 0

  def serialize(self):
    ret = {}
    ret['points'] = [{'id': p.id, 'pt': p.pt.tolist(), 'color': p.color.tolist()} for p in self.points]
    ret['frames'] = []
    for f in self.frames:
      ret['frames'].append({
        'id': f.id, 'K': f.K.tolist(), 'pose': f.pose.tolist(), 'h': f.h, 'w': f.w, 
        'kpus': f.kpus.tolist(), 'des': f.des.tolist(),
        'pts': [p.id if p is not None else -1 for p in f.pts]})
    ret['max_frame'] = self.max_frame
    ret['max_point'] = self.max_point
    return json.dumps(ret)

  def deserialize(self, s):
    ret = json.loads(s)
    self.max_frame = ret['max_frame']
    self.max_point = ret['max_point']
    self.points = []
    self.frames = []

    pids = {}
    for p in ret['points']:
      pp = Point(self, p['pt'], p['color'], p['id'])
      self.points.append(pp)
      pids[p['id']] = pp

    for f in ret['frames']:
      ff = Frame(self, None, f['K'], f['pose'], f['id'])
      ff.w, ff.h = f['w'], f['h']
      ff.kpus = np.array(f['kpus'])
      ff.des = np.array(f['des'])
      ff.pts = [None] * len(ff.kpus)
      for i,p in enumerate(f['pts']):
        if p != -1:
          ff.pts[i] = pids[p]
      self.frames.append(ff)

  def add_point(self, point):
    ret = self.max_point
    self.max_point += 1
    self.points.append(point)
    return ret

  def add_frame(self, frame):
    ret = self.max_frame
    self.max_frame += 1
    self.frames.append(frame)
    return ret

  # *** optimizer ***
  
  alt = False
  def optimize(self, local_window=LOCAL_WINDOW, fix_points=False, verbose=False):
    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    if self.alt:
      principal_point = (self.frames[0].K[0][2], self.frames[0].K[1][2])
      cam = g2o.CameraParameters(self.frames[0].K[0][0], principal_point, 0)
      cam.set_id(0)
      opt.add_parameter(cam)

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

    if local_window is None:
      local_frames = self.frames
    else:
      local_frames = self.frames[-local_window:]

    graph_frames, graph_points = {}, {}

    # add frames to graph
    for f in (local_frames if fix_points else self.frames):
      if not self.alt:
        pose = np.linalg.inv(f.pose)
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
        sbacam = g2o.SBACam(se3)
        sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[0][2], f.K[1][2], 0.0)
        v_se3 = g2o.VertexCam()
        v_se3.set_estimate(sbacam)
      else:
        pose = f.pose
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)

      v_se3.set_id(f.id * 2)
      v_se3.set_fixed(f.id <= 1 or f not in local_frames)
      #v_se3.set_fixed(f.id != 0)
      opt.add_vertex(v_se3)

      # confirm pose correctness
      est = v_se3.estimate()
      assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
      assert np.allclose(pose[0:3, 3], est.translation())

      graph_frames[f] = v_se3

    # add points to frames
    for p in self.points:
      if not any([f in local_frames for f in p.frames]):
        continue

      pt = g2o.VertexSBAPointXYZ()
      pt.set_id(p.id * 2 + 1)
      pt.set_estimate(p.pt[0:3])
      pt.set_marginalized(True)
      pt.set_fixed(fix_points)
      opt.add_vertex(pt)

      graph_points[p] = pt

      # add edges
      for f, idx in zip(p.frames, p.idxs):
        if f not in graph_frames:
          continue
        if not self.alt:
          edge = g2o.EdgeProjectP2MC()
        else:
          edge = g2o.EdgeProjectXYZ2UV()
          edge.set_parameter_id(0, 0)
        edge.set_vertex(0, pt)
        edge.set_vertex(1, graph_frames[f])
        uv = f.kpus[idx]
        edge.set_measurement(uv)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(robust_kernel)
        opt.add_edge(edge)
        
    if verbose:
      opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(50)

    # put frames back
    for f in graph_frames:
      est = graph_frames[f].estimate()
      R = est.rotation().matrix()
      t = est.translation()
      if not self.alt:
        f.pose = np.linalg.inv(poseRt(R, t))
      else:
        f.pose = poseRt(R, t)

    # put points back (and cull)
    if not fix_points:
      culled_pt_count = 0
      for p in graph_points:
        est = graph_points[p].estimate()
        p.pt = np.array(est)

        # <= 4 match point that's old
        old_point = len(p.frames) <= 4 and p.frames[-1].id+7 < self.max_frame

        # compute reprojection error
        errs = []
        for f,idx in zip(p.frames, p.idxs):
          uv = f.kpus[idx]
          proj = np.dot(np.dot(f.K, f.pose[:3]),
                        np.array([est[0], est[1], est[2], 1.0]))
          proj = proj[0:2] / proj[2]
          errs.append(np.linalg.norm(proj-uv))

        # cull
        if old_point or np.mean(errs) > 2:
          culled_pt_count += 1
          self.points.remove(p)
          p.delete()

      print("Culled:   %d points" % (culled_pt_count))

    return opt.active_chi2()

