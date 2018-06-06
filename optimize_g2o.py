import g2o
import numpy as np
from helpers import poseRt

def optimize(frames, points, local_window, fix_points, verbose=False, rounds=50):
  if local_window is None:
    local_frames = frames
  else:
    local_frames = frames[-local_window:]

  # create g2o optimizer
  opt = g2o.SparseOptimizer()
  solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
  solver = g2o.OptimizationAlgorithmLevenberg(solver)
  opt.set_algorithm(solver)

  # add normalized camera
  cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)         
  cam.set_id(0)                       
  opt.add_parameter(cam)   

  robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
  graph_frames, graph_points = {}, {}

  # add frames to graph
  for f in (local_frames if fix_points else frames):
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
  for p in points:
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
      edge = g2o.EdgeProjectXYZ2UV()
      edge.set_parameter_id(0, 0)
      edge.set_vertex(0, pt)
      edge.set_vertex(1, graph_frames[f])
      edge.set_measurement(f.kps[idx])
      edge.set_information(np.eye(2))
      edge.set_robust_kernel(robust_kernel)
      opt.add_edge(edge)
      
  if verbose:
    opt.set_verbose(True)
  opt.initialize_optimization()
  opt.optimize(rounds)

  # put frames back
  for f in graph_frames:
    est = graph_frames[f].estimate()
    R = est.rotation().matrix()
    t = est.translation()
    f.pose = poseRt(R, t)

  # put points back
  if not fix_points:
    for p in graph_points:
      p.pt = np.array(graph_points[p].estimate())

  return opt.active_chi2()

