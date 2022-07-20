# parent class of h36.py

import numpy as np
from research_project.VideoPose3D.common.skeleton import Skeleton

class MocapDataset():
  def __init__(self, cfg, fps, skeleton):
    self._skeleton = skeleton
    self._fps = fps
    self._data = None #Must be silled by sibclass
    self._cameras = None #Must be filled by subclass

  def remove_joints(self, joints_to_remove):
    kept_joints = self._skeleton.remove_joints(joints_to_remove)
    for subject in self._data.keys():
      for action in self._data[subject].keys():
        s = self._data[subject][action]
        if 'positions' in s:
          s['positions'] = s['positions'][:, kept_joints]

  def __getitem__(self, idx):
    return self._data[idx]
  
  def subjects(self): #returns the list of subjects in h3.6 dataset
    return self._data.keys() #retrieved from self._data

  def fps(self):
    return self._fps

  def skeleton(self):
    return self._skeleton

  def cameras(self):
    return self._cameras

  
