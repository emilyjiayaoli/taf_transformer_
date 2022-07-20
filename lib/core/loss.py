# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import no_type_check

import torch
import torch.nn as nn
import numpy as np

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.video_frames = 25 ##### temporaryneed to be passed & specified  from cfg

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        #print('batch_size is',batch_size)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1) #split so that every single element is a tensor
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        #print('heatmaps_pred size is', len(heatmaps_pred)) # 17 but heatmaps_pred is a tuple
        #print('heatmaps_gt size is', heatmaps_gt[0].shape)
        
        loss = 0
        #for frame in range(batch_size):
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze() #heatmap_pred[idx].shape is (25,1,3072) removes tensors shape of size 1
            heatmap_gt = heatmaps_gt[idx].squeeze()
            print('heatmap_pred shape', heatmap_pred.shape)
            print('heatmap_gt shape', heatmap_gt.shape)

            """ Print out heat maps
            print("heatmap_gt:", heatmap_gt)
            
            heatmap_gt1 = heatmap_gt.cpu().numpy()
            np.save("heatmap_gt.npy", heatmap_gt1)
            
            print('heatmap_pred size is', heatmap_pred.shape)
            print('heatmap_gt size is', heatmap_gt.shape)
            print('target_weight[:,idx] shape is', target_weight[:, idx].shape)
            print('target_weight shape is', target_weight.shape)
            print('shape1', heatmap_pred.mul(target_weight[:, idx]).shape)
            print('shape 2',heatmap_gt.mul(target_weight[:, idx]).shape)

            """
            if self.use_target_weight: #not using this atm
                loss += 0.5 * self.criterion(
                  heatmap_pred.mul(target_weight[:, idx]),
                  heatmap_gt.mul(target_weight[:, idx]))
      
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        #print('loss from loss.py is', loss)

        final_loss = loss / num_joints
        print('JointsMSELoss loss is ', final_loss) #0.0076
        return final_loss


class TemporalAffinityLoss(nn.Module):
    def __init__(self, use_target_weight):
      super(TemporalAffinityLoss, self).__init__()
      self.criterion = nn.MSELoss(reduction='mean') #obviously change bec we don't want nn.MSELoss
      #self.use_target_weight = use_target_weight

    def forward(self, output, taf_target):
      #print('taf_pred shape', output.shape) #taf_pred shape torch.Size([24, 17, 64, 48, 2])
      #print('taf_gt shape', taf_target.shape) #taf_gt shape torch.Size([24, 17, 64, 48, 2])

      batch_size = output.size(0) #aka the number of frames per image
      #tafs_size = batch_size #temporary when output of the model is still 25,68,48 instead of 24 for taf
      num_joints = output.size(1)
      tafs_pred = output.reshape((batch_size, num_joints, -1)).split(1, 0) #output (25,17,64,48) reshaped into (25,17,3072) then split into a tuple with len = specified by the 2nd parameter, which is dimension 0, which means that len(taf_pred) = 25, along each frame
      tafs_gt = taf_target.reshape((batch_size, num_joints, -1)).split(1, 0) 
      #print('taf_pred size is', len(tafs_pred)) #24
      #print('tafs_gt size is',len(tafs_gt))#24
      loss = 0
      
      '''for frame in range(batch_size):
          loss_per_frame = 0
          for idx in range(num_joints):
              heatmap_pred = taf_pred[frame][idx].squeeze() #removes tensors shape of size 1
              heatmap_gt = taf_gt[frame][idx].squeeze()

              loss_per_frame += 0.5 * self.criterion(taf_pred, taf_gt)

          loss += loss_per_frame
        final_loss = loss/(batch_size * num_joints)
        print('taf1 loss is ', loss)
        return final_loss'''
      
      for frame in range(batch_size):
            taf_pred = tafs_pred[frame].squeeze() #tafs_pred[frame].shape is (1,17,3072) ; .squeeze() removes tensors shape of size 1
            taf_gt = tafs_gt[frame].squeeze()
            #print('taf_pred shape', taf_pred.shape) #(17, 6144)
            #print('taf_gt shape', taf_gt.shape) #(17, 6144)
            loss += 0.5 * self.criterion(taf_pred, taf_gt.cuda())
      
      final_loss = loss / batch_size
      #print('taf1 loss is ', final_loss) #0.0101
      return final_loss

class JointsOHKMMSELoss(nn.Module):

    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


    '''
class JointsMSELoss2(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss2, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        #self.use_target_weight = use_target_weight
        self.video_frames = 25 ##### temporaryneed to be passed & specified  from cfg

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        #print('batch_size is',batch_size)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1) #split so that every single element is a tensor
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        #print('heatmaps_pred size is', heatmaps_pred[0].shape)
        #print('heatmaps_gt size is', heatmaps_gt[0].shape)
        
        final_loss = 0.5 * self.criterion(heatmaps_pred.squeeze(), heatmaps_gt.squeeze())
        #final_loss += 0.5 * self.criterion(heatmaps_pred, heatmaps_gt)
        #final_loss = loss / num_joints

        print('JointsMSELoss2 loss is ', final_loss)

        return final_loss

'''