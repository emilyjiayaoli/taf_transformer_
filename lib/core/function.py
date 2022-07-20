# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

import ipdb


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion1, criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):

    print("train loader type",train_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train() # switch to train mode
    end = time.time()
    

    print('|||Entering batch training loop function.py')
    #track = 0
    for i, (input, target, target_weight, video_taf_target, meta) in enumerate(train_loader):
      print("||| inside for loop for train")

      input = torch.permute(torch.squeeze(input),(0, 3, 1, 2)).float().cuda()
      target_weight = torch.squeeze(target_weight,0)
      target = torch.squeeze(target,0)
      video_taf_target = torch.squeeze(video_taf_target,0)
    
      print('input size at dimension 0 is ', input.size()) #250, 256, 192, 3
      
      #meta should be empty if the data_numpy was none in coco.py, so skips the iteration
      #print("||| Checking if meta is empty...")
      if not meta:
        print("||| Meta was empty, so onto next iteration ", i+1, " of train loop in functions.py")
        continue
    
      # measure data loading time
      data_time.update(time.time() - end)

      # compute output
      outputs1, outputs2 = model(input)
      '''
      print('outputs2 type is', type(outputs1))
      print('outputs shape is', outputs1.shape) #outputs2 shape is torch.Size([21, 17, 3072])
      print('outputs shape is', outputs2.shape)
      '''
      target = target.cuda(non_blocking=True) #moving target labels onto the cuda gpu
      target_weight = target_weight.cuda(non_blocking=True)

      #ipdb.set_trace()

      #calculates the loss based of if the output is a list or else a single item
      if isinstance(outputs1, list):
          loss1 = criterion1(outputs1[0], target, target_weight)
          for output in outputs1[1:]:
              loss1 += criterion1(output, target, target_weight)
      else:
          #output = outputs1
          loss1 = criterion1(outputs1, target, target_weight)

      #calculate the TAF loss:
      if isinstance(outputs2, list):
          loss2 = criterion2(outputs2[0], target, target_weight)
          for output in outputs2[1:]:
              loss2 += criterion2(output, video_taf_target)
      else:
          #output = outputs2
          loss2 = criterion2(outputs2, video_taf_target)

      torch.cuda.memory_summary(device=None, abbreviated=False)

      # compute gradient and do update step
      optimizer.zero_grad() #clearing out the gradients for each batch
      loss = loss1 + loss2
      loss.backward()
      optimizer.step()

      # measure accuracy and record loss
      losses.update(loss.item(), input.size(0)) #.size(0) is the batch size

      _, avg_acc, cnt, pred = accuracy(outputs1.detach().cpu().numpy(),
                                      target.detach().cpu().numpy()) #### temporarily is output1 only bec it refeences accuracy in evaluate.py
      acc.update(avg_acc, cnt)

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % config.PRINT_FREQ == 0:
          msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0)/batch_time.val,
                    data_time=data_time, loss=losses, acc=acc)
          logger.info(msg)

          writer = writer_dict['writer']
          global_steps = writer_dict['train_global_steps']
          writer.add_scalar('train_loss', losses.val, global_steps)
          writer.add_scalar('train_acc', acc.val, global_steps)
          writer_dict['train_global_steps'] = global_steps + 1

          prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

          #print('meta is', meta)
          #save_debug_images(config, input, meta, target, pred*4, outputs1,
          #                  prefix) ##### temp is only for output1 only
        


def validate(config, val_loader, val_dataset, model, criterion1, criterion2, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #num_samples = len(val_dataset)
    num_samples = val_dataset.num_samples #is the total number of samples/images (not batches) there are
    
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()

        print('|||| entering for loop in validate in function.py....')
        

        for i, (input, target, target_weight, video_taf_target, meta) in enumerate(val_loader):
            print("||| inside for loop for validation at index ", i)

            input = torch.permute(torch.squeeze(input),(0, 3, 1, 2)).float().cuda()
            target_weight = torch.squeeze(target_weight,0)
            target = torch.squeeze(target,0)
            video_taf_target = torch.squeeze(video_taf_target,0)
          
            #print('input/batch size at dimension 0 is ', input.size(0)) #250, 256, 192, 3
          
            outputs1, outputs2 = model(input)

            ''' commenting out for rn
            if isinstance(outputs1, list):
                output = outputs1[-1]
            else:
                output = outputs1
            
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5
            '''
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss1 = criterion1(outputs1, target, target_weight)
            loss2 = criterion2(outputs2, video_taf_target)
            loss = loss1 + loss2
            print('     loss (total) is', loss)

            num_images = input.size(0)
            
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            #_, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            _, avg_acc, cnt, pred = accuracy(outputs1.cpu().numpy(), ### temp is just based of outputs1
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, outputs1.clone().cpu().numpy(), c, s) ### temp as well as here

            '''
            print('num_images is ', num_images)
            print('preds[:, :, 0:2] shape is', preds[:, :, 0:2].shape)
            print('all_preds shape is ', all_preds.shape)
            print('idx',idx)
            print('all_preds[idx:idx + num_images, :, 0:2] shape is ', all_preds[idx:idx + num_images, :, 0:2].shape)
            '''
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['frames_dir_list'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                #save_debug_images(config, input, meta, target, pred*4, output,
                #                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        print('     name_values is ', name_values)
        print('     perf_indicator is ', perf_indicator)
        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
