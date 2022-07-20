# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
# added
from core.loss import TemporalAffinityLoss
from core.loss import JointsMSELoss
# added

#
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

from models import pose_tokenpose_l_taf


from dataset.h36 import h36
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args() #returns user argument
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train') # creates a logger that is located at 'output/coco/pose_tokenpose_l/tokenpose_L_D24_256_192_patch43_dim192_depth24_heads12/'

    #calls on info() from imported create_logger() to print out the cfg
    logger.info(pprint.pformat(args)) 
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK #TRUE
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC #FALSE
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED #TRUE


    ##1## Initialize entire model: models.pose_tokenpose_l.get_pose_net(cfg, is_train=True) returns an object of the TokenPose_L class, which initalizes the tokenpose model by also initializing objects from TokenPose_L_base and HRNET_base class
    '''
    model = eval('pose_tokenpose_l_taf.get_pose_net')(
        cfg, is_train=True
    )
    '''
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    ) #works on both taf & normal l
    
    print("Model initialized in train.py line 95")


    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))
    #print('right before')

    logger.info(get_model_summary(model, dump_input))

    
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()# #model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()

    print('hi')
    ##2## Initialize loss function (criterion) and optimizer
  
    
    criterion1 = JointsMSELoss( #temporarily switched back to jointmseloss, should be tafloss
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    print("||Criterion1 (joint position loss) initialized: type is",type(criterion1), ' in train.py line 127')

    criterion2 = TemporalAffinityLoss( #temporarily switched back to jointmseloss, should be tafloss
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    print("||Criterion2 (TAF loss) initialized: type is",type(criterion2), ' in train.py line 127')
    

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    print("Initializing training dataset...")
    ##3.1## Initialize training dataset
      #dataset.coco(cfg, '/content/drive/MyDrive/*learning2022/research_project/TokenPose/data/coco', 'train2017', True, transforms.Compose([transforms.ToTensor(),normalize,]))
      #dataset folder, coco.py --> can use file name to call the only class in the file, which is COCODataset
    '''
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT
        )
    '''
    #currently cfg.DATASET.ROOT is not being used, should also add in an cfg.DATASET.TRAIN_SET and is_train to deal with testing situation & maybe even inference
    train_dataset = eval(cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT + cfg.DATASET.TRAIN_SET, cfg.DATASET.TRAIN_SET
        )

    print("||TRAINING DATASET initialized: train_dataset type is",type(train_dataset), ' in train.py line 140') # --> type is class 'dataset.coco.COCODataset'

  
    ##3.2## Initialize validation dataset
    valid_dataset = eval(cfg.DATASET.DATASET)(
      cfg, cfg.DATASET.ROOT + cfg.DATASET.TEST_SET, cfg.DATASET.TEST_SET
    )
    '''
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    print('||VALIDATION DATASET initialized: valid_dataset type is ', type(valid_dataset), ' in train.py line 150')

    '''

    ##4.1## Loading Training Data by initializing train_loader
    train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
      shuffle=cfg.TRAIN.SHUFFLE,
      num_workers=cfg.WORKERS,
      #pin_memory=cfg.PIN_MEMORY
  )
    print('||train_loader initialized in train.py line 161')

    
    ##4.2## Loading Validation Data by initializing valid_loader
    valid_loader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
      shuffle=cfg.TRAIN.SHUFFLE,
      num_workers=cfg.WORKERS,
    )
    '''
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    print('||valid_loader initialized in train.py line 171')
    '''

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    #print('final output dir', final_output_dir)

    ##5## Loading check points if there are any
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file) #loads the checkpoint file to continue off of the epoch, and perf
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict']) #this is where the previous weights/parameters gets loaded into the model

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    #?sets the learning rate
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    ) if cfg.TRAIN.LR_SCHEDULER is 'MultiStepLR' else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)
    
    print("||All checkpoints loaded!")


    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))
        # lr_scheduler.step()
        print("|ct1 - ")

        ## train for one epoch
        train(cfg, train_loader, model, criterion1, criterion2, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        print("||Training for current epoch {} completed --> train.py".format(epoch))

      
        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion1, criterion2,
            final_output_dir, tb_log_dir, writer_dict
        )
        print("||Validation for current epoch {} completed --> train.py".format(epoch))
        
        lr_scheduler.step()

        
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        #logging more info
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
