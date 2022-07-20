# subclass of MocapDataset
# - adjusts raw dataset according to camera parameters
# - initializes self._data, which gets passed into the iterative training loop in functions.py

from locale import normalize
import numpy as np
import torch
import cv2
import copy
import os
from VideoPose3D.common.skeleton import Skeleton
from VideoPose3D.common.camera import normalize_screen_coordinates, image_coordinates
import pathlib

import random
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

from collections import defaultdict
from collections import OrderedDict
from nms.nms import oks_nms
from nms.nms import soft_oks_nms
import json

import matplotlib.pyplot as plt
import numpy as np
import math

#logger = logging.getLogger(__name__)

import ipdb

from VideoPose3D.common.mocap_dataset import MocapDataset #parent class

h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70, # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    },
]

h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S2': [
        {},
        {},
        {},
        {},
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}

class h36(MocapDataset):
  def __init__(self, cfg, H36_image_data_root, image_set, remove_static_joints=True): #path = path to ground truth annotations
    super().__init__(cfg, 50) #idk why the last parameter couldn't be taken h36m_skeleton

    self.num_joints = cfg.MODEL.NUM_JOINTS #17

    #Load ground truth files (both joints & taf)
    self.gt_joints_path = "/content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/data_2d_h36m_gt.npz" #path of groundtruth 3d joint anotations (not converted into heatmap yet)
    self.joints_gt = np.load(self.gt_joints_path, allow_pickle=True)['positions_2d'].tolist()
    self.gt_taf_folder_path = "/content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/H36_taf_gt" #path of groundtruth taf heatmaps
    #self.taf_gt = np.load(self.gt_taf_path,allow_pickle=True)
    self.frame_rate = 5

    #Other image/heatmaps properties --> from JointsDataset.py
    self.color_rgb = cfg.DATASET.COLOR_RGB
    self.output_path = cfg.OUTPUT_DIR
    self.data_format = cfg.DATASET.DATA_FORMAT

    #self.heatmap_scale = cfg.DATASET.HEATMAP_SCALE

    self.scale_factor = cfg.DATASET.SCALE_FACTOR
    self.rotation_factor = cfg.DATASET.ROT_FACTOR
    self.flip = cfg.DATASET.FLIP
    self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
    self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
    self.color_rgb = cfg.DATASET.COLOR_RGB

    self.oks_thre = cfg.TEST.OKS_THRE
    self.in_vis_thre = cfg.TEST.IN_VIS_THRE
    self.soft_nms = cfg.TEST.SOFT_NMS

    self.target_type = cfg.MODEL.TARGET_TYPE
    self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
    self.original_image_size = np.array(cfg.MODEL.ORIGINAL_IMAGE_SIZE) #1000 x 1000
    self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
    self.sigma = cfg.MODEL.SIGMA
    self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
    self.joints_weight = 1

    self.image_set = image_set
    #self._cameras
    self.H36_image_data_root = H36_image_data_root
    #self.H36_image_data_root = "/content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/H36_video_data"
    '''
    self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
    for cameras in self._cameras.values():
      for i, cam in enumerate(cameras):
        cam.update(h36m_cameras_intrinsic_params[i])
        for k,v in cam.items():
          if k not in ['id','res_w', 'res_h']:
            cam[k] = np.array(v,dtype='float32')
        
                        
                # Normalize camera frame
        cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
        cam['focal_length'] = cam['focal_length']/cam['res_w']*2
        if 'translation' in cam:
            cam['translation'] = cam['translation']/1000 # mm to meters
        
        # Add intrinsic parameters vector
        cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                            cam['center'],
                                            cam['radial_distortion'],
                                            cam['tangential_distortion']))
    '''
    print("data")
    self.video_list = self._get_video_list() #total number of videos
    self._data = self._generate_db() #generates the equivalent of self.db where each value is a {} that contains image file path, and ground truth 3d joint position
    self.data_len = len(self._data)
    self.frames_per_vid = 25
    self.num_samples = self.frames_per_vid * self.data_len
    print("after")
    #from h36m_dataset.py in VideoPose3D

    '''
    if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8
    '''
  def __len__(self):
        return self.data_len

  def _get_video_list(self):
    total_video_list = []
    subjects_list = os.listdir(self.H36_image_data_root)
    for subject in subjects_list:
      subject_videos_list = os.listdir(os.path.join(self.H36_image_data_root,subject))
      total_video_list.extend(subject_videos_list)
    return total_video_list
    
  def _generate_db(self):
    gt_db = []
    for video_name in self.video_list:
      gt_db.append(self._generate_db_kernal(video_name))
      print(video_name)
    return gt_db

  def _generate_db_kernal(self, vid_name=str): #loads gt for both keypoints & taf
    
    db_rec = {} #each db_rec is for a new video

    # extract subject from vid_name ex 'S1_000007_Greeting 1.58860488.mp4'
    indices_w_underscore = []
    import re
    for i in re.finditer("_", vid_name):
      indices_w_underscore.append(i.start())

    #retrieving subject, action, video index
    subject = vid_name[:indices_w_underscore[0]] #ex: S1
    video_index = vid_name[indices_w_underscore[0]+1 : indices_w_underscore[1]] #ex: 000002
    first_period_index = vid_name.index(".")
    #print('first_period_index', first_period_index)
    action = vid_name[indices_w_underscore[1]+1 : first_period_index] #ex: Greeting 1

    print('_generate_db_kernal stage 1')

    #getting video number
    subject_folder_name =  subject
    subject_folder_directory = os.path.join(self.H36_image_data_root, subject_folder_name) #ex: /content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/H36_image_data/S1
    video_folders_list = os.listdir(subject_folder_directory)

    vid_name_wo_mp4 = pathlib.Path(vid_name).with_suffix(".mp4").stem #S1_000007_Greeting 1.58860488  
    video_serial = int(vid_name_wo_mp4[first_period_index+1:]) #int('58860488')

    videos_serial_list = []
    #print('folder list', video_folders_list)
    for video in video_folders_list: #going through every video in that certain video_folders_list to find the 4 serials that are for the same action, sort then and then return index as video number --> pretty inefficient rn
      #print('action', action)
      if action in video:
        new_name = action + ' 1'
        new_name2 = action + ' 2'
        if new_name not in video and new_name2 not in video:
          temp_video_wo_mp4 = pathlib.Path(video).with_suffix(".mp4").stem
          temp_first_period_index = video.index(".")
          temp_video_serial = int(temp_video_wo_mp4[temp_first_period_index+1:])
          #print('temp_video_serial', temp_video_serial)
          videos_serial_list.append(temp_video_serial)
    #print('videos_serial_list', videos_serial_list)
    if len(set(videos_serial_list)) != 4:
      #print(len(videos_serial_list))
      print('ERROR: total video number is not 4 in h36.py')
      raise ValueError
    videos_serial_list = sorted(videos_serial_list)
  
    for i in range(len(videos_serial_list)):
      if video_serial == videos_serial_list[i]:
        video_number = i



    print('_generate_db_kernal stage 2 video number generated',  video_number)

    #adding video folder directory for specified video to db_rec --> str
    video_folder_directory = os.path.join(subject_folder_directory, vid_name) #ex: /content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/H36_image_data/H36_S1_50fps/_ALL.55011271.mp4
    db_rec['video_dir'] = video_folder_directory
    frame_list = os.listdir(video_folder_directory)

    #adding frames_dir_list to db_rec --> list
    frame_dir_list = []
    frames_num = 0
    #for frame_name in range(frame_list): #just added the next line so that all frame dirs in frame_dir_list are sampled frames
    for frame_num in range(0, len(frame_list), self.frame_rate): #creates a list of the entire directory of each frame instead of just image names like 1.png
      #frame_name = video_name[] + '_' + frame_list[frame_num] + '.jpg'
      frame_dir_list.append(os.path.join(video_folder_directory, frame_list[frame_num]))
      frames_num+=1
    db_rec['frames_dir_list'] = frame_dir_list[:25] #ex: ['.../1.png', '.../2.png',...]

    #adding number of frames to db_rec --> int
    db_rec['frames_num'] = frames_num

    #adding positions_2d ground truth & temporal affinity fields ground truth to db_rec
    # add the sampling of frames
    
    positions_2d_every_frame = np.array(self.joints_gt[subject][action][video_number])
    positions_2d_all_sampled_frames = []
    for frame in range(0,len(positions_2d_every_frame), self.frame_rate):
      positions_2d_all_sampled_frames.append(positions_2d_every_frame[frame]) #MAIN gt 2d positions for heatmaps
    db_rec['positions_2d_all_frames'] = positions_2d_all_sampled_frames[:25]

    print('_generate_db_kernal stage 3')

    '''
    ## TEMPORAL AFFINITY FIELDS GENERATION (previous version)
      #compute TAF ground truth for a specific video
      x, y, z = self.xyt_retriever(subject, action, video_number)
      print('_generate_db_kernal stage 3.1')
      taf_all_frames = self.compute_TAF(x, y, z) #main
      print('_generate_db_kernal stage 3.2')
      #downsampling taf_maps to have the correct size ---> ######could add this to the code that generates the taf ground truth to speed things up
      
      print('_generate_db_kernal stage 3.3')

      db_rec['taf_all_frames'] = taf_all_frames
      '''

    ## TEMPORAL AFFINITY FIELDS GENERATION (current)
    vid_name = "vid"+str(video_number)+".npy"
    video_taf_path = os.path.join(self.gt_taf_folder_path, subject, action, vid_name)
    db_rec['taf_all_frames'] = np.load(video_taf_path)[:24]
    '''
    if len(np.load(video_taf_path)[0])>300:
      db_rec['taf_all_frames'] = np.load(video_taf_path)[:300]
    else:
      db_rec['taf_all_frames'] = np.load(video_taf_path)
    '''

    print('_generate_db_kernal stage 4')

    #creating positions_2d_all_frames_vis for heatmap generation --> keypoint position heatmaps will be generated later
    positions_2d_all_frames = np.array(db_rec['positions_2d_all_frames'].copy())
    positions_2d_all_frames_vis = positions_2d_all_frames
    #print('shape of positions_2d_all_frames', positions_2d_all_frames.shape)
    for frames in range(len(positions_2d_all_frames)):
      #print('shape per frame', positions_2d_all_frames[frames].shape)
      for joint in range(self.num_joints):
        #print('length should be 2 is:', len(positions_2d_all_frames[frames][joint]))
        for point in range(len(positions_2d_all_frames[frames][joint])):
          if positions_2d_all_frames[frames][joint][point] > 0:
            positions_2d_all_frames_vis[frames][joint][point] = 1
    db_rec['positions_2d_all_frames_vis'] = positions_2d_all_frames_vis
    
    print('len(positions_2d_all_frames) is ', len(db_rec['positions_2d_all_frames']), 'while len(taf_all_frames+1 is ', len(db_rec['taf_all_frames'])+1)
    assert len(db_rec['positions_2d_all_frames']) == len(db_rec['taf_all_frames'])+1, "taf gt & 2d positions gt are mismatched - 2d positions gt should have one more frame compared to taf gt per video"

    print("db_rec successfully created for video_name ", vid_name, " in _generate_db_kernal() in h36.py")
    '''
    center, scale = self._box2cs(obj['clean_bbox'][:4])
    
    db_rec['center'] = 
    db_rec['scale'] = 
    db_rec['score'] = 
    '''
    #may or may not need to add the equivalent of "joints_3d_vis" as a key to db_rec

    print('_generate_db_kernal stage 5')
    return db_rec
  
  def generate_target(self, joints, joints_vis): #taken from JointsDataset.py in tokenpose
        '''
        :param joints:  [num_joints, 2]
        :param joints_vis: [num_joints, 2]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0] #so target_weight is basically a 0 or 1 value for x coordinate of each joint. the shape is (17,1)

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],  # used to be self.heatmap_size[1], self.heatmap_size[0
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3 #sigma = 2

            for joint_id in range(self.num_joints):
              
                #ipdb.set_trace()

                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size) # parameters are [102,302], 1, 6 --> returns either 0 or target-weight[joint_id], which is 1
                
                if target_weight[joint_id] == 0:
                    continue

                #heatmap_scale = self.image_size[0]/self.heatmap_size[0] --> need to figure out if the magnitude of the pixels (rather than the position the numbered pixels are in)affect the heatmaps generated

                mu_x = joints[joint_id][0]#/self.heatmap_scale #newly added: accounts for the downsized heatmap size
                mu_y = joints[joint_id][1]#/self.heatmap_scale 

                #print('sdfjasldfkasdlfs in h36')
                
                # 生成过程与hrnet的heatmap size不一样
                x = np.arange(0, self.original_image_size[0], 1, np.float32) #used to be self.heatmap_size
                y = np.arange(0, self.original_image_size[1], 1, np.float32) # [0,1,2,3,4,...heatmap width]
                y = y[:, np.newaxis] #[[0],[1],[2],[3],[4],...heatmap width]

                v = target_weight[joint_id]
                from google.colab.patches import cv2_imshow
                from sklearn.preprocessing import normalize
                if v > 0.5:
                    joint_target = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2)) #shape: (48,64)
                    #####
                    #target =(target - np.min(target)) / (np.max(target) - np.min(target))

                    '''
                    print('target type is ',type(target))
                    print('target size is ', target.shape)
                    #print('target is nonzero at', np.nonzero(target[0]))
                    print('target[0][402][625] is', target[0][400][625])
                    print('max in target is ', np.amax(target))
                    print('min in target is ', np.amin(target))

                    x = target[0][398:403,624:637]
                    #print('x after',x)
                    plt.imshow(target[0], cmap='hot', interpolation='nearest')
                    plt.savefig('foo.png')

                    #print('got here1')
                    '''
                    joint_target = self.downsize(joint_target) #it might be a problem bec target is not a cv2 image
                    joint_target = self.downsize(joint_target)
                    joint_target = self.downsize(joint_target)
                    #print('target.shape is ',joint_target.shape)
                    target[joint_id] = joint_target[30:94, 38:86] #[102:358,154:346] #downsamples heatmap size to be (256,192)
                    #print('final target.shape is ',target.shape)

                    #ipdb.set_trace()
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        
        return target, target_weight


  def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # [93,103]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] #[109,134]
        #print('ul is ', ul)
        #print('br is ', br)
        if ul[0] >= self.original_image_size[0] or ul[1] >= self.original_image_size[1] or br[0] < 0 or br[1] < 0: #usesd to be heatmap_size
            # If not, just return the image as is
            target_weight = 0

        return target_weight
  
  def downsize(self, input_heatmap): #(shape is 1000, 1000)
        downsized_input_heatmap = np.zeros((int(input_heatmap.shape[0]/2), int(input_heatmap.shape[1]/2)))
        for r, row in enumerate(range(input_heatmap.shape[0])[::2]):
          for c, col in enumerate(range(input_heatmap.shape[1])[::2]):
            downsized_input_heatmap[r][c] = input_heatmap[row][col]
        return downsized_input_heatmap

  def __getitem__(self, idx): 
    '''
    loads video, retrieves ground truth labels from self._data (previously generated database) and pass it into training/testing loop
    '''
    #generate self.db w/ image path, 3d joint positions

    print('Loading in data for video index {}'.format(idx))

    db_rec = copy.deepcopy(self._data[10]) #for a specific video

    frames_dir_list = db_rec['frames_dir_list']
    #print('frames_dir_list length is', len(frames_dir_list))
    video_dir = db_rec['video_dir']
    
    video_numpy = [] #3d array size (frame_num, img_length = image_size[0], img_width = image_size[1])
    video_heatmap_target = [] #4d array size (frame_num, numjoints=17, length=heatmap[0], width=heatmap[1])
    video_heatmap_target_weight = [] #3d array size (frame_num, numjoints=17, 1)
    #video_meta = [] # [{},{},{}] where len(video_meta) = total number of frames in this video

    for index, frames_dir in enumerate(frames_dir_list): #setting max frame number
      #reading image at every frame
      image_numpy = cv2.imread(frames_dir, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

      #deals w case if image_numpy is none
      if image_numpy is None:
       print("Unable to load data at frame ", frames_dir, " in video ", video_dir," --skipped, next image...")
       return torch.zeros(3, 256, 192),torch.zeros(17, 64, 48), torch.zeros(17,1), torch.zeros(17, 64, 48), {'video_dir': video_dir, #temporary size from coco
                #'filename': '',
                #'imgnum': '',
                'position_2d_all_frames': torch.zeros(17, 3),
                'positions_2d_all_frames_vis': torch.zeros(17, 3),
                'center': torch.zeros(2),
                'scale': torch.zeros(2),
                'rotation': 0,
                'score': 0
                }
      if self.color_rgb:
        ''' no more centercropping, downsampling then cropping sides instead
        ##add center cropping bec current image size is too large
        transform = transforms.CenterCrop((self.image_size[0],self.image_size[1])) # (192,256)'''
        image_numpy = cv2.pyrDown(image_numpy)
        #image_numpy = image_numpy[98:402,136:364] #cropping out the sides into (304,228)
        image_numpy = image_numpy[102:358,154:346] #croppinng sides into (256,192)

        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)

        #extracting current frame number from frames_dir
        #import pathlib
        ''' #old way of finding cur_frame number according to the frame name, but after samplinng frames at a diff rate, it nolonger matches
        frame_dir_name = frames_dir
        frame_dir_name = pathlib.Path(frame_dir_name).with_suffix("") #ex: 'Discussion.55011271.mp4_frame2.jpg'--> 'Discussion.55011271.mp4_frame2'
        frame_dir_name = frame_dir_name[frame_dir_name.index('frame')+5:] #ex: 'Discussion.55011271.mp4_frame2' --> '2'
        cur_frame_ct = int(frame_dir_name) # MAIN current frame number
        '''

        cur_frame_ct = index

        positions_2d_cur_frame = db_rec['positions_2d_all_frames'].copy()[cur_frame_ct]
        positions_2d_cur_frame_vis = db_rec['positions_2d_all_frames_vis'].copy()[cur_frame_ct]

        positions_2d_cur_frame_heatmap = positions_2d_cur_frame.copy()

        #code self.generate_target in MocapDataset
        frame_heatmap_target, frame_heatmap_target_weight = self.generate_target(positions_2d_cur_frame_heatmap, positions_2d_cur_frame_vis) #generating target & target weight
        #frame_heatmap_target = torch.from_numpy(frame_heatmap_target)#.cuda() #turns frame_heatmap_target into tensor
        #frame_heatmap_target_weight = torch.from_numpy(frame_heatmap_target_weight)#.cuda()
        '''
        frame_meta = {'frames_dir': frames_dir, #temporary size from coco
                #'filename': '',
                #'imgnum': '',
                'position_2d_cur_frame': torch.tensor(positions_2d_cur_frame),
                'positions_2d_cur_frame_vis': torch.tensor(positions_2d_cur_frame_vis),
                #'center': torch.zeros(2),
                #'scale': torch.zeros(2),
                #'rotation': 0,
                #'score': 0
                }
        '''
        video_numpy.append(image_numpy)
        video_heatmap_target.append(frame_heatmap_target)
        video_heatmap_target_weight.append(frame_heatmap_target_weight)


        # need to pass in taf ground truth map as well
        #video_meta.append(frame_meta)
        #print('done loading frames_dir ', type(frames_dir), ' at index ', index)

    center = torch.zeros(17, 2)
    for i in range(len(center)):
      center[i][0] = 128
      center[i][1] = 98


    video_meta = {'video_dir': video_dir, #temporary size from coco
                #'filename': '',
                #'imgnum': '',
                'frames_dir_list': frames_dir_list, #list of all the frames directory
                'positions_2d_all_frames': torch.tensor(db_rec['positions_2d_all_frames'].copy()),
                'positions_2d_all_frames_vis': torch.tensor(db_rec['positions_2d_all_frames_vis'].copy()),
                'center': torch.tensor([128,98]), #torch.zeros(2)
                'scale': torch.ones(2), #torch.zeros(2)
                'rotation': 0,
                'score': 0
                }

    
    input = torch.tensor(video_numpy)#.cuda()
    #print('type', type(video_heatmap_target))
    video_heatmap_target = torch.tensor(video_heatmap_target)
    video_heatmap_target_weight = torch.tensor(video_heatmap_target_weight)
    video_taf_target = torch.tensor(db_rec['taf_all_frames'])#.cuda() #shape: (frames, joints, 64, 48, 2)
   
  
    return input, video_heatmap_target, video_heatmap_target_weight, video_taf_target, video_meta

  def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                *args, **kwargs):
      rank = cfg.RANK
      img_path = list(img_path)
      '''
      print('output_dir is ', output_dir)
      print('preds shape is ', preds.shape)
      print('all_boxes shape is', all_boxes.shape)
      print('image path is', img_path)
      '''
      #creating directories/prepping output_dir/results/keypoints_{}_results_{}.json
      res_folder = os.path.join(output_dir, 'results') 
      if not os.path.exists(res_folder):
          try:
              os.makedirs(res_folder)
          except Exception:
              logger.error('Fail to make {}'.format(res_folder))

      res_file = os.path.join( # output_dir/results/keypoints_{}_results_{}.json
          res_folder, 'keypoints_{}_results_{}.json'.format(
              self.image_set, rank)
      )

      # person x (keypoints)
      _kpts = []
      '''
      print('pred len is', len(preds)) #300 ; preds shape is  (300, 17, 3)
      print('img_path len is', len(img_path))
      '''
      for idx, kpt in enumerate(preds): #kpt is size(17,2) in (25,17,2)
          #print('img_path[idx][-33:-4] is', type(img_path[idx][-33:-4]))
          _kpts.append({
              'keypoints': kpt,
              'center': all_boxes[idx][0:2],
              'scale': all_boxes[idx][2:4],
              'area': all_boxes[idx][4],
              'score': all_boxes[idx][5],
              'image': img_path[idx][-33:-4] #used to be int(img_path[idx][-16:-4]) ---> is currently a tuple, which might be an issue
          })

      # image x person x (keypoints)
      kpts = defaultdict(list)
      for kpt in _kpts: #_kpts is a list of dicts
          kpts[kpt['image']].append(kpt) #kpts = ['img id':{}]

      # rescoring and oks nms
      num_joints = self.num_joints #
      in_vis_thre = self.in_vis_thre #
      oks_thre = self.oks_thre #
      oks_nmsed_kpts = []
      for img in kpts.keys():
          img_kpts = kpts[img] #retrieving the dicts for each image
          print('kpts type is ', type(img_kpts)) #should be dict
          for n_p in img_kpts:
              box_score = n_p['score']
              kpt_score = 0
              valid_num = 0
              for n_jt in range(0, num_joints): #loop through the joints
                  t_s = n_p['keypoints'][n_jt][2] #retrieve t_s aka the pred for that key point and we change the kpt_score if the it is greated than in_vis_there
                  if t_s > in_vis_thre: # >0.2
                      kpt_score = kpt_score + t_s
                      valid_num = valid_num + 1
              if valid_num != 0:
                  kpt_score = kpt_score / valid_num
              # rescoring
              n_p['score'] = kpt_score * box_score

          if self.soft_nms: #
              keep = soft_oks_nms(
                  [img_kpts[i] for i in range(len(img_kpts))],
                  oks_thre
              )
          else:
              keep = oks_nms( #imported in
                  [img_kpts[i] for i in range(len(img_kpts))],
                  oks_thre
              )

          if len(keep) == 0:
              oks_nmsed_kpts.append(img_kpts)
          else:
              oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])
      
      self._write_coco_keypoint_results(
          oks_nmsed_kpts, res_file)
      if 'test_set' not in self.image_set:
          info_str = self._do_python_keypoint_eval(
              res_file, res_folder)
          name_value = OrderedDict(info_str)
          return name_value, name_value['AP']
      else:
          return {'Null': 0}, 0 #wouldn't it return this for test_set everytime?

  def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

  '''
  def _generate_db(self, path, remove_static_joints):
    gt_db = []
    for index in self.image_set_index:
      gt_db.extend(self._generate_db_kernal(index))
    return gt_db
    
  def _generate_db_kernal(self, index):
    #Load serialized dataset of ground truth
    gt_joints_path = "/content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/data_2d_h36m_gt.npz" #path of groundtruth 3d joint anotations (not converted into heatmap yet)
    gt_taf_path = "/content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/h36_taf_gt.npy" #path of groundtruth taf heatmaps
    
    joints_data = np.load(gt_joints_path,allow_pickle=True)['positions_3d'].items()
    img_path = getimgpath() ###finish & code getimgpath

    db = {}
    for subject, actions in joints_data.items():
      self._data[subject] = {}
      for action_name, positions in actions.items():
        db[subject][actions][action_name] = {
          'joints_3d': positions,
          'cameras': self._cameras[subject],
        }

  def _generate_db_kernal(self): #### OLD VERSION WHERE db_rec for EVERY SINGLE VIDEO IS ADDED IN, newer version above only does it for the video_name passed in
    
    #Load ground truth files (both joints & taf)
    gt_joints_path = "/content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/data_2d_h36m_gt.npz" #path of groundtruth 3d joint anotations (not converted into heatmap yet)
    joints_gt = np.load(gt_joints_path, allow_pickle=True)['positions_2d'].tolist()
    gt_taf_path = "/content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/h36_taf_gt.npy" #path of groundtruth taf heatmaps
    taf_gt = np.load(gt_taf_path,allow_pickle=True)

    #defining some variables to parse through
    subjects_list = h36m_cameras_extrinsic_params.keys() #['S1','S2','S3',...]
    indexed_subjects_list = subjects_list[:1] #temperary because currently I only have data for s1

    for subject in indexed_subjects_list:
      subject_folder_name =  "H36_" + subject + "_50fps" #ex: H36_S1_50fps
      subject_folder_directory = os.path.join(self.H36_image_data_root, subject_folder_name) #ex: /content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/H36_image_data/H36_S1_50fps
      video_folders_list = os.lisdir(subject_folder_directory)

      for index, video_name in enumerate(video_folders_list): #ex: _ALL.55011271.mp4
        db_rec = {} #each db_rec is for a new video

        action_name = video_name[7:-13] #ex: '_ALL.55011271.mp4' --> '_ALL'

        if index <= 3: #decides video number (0-3) based on index in video_folders_list --> assumes that there are 4 videos per action and there's no missing videos
          video_number = index
        elif index % 4 == 0:
          video_number = 0
        elif index % 4 == 1:
          video_number = 1
        elif index % 4 == 2:
          video_number = 2
        elif index % 4 == 3:
          video_number = 3
        else:
          print('video index out of bounds')
          raise KeyError


        video_folder_directory = subject_folder_directory + video_name #ex: /content/drive/MyDrive/*learning2022/research_project/VideoPose3D/data/H36_image_data/H36_S1_50fps/_ALL.55011271.mp4
        db_rec['video_dir'] = video_folder_directory
        frame_list = os.lisdir(video_folder_directory)
        frame_dir_list = []
        for frame_name in frame_list: #creates a list of the entire directory of each frame instead of just image names like 1.png
          frame_dir_list.append(video_folder_directory+frame_name)
        db_rec['frames_dir_list'] = frame_dir_list #ex: ['.../1.png', '.../2.png',...]
        db_rec['frames_num'] = len(frame_list)

        db_rec['positions_2d_all_frames'] = joints_gt[subject][action_name][video_number] #MAIN gt 2d positions for heatmaps
        db_rec['taf_all_frames'] = taf_gt[subject][action_name][video_number] #MAIN gt temporal affinity fields

        assert len(db_rec['positions_2d_all_frames']) == len(db_rec['taf_all_frames'])+1, "taf gt & 2d positions gt are mismatched - 2d positions gt should have one more frame compared to taf gt per video"


      # Code from coco.py
      c = db_rec['center']
      s = db_rec['scale']
      score = db_rec['score'] if 'score' in db_rec else 1
      r = 0

      if self.is_train:
          if (np.sum(positions_2d_all_frames_vis[:, 0]) > self.num_joints_half_body
              and np.random.rand() < self.prob_half_body):
              c_half_body, s_half_body = self.half_body_transform(
                  positions_2d_all_frames, positions_2d_all_frames_vis
              )

              if c_half_body is not None and s_half_body is not None:
                  c, s = c_half_body, s_half_body

          sf = self.scale_factor
          rf = self.rotation_factor
          s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
          r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
              if random.random() <= 0.6 else 0

          if self.flip and random.random() <= 0.5:
              image_numpy = image_numpy[:, ::-1, :]
              joints, joints_vis = fliplr_joints(
                  positions_2d_all_frames, positions_2d_all_frames_vis, image_numpy.shape[1], self.flip_pairs)
              c[0] = image_numpy.shape[1] - c[0] - 1

      joints_heatmap = joints.copy()
      trans = get_affine_transform(c, s, r, self.image_size)
      trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

      input = cv2.warpAffine(
          image_numpy,
          trans,
          (int(self.image_size[0]), int(self.image_size[1])),
          flags=cv2.INTER_LINEAR)

      if self.transform:
          input = self.transform(input)

      for i in range(self.num_joints):
          if joints_vis[i, 0] > 0.0:
              joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
              joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)

      target, target_weight = self.generate_target(joints_heatmap, joints_vis)'''
    



    
    

