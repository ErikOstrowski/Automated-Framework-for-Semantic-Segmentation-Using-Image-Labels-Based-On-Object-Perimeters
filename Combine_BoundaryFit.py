import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *


#######################################################################################################
# Code to Combine results from BoundaryFit with the CAM style of PuzzleCAM
#######################################################################################################

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--data_dir', default= './Dataset/VOC2012/' , type=str)


###############################################################################
# Inference parameters
###############################################################################

parser.add_argument('--domain', default='train_aug', type=str)


parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)




slic = './Dataset/BoundaryFit_slic' # 
quick = './Dataset/BoundaryFit_quick' # 



if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()


    pred_dir = create_directory(f'./experiments/predictions/Combined_BoundaryCAM/')


    set_seed(args.seed)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    #scale = args.scales
    # for mIoU
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)


# Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]


    eval_timer.tik()



    
    predict_folder = './experiments/predictions/ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0/'
    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size
            #print(f'\n IMAGE ID: {image_id}')
            npy_path = pred_dir + image_id + '.npy'
            if os.path.isfile(npy_path):
                continue


            strided_size = get_strided_size((ori_h, ori_w), 4)
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

            predict_dict = np.load(os.path.join(predict_folder, image_id + '.npy'), allow_pickle=True).item()

            hr_cams = predict_dict['hr_cam']
            strided_cams = predict_dict['cam']
            
            # load BoundaryFit.py results
            slic2 = np.load(slic + image_id + '.npy')
            quick2 = np.load(quick + image_id + '.npy')


            slic2[slic2 > 0] = 1
            quick2[quick2 > 0] = 1


            slic2 = torch.from_numpy(slic2)
            quick2 = torch.from_numpy(quick2)

            
            hr_cams = torch.from_numpy(hr_cams)
 
            slic_hr = np.load(slic + image_id + '.npy')
            quick_hr = np.load(quick + image_id + '.npy')


 
            quick_hr[quick_hr > 0] = 1
            slic_hr[slic_hr > 0] = 1

            slic_hr = torch.from_numpy(slic_hr)
            quick_hr = torch.from_numpy(quick_hr)

            # Conform BoundaryFit.py results with the hr_cams of PuzzleCAM
            slic_hr = [resize_for_tensors(slic_hr.unsqueeze(0), strided_size)[0]]
            slic_hr = torch.sum(torch.stack(slic_hr), dim=0)
            quick_hr = [resize_for_tensors(quick_hr.unsqueeze(0), strided_size)[0]]
            quick_hr = torch.sum(torch.stack(quick_hr), dim=0)


            keys = torch.nonzero(torch.from_numpy(label))[:, 0]

            keys = np.pad(keys + 1, (1, 0), mode='constant')

            # Combination happens here, we multiply both BoundaryCAM.py results with the original result and then add the original result
            for c in range(strided_cams.shape[0]):
                strided_cams[c] = (strided_cams[c] * slic_hr[c + 1] * quick_hr[c + 1]) + (strided_cams[c])
                

            for c in range(hr_cams.shape[0]):
                hr_cams[c] = (hr_cams[c] * slic2[c + 1] * quick2[c + 1] ) + (hr_cams[c])
                

            # Save
            np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})

            sys.stdout.write(
                '\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100,
                                                                  (ori_h, ori_w), hr_cams.size()))
            sys.stdout.flush()
        print()

        if args.domain == 'train_aug':
            args.domain = 'train'
    






