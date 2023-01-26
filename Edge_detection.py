import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images
from PIL import Image
import cv2
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--segment", default='slic', type=str)
args = parser.parse_args()


segment = args.segment


if segment == 'quick':
	segs = './Dataset/USS_quick/'
	out = './Dataset/PM_quick/'
else:
	segs = './Dataset/USS_slic/'
	out = './Dataset/PM_slic/'


for obj in os.listdir(segs):
        # read image
        path = segs + obj
        image = cv2.imread(path)
        # gray image out
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blurr image to close holes in almost closed objects
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	# perform edge detection        
        edge = cv2.Canny(blurred, 1, 50)
        cv2.imwrite(out + obj,edge)
