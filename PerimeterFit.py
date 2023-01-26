from PIL import Image, ImageDraw
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--segment", default='slic', type=str)
parser.add_argument("--cam_path", default='./experiments/predictions/ResNeSt101@Puzzle@optimal@train@scale=0.5,1.0,1.5,2.0/' , type=str)
args = parser.parse_args()
segment = args.segment

cam_path =  args.cam_path


if segment == 'quick':
	predict_path = './Dataset/PerimeterFit_quick/'
	Edges_path ='./Dataset/PM_quick/'
	segs = './Dataset/USS_quick/'
else:
	predict_path = './Dataset/PerimeterFit_slic/'
	Edges_path ='./Dataset/PM_slic/'
	segs = './Dataset/USS_slic/'
	

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


counter=0


for obj in os.listdir(cam_path):
      if not(os.path.isfile(predict_path + obj)):
        name = obj[:-4]
        img = Image.open(segs + name + '.jpg')
        edges = Image.open(Edges_path + name + '.jpg')

        #PREVENT WHITE IN THE ORIGINAL IMAGE TO MAKE EDGES DISSAPEAR
        img = np.array(img)
        m = np.amax(img)
        img = (img / m)*255
        img = img.astype(int)


        # Threshold for edges
        edges = np.array(edges)
        if segment == 'quick':
        
          edges[edges < 200] = 0 # 255
          edges[edges > 0] = 255
    		
        else:
        
          edges[edges < 250] = 0
          edges[edges > 0] = 255

	# normalize edges
        edges[edges == 0] = 1
        edges[edges > 1] = 0
	
	# create the negative version of the edge map
        negative = Image.open(Edges_path + name + '.jpg')
        negative = np.array(negative)
        if segment == 'quick':
          negative[negative > 200] = 255
          negative[negative < 255] = 0
          negative[negative == 255] = 1
        else:
          negative[negative > 250] = 255
          negative[negative < 255] = 0
          negative[negative == 255] = 1

	#read intial CAM
        predict_dict2 = np.load(os.path.join(cam_path, name + '.npy')).item()
        cams = predict_dict2['hr_cam']
        cams2 = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.0)
        if segment == 'quick':
          rough_cam = cams2
          rough_cam[rough_cam > 0] = 1
          rough_cam[rough_cam < 1] = 0
        else:
          rough_cam = cams2 
          rough_cam[rough_cam > .050] = 255
          rough_cam[rough_cam < 255] = 0

	# create space for result mask
        end = np.zeros(cams2.shape) # [num_classes, H, W]
        
	# loop for every class
        for cla in range(cams2.shape[0]):

            # prepare img for 
            add = img
            add = Image.fromarray((add).astype(np.uint8))

            rep_value = (0, 0, 0)

            # Floodfill to go over every pixel, if pixel is negative, turn the patch negative
            _, h, w = rough_cam.shape
            for k in range(h - 1):
                        for j in range(w - 1):
                            if rough_cam[cla, (k), (j)] == 0:
                                for b in range(3):
                                    ImageDraw.floodfill(add, (j, k), rep_value, thresh=10)
            
            add = np.array(add)
            add = np.sum(add, axis = 2)

            if segment == 'quick':
              add[add > 0] = 1 
              add = add + negative
              add[add > 0] = 255
            else:
              add[add > 0] = 255  
              add = (add + negative) * rough_cam[cla]
              add[add > 0] = 255
            


            #add = Image.fromarray((add).astype(np.uint8))
            #add.show()
            end[cla] = add
            #break
        #break
        out = predict_path + obj[:-4] + '.npy'
        np.save(out, end)
        counter +=1
        print(counter)

# restore np.load for future normal usage
np.load = np_load_old
