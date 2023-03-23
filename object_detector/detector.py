import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import joblib
import cv2
from config import *
from skimage import color
import matplotlib.pyplot as plt 
import os 
import glob
import csv
def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.

    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detector(filename):
    im = cv2.imread(filename)
    min_wdw_sz = (64, 128)
    step_size = (5, 5)
    downscale = 1.25

    clf = joblib.load(os.path.join(model_path, 'svm.model'))

    #List to store the detections
    detections = []
    #The current scale of the image 
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale = downscale):
        #The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            fd = hog(im_window, orientations = 9, pixels_per_cell= (8,8), cells_per_block = (2,2), visualize = False)

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)
            #print('clf.decision_function(fd):',clf.decision_function(fd))
            if pred == 1:
                
                if clf.decision_function(fd) > 0.5  :
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                    int(min_wdw_sz[0] * (downscale**scale)),
                    int(min_wdw_sz[1] * (downscale**scale))))
                 

            
        scale += 1

    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)
        
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print ("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    #print ("shape, ", len(pick))
    #print(pick)
    for(xA, yA, xB, yB) in pick:
        # img_crop=clone[yA:yB,xA:xB]
        # # print(type(filename))
        # # print([xA,yA,xB,yB, filename[82:]])
        # data=[xA,yA,xB,yB, filename[82:]]
        # with open(r'/home/quocanhlee/Desktop/human-detector-master/object_detector/results/test_level5_SVM.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(data)
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #     plt.axis("off")
    #     plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
    #     plt.title("People crop:")
    #     plt.show()
        
    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.title("Raw Detection before NMS")
    # plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()
    
   
def test_folder(foldername):

    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        detector(filename)

if __name__ == '__main__':
    foldername = r'/home/quocanhlee/Desktop/human-detector-master/object_detector/test_image/Level 3'
    test_folder(foldername)
