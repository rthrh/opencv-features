import cv2
import numpy as np
# import gnumpy as gpu


import random

#quantize
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq


def quantize(img):
    # reshaping the pixels matrix
    pixel = reshape(img,(img.shape[0]*img.shape[1],3))

    # performing the clustering
    centroids,_ = kmeans(pixel,8) # six colors will be found
    # quantization
    qnt,_ = vq(pixel,centroids)

    # reshaping the result of the quantization
    centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))
    clustered = centroids[centers_idx]
    return clustered

def main():
    # Start streaming images from your computer camera
    feed = cv2.VideoCapture(0) 

    # Define the parameters needed for motion detection
    alpha = 0.02 # Define weighting coefficient for running average
    motion_thresh = 35 # Threshold for what difference in pixels  
    running_avg = None # Initialize variable for running average

    kernel = np.ones((3,3),np.uint8)

    k = 31
    rand1 = 1
    toggleflag = 0
    frameNum = 0
    while True:
        frameNum += 1
        #get random numpy
        if frameNum % 5 == 0:    
            if toggleflag == 0:
                rand1 = rand1 + 1
            else:
                rand1 = rand1 - 1
                
            
            if rand1 >= 9:
                toggleflag = 1
            elif rand1 <= 1:
                toggleflag = 0
        
        # rand1 = random.randint(1,5)
        rand2 = random.randint(1,2)
        rand2 = rand1
        
        current_frame = feed.read()[1]
        width,height,channels = current_frame.shape
        # test_frame = current_frame
        

        current_frame = cv2.medianBlur(current_frame,5)
        current_frame = cv2.GaussianBlur(current_frame, (5,5),0)
        # quantized_frame = quantize(current_frame.astype('float')).astype('uint8')
        
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # denoised_frame = cv2.fastNlMeansDenoising(gray_frame,None)
        
        
        dil_frame = cv2.dilate(current_frame, kernel, iterations=rand1)
        ero_frame = cv2.erode(current_frame, kernel, iterations=rand2)
        morph_frame = dil_frame - ero_frame
        # morph_frame = cv2.dilate(morph_frame, kernel, iterations=1)
        

        # smooth_frame = cv2.GaussianBlur(gray_frame, (k, k), 0)
      

        # thresh_frame = cv2.adaptiveThreshold(gray_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                # cv2.THRESH_BINARY,11,2)
        

        # morph_thresh_frame = cv2.adaptiveThreshold(morph_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                # cv2.THRESH_BINARY,11,2)            

        # out_image = morph_thresh_frame - thresh_frame
        
        # If this is the first frame, making running avg current frame
        if running_avg is None:
            running_avg = np.float32(morph_frame) 
        # Find absolute difference between the current smoothed frame and the running average
        diff = cv2.absdiff(np.float32(morph_frame), np.float32(running_avg))
        # Then add current frame to running average after
        cv2.accumulateWeighted(np.float32(morph_frame), running_avg, alpha)
        # For all pixels with a difference > thresh, turn pixel to 255, otherwise 0
        _, subtracted = cv2.threshold(diff, motion_thresh, 1, cv2.THRESH_BINARY)

            
        
        cv2.imshow('morph_frame', morph_frame)
        # cv2.imshow('thresh frame', thresh_frame)
        # cv2.imshow('morph_thresh_frame', morph_thresh_frame)
        # cv2.imshow('filter frame', filter_frame)
        # cv2.imshow('out_image', ero_frame)
        # cv2.imshow('quantize', quantized_frame)
        # break
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
    feed.release()
    
    
    
    
main()