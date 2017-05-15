import cv2
import numpy as np
# import gnumpy as gpu


import random

#quantize
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq




def quantize(img,color_depth = 2):
    bit_rate_start = np.array(8).astype('uint8')
    bit_rate = np.array(color_depth).astype('uint8')

    rows = img.shape[0]
    cols = img.shape[1]
    
    # for i in range(0,rows):
        # for j in range(0,cols):
            
    
    
    img_out = np.empty_like(img)
    
    img_out[:] = np.round(pow(2,bit_rate_start)*img[:])/pow(2,bit_rate);
    

    # if color_depth == 8:
        # print ('Input bit rate per channel is the same as output bit rate')
        # return img
        
    # else:
    return img_out

def pixelize(img,block_size = 8):

    rang = block_size
    rows = img.shape[0]
    cols = img.shape[1]
    
    img_out = np.empty_like(img)
    
    for i in range(0,rows,block_size):
        for j in range(0,cols,block_size):
            if (i < rows - block_size) and (j < cols - block_size):
                mean = np.array([np.mean(img[i:i+rang,j:j+rang,0]),np.mean(img[i:i+rang,j:j+rang,1]),np.mean(img[i:i+rang,j:j+rang,2])])
                img_out[i:i+rang,j:j+rang,:] = mean.astype('uint8')
            else:
                if (i >= rows - block_size):
                    mean = np.array([np.mean(img[i:-1,j:j+rang,0]),np.mean(img[i:-1,j:j+rang,1]),np.mean(img[i:-1,j:j+rang,2])])
                    img_out[i:-1,j:j+rang,:] = mean.astype('uint8') #pionowe      
                elif (j >= cols - block_size):
                    mean = np.array([np.mean(img[i:i+rang,j:-1,0]),np.mean(img[i:i+rang,j:-1,1]),np.mean(img[i:i+rang,j:-1,2])])
                    img_out[i:i+rang,j:-1,:] = mean.astype('uint8') # poziome               
                else:
                    pass
    return img_out
                



def main():
    # Start streaming images from your computer camera
    feed = cv2.VideoCapture(0) 

    # Define the parameters needed for motion detection
    alpha = 0.02 # Define weighting coefficient for running average
    motion_thresh = 35 # Threshold for what difference in pixels  
    running_avg = None # Initialize variable for running average

    kernel = np.ones((3,3),np.uint8)

    k = 31
    toggleflag = 0
    frameNum = 0
    while True:
        frameNum += 1
        
        current_frame = feed.read()[1]
        width,height,channels = current_frame.shape
        
        #IMAGE SMOOTH
        current_frame = cv2.medianBlur(current_frame,5)
        current_frame = cv2.GaussianBlur(current_frame, (5,5),0)
        
        #RGB -> GRAY
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        #MORPHOLIGAL FILTER
        dil_frame = cv2.dilate(current_frame, kernel, iterations=1)
        ero_frame = cv2.erode(current_frame, kernel, iterations=1)
        morph_frame = dil_frame - ero_frame

        #PIXELIZE
        pixel_frame = pixelize(current_frame,block_size = 12)

        #QUANTIZE
        quantized_frame = quantize(current_frame,color_depth =5)
        unikaty = np.unique(quantized_frame)
        print('unikaty:',len(unikaty))
        
        
        #DIFF CALCULATIONS
        # If this is the first frame, making running avg current frame
        if running_avg is None:
            running_avg = np.float32(morph_frame) 
        # Find absolute difference between the current smoothed frame and the running average
        diff = cv2.absdiff(np.float32(morph_frame), np.float32(running_avg))
        # Then add current frame to running average after
        cv2.accumulateWeighted(np.float32(morph_frame), running_avg, alpha)
        # For all pixels with a difference > thresh, turn pixel to 255, otherwise 0
        _, subtracted = cv2.threshold(diff, motion_thresh, 1, cv2.THRESH_BINARY)

        
        
        #
        pix_dil = cv2.dilate(pixel_frame, kernel, iterations=1)   
        pix_ero = cv2.erode(pixel_frame, kernel, iterations=1)
        test_out = pix_dil - pix_ero
        
        
        #
        diff_image = current_frame - quantized_frame
        
        #DISPLAY
        cv2.imshow('morph_frame', morph_frame)
        cv2.imshow('pixel_frame', pixel_frame)
        # cv2.imshow('current_frame', current_frame)
        # cv2.imshow('filter frame', filter_frame)
        cv2.imshow('quantized_frame', quantized_frame)
        cv2.imshow('test_out', test_out)
        
        # cv2.imshow('diff_image', diff_image)
        # break
        
        
        
        
        #USER KEYBOARD INPUT
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
    feed.release()
    
    
    
    
main()