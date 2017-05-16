import cv2
import numpy as np
# import gnumpy as gpu


import random

from pst import *

#quantize
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq
from numpy.fft import fft2, ifft2, fftshift


def low_pass_filter(image,lpf=0.5):
    L = 0.5

    x = np.linspace(-L, L, image.shape[1])
    y = np.linspace(-L, L, image.shape[0])

    X, Y = np.meshgrid(x, y)

    THETA, RHO = np.arctan2(Y, X), np.hypot(X, Y)


    sigma = (lpf ** 2) / np.log(2)

    image_f = fft2(image)   
    image_f = image_f * fftshift(np.exp(-(RHO / np.sqrt(sigma))**2))
    image_filtered = np.real(ifft2(image_f))
    return image_filtered

def equalize(img):
    img_out = np.empty_like(img)
    cv2.equalizeHist(img,img_out)
    return img_out

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
    #### TODO #### IS NOT WORKING CORRECTLY WITH DOWN AND RIGHT SIDE OF IMAGE

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
        smooth_frame = cv2.medianBlur(current_frame,5)
        smooth_frame = cv2.GaussianBlur(smooth_frame, (5,5),0)
        
        # current_frame = equalize(current_frame)
        
        #RGB -> GRAY
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        #LOW PASS FILTER
        # low_pass_frame = low_pass_filter(gray_frame)
        # gray_frame = low_pass_frame
        
        #MORPHOLIGAL FILTER
        dil_frame = cv2.dilate(gray_frame, kernel, iterations=1)
        ero_frame = cv2.erode(gray_frame, kernel, iterations=1)
        morph_frame = dil_frame - ero_frame
        
        
        # BINARIZATION + EDGES
        th,morph_frame_bin = cv2.threshold(morph_frame,55,255,cv2.THRESH_BINARY)
        closing_frame =  cv2.erode(cv2.dilate(morph_frame_bin, kernel, iterations=1),kernel,iterations=1)

        #SOBEL FILTER
        sobel_frame = cv2.Sobel(smooth_frame, cv2.CV_8U, dx=1, dy=1) 
        
        
        
        #PIXELIZE
        pixel_frame = pixelize(smooth_frame,block_size = 12)

        #QUANTIZE
        # quantized_frame = quantize(smooth_frame,color_depth =5)

        #Phase Stretch Transform
        # pst_frame = pst(gray_frame, lpf=0.5, phase_strength=0.9, warp_strength=1.0, thresh_min=-0.7, thresh_max=0.9, morph_flag=False)
        
        # dil_frame = cv2.dilate(pst_frame, kernel, iterations=1)
        # ero_frame = cv2.erode(pst_frame, kernel, iterations=1)
        # pst_morph_frame = dil_frame - ero_frame        
        
        
        #PYRAMIDE MEAN SHIFT FILTERING --> 
        pyramide_frame = cv2.pyrMeanShiftFiltering(smooth_frame, 4,16,1)
        
        
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

        

        #DISPLAY
        cv2.imshow('gray_frame', gray_frame)
        # cv2.imshow('low_pass_frame', low_pass_frame)
        
        # cv2.imshow('current_frame', current_frame)
        # cv2.imshow('filter frame', filter_frame)
        cv2.imshow('pyramide_frame', pyramide_frame)
        # cv2.imshow('pst_frame', pst_frame)
        cv2.imshow('sobel_frame', sobel_frame)
        
        # break
        
        
        
        
        #USER KEYBOARD INPUT
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
    feed.release()
    
    
    
    
main()