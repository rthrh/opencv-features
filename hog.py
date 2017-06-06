import cv2
import numpy as np
# import gnumpy as gpu
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths


def gamma_correction(img, correction = 2.0):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255.0)



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
        
        #GAMMA CORRECTION
        gamma_frame = gamma_correction(current_frame,correction=1.8)
        
        
        #IMAGE SMOOTH
        smooth_frame = cv2.medianBlur(gamma_frame,5)
        smooth_frame = cv2.GaussianBlur(smooth_frame, (5,5),0)
        

        
        
        # current_frame = equalize(current_frame)
        
        #RGB -> GRAY
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        

        
        #Histogram of Gradients (HOG)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        (rects, weights) = hog.detectMultiScale(gray_frame,winStride=(4, 4),padding=(8, 8),scale=1.05)
        

        image = current_frame
        orig = image.copy()
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
     
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
     
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
     
        # show some information on the number of bounding boxes
        # filename = imagePath[imagePath.rfind("/") + 1:]
        # print("[INFO] {}: {} original boxes, {} after suppression".format(
            # filename, len(rects), len(pick)))
     
        # show the output images
        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        print (len(rects))
        cv2.waitKey(1)
        
        
        
        #USER KEYBOARD INPUT
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
    feed.release()
    
    
    
    
main()