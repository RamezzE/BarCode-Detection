import numpy as np
import argparse
import imutils
import cv2

capture = cv2.VideoCapture(0) # 0 is webcam

while True:
    isTrue, frame = capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Calculate the minimum and maximum pixel values in the image
    min_val = np.min(gray)
    max_val = np.max(gray)

    # Define the desired minimum and maximum values after stretching
    new_min = 0
    new_max = 255

    # Apply contrast stretching using the formula:
    # new_pixel = (pixel - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
    stretched = ((gray - min_val) * (new_max - new_min) / (max_val - min_val) + new_min).astype(np.uint8)
    
    # gray = stretched

    ddepth = cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.add(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # blur and threshold the image
    blurred = cv2.blur(gradient, (5, 5))
    (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.intp(box)

    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)
    
    
    cv2.imshow('Video', frame)
    cv2.imshow('Gradient', gradient)
    # cv2.imshow('blurred', blurred)
    # cv2.imshow("Threshold", thresh)
    cv2.imshow("Closed", closed)
    # cv2.imshow("Stretched", stretched)


    # frame is displayed for n seconds to wait for input 
    if cv2.waitKey(1) & 0xFF == ord('d'): 
        break
    
capture.release()
cv2.destroyAllWindows()

