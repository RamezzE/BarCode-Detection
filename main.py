import numpy as np
import argparse
import imutils
import cv2
import os

# load the image and convert it to grayscale


# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one

# closed = thresh



def rescaleFrame(frame, scale=0.5):
    # Works for Images, Videos and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Replace 'your_folder_path' with the actual path to your folder containing images
folder_path = 'Images'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate over the files
for file in files:
    # Check if the file is an image (you can customize the file extension check)
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Construct the full path to the image
        image_path = os.path.join(folder_path, file)
        
        # Read the image
        image = cv2.imread(image_path)
        
        if image.shape[0] > 800 or image.shape[1] > 1500:
            image = rescaleFrame(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Grayscale", gray)
        # cv2.waitKey(0)

        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction using OpenCV 2.4

        # ddepth = cv2.CV_64F
        # ksize = -1
        # gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=ksize)
        # gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=ksize)

        # subtract the y-gradient from the x-gradient
        # gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.Canny(gray, 150, 250)
        # gradient = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=1, ksize=1)
        gradient = cv2.convertScaleAbs(gradient)

        # cv2.imshow("Gradient X", gradX)
        # cv2.imshow("Gradient Y", gradY)
        # cv2.imshow("Gradient", gradient)
        # cv2.waitKey(0)

        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow("Closed", closed)
        # cv2.waitKey(0)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=1)
        closed = cv2.dilate(closed, None, iterations=1)

        # if image.shape[0] < 500 or image.shape[1] < 500:
        # 	# dont construct a closing kernel
        # 	closed = thresh

        # cv2.imshow("Closed", closed)
        # cv2.waitKey(0)
        
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        sortedContours = sorted(cnts, key=cv2.contourArea, reverse=True)

        # compute the rotated bounding box of the largest contour

        for cnt in sortedContours[:1]:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # draw a bounding box arounded the detected barcode
            cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
            
        cv2.imshow("Image", image)
        cv2.waitKey(0)

        