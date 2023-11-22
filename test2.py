import os
import cv2
import numpy as np
import imutils

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

        
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        # cv2.imshow('blur', blur)
        # cv2.waitKey()

        # Threshold and morph close
        # thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('thresh', thresh)
        # cv2.waitKey()

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('thresh', thresh)
        # cv2.waitKey()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,9))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # kernel = np.ones((5,5),np.uint8)
        # close = cv2.erode(close,kernel,iterations=1)
        # close = cv2.dilate(close, kernel, iterations=1)

        # cv2.imshow('close', close)
        # cv2.waitKey()

        # Find contours and filter using aspect ratio and area
        cnts = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        sortedContours = sorted(cnts, key=cv2.contourArea, reverse=True)

        # compute the rotated bounding box of the largest contour

        for cnt in sortedContours[:5]:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # draw a bounding box arounded the detected barcode
            cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
            
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        
    