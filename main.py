import numpy as np
import imutils
import cv2
import os

def rescaleFrame(frame, scale=0.5):
    # Works for Images, Videos and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def detectBarcode(image, showProcess=False):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # blurred = cv2.bilateralFilter(gray, 9, 25, 75)
    # blurred = cv2.bilateralFilter(gray, 9, 25, 25)
    blurred = cv2.medianBlur(gray, 3)
    
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edgeDetection = cv2.Canny(blurred, 100, 250)
    edgeDetection = cv2.convertScaleAbs(edgeDetection)
    
    # perform a series of erosions and dilations
    # kernel = np.ones((3, 3), np.uint8)
    # erode_dilate = cv2.dilate(edgeDetection, kernel, iterations=2)
    # kernel = np.ones((9, 9), np.uint8)
    # erode_dilate = cv2.erode(erode_dilate, kernel, iterations=1)
    
    kernel = np.ones((7, 7), np.uint8)
    erode_dilate = cv2.dilate(edgeDetection, kernel, iterations=1)
    erode_dilate = cv2.erode(erode_dilate, kernel, iterations=2)

    # erode_dilate = edgeDetection
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(erode_dilate, cv2.MORPH_CLOSE, kernel)

    # if image.shape[0] < 500 or image.shape[1] < 500:
    # 	# dont construct a closing kernel
    # 	closed = thresh
    
    # closed = erode_dilate
    
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sortedContours = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # loop over the contours
    # compute the rotated bounding box of the largest contour

    for cnt in sortedContours[:1]:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # draw a bounding box arounded the detected barcode
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        
    if showProcess:
        # cv2.imshow("1- Gray", gray)
        cv2.imshow("1- Blurred", blurred)
        cv2.imshow("2- Canny", edgeDetection)
        cv2.imshow("3- Erode and Dilate", erode_dilate)
        cv2.imshow("4- Closed", closed)

    imageName = image_path.split('\\')[-1]
    cv2.imshow(imageName, image)
    cv2.waitKey(0)

folderPath = 'Images'

# List all files in the folder
files = os.listdir(folderPath)

for file in files:
    
    # Check if the file is an image
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Construct the full path to the image
        image_path = os.path.join(folderPath, file)
        
        # Read the image
        image = cv2.imread(image_path)
        
        if image.shape[0] > 800 or image.shape[1] > 1500:
            image = rescaleFrame(image)
            
        detectBarcode(image, False)
        