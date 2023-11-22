import numpy as np
import argparse
import imutils
import cv2

def rescaleFrame(frame, scale=0.5):
    # Works for Images, Videos and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
if image.shape[0] > 800 or image.shape[1] > 1500:
    image = rescaleFrame(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
cv2.imshow('blur', blur)
cv2.waitKey()

# Threshold and morph close
# thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('thresh', thresh)
cv2.waitKey()

_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('thresh', thresh)
cv2.waitKey()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,9))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# kernel = np.ones((5,5),np.uint8)
# close = cv2.erode(close,kernel,iterations=1)
# close = cv2.dilate(close, kernel, iterations=1)

cv2.imshow('close', close)
cv2.waitKey()

# Find contours and filter using aspect ratio and area
cnts = cv2.findContours(close.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
sortedContours = sorted(cnts, key=cv2.contourArea, reverse=True)

# compute the rotated bounding box of the largest contour

for cnt in sortedContours[:3]:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # draw a bounding box arounded the detected barcode
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    
cv2.imshow("Image", image)
cv2.waitKey(0)


# Find contours and filter using threshold area
# cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# min_area = 100
# max_area = 1500
# image_number = 0
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area > min_area and area < max_area:
#         x,y,w,h = cv2.boundingRect(c)
#         ROI = image[y:y+h, x:x+w]
#         cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#         image_number += 1

# cv2.imshow('image', image)
# cv2.waitKey()
