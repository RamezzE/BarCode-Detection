import numpy as np
import argparse
import imutils
import cv2

def rescaleFrame(frame, scale = 0.5):
    # Works for Images, Videos and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])

if image.shape[0] > 800 or image.shape[1] > 1500:
    image = rescaleFrame(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

# normalize float versions
norm = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# scale to uint8
norm = (255*norm).astype(np.uint8)

cv2.imshow("Normalized", norm)
cv2.waitKey(0)

gray = norm

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4

ddepth = cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
# gradX = cv2.Scharr(gray, ddepth=ddepth, dx=1, dy=0)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
# gradY = cv2.Scharr(gray, ddepth=ddepth, dx=0, dy=1)
# gradient = np.sqrt(gradX**2 + gradY**2)
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
# gradient = cv2.add(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# cv2.imshow("Gradient X", gradX)
# cv2.imshow("Gradient Y", gradY)
cv2.imshow("Gradient", gradient)
cv2.waitKey(0)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))

print(np.mean(image))

# if the image is bright, dont blur
# probably only the barcode exists in the image on a white background
if np.mean(image) > 150:
    # dont blur
	blurred = gradient

(_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)


cv2.imshow("Threshold", thresh)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Closed", closed)
cv2.waitKey(0)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

# if image.shape[0] < 500 or image.shape[1] < 500:
# 	# dont construct a closing kernel
# 	closed = thresh

cv2.imshow("Closed", closed)
cv2.waitKey(0)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one

# closed = thresh

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
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)