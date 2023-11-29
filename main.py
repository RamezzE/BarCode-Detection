import numpy as np
import imutils
import cv2
import os


def find_rectangles_within_contour(image, img2, contour, showProcess=False):
    # Create an empty mask for the specified contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1,
                     (255, 255, 255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, mask)

    # Convert the masked image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 3)

    # Perform edge detection, then perform dilation + erosion to
    # close gaps in between object edges
    edgeDetection = cv2.Canny(blurred, 100, 250)
    edgeDetection = cv2.convertScaleAbs(edgeDetection)

    kernel = np.ones((7, 7), np.uint8)

    # Calculate the mean brightness of the image
    brightness = calculate_brightness(image)

    if brightness > 150:  # Adjust thresholds for bright images
        erode_dilate = edgeDetection
        # erode_dilate = cv2.dilate(edgeDetection, kernel, iterations=1)
    else:
        erode_dilate = cv2.dilate(edgeDetection, kernel, iterations=1)
        erode_dilate = cv2.erode(erode_dilate, kernel, iterations=2)

    cnts = cv2.findContours(
        erode_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rectangle_contours = []
    for cnt in cnts:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            rectangle_contours.append(approx)
        elif len(approx) >= 4:
        # Check if the contour has almost equal side lengths, indicating a square
            sides = cv2.arcLength(approx, True) / 4
            approx_area = cv2.contourArea(approx)
            if 0.9 <= cv2.contourArea(approx) <= 1.1 * sides * sides and cv2.isContourConvex(approx):
                rectangle_contours.append(approx)
        else:
            return
        
        cnt = approx
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    if showProcess:
        cv2.imshow("1- Blurred", blurred)
        cv2.imshow("2- Canny", edgeDetection)
        cv2.imshow("3- Erode and Dilate", erode_dilate)
        
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def calculate_brightness(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the mean brightness of the image
    brightness = np.mean(gray)

    return brightness


def detectBarcodes(image, showProcess=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 3)

    # Perform edge detection, then perform dilation + erosion to
    # close gaps in between object edges
    edgeDetection = cv2.Canny(blurred, 100, 250)
    edgeDetection = cv2.convertScaleAbs(edgeDetection)

    kernel = np.ones((7, 7), np.uint8)

    # Calculate the mean brightness of the image
    brightness = calculate_brightness(image)

    if brightness > 150:  # Adjust thresholds for bright images
        erode_dilate = edgeDetection
        # erode_dilate = cv2.dilate(edgeDetection, kernel, iterations=1)
    else:
        erode_dilate = cv2.dilate(edgeDetection, kernel, iterations=1)
        erode_dilate = cv2.erode(erode_dilate, kernel, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(erode_dilate, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    largestContourArea = cv2.contourArea(
        sorted(cnts, key=cv2.contourArea, reverse=True)[0])

    for cnt in cnts:
        # Filter contours based on area
        if cv2.contourArea(cnt) < (largestContourArea/2):
            continue

        find_rectangles_within_contour(image, erode_dilate, cnt, True)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Draw a bounding box around each detected barcode
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    if showProcess:
        cv2.imshow("1- Blurred", blurred)
        cv2.imshow("2- Canny", edgeDetection)
        cv2.imshow("3- Erode and Dilate", erode_dilate)
        cv2.imshow("4- Closed", closed)

    # imageName = image_path.split(os.path.sep)[-1]
    # cv2.imshow(imageName, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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

        detectBarcodes(image, False)
