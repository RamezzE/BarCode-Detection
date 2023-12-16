import numpy as np
import cv2
import os
import imutils

# True :: Show steps
# False :: Hide steps
showProcess = False


def rescale_frame(frame, scale=0.5):
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


# Refine & Validate Barcode Regions
def detect_barcodes2(image, contour):
    # Draw the contour on a mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    masked_image = cv2.bitwise_and(image, mask)

    if showProcess:
        cv2.imshow("Masked Image", masked_image)

    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Canny Edge Detection
    edge_detection = cv2.Canny(blurred, 50, 225, apertureSize=3)

    if showProcess:
        cv2.imshow("Edge", edge_detection)
        cv2.waitKey(0)

    # Hough Line Transform - Detect lines
    lines = cv2.HoughLinesP(edge_detection, rho=1.25, theta=np.pi / 150, threshold=25, minLineLength=100, maxLineGap=10)

    blank = np.zeros(image.shape, dtype=np.uint8)

    if lines is None or len(lines) < 2:
        print("No lines or < 2")
        return False

    for i in range(len(lines)):
        cv2.line(blank, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
                 cv2.LINE_AA)

    if showProcess:
        cv2.imshow("All Lines", blank)
        cv2.waitKey(0)

    if len(lines) < 10:
        return contour

    red_pixels_x = np.where(blank[:, :, 2] == 255)[1]

    x_min = np.min(red_pixels_x)
    x_max = np.max(red_pixels_x)

    red_pixels_y = np.where(blank[:, :, 2] == 255)[0]

    y_min = np.min(red_pixels_y)
    y_max = np.max(red_pixels_y)

    print("Min X:", x_min)
    print("Max X:", x_max)
    print("Min Y:", y_min)
    print("Max Y:", y_max)

    cnt = contour.copy()

    for point in cnt:
        x, y = point[0]

        if x < x_min:
            x = x_min
        elif x > x_max:
            x = x_max

        if y < y_min:
            y = y_min
        elif y > y_max:
            y = y_max

        point[0] = [x, y]

    return cnt


# Detect Barcode Regions
def detect_barcodes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.medianBlur(gray, 3)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edge_detection = cv2.Canny(blurred, 75, 225)
    edge_detection = cv2.convertScaleAbs(edge_detection)

    if showProcess:
        cv2.imshow("1 - Edge", edge_detection)

    kernel = np.ones((7, 7), np.uint8)

    brightness = calculate_brightness(image)

    # Skip dilation and erosion if image is bright
    if brightness > 170:
        erode_dilate = edge_detection
    else:
        erode_dilate = cv2.dilate(edge_detection, kernel, iterations=1)
        if showProcess:
            cv2.imshow("2 - Dilate", erode_dilate)
        erode_dilate = cv2.erode(erode_dilate, np.ones((11, 11), np.uint8), iterations=1)
        if showProcess:
            cv2.imshow("3 - Erode", erode_dilate)

    # Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(erode_dilate, cv2.MORPH_CLOSE, kernel)

    if showProcess:
        cv2.imshow("4 - Closed", closed)

    cnts = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        print("No contours found")
        return image

    largest_contour_area = cv2.contourArea(
        sorted(cnts, key=cv2.contourArea, reverse=True)[0])

    for cnt in cnts:
        # Filter contours based on area
        if cv2.contourArea(cnt) < (largest_contour_area / 2):
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        box_original = box.copy()

        cnt = detect_barcodes2(image, cnt)

        if cnt is False:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        # cv2.drawContours(image, [box_original], -1, (255, 0, 0), 3)

    return image

capture = cv2.VideoCapture(0) # 0 is webcam

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        print("Failed to capture frame")
        break

    # Resize the frame if needed
    # frame = rescale_frame(frame, scale=0.5)

    frame = detect_barcodes(frame)  # Detect barcode in frame

    # Display the frame
    cv2.imshow("Barcode Detection", frame)

    # Exit the loop if 'd' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

# Release the capture object and close all windows
capture.release()
cv2.destroyAllWindows()

