import numpy as np
import cv2
import os
import imutils

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

def find_parallel(lines):
    max_parallel_count = 0
    best_line_index = -1
    best_parallel_indices = []

    for i in range(len(lines)):
        parallel_indices = []

        for j in range(len(lines)):
            if i == j:
                continue

            # Extract line coordinates
            x1_i, y1_i, x2_i, y2_i = lines[i][0]
            x1_j, y1_j, x2_j, y2_j = lines[j][0]

            # Check if the lines are approximately parallel
            delta_x_i = x2_i - x1_i
            delta_y_i = y2_i - y1_i
            delta_x_j = x2_j - x1_j
            delta_y_j = y2_j - y1_j

            # Avoid division by zero
            if delta_x_i == 0 or delta_x_j == 0:
                continue

            slope_i = delta_y_i / delta_x_i
            slope_j = delta_y_j / delta_x_j

            if abs(slope_i - slope_j) < 0.05:
                parallel_indices.append(j)

        if len(parallel_indices) > max_parallel_count:
            max_parallel_count = len(parallel_indices)
            best_line_index = i
            best_parallel_indices = parallel_indices

    return best_line_index, best_parallel_indices

def detectBarcodes2(image, contour):
    showProcess = True

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    masked_image = cv2.bitwise_and(image, mask)
    cv2.imshow("Masked Image", masked_image)

    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.medianBlur(gray, 7)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edgeDetection = cv2.Canny(blurred, 50, 225, apertureSize=3)

    if showProcess:
        cv2.imshow("Edge", edgeDetection)
        cv2.waitKey(0)

    lines = cv2.HoughLinesP(edgeDetection, rho=1, theta=np.pi / 180, threshold=25, minLineLength=100, maxLineGap=10)

    blank = np.zeros(image.shape, dtype=np.uint8)

    if lines is None:
        print("No lines")
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

    # Find the minimum and maximum x-coordinates
    x_min = np.min(red_pixels_x)
    x_max = np.max(red_pixels_x)

    # Find the y-coordinates where red channel is 255
    red_pixels_y = np.where(blank[:, :, 2] == 255)[0]

    # Find the minimum and maximum y-coordinates
    y_min = np.min(red_pixels_y)
    y_max = np.max(red_pixels_y)

    # Print the results
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

    # Draw the contour on the image
    return cnt

def detectBarcodes(image, showProcess=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 3)

    edgeDetection = cv2.Canny(blurred, 50, 225)
    edgeDetection = cv2.convertScaleAbs(edgeDetection)

    if showProcess:
        cv2.imshow("1 - Edge", edgeDetection)

    kernel = np.ones((7, 7), np.uint8)

    # Calculate the mean brightness of the image
    brightness = calculate_brightness(image)

    if brightness > 170:  # Adjust thresholds for bright images
        erode_dilate = edgeDetection
    else:
        erode_dilate = cv2.dilate(edgeDetection, kernel, iterations=1)
        if showProcess:
            cv2.imshow("2 - Dilate", erode_dilate)
        erode_dilate = cv2.erode(erode_dilate, np.ones((11, 11), np.uint8), iterations=1)
        if showProcess:
            cv2.imshow("3 - Erode", erode_dilate)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(erode_dilate, cv2.MORPH_CLOSE, kernel)

    if showProcess:
        cv2.imshow("4 - Closed", closed)

    cnts = cv2.findContours(
        closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    largestContourArea = cv2.contourArea(
        sorted(cnts, key=cv2.contourArea, reverse=True)[0])

    for cnt in cnts:
        # Filter contours based on area
        if cv2.contourArea(cnt) < (largestContourArea / 2):
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cnt_original = box.copy()

        cnt = detectBarcodes2(image, cnt)

        if cnt is False:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Draw a bounding box around each detected region
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        cv2.drawContours(image, [cnt_original], -1, (255, 0, 0), 1)

    image_name = os.path.basename(image_path)
    cv2.imshow(f"Detected Barcode {image_name}", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
