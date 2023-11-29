import numpy as np
import imutils
import cv2
import os

def find_rectangles_within_contour(image, contour, showProcess=False):
    
    original = image.copy()
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    # Loop through each point in the contour
    for point in contour:
        x, y = point[0]

        # Update min and max values
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    
    # cv2.imshow("Mask", mask)
    
    masked_image = cv2.bitwise_and(image, mask)
    # cv2.imshow("Masked Image", masked_image)
    # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # masked_image = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV)[1]
    masked_image = cv2.Canny(masked_image, 50, 200)
    # invert 
    # masked_image = cv2.bitwise_not(masked_image)
    
    lines = cv2.HoughLinesP(image=masked_image,rho=1.5,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=20,maxLineGap=80)

    blank = np.zeros(image.shape, dtype=np.uint8)
    
    copy = image.copy()
    
    if lines is not None:
        a = lines.shape[0]
        for i in range(a):
            # cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(blank, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        
    
    cv2.imshow("Masked Image", masked_image)
    cv2.imshow("Image", image)
    cv2.imshow("Lines", blank)
    cv2.waitKey(0)
    
    # mask lines with original
    blank2 = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    blank2 = cv2.threshold(blank2, 0, 255, cv2.THRESH_BINARY_INV)[1]
    blank2 = cv2.bitwise_not(blank2)
    
    masked = cv2.bitwise_and(image, image, mask=blank2)
    cv2.imshow("Masked Image2", masked)
    cv2.waitKey(0)
    
    thresh = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=1)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    
    
    x_min, x_max = -1, -1
    
    done = False
    for i in range(min_x, max_x):
        if (done):
            break
        for j in range(min_y, max_y):
            b,g,r = blank[j, i]
            if r == 255:
            # if thresh[j,i] < 10:
                x_min = i
                cv2.circle(copy, (i, j), 2, (255, 0 , 0), 3)
                print ("X_MIN", x_min)
                done = True
                break
            
    done = False
    
    for i in range(max_x, min_x, -1):
        if (done):
            break
        for j in range(min_y, max_y):
            # if thresh[j,i] < 10:
            b,g,r = blank[j, i]
            if r == 255:    
                x_max = i
                cv2.circle(copy, (i, j), 2, (255, 0, 0), 3)
                print("X_MAX", x_max)
                done = True
                break
    
    if x_min == -1 or x_max == -1:
        return contour
    
    # contour[:, 0] = np.clip(contour[:, 0], x_min, x_max)
    for point in contour:
        x, y = point[0]

        if x < x_min:
            x = x_min
        elif x > x_max:
            x = x_max
            
        point[0] = [x, y]
            
    img = original.copy()
    
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    
    print("DONEE")
    cv2.imshow("Final Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return contour


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

        find_rectangles_within_contour(image, cnt, True)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Draw a bounding box around each detected barcode
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

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
