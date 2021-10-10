import numpy as np
import cv2
import scipy, scipy.misc, scipy.cluster
import still_image_processing as sip
import math
def mostColor(image):
    NUM_CLUSTERS = 2
    shape = image.shape
    ar = image.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    codes, _ = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, _ = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, _ = scipy.histogram(vecs, len(codes))    # count occurrences
    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    return int(peak[0]), int(peak[1]), int(peak[2])

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def regionOfInterest(image, topOffset, bottomOffset, leftOffset=0, rightOffset=0):

    point1 = [leftOffset, image.shape[0] - bottomOffset]
    point2 = [leftOffset, topOffset]
    point3 = [image.shape[1] - rightOffset, topOffset]
    point4 = [image.shape[1] - rightOffset, image.shape[0] - bottomOffset]
    rip1 = [point1, point2, point3, point4]  # Region of interest

    vertices = [np.array(rip1, dtype=np.int32)]
    return sip.region_of_interest(image, vertices)

def mask(image):
    lower = [183, 178, 165]
    upper = [255, 255, 255]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output

def detect(_, c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "rectangle"
    return shape

def select_rgb_white_yellow(image):
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def textDisplay(img, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (img.shape[1] / 6, 200)
    fontScale = 3
    fontColor = color #(255,255,255)
    lineType  = 5

    img = cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale,fontColor, lineType)
    return img

def position(right, left):
    pos = -1

    if right == "Continuous" and left == "Continuous":
        pos = 2
    elif right == "Dotted" and left == "Dotted":
        pos = 2
    elif right == "Unknown" and left == "Unknown":
        pos = 0
    elif right == "Dotted" and left == "Continuous":
        pos = 1
    elif right == "Continuous" and left == "Dotted":
        pos = 3
    elif right == "Continuous" and left == "Unknown":
        pos = 3
    elif right == "Unknown" and left == "Continuous":
        pos = 1
    elif (right == "Dotted" and left == "Unknown") or (right == "Unknown" and left == "Dotted"):
        pos = 0
    elif (right != "Continuous" or right != "Dotted") \
            or (left != "Continuous" or left != "Dotted"):
        pos = -1
    return pos

def triangleOfObject(x1, x2, y1, distance, maxHeight, centerX):
    shortX = math.fabs(x2 - x1)
    alpha = math.acos(shortX / distance)
    longX = math.fabs(centerX - x2)
    longY = math.tan(alpha) * (shortX + longX)
    shortY = maxHeight - y1
    wholeH = longY + shortY
    return wholeH, alpha

def closing (image):
    kernal = np.ones((20, 20), np.int8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
    return image