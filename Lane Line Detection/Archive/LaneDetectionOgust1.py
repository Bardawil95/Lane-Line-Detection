import still_image_processing as sip, numpy as np, cv2, time, math
from lanesDetectionFuncs import mostColor as mc, regionOfInterest, rescale_frame, textDisplay, position

# Video Sets
# cap = cv2.VideoCapture("vdo0033-720p_cut.mp4")
# cap = cv2.VideoCapture("VDO_0004.mp4")
cap = cv2.VideoCapture("Datasets/Dataset7-1.mp4")

# Setting black, white and blue colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (255, 0, 0)

# Setting Region of Interset dimensions
leftOffset = 0
rightOffset = 0
topOffset = 500
bottomOffset = 60

center_y = 450
center_size = 110
halfCenterSize = center_size / 2

# Setting offset values for most color used
'''
plusOffset = 200
minusOffset = 200
'''
# value for rectangle's distance (x-coordinate)
xDist = 10
yThresh = 5
# Object's minimum area threshold
minAreaThresh = 30

# Height and Width center points
centerHeight = 360
centerWidth = 640

# Lower threshold white value
lowerWhite = [150, 150, 150]

# Threshold values for yellow mask
'''
lowerY = [30, 115, 125]
upperY = [155, 200, 255]
'''

lowerY = [40, 125, 135]
upperY = [135, 180, 235]

# Distance Threshold
distThresh = 170

# Position of the vehicle decision parameters
current_status = -1
status = -1
nbr_status_chg = 5
status_steps = 0


def pV(image):
    left = ""
    right = ""
    distance = []
    # Region of interest frame
    roi_image = regionOfInterest(image, topOffset, bottomOffset, leftOffset, rightOffset)
    roi_image_copy = roi_image.copy()
    y = center_y
    x = ((image.shape[1] - leftOffset - rightOffset) / 2) + leftOffset

    # Points for most used color region of interest
    p1 = [x - halfCenterSize, y + halfCenterSize]
    p2 = [x - halfCenterSize, y - halfCenterSize]
    p3 = [x + halfCenterSize, y - halfCenterSize]
    p4 = [x + halfCenterSize, y + halfCenterSize]
    rip2 = [p1, p2, p3, p4]
    vertices = [np.array(rip2, dtype=np.int32)]

    # Region of interest for the most used color
    roi_image_most = sip.region_of_interest(image, vertices)
    roi_image_most = roi_image_most[y - halfCenterSize:y + halfCenterSize, x - halfCenterSize:x + halfCenterSize]
    blue, green, red = mc(roi_image_most)

    img_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Mask to show objects in white color (White mask)
    lower_most = np.array(lowerWhite)
    upper_most = np.array(white)
    white_mask = cv2.inRange(roi_image, lower_most, upper_most)

    # Mask to show objects in white color (Yellow mask)
    lower = np.uint8(lowerY)
    upper = np.uint8(upperY)
    yellow_mask = cv2.inRange(roi_image, lower, upper)
    kernal = np.ones((20, 20), np.int8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernal)

    # Merging yellow and white masks
    mask_yw = cv2.bitwise_or(yellow_mask, white_mask)
    mask_yw = cv2.bitwise_and(img_gray, mask_yw)
    threshYW = cv2.threshold(mask_yw, 120, 255, cv2.THRESH_BINARY)[1]

    # Step 2: Getting contour value
    _, contours, hierarchy = cv2.findContours(threshYW, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    threshYWC = threshYW.copy()
    threshYW = rescale_frame(threshYW, 50)
    # cv2.imshow("threshYW", threshYW)
    areaArr = []
    areaArr2 = []
    i = 0
    centerImage = (centerWidth, centerHeight)

    # Step 3: Area of each contour to eliminate small objects (noise)
    contoursnew = []
    for c in contours:
        # Area of each object
        areaArr.append(cv2.contourArea(c))
        # Elimination of small objects
        if areaArr[i] < minAreaThresh:
            cv2.drawContours(threshYWC, [c], 0, black, 20)
        else:
            cv2.drawContours(threshYWC, [c], 0, white, 5)
            areaArr2.append(cv2.contourArea(c))
            contoursnew.append(c)
        i += 1

    # Step 4: Xtreme points of objects
    topmost = []
    bottommost = []
    cx = []
    cy = []
    i = 0
    for c in contoursnew:
        tempA = c[c[:, :, 1].argmin()][0]
        tempB = c[c[:, :, 1].argmax()][0]
        topmost.append(tempA)  # Points of objects where x and y are the minimum.
        bottommost.append(tempB)  # Points of objects where x and y are the maximum.
        cx.append((bottommost[i][0] + topmost[i][0]) / 2)  # x-coordinate center of object
        i += 1
    # Step 5: elimination of non-rectangular areas
    i = 0
    for c in contoursnew:
        y = bottommost[i][1] - topmost[i][1]  # y-coordinate distance for object
        x = bottommost[i][0] - topmost[i][0]  # x-coordinate distance for object
        distance_tmp = math.sqrt(pow(x, 2) + pow(y, 2))
        distance.append(distance_tmp)  # Distance of an object
        # print ">>>[(", bottommost[i][0], ",", bottommost[i][1], ")(", topmost[i][0], ",", topmost[i][1], ")], DX=", x, " DY=", y, "Distance[i]=", distance[i]
        area2 = distance[i] * xDist
        if areaArr2[i] > area2 or y < yThresh:  # Elimination of non-rectangular objects
            cv2.drawContours(threshYWC, [c], 0, black, 30)
            np.delete(distance, i)
        else:
            cv2.drawContours(threshYWC, [c], 0, white, 3)
            # cv2.line(roi_image, tuple(topmost[i]), tuple(bottommost[i]), (0, 255, 0), 15)
        i += 1

    kernal = np.ones((20, 20), np.int8)
    threshYWC = cv2.morphologyEx(threshYWC, cv2.MORPH_CLOSE, kernal)

    # Step 6-1: Detect left lines and decide whether the lines are dotted or continous.
    cxtemp = 0
    itmp = 0
    i = 0
    for i in range(len(distance)):
        # Specify left lane lines from the center in black color. (Applied on ROI image)

        if cx[i] < centerImage[0]:
            if cx[i] > cxtemp:
                cxtemp = cx[i]
                itmp = i

    if cxtemp != 0:
        if distance[itmp] < distThresh:  # If the left line is dotted (Black color)
            cv2.line(roi_image, tuple(topmost[itmp]), tuple(bottommost[itmp]), black, 15)
            left = "Dotted"
        else:  # If the left line is continuous (Blue color)
            cv2.line(roi_image, tuple(topmost[itmp]), tuple(bottommost[itmp]), blue, 15)
            left = "Continuous"
    else:
        left = "Unknown"

    # Step 6-2: Detect right lines and decide whether the lines are dotted or continous.
    cxtemp2 = 1000000
    itmp2 = 0
    i = 0
    for i in range(len(distance)):
        # Specify right lane lines from the center in black color. (Applied on ROI image)
        if cx[i] > centerImage[0]:
            if cx[i] < cxtemp2:
                cxtemp2 = cx[i]
                itmp2 = i

    if cxtemp2 != 0:
        if distance[itmp2] < distThresh:  # If the right line is dotted (Black color)
            cv2.line(roi_image, tuple(topmost[itmp2]), tuple(bottommost[itmp2]), black, 15)
            right = "Dotted"
        else:  # If the right line is continuous (Blue color)
            cv2.line(roi_image, tuple(topmost[itmp2]), tuple(bottommost[itmp2]), blue, 15)
            right = "Continuous"
    else:
        right = "Unknown"

    current_status = position(right, left)
    '''
    if current_status is -1:
        textDisplay(roi_image, "Undefined", white)
    elif current_status is 0:
        textDisplay(roi_image, "Unknown", white)
    elif current_status is 1:
        textDisplay(roi_image, "Left", white)
    elif current_status is 2:
        textDisplay(roi_image, "Middle", white)
    elif current_status is 3:
        textDisplay(roi_image, "Right", white)
    '''
    return roi_image, roi_image_copy, roi_image_most, threshYW, yellow_mask, threshYWC, white_mask, img_gray, current_status


while 1:
    ch = cv2.waitKey(1)
    if cap.read() is None:
        "Error Video!"
        break
    _, frame = cap.read()
    if frame is None or ch & 0xFF == ord('q'):
        break
    roi_image, roi_image_copy, roi_image_most, threshYW, yellow_mask, threshYWC, white_mask, img_gray, current_status = pV(
        frame)
    threshYWC = rescale_frame(threshYWC, 60)
    roi_image = rescale_frame(roi_image, 60)

    if current_status != status:
        if status_steps >= nbr_status_chg or status == -1:
            status = current_status
            status_steps = 0
        else:
            status_steps += 1
    else:
        status_steps = 0

    if status is -1:
        textDisplay(roi_image, "Undefined", white)
    elif status is 0:
        textDisplay(roi_image, "Unknown", white)
    elif status is 1:
        textDisplay(roi_image, "Left", white)
    elif status is 2:
        textDisplay(roi_image, "Middle", white)
    elif status is 3:
        textDisplay(roi_image, "Right", white)

    cv2.imshow("Merged Mask", threshYWC)
    cv2.imshow("Original (ROI)", roi_image)
    # time.sleep(0.5)
cv2.waitKey(0)
cv2.destroyAllWindows()