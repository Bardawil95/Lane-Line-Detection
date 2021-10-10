import still_image_processing as sip, numpy as np, cv2, time, math
import lanesDetectionFuncs as laneFuncs

# Video Set
cap = cv2.VideoCapture("Dataset8.avi")

# Setting black, white and blue colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (255, 0, 0)

# Setting Region of Interest dimensions
leftOffset = 0
rightOffset = 0
topOffset = 700
bottomOffset = 180

center_y = 450
center_size = 110
halfCenterSize = center_size / 2

# Setting offset values for most color used
'''
plusOffset = 200
minusOffset = 200
'''
# value for rectangle's distance (x-coordinate)
xDist = 25
yThresh = 5
maxDistFromCenter = 160
minDriftVertical = 5
maxHeight = 720
lowH = 100
highH = 360
# Object's minimum area threshold
minAreaThresh = 150

# Height and Width center points

centerWidth = 850

# Lower threshold white value
lowerWhite = [150, 150, 150]

# Threshold values for yellow mask

lowerY = [40, 125, 135]
upperY = [135, 180, 235]

# Distance Threshold
distThresh = 300

# Position of the vehicle decision parameters
current_status = -1
status = -1
nbr_status_chg = 15
status_steps = 0


def pV(image):
    # Region of interest frame
    roi_image = laneFuncs.regionOfInterest(image, topOffset, bottomOffset, leftOffset, rightOffset)
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
    roi_image_most = roi_image_most[int(y - halfCenterSize):int(y + halfCenterSize), int(x - halfCenterSize):int(x + halfCenterSize)]
    blue = laneFuncs.mostColor(roi_image_most)

    img_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Step 0-1: Mask to show objects in white color (White mask)
    lower_most = np.array(lowerWhite)
    upper_most = np.array(white)
    white_mask = cv2.inRange(roi_image, lower_most, upper_most)

    # Step 0-2: Mask to show objects in white color (Yellow mask)
    lower = np.uint8(lowerY)
    upper = np.uint8(upperY)
    yellow_mask = cv2.inRange(roi_image, lower, upper)
    kernal = np.ones((20, 20), np.int8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernal)

    # Step 1: Merging yellow and white masks
    mask_yw = cv2.bitwise_or(yellow_mask, white_mask)
    mask_yw = cv2.bitwise_and(img_gray, mask_yw)
    threshYW = cv2.threshold(mask_yw, 120, 255, cv2.THRESH_BINARY)[1]


    # Step 2: Getting contour values
    threshYWC = threshYW.copy()


    # closing
    kernal = np.ones((20, 20), np.int8)
    threshYWC = cv2.morphologyEx(threshYWC, cv2.MORPH_CLOSE, kernal)

    temp = threshYWC.copy()
    temp = laneFuncs.rescale_frame(temp, 40)
    cv2.imshow("(0) Black & White", temp)

    contours_step2, _ = cv2.findContours(threshYWC, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    # Step 3: Area of each contour to eliminate small objects (noise)
    areaArr = []
    areaArr_step3 = []
    contours_step3 = []
    topmost_step3 = []
    bottommost_step3 = []
    i = 0
    for c in contours_step2:
        # Area of each object
        areaArr.append(cv2.contourArea(c))
        area = cv2.contourArea(c)
        tempA = c[c[:, :, 1].argmin()][0]
        tempB = c[c[:, :, 1].argmax()][0]
        topmost_step3.append(tempA)  # Points of objects where x and y are the minimum.
        bottommost_step3.append(tempB)  # Points of objects where x and y are the maximum.
        # Elimination of small objects
        if area < minAreaThresh: #minAreaThresh:
            cv2.fillPoly(threshYWC, [c], black)
            cv2.drawContours(threshYWC, [c], 0, black, 30)
        else:
            cv2.drawContours(threshYWC, [c], 0, white, 5)
            areaArr_step3.append(cv2.contourArea(c))
            contours_step3.append(c)
        i += 1


    # Step 4: Calculation of extreme points of each objects
    topmost_step4 = []
    bottommost_step4 = []
    distance_step4 = []
    i = 0
    y_step4 =[]
    for c in contours_step3:
        tempA = c[c[:, :, 1].argmin()][0]
        tempB = c[c[:, :, 1].argmax()][0]
        topmost_step4.append(tempA)  # Points of objects where x and y are the minimum.
        bottommost_step4.append(tempB)  # Points of objects where x and y are the maximum.
        y = math.fabs(tempA[1] - tempB[1])  # y-coordinate distance for object
        y_step4.append(y)
        x = math.fabs(tempA[0] - tempB[0])  # x-coordinate distance for object
        distance_step4.append(math.sqrt(pow(x, 2) + pow(y, 2)))  # Distance of an object
        i += 1

    # Step 5: elimination of non-rectangular areas and removing horizontal lines (y is almost zero)
    contours_step5 = []
    topmost_step5 = []
    bottommost_step5 = []
    distance_step5 = []
    i = 0

    for c in contours_step3:
        area_tmp = distance_step4[i] * xDist
        if areaArr_step3[i] > area_tmp or y_step4[i] < yThresh:  # Elimination of non-rectangular objects and removing horizontal lines (y is almost zero)
            cv2.drawContours(threshYWC, [c], 0, black, 30)
            cv2.fillPoly(threshYWC, [c], black)
        else:
            cv2.drawContours(threshYWC, [c], 0, white, 3)
            distance_step5.append(distance_step4[i])
            bottommost_step5.append(bottommost_step4[i])
            topmost_step5.append(topmost_step4[i])
            contours_step5.append(c)
        i += 1

    threshYWC = laneFuncs.rescale_frame(threshYWC, 40)
    cv2.imshow("Step 5: Non-rectangular objects & horizontal lines elimination", threshYWC)

    # Step 6: Elimination of objects that are vertical and away from the center of the vehicle
    contours_step6 = []
    topmost_step6 = []
    bottommost_step6 = []
    distance_step6 = []
    i = 0

    for c in contours_step5:
        if math.fabs(topmost_step5[i][0] - centerWidth) > maxDistFromCenter and math.fabs(topmost_step5[i][0] - bottommost_step5[i][0]) < minDriftVertical:
            cv2.drawContours(threshYWC, [c], 0, black, 30)
            cv2.fillPoly(threshYWC, [c], black)
        else:
            distance_step6.append(distance_step5[i])
            bottommost_step6.append(bottommost_step5[i])
            topmost_step6.append(topmost_step5[i])
            contours_step6.append(c)
        i += 1

    # Step 7: Elimination of lines directed to outside the screen
    contours_step7 = []
    topmost_step7 = []
    bottommost_step7 = []
    distance_step7 = []
    centerX_step7 = []
    i = 0

    for c in contours_step6:
        if topmost_step6[i][0] < centerWidth and bottommost_step6[i][0] < centerWidth and topmost_step6[i][0] < bottommost_step6[i][0]:
                cv2.drawContours(threshYWC, [c], 0, black, 30)
                cv2.fillPoly(threshYWC, [c], black)
        else:
            if topmost_step6[i][0] > centerWidth and bottommost_step6[i][0] > centerWidth and topmost_step6[i][0] > bottommost_step6[i][0]:
                cv2.drawContours(threshYWC, [c], 0, black, 30)
                cv2.fillPoly(threshYWC, [c], black)
            else:
                distance_step7.append(distance_step6[i])
                bottommost_step7.append(bottommost_step6[i])
                topmost_step7.append(topmost_step6[i])
                contours_step7.append(c)
                centerX_step7.append((bottommost_step6[i][0] + topmost_step6[i][0]) / 2)  # x-coordinate center of object
                i += 1
    # Step 8: Elimination of lines directed to inside the screen but not toward the center view
    contours_step8 = []
    topmost_step8 = []
    bottommost_step8 = []
    distance_step8 = []
    centerX_step8 = []
    i = 0

    for c in contours_step7:
        wholeH, _ = laneFuncs.triangleOfObject(bottommost_step7[i][0], topmost_step7[i][0], bottommost_step7[i][1], distance_step7[i], maxHeight, centerWidth)
        if wholeH < lowH or wholeH > highH:
            cv2.drawContours(threshYWC, [c], 0, black, 30)
            cv2.fillPoly(threshYWC, [c], black)

        else:
            distance_step8.append(distance_step7[i])
            bottommost_step8.append(bottommost_step7[i])
            topmost_step8.append(topmost_step7[i])
            centerX_step8.append((bottommost_step7[i][0] + topmost_step7[i][0]) / 2)  # x-coordinate center of object
            contours_step8.append(c)

        i += 1

    # Step 9-1: Detect left lines and decide whether the lines are dotted or continous.
    alphatemp = 0
    itmp = -1
    for i in range(len(distance_step8)):
        # Specify left lane lines from the center in black color. (Applied on ROI image)
        if bottommost_step8[i][0] < centerWidth: # the line is on the left
            _, alpha = laneFuncs.triangleOfObject(bottommost_step8[i][0], topmost_step8[i][0], bottommost_step8[i][1], distance_step8[i], maxHeight, centerWidth)

            if alpha > alphatemp or itmp == -1:
                alphatemp = alpha
                itmp = i

    if itmp != -1:
        if distance_step8[itmp] < distThresh:  # If the left line is dotted (Black color)
            cv2.line(roi_image, tuple(topmost_step8[itmp]), tuple(bottommost_step8[itmp]), black, 15)
            left = "Dotted"
        else:  # If the left line is continuous (Blue color)
            cv2.line(roi_image, tuple(topmost_step8[itmp]), tuple(bottommost_step8[itmp]), blue, 15)
            left = "Continuous"
    else:
        left = "Unknown"

    # Step 9-2: Detect right lines and decide whether the lines are dotted or continous.
    alphatemp = 0
    itmp = -1
    for i in range(len(distance_step8)):
        # Specify right lane lines from the center in black color. (Applied on ROI image)
        if bottommost_step8[i][0] > centerWidth:  # the line is on the right
            _, alpha = laneFuncs.triangleOfObject(bottommost_step8[i][0], topmost_step8[i][0], bottommost_step8[i][1], distance_step8[i], maxHeight, centerWidth)
            if alpha > alphatemp or itmp == -1:
                alphatemp = alpha
                itmp = i

    if itmp != -1:
        if distance_step8[itmp] < distThresh:  # If the right line is dotted (Black color)
            cv2.line(roi_image, tuple(topmost_step8[itmp]), tuple(bottommost_step8[itmp]), black, 15)
            right = "Dotted"
        else:  # If the right line is continuous (Blue color)
            cv2.line(roi_image, tuple(topmost_step8[itmp]), tuple(bottommost_step8[itmp]), blue, 15)
            right = "Continuous"
    else:
        right = "Unknown"

    # Step 10: determine the location of the vehicles (Left, Right, Middle or Unknown)
    current_status = laneFuncs.position(right, left)

    return roi_image, threshYWC, current_status #roi_image_copy, roi_image_most, threshYW, yellow_mask, threshYWC, white_mask, img_gray,


while 1:
    ch = cv2.waitKey(1)
    if cap.read() == None:
        "Error Video!"
        break
    _, frame = cap.read()
    roi_image, threshYWC, current_status = pV(frame)
    roi_image = laneFuncs.rescale_frame(frame, 40)

    if current_status != status:
        if status_steps >= nbr_status_chg or status == -1:
            status = current_status
            status_steps = 0
        else:
            status_steps += 1
    else:
        status_steps = 0

    # if status == -1:
    #     laneFuncs.textDisplay(roi_image, "Undefined", white)
    # elif status == 0:
    #     laneFuncs.textDisplay(roi_image, "Unknown", white)
    # elif status == 1:
    #     laneFuncs.textDisplay(roi_image, "Left", white)
    # elif status == 2:
    #     laneFuncs.textDisplay(roi_image, "Middle", white)
    # elif status == 3:
    #     laneFuncs.textDisplay(roi_image, "Right", white)

    if status == -1:
        print("Undefined")
    elif status == 0:
        print("Unknown")
    elif status == 1:
        print("Left")
    elif status == 2:
        print("Middle")
    elif status == 3:
        print("Right")

    cv2.imshow("(7) Original (ROI)", roi_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()