import still_image_processing as sip, numpy as np, cv2, time, math
from lanesDetectionFuncs import mostColor as mc, regionOfInterest, rescale_frame, select_rgb_white_yellow as srgbwy

#Video Sets
#cap = cv2.VideoCapture("vdo0033-720p_cut.mp4")
#cap = cv2.VideoCapture("VDO_0004.mp4")
cap = cv2.VideoCapture("Datasets/Dataset7-1.mp4")

image = cv2.imread("Other/solidWhiteCurve.jpg")

# Setting black and white colors
black = (0, 0, 0)
white = (255, 255, 255)

# Setting RGB colors
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

#Setting Region of Interset dimensions
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
lowerWhite = [160, 160, 160]

# Yellow mask threshold values
lowerY = [40, 125, 135]
upperY = [135, 180, 235]
def pV(image):
    distance = []
    # Region of interest frame
    roi_image = regionOfInterest(image, topOffset, bottomOffset, leftOffset, rightOffset)
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
    '''
    # Mask to show objects in white color
    lower_most = np.array(lowerWhite)
    upper_most = np.array(white)
    mask_most = cv2.inRange(roi_image, lower_most, upper_most)
    masked_roi = cv2.bitwise_and(roi_image,roi_image, mask=mask_most)
    gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
    '''
    # Step 0: Yellow detection
    gray_y = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    lower = np.uint8(lowerY)
    upper = np.uint8(upperY)
    yellow_mask = cv2.inRange(roi_image, lower, upper)

    mask_y = cv2.bitwise_and(gray_y, yellow_mask)
    threshY = cv2.threshold(mask_y, 120, 255, cv2.THRESH_BINARY)[1]

    '''
    kernal = np.ones((20, 20), np.int8)
    threshY = cv2.morphologyEx(threshY, cv2.MORPH_CLOSE, kernal)
    '''

    # Step 1: Black and white image
#    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]


    # Step 2: Getting contour value
    _, contours, hierarchy = cv2.findContours(threshY, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    threshYe = threshY.copy()
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
            cv2.drawContours(threshYe, [c], 0, black, 20)
        else:
            cv2.drawContours(threshYe, [c], 0, white, 5)
            areaArr2.append(cv2.contourArea(c))
            contoursnew.append(c)
        i += 1

    # Step 4: Xtreme points of objects
    topmost = []
    bottommost = []
    cx = []
    i = 0
    for c in contoursnew:
        tempA = c[c[:, :, 1].argmin()][0]
        tempB = c[c[:, :, 1].argmax()][0]
        topmost.append(tempA)  # Points of objects where x and y are the minimum.
        bottommost.append(tempB)  # Points of objects where x and y are the maximum.
        cx.append((bottommost[i][0] + topmost[i][0]) / 2)  # x-coordinate center of object
        i += 1

    # Step 5: elimination of non-rectangular areas
    #print "#################### step 5 #################################"
    i = 0
    for c in contoursnew: #range(len(topmost)):

        y = bottommost[i][1] - topmost[i][1]  # y-coordinate distance for object
        x = bottommost[i][0] - topmost[i][0]  # x-coordinate distance for object
        print "Y = ", y, "  X = ", x
        distance_tmp = math.sqrt(pow(x, 2) + pow(y, 2))
        distance.append(distance_tmp)  # Distance of an object
        area2 = distance[i] * xDist
        if areaArr2[i] > area2 or y < yThresh:  # Elimination of non-rectangular objects
            print areaArr2[i], "  >  ", area2
            cv2.drawContours(threshYe, [c], 0, black, 30)
        else:
            print areaArr2[i], "  <=  ", area2
            cv2.drawContours(threshYe, [c], 0, white, 3)
            cv2.line(roi_image, tuple(topmost[i]), tuple(bottommost[i]), blue, 5)

    kernal = np.ones((20, 20),np.int8)
    threshYe = cv2.morphologyEx(threshYe,cv2.MORPH_CLOSE,kernal)
    '''
       # Step 6-1: Detect left lines and decide whether the lines are dotted or continous.
       cxtemp = 0
       for i in range(len(bottommost)):
         # Specify left lane lines from the center in black color. (Applied on ROI image)
           if cx[i] < centerImage[0]:
               if cx[i] < cxtemp:
                   continue
               else:
                   if distance[i] < distThresh:       # If the left line is dotted (Black color)
                       cv2.line(roi_image, tuple(topmost[i]), tuple(bottommost[i]), black, 10)
                   else:                               # If the left line is continuous (Blue color)
                       cv2.line(roi_image, tuple(topmost[i]), tuple(bottommost[i]), blue, 10)
                   cxtemp = cx[i]

       # Step 6-2: Detect right lines and decide whether the lines are dotted or continous.
       cxtemp = 10000000
       for i in range(len(bottommost)):
           # Specify right lane lines from the center in black color. (Applied on ROI image)
           if cx[i] > centerImage[0]:
               if cx[i] > cxtemp:
                   cv2.drawContours(image, [c], 0, black, 20)
               else:
                   if distance[i] < distThresh:       # If the right line is dotted (Black color)
                       cv2.line(roi_image, tuple(topmost[i]), tuple(bottommost[i]), black, 10)
                   else:                               # If the right line is continuous (Blue color)
                       cv2.line(roi_image, tuple(topmost[i]), tuple(bottommost[i]), blue, 10)
           cxtemp = cx[i]
       '''
    threshCanny = cv2.Canny(threshYe, 50, 150)
    line_image = 0
    output = sip.weighted_img(line_image, image, p1=0.8, p2=1., p3=0.)

    return threshYe, roi_image, roi_image_most, threshCanny, line_image, output

while 1:
    ch = cv2.waitKey(1)
    if cap.read() is None:
        "Error Video!"
        break
    _,frame = cap.read()
    if frame is None or ch & 0xFF == ord('q'):
        break
    threshYe, roi_image, roi_image_most, threshCanny, line_image, output = pV(frame)
    threshYe = rescale_frame(threshYe, 50)
    cv2.imshow("Yellow", threshYe)
cv2.waitKey(0)
cv2.destroyAllWindows()
