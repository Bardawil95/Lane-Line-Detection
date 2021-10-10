import numpy as np
import cv2
import still_image_processing as sip
import LaneDetection as nt
def processVid(image):
    imshape = image.shape
    height = imshape[0]
    width = imshape[1]

    y1 = height / 2 + height / 4.5
    y2 = height / 2 + height / 9

    bottomLeft = [0, y1]
    topLeft = [0, y2]
    topRight = [width, y2]
    bottomRight = [width, y1]

    rip = [bottomLeft, topLeft, topRight, bottomRight]      # Region of interest

    vertices = [np.array(rip, dtype=np.int32)]
    roi_image = sip.region_of_interest(image, vertices)

    #roi_image_masked = nt.mask(roi_image)

    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    roi_image_masked = nt.mask(roi_image)
    img_hsv = cv2.cvtColor(roi_image, cv2.COLOR_RGB2HSV)
    # hsv = [hue, saturation, value]
    # more accurate range for yellow since it is not strictly black, white, r, g, or b

    lower_yellow = np.array([20, 100, 100], dtype="uint16")

    upper_yellow = np.array([30, 255, 255], dtype="uint16")

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    kernel_size = 5
    gauss_gray = cv2.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)

    # same as quiz values
    low_threshold = 50
    high_threshold = 250
    canny_edges = cv2.Canny(gauss_gray, low_threshold, high_threshold)

    # rho and theta are the distance and angular resolution of the grid in Hough space
    # same values as quiz
    rho = 2
    theta = np.pi / 180
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 20
    min_line_len = 50
    max_line_gap = 200

    roi_image_masked = cv2.dilate(roi_image_masked, (5, 5))

    #roi_image_masked = cv2.Canny(roi_image_masked, low_threshold, high_threshold)
    roi_image_masked_canny = cv2.Canny(roi_image_masked, low_threshold, high_threshold)
    line_image = sip.hough_lines(canny_edges, rho, theta, threshold, min_line_len, max_line_gap)
    line_image_masked = sip.hough_lines(roi_image_masked_canny, rho, theta, threshold, min_line_len, max_line_gap)
    result = sip.weighted_img(line_image_masked, image, p1=0.8, p2=1., p3=0.)

    return roi_image, mask_yw_image, roi_image_masked, gray_image, canny_edges, gauss_gray, line_image, line_image_masked, result