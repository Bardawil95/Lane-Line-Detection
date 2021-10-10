from transform import four_point_transform
import numpy as np
import argparse
import cv2

image = cv2.imread("solidWhiteCurve.jpg")
bl = [180, 540] #Bottom Left point
tl = [417, 354] #Top Left
tr = [580, 354] #Bottom Left point
br = [900, 540] #Top Left

warped, cannyWarped = four_point_transform(image, bl, tl, tr, br)#, pts)#, vertices)
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.imshow("Canny Warped", cannyWarped)
cv2.waitKey(0)
cv2.destroyAllWindows()