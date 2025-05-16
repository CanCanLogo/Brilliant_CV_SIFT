import cv2
import matplotlib.pyplot as plt

img1 = cv2.cvtColor(cv2.imread("./vector3.jpg"), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread("./Vector_Tsoi.jpg"), cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)

matches = sorted(matches, key=lambda  x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

plt.imshow(img3)
plt.show()