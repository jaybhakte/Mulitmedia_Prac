# Mulitmedia_Prac
1.
Aim: Write a program in python to read and display an image.
Softwares Used: Jupyter Notebook, Opencv
Code:
import cv2
from matplotlib import pyplot as plt
print(cv2.__version__)

img = cv2.imread("dolphin.jpg")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Over the Clouds-Original Color Image")
Output:

2.
Aim: Write a program in python to read an image and print a grayscale
image.
Softwares Used: Jupyter Notebook, Opencv
Code:
import cv2
from matplotlib import pyplot as plt
print(cv2.__version__)

img = cv2.imread("dolphin.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Over the Clouds_gray")

Output:

3.
Aim: Write a program in python to scale an image.
Softwares Used: Jupyter Notebook, Opencv
Code:
import cv2
import numpy as np
FILE_NAME = 'dolphin.jpg'
try:
# Read image from disk.
img = cv2.imread(FILE_NAME)
# Get number of pixel horizontally and vertically.
(height, width) = img.shape[:2]
# Specify the size of image along with interpolation methods.
# cv2.INTER_AREA is used for shrinking, whereas cv2.INTER_CUBIC
# is used for zooming.
res = cv2.resize(img, (int(width / 2), int(height / 2)),interpolation =
cv2.INTER_CUBIC)
# Write image back to disk.
cv2.imwrite('result.jpg', res)
except IOError:
print ('Error while reading files !!!')

Output:

Practical No.4
Aim: Write a program in python to rotate an image.
Softwares Used: Jupyter Notebook, Opencv
Code:
import cv2
import numpy as np
FILE_NAME = 'dolphin.jpg'
try:
# Read image from the disk.
img = cv2.imread(FILE_NAME)
# Shape of image in terms of pixels.
(rows, cols) = img.shape[:2]
# getRotationMatrix2D creates a matrix needed for transformation.
# We want matrix for rotation w.r.t center to 45 degree without scaling.
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
res = cv2.warpAffine(img, M, (cols, rows))
# Write image back to disk.
cv2.imwrite('result.jpg', res)
except IOError:
print ('Error while reading files !!!')

Output:

Practical No.5
Aim: Write a program in python to translate an image.
Softwares Used: Jupyter Notebook, Opencv
Code:
import cv2
import numpy as np
FILE_NAME = 'dolphin.jpg'
# Create translation matrix.
# If the shift is (x, y) then matrix would be
# M = [1 0 x]
# [0 1 y]
# Let's shift by (100, 50).
M = np.float32([[1, 0, 100], [0, 1, 50]])
try:
# Read image from disk.
img = cv2.imread(FILE_NAME)
(rows, cols) = img.shape[:2]
# warpAffine does appropriate shifting given the
# translation matrix.
res = cv2.warpAffine(img, M, (cols, rows))
# Write image back to disk.
cv2.imwrite('result.jpg', res)
except IOError:
print ('Error while reading files !!!')

Practical No.6
Aim: Write a program in python for edge detection an image.
Softwares Used: Jupyter Notebook, Opencv
Code:
import cv2
import numpy as np
FILE_NAME = 'dolphin.jpg'
try:
# Read image from disk.
img = cv2.imread(FILE_NAME)
# Canny edge detection.
edges = cv2.Canny(img, 100, 200)
# Write image back to disk.
cv2.imwrite('result.jpg', edges)
except IOError:
print ('Error while reading files !!!')

Practical No.7

Aim: Write a program in python for Point Processing in image.
Softwares Used: Jupyter Notebook, Opencv
Code:
import cv2
import numpy as np

# Image negative
img = cv2.imread('food.jpeg',0)
# To ascertain total numbers of
# rows and columns of the image,
# size of the image
m,n = img.shape
# To find the maximum grey level
# value in the image
L = img.max()
# Maximum grey level value minus
# the original image gives the
# negative image
img_neg = L-img
# convert the np array img_neg to
# a png image
cv2.imwrite('Dolphin_Negative.png', img_neg)
# Thresholding without background
# Let threshold =T
# Let pixel value in the original be denoted by r
# Let pixel value in the new image be denoted by s
# If r<T, s= 0
# If r>T, s=255
T = 150
# create an array of zeros

img_thresh = np.zeros((m,n), dtype = int)
for i in range(m):
for j in range(n):
if img[i,j] < T:
img_thresh[i,j]= 0
else:
img_thresh[i,j] = 255

# Convert array to png image
cv2.imwrite('Dolphin_Thresh.png', img_thresh)
# the lower threshold value
T1 = 100
# the upper threshold value
T2 = 180
# create an array of zeros
img_thresh_back = np.zeros((m,n), dtype = int)
for i in range(m):
for j in range(n):
if T1 < img[i,j] < T2:
img_thresh_back[i,j]= 255
else:
img_thresh_back[i,j] = img[i,j]
# Convert array to png image
cv2.imwrite('Dolphin_Thresh_Back.png', img_thresh_back)

Practical No.8

Aim: Write a program in python to insert watermark in image.
Softwares Used: Jupyter Notebook, Opencv
Code:
# import all the libraries
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
# image opening
image = Image.open("dolphin.jpg")
# this open the photo viewer
image.show()
plt.imshow(image)
# text Watermark
watermark_image = image.copy()
draw = ImageDraw.Draw(watermark_image)
# ("font type",font size)
w, h = image.size
x, y = int(w / 2), int(h / 2)
if x > y:
font_size = y
elif y > x:
font_size = x
else:
font_size = x
font = ImageFont.truetype("arial.ttf", int(font_size/6))
# add Watermark
# (0,0,0)-black color text
draw.text((x, y), "dolphin", fill=(0, 0, 0), font=font, anchor='ms')
plt.subplot(1, 2, 1)
plt.title("black text")

plt.imshow(watermark_image)
# add Watermark
# (255,255,255)-White color text
draw.text((x, y), "dolphin", fill=(255, 255, 255), font=font, anchor='ms')
plt.subplot(1, 2, 2)
plt.title("white text")
plt.imshow(watermark_image)
