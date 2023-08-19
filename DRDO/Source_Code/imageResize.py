import cv2
import os

dir = '/Users/anishpawar/IITB/DRDO/Images/PNG'
dst = '/Users/anishpawar/IITB/DRDO/Images/1536'
paths = os.listdir(dir)

for i in paths:
    image = cv2.imread(f'{dir}/{i}')
    w, h, c = image.shape
    aspect = w/h
    image = cv2.resize(image, (1536, 1536))
    cv2.imwrite(f'{dst}/{i}', image)
    print(w, h, c)
    # cv2.imshow('GG', image)
    # cv2.waitKey(0)
