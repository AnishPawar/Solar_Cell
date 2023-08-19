import cv2
img1 = cv2.imread('static/processed_images/heatmap.png')
img2 = cv2.imread('static/processed_images/og.png')
k  = cv2.addWeighted(img1,0.7,img2,0.9,0.0)
cv2.imshow('k,',k)
cv2.waitKey(0)