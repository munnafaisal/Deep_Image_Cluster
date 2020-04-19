import cv2

img = cv2.imread('Cropped_Image_2/Cropped_Image/0.jpg')
img_resz = cv2.resize(img,(400,800))

img_resnet = cv2.resize(img_resz,(224,224))
cv2.imshow("ORG", img)
cv2.imshow("resnet", img_resnet)
cv2.waitKey(0)