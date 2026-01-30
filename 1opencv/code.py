import cv2

image = cv2.imread(
    r"D:\me\Mohamed\youtube courses\Learn_Computer_Vision_in_30_Days\assets\LauraPalmer.jpg"
)
cv2.imshow("Laura Palmer", image)
cv2.waitKey(0)
