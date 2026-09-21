import cv2
import numpy as np
image = cv2.imread("image.jpg")
if image is None:
    print("Image Not Found!")
else:
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Format Conversion
image = cv2.imread("image.jpg")
cv2.imwrite("newimage.png", image)
# Color Space Operations
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
#smoothing
image = cv2.imread("image.jpg")
smoothed_image = cv2.GaussianBlur(image, (51, 51), 0)
cv2.imshow("Smoothed Image", smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#sharpening
image = cv2.imread("image.jpg")
kernel = np.array([[0,-2,0],
                    [-2,9,-2],
                    [0,-2,0]])
sharpened_image = cv2.filter2D(image, -1, kernel)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#histogram
image = cv2.imread("image.jpg",0)
equalized = cv2.equalizeHist(image)
cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()