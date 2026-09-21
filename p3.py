import cv2

# STEP 1: READ THE IMAGE

image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found!")
    print("Please check the image name and folder.")
else:

    # Display original image
    cv2.imshow("Original Image", image)

    # STEP 2: CONVERT IMAGE TO GRAYSCALE


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Grayscale Image", gray)

    # STEP 3: SOBEL EDGE DETECTION

    # Detect vertical edges
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    # Detect horizontal edges
    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # Convert values to absolute values
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Combine horizontal and vertical edges
    sobel_combined = cv2.addWeighted(
        sobel_x,
        0.5,
        sobel_y,
        0.5,
        0
    )

    cv2.imshow("Sobel X - Vertical Edges", sobel_x)
    cv2.imshow("Sobel Y - Horizontal Edges", sobel_y)
    cv2.imshow("Sobel Combined", sobel_combined)

    # STEP 4: CANNY EDGE DETECTION

    canny_edges = cv2.Canny(
        gray,
        100,
        200
    )

    cv2.imshow("Canny Edge Detection", canny_edges)

    # STEP 5: FAST CORNER DETECTION

    # Create FAST detector
    fast = cv2.FastFeatureDetector_create()

    # Detect corners/keypoints
    keypoints = fast.detect(
        gray,
        None
    )

    # Draw detected corners on original image
    fast_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0)
    )

    # Display FAST result
    cv2.imshow("FAST Corner Detection", fast_image)

    print("Number of corners detected:", len(keypoints))

    cv2.waitKey(0)
    cv2.destroyAllWindows()