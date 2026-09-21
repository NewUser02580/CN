import cv2
import matplotlib.pyplot as plt

# Read images
img1 = cv2.imread("image.jpg")
img2 = cv2.imread("image.jpg")

if img1 is None:
    print("Error: image.jpg not found")
    exit()

if img2 is None:
    print("Error: image.jpg not found")
    exit()

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# ============================================================
# PART 1: SIFT
# ============================================================

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)

print("SIFT Keypoints in Image :", len(kp1_sift))
print("SIFT Keypoints in Image :", len(kp2_sift))


# Create FLANN matcher for SIFT
index_params = dict(
    algorithm=1,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)

# Match descriptors
matches_sift = flann.knnMatch(
    des1_sift,
    des2_sift,
    k=2
)

# Lowe's ratio test
good_sift = []

for m, n in matches_sift:
    if m.distance < 0.7 * n.distance:
        good_sift.append(m)

print("Good SIFT Matches:", len(good_sift))


# Draw SIFT matches
sift_result = cv2.drawMatches(
    img1,
    kp1_sift,
    img2,
    kp2_sift,
    good_sift,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# ============================================================
# PART 2: ORB
# ============================================================

# Create ORB detector
orb = cv2.ORB_create(
    nfeatures=1000
)

# Detect keypoints and descriptors
kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)

print("ORB Keypoints in Image 1:", len(kp1_orb))
print("ORB Keypoints in Image 2:", len(kp2_orb))


# ORB uses binary descriptors,
# so use BFMatcher with Hamming distance
bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

# Find two best matches
matches_orb = bf.knnMatch(
    des1_orb,
    des2_orb,
    k=2
)

# Lowe's ratio test
good_orb = []

for m, n in matches_orb:
    if m.distance < 0.7 * n.distance:
        good_orb.append(m)

print("Good ORB Matches:", len(good_orb))


# Draw ORB matches
orb_result = cv2.drawMatches(
    img1,
    kp1_orb,
    img2,
    kp2_orb,
    good_orb,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

# Convert BGR to RGB
sift_result = cv2.cvtColor(
    sift_result,
    cv2.COLOR_BGR2RGB
)

orb_result = cv2.cvtColor(
    orb_result,
    cv2.COLOR_BGR2RGB
)


# Display both results
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.imshow(sift_result)
plt.title("SIFT Feature Matching")
plt.axis("off")

plt.subplot(2, 1, 2)
plt.imshow(orb_result)
plt.title("ORB Feature Matching")
plt.axis("off")

plt.tight_layout()
plt.show()
