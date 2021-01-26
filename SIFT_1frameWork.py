import cv2
import numpy as np
from matplotlib import pyplot as plt

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

figsize = (100, 100)

# ImRead
Query_img = cv2.imread("Query_Grayscale.jpg", cv2.IMREAD_GRAYSCALE)  # Query_image
train_img = cv2.imread("DSC_0008.JPG", cv2.IMREAD_GRAYSCALE)   # Train_image
Itay_Alex = cv2.imread("Itay&Alex_RGB.png")                        # itayAlex_image
Itay_Alex = cv2.resize(Itay_Alex, (2009, 2842))


RGB_Itay_Alex = cv2.cvtColor(Itay_Alex, cv2.COLOR_BGR2RGB)
train = cv2.imread("DSC_0008.JPG")
RGB_train = cv2.cvtColor(train, cv2.COLOR_BGR2RGB)

height, width, m = RGB_Itay_Alex.shape
Blank_Mask = 255*np.ones((height, width, m), dtype=np.uint8)

plt.figure(figsize=figsize)
plt.imshow(Query_img, cmap="gray", vmin=0, vmax=255)
plt.title("query_img")
plt.show()
plt.figure(figsize=figsize)
plt.imshow(train_img, cmap="gray", vmin=0, vmax=255)
plt.title("train_img")
plt.show()
plt.figure(figsize=figsize)
plt.imshow(RGB_Itay_Alex, cmap="gray", vmin=0, vmax=255)
plt.title("RGB_Itay_Alex")
plt.show()

# Features
sift = cv2.xfeatures2d.SIFT_create()
kp_query_img, desc_query_img = sift.detectAndCompute(Query_img, None)
kp_train_img, desc_train_img = sift.detectAndCompute(train_img, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Matches
matches = flann.knnMatch(desc_query_img, desc_train_img, k=2)

# Matches Filter
good_points = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good_points.append(m)

# DrawMatches
MatchesPlot = cv2.drawMatches(Query_img, kp_query_img, train_img, kp_train_img, good_points, train_img, flags=2)

plt.figure(figsize=figsize)
plt.imshow(MatchesPlot, cmap="gray", vmin=0, vmax=255)
plt.title("MatchesPlot")
plt.show()

# Homography
if len(good_points) > 15:
    query_pts = np.float32([kp_query_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    train_pts = np.float32([kp_train_img[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
    H_matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # WarpPerspective
    h, w = Query_img.shape
    # pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, H_matrix)

    RGB_Itay_Alex_warped = cv2.warpPerspective(RGB_Itay_Alex, H_matrix, (train_img.shape[1], train_img.shape[0]))

    plt.figure(figsize=figsize)
    plt.imshow(RGB_Itay_Alex_warped, cmap="gray", vmin=0, vmax=255)
    plt.title("warpPerspective")
    plt.show()

    Blank_Mask_warped = cv2.warpPerspective(Blank_Mask, H_matrix, (train_img.shape[1], train_img.shape[0]))
    plt.figure(figsize=figsize)
    plt.imshow(Blank_Mask_warped, cmap="gray", vmin=0, vmax=255)
    plt.title("Blank_Mask_warped")
    plt.show()

    Inv_Blank_Mask_warped = cv2.bitwise_not(Blank_Mask_warped)
    plt.figure(figsize=figsize)
    plt.imshow(Inv_Blank_Mask_warped, cmap="gray", vmin=0, vmax=255)
    plt.title("Inv_Blank_Mask_warped")
    plt.show()

    Mask_Result = cv2.bitwise_and(Inv_Blank_Mask_warped, RGB_train)
    plt.figure(figsize=figsize)
    plt.imshow(Mask_Result, cmap="gray", vmin=0, vmax=255)
    plt.title("Mask_Result")
    plt.show()

    Result = cv2.add(RGB_Itay_Alex_warped, Mask_Result)
    plt.figure(figsize=figsize)
    plt.imshow(Result, cmap="gray", vmin=0, vmax=255)
    plt.title("Result")
    plt.show()

