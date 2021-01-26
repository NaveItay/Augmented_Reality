import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import stl

# Config
square_size = 3.5       # cm
pattern_size = (7, 4)
img_mask = './chess_2/*.JPG'
figsize = (20, 20)

# Using '*' pattern
print('\nChess Photos Loaded:')
for name in glob.glob(img_mask):
    print(name)

img_names = glob.glob(img_mask)
num_images = len(img_names)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = cv2.imread(img_names[0]).shape[:2]


plt.figure(figsize=figsize)

for i, fn in enumerate(img_names):
    print('processing %s... ' % fn)
    imgBGR = cv2.imread(fn)

    if imgBGR is None:
        print("Failed to load", fn)
        continue

    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    # # if you want to better improve the accuracy... cv2.findChessboardCorners already uses cv2.cornerSubPix
    # if found:
    #     term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    #     cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if not found:
        print('chessboard not found')
        continue

    if i < 12:
        img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
        plt.subplot(4, 3, i+1)
        plt.imshow(img_w_corners)

    print('           %s... OK' % fn)
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)


plt.show()

# calculate camera distortion
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)




print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())


# undistort the image with the calibration
plt.figure(figsize=figsize)
for i, fn in enumerate(img_names):

    imgBGR = cv2.imread(fn)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)

    if i < 4:
        plt.subplot(2, 2, i+1)
        plt.imshow(dst)

plt.show()
print('Done')

objectPoints = 3 * square_size * np.array(
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


plt.figure(figsize=figsize)
for i, fn in enumerate(img_names):

    imgBGR = cv2.imread(fn)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

    dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)

    imgpts = cv2.projectPoints(objectPoints, _rvecs[i], _tvecs[i], camera_matrix, dist_coefs)[0]
    drawn_image = draw(dst, imgpts)

    # Filename
    filename = 'savedImage' + str(i) + '.jpg'

    # Using cv2.imwrite() method
    # Saving the image
    outputpath = 'D:/computer vision/PROJECT_2/output_folder'
    cv2.imwrite(os.path.join(outputpath, filename), drawn_image)

    #cv2.imwrite(filename, drawn_image)

    if i < 12:
        plt.subplot(4, 3, i + 1)
        plt.imshow(drawn_image)

plt.show()



# Using an existing stl file:
your_mesh = stl.Mesh.from_file('Baby_Yoda_Smile_Reach.stl')

# Or creating a new mesh (make sure not to overwrite the `mesh` import by
# naming it `mesh`):
VERTICE_COUNT = 100
data = np.zeros(VERTICE_COUNT, dtype=stl.Mesh.dtype)
your_mesh = stl.Mesh(data, remove_empty_areas=False)

# The mesh normals (calculated automatically)
your_mesh.normals
# The mesh vectors
your_mesh.v0, your_mesh.v1, your_mesh.v2
# Accessing individual points (concatenation of v0, v1 and v2 in triplets)
#assert (your_mesh.points[0][0:3] == your_mesh.v0[0]).all()
#assert (your_mesh.points[0][3:6] == your_mesh.v1[0]).all()
#assert (your_mesh.points[0][6:9] == your_mesh.v2[0]).all()
#assert (your_mesh.points[1][0:3] == your_mesh.v0[1]).all()

#your_mesh.save('new_stl_file.stl')
