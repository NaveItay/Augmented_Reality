import cv2
import numpy as np
import trimesh
import pyrender
import math

class basic_cube_render():
    def __init__(self):
        # x = 1.2
        x = 0.5
        # y = 1.9
        y = 7.2
        z = -5

        self.objectPoints = 110*np.array([[0 + x, 0 + y, 0], [0 + x, 1 + y, 0], [1 + x, 1 + y, 0], [1 + x, 0 + y, 0], [0 + x, 0 + y, -1],
                                        [0 + x, 1 + y, -1], [1 + x, 1 + y, -1], [1 + x, 0 + y, -1]], dtype=float)

    def draw(self, img, rvec, tvec):
        imgpts = cv2.projectPoints(self.objectPoints, rvec, tvec, camera_matrix, dist_coeffs)[0]

        imgpts = np.int32(imgpts).reshape(-1, 2)

        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        return img

def rot_x(t):
    ct = np.cos(t)
    st = np.sin(t)
    m = np.array([[1, 0, 0],
                  [0, ct, -st],
                  [0, st, ct]])
    return m

class mesh_render():

    def __init__(self, mesh):

        # rotate 180 around x because the Z dir of the reference grid is down
        # T = np.eye(4)
        T[0:3, 0:3] = rot_x(np.pi)
        mesh.apply_transform(T)
        # Load the trimesh and put it in a scene
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
        scene.add(mesh)

        # add temp cam
        self.camera = pyrender.IntrinsicsCamera(camera_matrix[0, 0],
                                                camera_matrix[1, 1],
                                                camera_matrix[0, 2],
                                                camera_matrix[1, 2], zfar=10000, name="cam")
        light_pose = np.array([
            [1.0, 0, 0, 0.0],
            [0, 1.0, 0.0, 10.0],
            [0.0, 0, 1, 100.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.cam_node = scene.add(self.camera, pose=light_pose)

        # Set up the light -- a single spot light in z+
        light = pyrender.SpotLight(color=255 * np.ones(3), intensity=3000.0,
                                   innerConeAngle=np.pi / 16.0)
        scene.add(light, pose=light_pose)

        self.scene = scene
        height, width, m = 1080, 1920, 3
        # height, width, m = Result.shape
        self.r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        # add the A flag for the masking
        self.flag = pyrender.constants.RenderFlags.RGBA

    def draw(self, img, rvec, tvec):
        # ===== update cam pose
        camera_pose = np.eye(4)
        res_R, _ = cv2.Rodrigues(rvec)

        # opengl transformation
        # https://stackoverflow.com/a/18643735/4879610
        camera_pose[0:3, 0:3] = res_R.T
        camera_pose[0:3, 3] = (-res_R.T @ tvec).flatten()
        # 180 about x
        camera_pose = camera_pose @ np.array([[1, 0, 0, 0],
                                              [0, -1, 0, 0],
                                              [0, 0, -1, 0],
                                              [0, 0, 0, 1]])

        self.scene.set_pose(self.cam_node, camera_pose)

        # ====== Render the scene
        # color, depth = self.r.render(self.scene)
        color, depth = self.r.render(self.scene, flags=self.flag)
        img[color[:, :, 3] != 0] = color[:, :, 0:3][color[:, :, 3] != 0]
        return img

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

Drill_Size = 500
Fox_Size = 220

mesh = trimesh.load('models/drill.obj')
# normalize bounding box from (0,0,0) to max(30)
mesh.rezero()  # set th LOWER LEFT (?) as (0,0,0)
T = np.eye(4)
T[0:3, 0:3] = Drill_Size * np.eye(3)*(1 / np.max(mesh.bounds))
mesh.apply_transform(T)
# rotate to make the drill standup
T = np.eye(4)
T[0:3, 0:3] = rot_x(np.pi/2)
mesh.apply_transform(T)

fox = trimesh.load('models/fox.obj')
# normalize bounding box from (0,0,0) to max(30)
fox.rezero()  # set th LOWER LEFT (?) as (0,0,0)
T = np.eye(4)
T[0:3, 0:3] = Fox_Size * np.eye(3)*(1 / np.max(fox.bounds))
fox.apply_transform(T)
T = np.eye(4)
T[0:3, 0:3] = rot_x(np.pi/2)
fox.apply_transform(T)

camera_matrix = np.float32([[4.33361171e+03, 0.00000000e+00, 2.99003303e+03],
                            [0.00000000e+00, 4.41254035e+03, 2.43338726e+03],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# rms = 2.725661584054838

dist_coeffs = np.float32([[-1.05783249e-02, 2.71079802e-01, -2.68722164e-04,
                           -2.64562464e-03, 2.47868056e-01]])

# ImRead
Query_img = cv2.imread("Query_Grayscale.jpg", cv2.IMREAD_GRAYSCALE)  # Query_image
RGB_Itay_Alex = cv2.imread("INPUT_IMG.jpg")
# RGB_Itay_Alex = cv2.cvtColor(Itay_Alex, cv2.COLOR_BGR2RGB)     # itayAlex_image

height, width, m = RGB_Itay_Alex.shape
Blank_Mask = 255*np.ones((height, width, m), dtype=np.uint8)

# Sift Feature
sift = cv2.xfeatures2d.SIFT_create()
kp_Query_img, desc_Query_img = sift.detectAndCompute(Query_img, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
Flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture("DSC_0012.MOV")

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('Final_V2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (864, 972))

# cube
cube = basic_cube_render()

# drill
drill = mesh_render(mesh)

# fox
FOX = mesh_render(fox)

while cap.isOpened():
    ret, current_frame = cap.read()

    Train_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)  # Train_image

    kp_GrayFrame, desc_GrayFrame = sift.detectAndCompute(Train_img, mask=None)

    matches = Flann.knnMatch(desc_Query_img, desc_GrayFrame, k=2)

    # Matches Filter
    good_points = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_points.append(m)

    # DrawMatches
    img3 = cv2.drawMatches(Query_img, kp_Query_img, Train_img, kp_GrayFrame, good_points, Train_img, flags=2)

    # Homography
    if len(good_points) > 15:
        query_pts = np.float32([kp_Query_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_GrayFrame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        H_matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # PutText
        cv2.putText(img3, "Matches amount:", (1250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img3, str(len(good_points)), (1500, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # WarpPerspective
        h, w = Query_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H_matrix)

        # ObjectPoints and ImagePoints
        zer = np.zeros((pts.shape[0], 1, 1))
        ObjectPoints = np.float32(np.append(pts, zer, axis=2))
        ImagePoints = dst

        # solvePnP
        _ret, rotationVectors, translationVectors = cv2.solvePnP(ObjectPoints, ImagePoints, camera_matrix, dist_coeffs)

        # _ret, rotationVectors, translationVectors, inliers = cv2.solvePnPRansac(ObjectPoints, ImagePoints,
        #                                                                         camera_matrix, dist_coeffs)

        # WarpPerspective
        h, w = Query_img.shape
        RGB_Itay_Alex_warped = cv2.warpPerspective(RGB_Itay_Alex, H_matrix, (Train_img.shape[1], Train_img.shape[0]))
        Blank_Mask_warped = cv2.warpPerspective(Blank_Mask, H_matrix, (Train_img.shape[1], Train_img.shape[0]))

        Inv_Blank_Mask_warped = cv2.bitwise_not(Blank_Mask_warped)
        Mask_Result = cv2.bitwise_and(Inv_Blank_Mask_warped, current_frame)
        Result = cv2.add(RGB_Itay_Alex_warped, Mask_Result)

        # Draw cube
        cube_img = cube.draw(Result, rotationVectors, translationVectors)

        # Draw Fox
        fox_img = FOX.draw(cube_img, rotationVectors, translationVectors)

        # Draw drill
        drill_img = drill.draw(fox_img, rotationVectors, translationVectors)

        imgStack = stackImages(0.45, ([drill_img, ], [img3, ]))

        if ret is True:
            # Write the frame into the file 'output.avi'
            out.write(imgStack)

        cv2.imshow("Homography", imgStack)
    else:
        imgStack = stackImages(0.45, ([drill_img, ], [img3, ]))
        cv2.imshow("Homography", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()