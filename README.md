# Augmented Reality using python
![title](/Images/introduction.PNG)
## Project Goals:
* Goal A: Find an image in a short video and replace it with another image.
* Goal B: Perform a camera calibration.
* Goal C: Insert a 3D object into the video.
  
  
  
###### Execution stages (Goal A – replace image):
> *  Working on a photo (One frame)
>    - Take a photo with the image in it.
>         
>           
>         ![title](/Images/train_img.PNG)
>      -  __In this photo we can see the train image__
>    
>
>    - Find a good reference to the image.
>         
>           
>         ![title](/Images/Query_img.PNG)
>
>    - Choose/Create an image to put into the frame.
>         
>           
>         ![title](/Images/input_image.PNG)
>
>    - Find matches with sift algorithm (Photo and reference).
>         
>           
>         ![title](/Images/MachesPlot.PNG)
>      -  __Find key points and descriptors in query and train image.__
>      ```
>      # Sift Feature
>      sift = cv2.xfeatures2d.SIFT_create()
>      kp_Query_img, desc_Query_img = sift.detectAndCompute(Query_img, None)
>      ...
>      ....
>      while cap.isOpened():
>          ..
>          ...
>          kp_GrayFrame, desc_GrayFrame = sift.detectAndCompute(Train_img, mask=None)
>      ```
>
>
>
>      -  __Flann Based Matcher has been used because it works faster than BF matcher for large Data sets, it contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features.__
>      ```
>      # Feature matching
>      index_params = dict(algorithm=0, trees=5)
>      search_params = dict()
>      Flann = cv2.FlannBasedMatcher(index_params, search_params)
>      ...
>      ....
>      while cap.isOpened():
>          ..
>          ...
>          matches = Flann.knnMatch(desc_Query_img, desc_GrayFrame, k=2)
>      ```     
>      
>      
>      
>      -  __Filter the matches using the distance ratio on the descriptors.__
>      ```
>      # Matches Filter
>      good_points = []
>      for m, n in matches:
>          if m.distance < 0.5 * n.distance:
>              good_points.append(m)
>      ```       
>      
>    - Find matches with sift algorithm (Photo and reference).   
>      -  __Convert to coordinates (x, y)__
>      -  __Find H matrix with cv2.findhomography function__
>      ```
>      # Homography
>      query_pts = np.float32([kp_Query_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
>      train_pts = np.float32([kp_GrayFrame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
>      H_matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
>      matches_mask = mask.ravel().tolist()
>      ```     
>         ![title](/Images/H_matrix.PNG)
>
>
>    - Perform transformation (with H matrix) on input image (the image we want to put into the video).
>      ```
>      # WarpPerspective
>      RGB_Itay_Alex_warped = cv2.warpPerspective(RGB_Itay_Alex, H_matrix, (Train_img.shape[1], Train_img.shape[0]))
>      ```
>         ![title](/Images/warped.PNG)
>
>      -  __Before pasting this image inside the photo, we need to reset all the pixels that are in the position of the image.__
>
>
>    - Perform transformation (with same H matrix) to a blank white picture. 
>       
>         ![title](/Images/mask_warped.PNG)
>
>
>    - Inverse the blank mask.
>           
>         ![title](/Images/inv_mask_warped.PNG)
>
>    - Reset the pixels within the black pixel location the mask (bitwise_and)
>      ```
>      Mask_Result = cv2.bitwise_and(Inv_Blank_Mask_warped, current_frame)
>      ```
>         ![title](/Images/mask_result.PNG)
>
>    - Add input picture to the frame.
>      ```
>      Result = cv2.add(RGB_Itay_Alex_warped, Mask_Result)
>      ```
>         ![title](/Images/result_sift.PNG)


###### Execution stages (Goal B – Camera calibration):
> *  Working on a photo (One frame)
>    - Print Chessboard for calibration.
>      ```
>      # Config
>      square_size = 3.5   # cm
>      pattern_size = (7, 4)
>      ```
>         ![title](/Images/chessboard_for_calibration.PNG)
>
>    - Take few photos.
>         ![title](/Images/few_phot.PNG)
>
>    - Find corners with cv2 library.
>      ```
>      found, corners = cv2.findChessboardCorners(img, pattern_size)
>      ```
>         ![title](/Images/find_corners.PNG)
>
>    - Undistorted pictures.
>      ```
>      dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)
>      ```
>         ![title](/Images/undistort.PNG)
>
>    - Draw a cube on undistorted pictures.
>         ![title](/Images/Draw_cube.PNG)
>     
>    - Calibration matrix and distortion coefficients to add a 3d object to the video
>      (find rotation and translation vectors).  
>      ```
>      camera_matrix = np.float32([[4.33361171e+03, 0.00000000e+00, 2.99003303e+03], [0.00000000e+00, 4.41254035e+03, 2.43338726e+03], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
>                                 
>      dist_coeffs = np.float32([[-1.05783249e-02, 2.71079802e-01, -2.68722164e-04, -2.64562464e-03, 2.47868056e-01]])
>      ```



###### Execution stages (Goal C – Insert 3D object):
> *  Working on a photo (One frame)
>    - Load a 3D object and transform it (rotation and size)
>      ```
>      mesh = trimesh.load('models/drill.obj')
>      # normalize bounding box from (0,0,0) to max(30)
>      mesh.rezero()  # set th LOWER LEFT (?) as (0,0,0)
>      T = np.eye(4)
>      T[0:3, 0:3] = Drill_Size * np.eye(3)*(1 / np.max(mesh.bounds))
>      mesh.apply_transform(T)
>      # rotate to make the drill standup
>      T = np.eye(4)
>      T[0:3, 0:3] = rot_x(np.pi/2)
>      mesh.apply_transform(T)
>      ```
>
>    - Use the H matrix to get warped rectangle points.
>      -  __Object points – static 4 points (corners) with of the query image rectangle.__
>      -  __Image points – dynamic 4 points (corners) from the book location on the frame.__
>         ```
>         # WarpPerspective
>         h, w = Query_img.shape
>         pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
>         dst = cv2.perspectiveTransform(pts, H_matrix)
>         
>         # ObjectPoints and ImagePoints
>         zer = np.zeros((pts.shape[0], 1, 1))
>         ObjectPoints = np.float32(np.append(pts, zer, axis=2))
>         ImagePoints = dst
>         ```
>
>    - Use cv2.solvePnP algorithm to find rotation vector and translation vector.
>      ```
>      # solvePnP
>      _ret, rotationVectors, translationVectors = cv2.solvePnP(ObjectPoints, ImagePoints, camera_matrix, dist_coeffs)
>      ```
>      -  __camera matrix and distortion coefficients from camera calibration.__
>
>    - Draw a 3D object.
>      -  __The 3d object is colorless because the xlib library is not supported by windows and it is not the main purpose of this project.__
>      -  __If you got an error change the library to draw the image. (put the use of this library in TRY-EXCEPT block).__
>         ```
>         def make_uncurrent(self):
>           try:
>             import pyglet.gl.xlib
>             pyglet.gl.xlib.glx.glXMakeContextCurrent(self._window.context.x_display, 0,0, None)
>           except:
>             pass    
>         ```
>         ![title](/Images/3D_result.PNG)
>
>
>
>
>
>
>
>
>
>
>
>
>
