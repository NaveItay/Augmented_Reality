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
>
>      - Print Chessboard for calibration.
>        ```
>        # Config
>        square_size = 3.5   # cm
>        pattern_size = (7, 4)
>        ```
>           ![title](/Images/chessboard_for_calibration.PNG)
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
>
>
>
>
