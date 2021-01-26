# Augmented Reality using python
![title](/Images/introduction.PNG)
## Project Goals:
* Goal A: Find an image in a short video and replace it with another image.
* Goal B: Perform a camera calibration.
* Goal C: Insert a 3D object into the video.
  
  
  
###### Execution stages (Goal A – replace image):
> 1. Working on a photo (One frame)
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
>      
>      
>      
>      ```
>
>
>
>
