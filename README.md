
# HSV Video Object Tracker

This project is an interactive video object tracker built with Python and OpenCV.

The tracker allows the user to click on an object in the webcam stream. The selected pixel color is converted to the HSV color space, and the object is then tracked in real time using color segmentation.

Features: 

Real-time webcam video processing
Interactive object selection with mouse click
RGB/BGR to HSV color conversion
HSV thresholding with tolerance values
Special handling for red hue wrapping around 0/180
Morphological operations to reduce noise
Contour detection
Bounding box and center point visualization
Reset and quit keyboard controls
Method

The tracking pipeline is based on the following steps:

Capture frames from the webcam using OpenCV.
Let the user click on the target object.
Convert the selected pixel from BGR to HSV.
Build a binary mask around the selected HSV value.
Apply morphological operations to clean the mask.
Detect contours in the mask.
Select the largest contour as the tracked object.
Draw a bounding box, center point and tracking information.
Installation
## Why HSV?

HSV is more suitable than RGB for color-based tracking because it separates chromatic information from brightness. This makes it easier to define a color range around the selected object, even when lighting conditions change slightly.



Create a virtual environment:

python -m venv venv

Activate it:

On Linux/macOS:

source venv/bin/activate

On Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Usage

Run the tracker:

python src/tracker_hsv.py

Controls:

Left click: select the object to track
r: reset the tracker
q: quit

Technical Details :
The tracker uses the HSV color space because it separates color information from brightness more effectively than RGB/BGR.

The selected HSV value is used to create a color range:

lower = target_hsv - tolerance
upper = target_hsv + tolerance

A binary mask is then computed with:

cv2.inRange(hsv_frame, lower, upper)

The largest detected contour is considered to be the tracked object.

Limitations

This first version is intentionally simple and only uses color information.

Main limitations:

The tracker may fail if the background has a similar color.
It is sensitive to lighting changes.
It does not handle occlusions.
It only tracks the largest region matching the selected color.
It does not use motion prediction.
It does not assign persistent IDs to multiple objects.
Future Improvements

Possible improvements:

Add Kalman filtering for motion prediction.
Add MeanShift or CamShift tracking.
Add multi-object tracking.
Add object re-identification after occlusion.
Use YOLO for object detection.
Combine color features with shape or motion features.
Save the output video.
Add evaluation metrics.
Add a graphical interface.
Project Status

First functional version completed.

This version demonstrates a basic but complete computer vision pipeline for real-time object tracking using HSV color segmentation.

## Roadmap

- [x] Real-time webcam capture
- [x] Mouse-based target selection
- [x] HSV color segmentation
- [x] Morphological mask cleaning
- [x] Bounding box visualization
- [ ] Save tracked video output
- [ ] Add trajectory visualization
- [ ] Add Kalman filter prediction
- [ ] Add multi-object tracking
- [ ] Add deep learning detector

Author

Perez Benjamin
