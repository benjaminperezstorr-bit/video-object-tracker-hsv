# Method Explanation

## HSV Color Space

The HSV color space represents colors using three components:

- Hue: the color type
- Saturation: the intensity of the color
- Value: the brightness

HSV is useful for object tracking because it is often more robust than RGB when dealing with variations in lighting.

## Color Segmentation

After the user clicks on the target object, the selected pixel is converted from BGR to HSV. A tolerance range is then created around this HSV value.

Pixels inside this range are kept, while other pixels are removed.

## Morphological Operations

The binary mask may contain noise. To reduce this noise, erosion and dilation are applied.

Erosion removes small isolated pixels.

Dilation fills holes and strengthens the detected region.

## Contour Detection

Contours are extracted from the binary mask. The largest contour is selected as the object to track.

A bounding box is then drawn around the object.
