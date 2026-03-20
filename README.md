# Camera-Calibration
Camera Calibration toolbox using Python &amp; openCV
## Introduction
Cameras have been around in our lives for a while. When first appeared, they were very expensive. Fortunately, in the late 20th century, the pinhole camera was developed and sold at suitable prices such that everybody was able to afford it. However, as is the case with any trade off, this convenience comes at a price. This camera model is typically not good enough for accurate geometrical computations based on images; hence, significant distortions may result in images.

Camera calibration is the process of determining the relation between the camera’s natural units (pixels) and the real world units (for example, millimeters or inches). Technically, camera calibration estimates intrinsic (camera's internal characteristics such as focal length, skew, distortion) and extrinsic (its position and orientation in the world) parameters. Camera calibration is an important step towards getting a highly accurate representation of the real world in the captured images. It helps removing distortions as well.

## Installation
1. Clone or download this repository.

2. Make sure python 3.x is installed on your PC. To check if and which version is installed, run the following command:
```
python -V
```
If this results an error, this means that python isn’t installed on your PC! please install it from [the original website](https://www.python.org/)

3. (optional) it is recommended that you create a python virtual environment and install the necessary libraries in it to prevent versions collisions:
```
python -m venv CV
```
where CV is the environment name. Once you’ve created a virtual environment, you may activate it.
```
CV\Scripts\activate.bat
```

4. Install required libraries from the provided file (**requirements.txt**):
```
pip install -r requirements.txt
```
Make sure you provide the correct path of **requirements.txt**

5. DONE :) Run the script:
```
python calibration_GUI.py
```
