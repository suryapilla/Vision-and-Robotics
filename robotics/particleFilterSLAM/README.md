# Particle Filter SLAM

## Objective:
Implement simultaneous localization and mapping (SLAM) using encoder and IMU odometry, 2-D Li-
DAR scans, and RGBD measurements from a differential-drive robot. Use the odometry, and LiDAR
measurements to localize the robot and build a 2-D occupancy grid map of the environment. Use the
RGBD images to assign colors to your 2-D map of the floor.

### 2-D Ocuppancy Grid
![Alt Text](https://media.giphy.com/media/0DbwULmbO3ERJPBHPP/giphy.gif)

### 2-D texture map of the floor
![Alt Text](https://media.giphy.com/media/IZ7Ky27ORuudlzbNrR/giphy.gif)


## Sensors:

1. Encoder: Encoders count the rotations of the four wheels at 40 Hz. The encoder counter
is reset after each reading. For example, if one rotation corresponds to ` meters traveled, five
consecutive encoder counts of 0, 1, 0, −2, 3 correspond to (0 + 1 + 0 − 2 + 3) = 2 meters traveled
for that wheel. The data sheet indicates that the wheel diameter is 0.254 m and since there
are 360 ticks per revolution, the wheel travels 0.0022 meters per tic. Given encoder counts
[F R, F L, RR, RL] corresponding to the front-right, front-left, rear-right, and rear-left wheels,
the right wheels travel a distance of (F R + RR)/2 ∗ 0.0022 m, while the left wheels travel a
distance of (F L + RL)/2 ∗ 0.0022 m.

2. IMU: Linear acceleration and angular velocity data is provided from an inertial measurement
unit. The IMU data is noisy, since it is collected from a moving robot with high-frequency
vibrations. You should consider applying a low-pass filter (e.g., with bandwidth around 10 Hz)
to reduce the measurement noise. We will only use is the yaw rate from the IMU as the angular
velocity in the differential-drive model in order to predict the robot motion. It is not necessary
to use the other IMU measurements.

3. LiDAR (Hokuyo): A horizontal LiDAR with 270◦ degree field of view and maximum range of 30 m
provides distances to obstacles in the environment. Each LiDAR scan contains 1081 measured
range values. The sensor is called Hokuyo UTM-30LX and its specifications can be viewed
online. The location of the sensor with respect to the robot body is shown in the provided robot
description file. Make sure you know how to interpret the LiDAR data and how to convert from
range measurements to (x, y) coordinates in the sensor frame, then to the body frame of the
robot, and finally to the world frame.

4. RGBD Camera (Kinect): An RGBD camera provides RGB images and disparity images. The depth camera is
located at (0.18, 0.005, 0.36) m with respect to the robot center and has orientation with roll 0 rad, pitch 0.36 rad, and yaw 0.021 rad.
 

## Dataset:
The dataset for two scenarios is provided in the data folder for encoder, LiDAR and IMU.

Download the dataRGBD dataset from the link and copy the dataRGBD "data" directory : https://drive.google.com/drive/folders/1Fn7YF4u-0bwKGcdKhu76zGfcxNydyXdr?usp=drive_link


## Code:

- Clone the repository
- create virtual environement (anaconda can be used to do it and instructions are given below):
```
conda create --name env_particleSLAM
conda activate env_particleSLAM
git clone https://github.com/suryapilla/Vision-and-Robotics/tree/particleSLAM/robotics/objectTracking
pip install -r requirements.txt
```
- utils.py contains all relevant functions required for the main function
- give correct paths in `config.yml`
- There are 3 sections in the main.py script:
	- Trajectory generation
	- Map generation
	- Texture Mapping
- Run main.py to generate all the 3 plots: Trajectory, Occupancy map and Texture Map
```
python main.py
```
- The script creates 3 png images for the above mentioned and saves in figures directory

#### *Please refer to the report.pdf for detailed explanation of the code implementation


 
