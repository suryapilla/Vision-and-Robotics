#%%
#%%
from utils import *
from utilsCam import *
import time 
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2mat, mat2euler, quat2mat
import jax
from jax import value_and_grad
import yaml
import argparse

#%%

# Read the data
# Given input data path
path = str(input("Enter the path to the test data directory: "))
# path = r"/home/surya/Desktop/ucsd/ECE276A/PR1/data"
with open("/home/surya/Desktop/ucsd/ECE276A/PR1/SuryaPillaCodeSubmission/config/config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# breakpoint()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=int,help="Enter the dataset number",choices=[1,2,3,4])
config['dataset'] = parser.parse_args().dataset
# breakpoint()
dataset = config['dataset']
cfile = path + "/cam/cam" + dataset + ".p"
ifile = path + "/imu/imuRaw" + dataset + ".p"

camd = read_data(cfile)
imud = read_data(ifile)

# Applying Scaling
scalingFactAcc, scalingFactGyro = scalingFacs()

imuData = imud["vals"].astype(np.float64)

# As the wz,wx,wy axis are flipped this is re-aligning
imuData[[3,4,5],:] = imuData[[4,5,3],:]

# Compensating for bias by taking first n points where there is no rotation and averaging
numpoint = 500
biasAccX = float(sum(imuData[0,:numpoint]) / numpoint)
biasAccY = float(sum(imuData[1,:numpoint]) / numpoint)
biasAccZ = float(sum(imuData[2,:numpoint]) / numpoint)

biasGyrX = float(sum(imuData[3,:numpoint]) / numpoint)
biasGyrY = float(sum(imuData[4,:numpoint]) / numpoint)
biasGyrZ = float(sum(imuData[5,:numpoint]) / numpoint)

# Adding bias and scaling, note that a negative sign is applied as the Ax, Ay axis are in negative given in the documentation
for i in range(len(imuData[:].T)):

    imuData[0,i] = scalingFactAcc*(-imuData[0,i] + biasAccX)
    imuData[1,i] = scalingFactAcc*(-imuData[1,i] + biasAccY)
    imuData[2,i] = scalingFactAcc*(imuData[2,i] - biasAccZ) + 1
    
    imuData[3,i] = scalingFactGyro*(imuData[3,i] - biasGyrX)
    imuData[4,i] = scalingFactGyro*(imuData[4,i] - biasGyrY)
    imuData[5,i] = scalingFactGyro*(imuData[5,i] - biasGyrZ)


# Apply Motion Model
ts = imud["ts"]
Qt_1= motionModel(ts,imuData[3:6,:])

imuRoll,imuPitch,imuYaw = quat2eul(Qt_1)


alpha = 0.01
eps = 0.0000001 # pertubation to avoid nans

ts = jnp.array(imud["ts"].astype(np.float64))

Qt_o = jnp.array(Qt_1)

numIter = 10
for i in trange(1,numIter):
  print(i)
  # print(costfn(Qt_o,ts,imuData))
  J  = jacrev(costfn)(Qt_o,ts,imuData)
  QR = Qt_o - (alpha*J)
  QR_norm = jnp.linalg.norm(QR,axis=1).reshape(-1,1)
  Qt_1_i = QR/(QR_norm+eps)

  Qt_o = Qt_1_i

# determining roll, pitch and yaw for the estimate quaternions 
imuRoll_o,imuPitch_o,imuYaw_o = quat2eul(Qt_o)

#%%>>>>>>>>>>>>>>>>>>>>>>>>>Comment  this section if plots are not required <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# plotting the row, pitch and yaw for VICON, Estimated, Optimized, comment out the VICON values if plotting for test data
plt.figure(1)
plt.title("Roll")
plt.plot(imud["ts"].T,imuRoll)
plt.plot(imud["ts"].T,imuRoll_o)
plt.xlabel("time(s)")
plt.ylabel("angle(radians)")
plt.legend(["Non-Optimized", "Optimized"])
plt.grid(True)

plt.figure(2)
plt.title("Pitch")
plt.plot(imud["ts"].T,imuPitch)
plt.plot(imud["ts"].T,imuPitch_o)
plt.xlabel("time(s)")
plt.ylabel("angle(radians)")
plt.legend(["Non-Optimized", "Optimized"])
plt.grid(True)

plt.figure(3)
plt.title("Yaw")
plt.plot(imud["ts"].T,imuYaw)
plt.plot(imud["ts"].T,imuYaw_o)
plt.xlabel("time(s)")
plt.ylabel("angle(radians)")
plt.legend(["Non-Optimized", "Optimized"])
plt.grid(True)


#%%>>>>>>>>>>>>>>>>>>>>>>>>>Comment  this section if Panorama is not desired <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
camd = read_data(cfile)

#%%
camData, camTs = camd['cam'], camd['ts'][0]
imgRow, imgCol, imgChan, imgCnt = camData.shape

# converting pixel coorinates to spherical coordinates (lamda,phi,1)
hFOV = 60
vFOV = 45

camLatLong = pix2spheric(hFOV,vFOV,imgRow,imgCol)
camLatLong[0,0,:]

#%% x,y,z = cos(phi)cos(lam), cos(phi)sin(lam), sin(phi) %%#
camCart = spheric2cart(imgRow,imgCol,camLatLong)
        
imuIdx = 0
panRows, panCols = 720, 1020
pamImg = np.zeros((int(panRows), int(panCols ), 3))

for i in trange(imgCnt):

    
    #finding the closest time for camera and imu
    imuIdx = np.argmax(imud["ts"]>camTs[i])

    # extract the rotation matrix from IMU/vicon data
    R_cam2World = quat2mat(Qt_o[imuIdx])
    
    # Apply rotation matrix and convert coordinates from camera to world coordinates
    worldCoord = R_cam2World@(camCart).reshape((-1, 3)).T
    
    # As there is 10cms of difference between IMU and camera along z axis a position vector corresponding to 10cm is added
    worldCart = worldCoord.T.reshape((imgRow, imgCol, 3)) + np.array([0,0,0.1])
    
    # convert cartesian coordinates to spherical coordinates
    worldSpheric = cart2spheric(worldCart[:, :, 0], worldCart[:, :, 1], worldCart[:, :, 2])


    worldSpheric = np.stack(worldSpheric, axis=-1)
    worldSpheric = worldSpheric[:, :, 0:2]

    # project spherical coords to cylinderical plane
    worldSpheric[:, :, 0] = (np.pi/2 + worldSpheric[:, :, 0])
    worldSpheric[:, :, 1] = (np.pi + worldSpheric[:, :, 1])
    
    
    # scale to image size
    worldSpheric[:, :, 0] = worldSpheric[:, :, 0]*panRows / np.pi
    worldSpheric[:, :, 1] = worldSpheric[:, :, 1]*panCols /(2 * np.pi)

    # convert to ints and copy image to panorama_img
    worldCyl = worldSpheric.astype(int)

    pamImg = pamImg.astype(int)
    
    pamImg[worldCyl [:, :, 0], worldCyl[:, :, 1]] = camData[:, :, :, i]

plt.figure(12)
plt.imshow(pamImg)
plt.show()


