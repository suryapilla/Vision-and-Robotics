#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

import argparse
import yaml

import numpy as np
from utils import *
from tqdm import trange

#%%

with open("config/config.yml", 'r') as stream:
    config = yaml.safe_load(stream)
    
dataset = config['DATASET']
path_figures = config['PATH_FIGURES']
occupany_fig = config['OCCUPANCY_GRID_FIG_NAME']
traj_fig = config['TRAJECTORY_FIG_NAME']
texture_fig = config['TEXTURE_MAP_FIG_NAME']
disp_path = config['DISPARITY_IMG_PATH'] + str(dataset) + "/"
rgb_path = config['RGB_IMG_PATH'] + str(dataset) + "/"

#%%

with np.load("./data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("./data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("./data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

with np.load("./data/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images


#%%

# Time sync
zt = encoder_counts.mean(axis=0)
imu_sync_idx = []
lidar_sync_idx = []
for i in range(encoder_stamps.shape[0]):
    imuIdx = np.argmax(imu_stamps > encoder_stamps[i])
    imu_sync_idx.append(imuIdx)
    lidarIdx = np.argmax(lidar_stamps > encoder_stamps[i])
    lidar_sync_idx.append(lidarIdx)

lidar_ranges = lidar_ranges[:,lidar_sync_idx]    
imu_angular_velocity_sync = imu_angular_velocity[:,imu_sync_idx]
imu_stamps_sync = imu_stamps[imu_sync_idx]

tau=[]
for i in range(encoder_stamps.shape[0]-1):
    tau.append(encoder_stamps[i+1]-encoder_stamps[i])

#%% Mapping and Trajectory generation

MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -30  #meters
MAP['ymin']  = -30
MAP['xmax']  =  30
MAP['ymax']  =  30 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'])) #DATA TYPE: char or int8
MAP['trajectory'] = np.zeros((MAP['sizex'],MAP['sizey']))

robot_posx = MAP['sizex']/2
robot_posy = MAP['sizey']/2

angles = np.arange(-135,135.25,0.25)*np.pi/180.0
ranges = lidar_ranges[:,0]

# take valid indices
indValid = np.logical_and((ranges < 30),(ranges > 0.1))
ranges = ranges[indValid]
angles = angles[indValid]

# xy position in the sensor frame
xs0 = ranges*np.cos(angles)
ys0 = ranges*np.sin(angles)

# convert from meters to cells
xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1


for i in range(xis.shape[0]):
    x_n,y_n = bresenham2D(robot_posx, robot_posy, xis[i], yis[i])
    MAP['map'][x_n[0:-1],y_n[0:-1]]+=np.log(4)
    MAP['map'][x_n[-1],y_n[-1]]+=np.log(1/4)

N = config['NUMBER_OF_PARTICLES']
Particles_N = np.zeros((N,3))
W0 = np.ones((N,1))[:,0]/N
k_init=0
Particle_x = []
Particle_y = []
Particle_x_upd = []
Particle_y_upd = []
traj = []
for k in trange(encoder_stamps.shape[0]-1):
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    ranges = lidar_ranges[:,k+1]
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    
    # Apply Prediction Step
    omega_t = imu_angular_velocity_sync[2,k]
    tau_t = tau[k]
    encoder_cnts_t = zt[k]
    
    x_t_1,y_t_1,theta_t_1 = motionModel_nParticles(Particles_N,omega_t,tau_t,encoder_cnts_t)
    Particles_N = np.array([x_t_1,y_t_1,theta_t_1]).T
    Particle_x.append(Particles_N[:,0])
    Particle_y.append(Particles_N[:,1])
    
    # Apply update step here
    x_t_1,y_t_1,theta_1,W0,Particles_N,k_init = updateStep(Particles_N,MAP,W0,ranges,angles,k_init)
    Particle_x_upd.append(x_t_1)
    Particle_y_upd.append(y_t_1)
    traj.append([x_t_1,y_t_1,theta_1])
    
    RotMat = np.array([[np.cos(theta_1),-np.sin(theta_1),x_t_1],[np.sin(theta_1),np.cos(theta_1),y_t_1],[0,0,1]])
    
    xs0 = ranges*np.cos(angles)
    ys0 = ranges*np.sin(angles)
    
    LidWorld = RotMat@np.array([xs0,ys0,np.ones(xs0.shape)])

    
    xR0 = LidWorld[0]
    yR0 = LidWorld[1]


    if (x_t_1 > MAP['xmax'] or x_t_1 < MAP['xmin'] or y_t_1 > MAP['ymax'] or y_t_1 < MAP['ymin']):
        continue
    
    x_st = np.ceil((x_t_1 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_st = np.ceil((y_t_1 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    
    # convert from meters to cells
    xis = np.ceil((xR0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((yR0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1


    for i in range(xis.shape[0]):
        
        x_n,y_n = bresenham2D(x_st, y_st, xis[i], yis[i])
        
        x_n_valid_ind = (x_n < MAP['sizex']) & (x_n >= 0)
        y_n_valid_ind = (y_n < MAP['sizey']) & (y_n >= 0)
        xy_valid_ind = x_n_valid_ind & y_n_valid_ind
        
        x_n = x_n[xy_valid_ind]
        y_n = y_n[xy_valid_ind]

        MAP['map'][x_n[0:-1],y_n[0:-1]]-=np.log(4)

        MAP['map'][x_n[-1],y_n[-1]]+=np.log(4)

traj_np = np.array(traj)      
#%%
plt.figure(1)   
sigmoid_map = np.exp(MAP['map'])/(1+np.exp(MAP['map']))
plt.imshow(sigmoid_map,cmap="gray")  
plt.savefig(path_figures + occupany_fig)

plt.figure(2)
plt.plot(Particle_x,Particle_y)
plt.plot(Particle_x_upd,Particle_y_upd)
plt.savefig(path_figures + traj_fig)

#%%
# Time sync for RGB camera
traj_np.shape
enc_sync_idx = []
disp_sync_idx = []
for i in range(rgb_stamps.shape[0]):
    
    encIdx = np.argmax(encoder_stamps > rgb_stamps[i])
    enc_sync_idx.append(encIdx)
    dispIdx = np.argmax(disp_stamps > rgb_stamps[i])
    disp_sync_idx.append(dispIdx)
traj_np_sync = traj_np[enc_sync_idx,:]
disp_stamps_sync = disp_stamps[disp_sync_idx]

#%%
# Time sync for IMU
def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

# Generation of texture map
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -30  #meters
MAP['ymin']  = -30
MAP['xmax']  =  30
MAP['ymax']  =  30 
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'],3)) #DATA TYPE: char or int8
MAP['trajectory'] = np.zeros((MAP['sizex'],MAP['sizey']))


for img_num in trange(0,rgb_stamps.shape[0]):
    disp_num = disp_sync_idx[img_num]
    # load RGBD image
    
    imd = cv2.imread(disp_path+'disparity' + str(dataset)  + '_' + str(disp_num+1) +'.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(rgb_path+'rgb' + str(dataset)  + '_' + str(img_num+1) +'.png')[...,::-1] # (480 x 640 x 3)
    # print(imc.shape)

    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd
    # print(z)
    # break
    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
    

    rgbu = rgbu[valid]
    rbgv = rgbv[valid]
    # roll = 0
    # pitch = 0.36
    # yaw = 0.021
    
    # Rx = np.array([[1,             0,             0],
    #                [0,  np.cos(roll), -np.sin(roll)],
    #                [0,  np.sin(roll),  np.cos(roll)]])
    
    # Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
    #                [0            , 1,             0],
    #                [-np.sin(pitch),0, np.cos(pitch)]])
    
    # Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
    #                [np.sin(yaw), np.cos(yaw),0],
    #                [0          ,           0,1]])
    
    # R = (Rz.dot(Ry)).dot(Rx)
    R = np.array([[ 0.9356905, -0.0196524,  0.3522742],
                    [0.0209985,  0.9997795, -0.0000000],
                    [-0.3521966,  0.0073972,  0.9358968 ]])
    # print(R)
    # print(xsdfgh)
    P = np.array([[0.18,0.005,0.36]])
    # print(P.shape)
    # optical to regular frame
    o_R_r = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    r_R_o = np.linalg.inv(o_R_r)
    
    X_o = np.array([x[valid],y[valid],z[valid]])
    # print(X_o.shape)
    img_reg = r_R_o.dot(X_o)
    # print(img_reg.shape)
    # Regular to sensor frame
    # img_sens = R.dot(img_reg) + np.array([img_reg[0]*np.ones(img_reg.shape[1]),img_reg[1]*np.ones(img_reg.shape[1]),img_reg[2]*np.ones(img_reg.shape[1])])
    img_sens = R.dot(img_reg) + P.T
    
    x_t_1 = traj_np_sync[img_num,0]
    y_t_1 = traj_np_sync[img_num,1]
    z_t_1 = 0.147
    
    theta_t_1 = traj_np_sync[img_num,2]
    
    RotMat = np.array([[np.cos(theta_t_1),-np.sin(theta_t_1),0,x_t_1],[np.sin(theta_t_1),np.cos(theta_t_1),0,y_t_1],[0,0,1,z_t_1],[0,0,0,1]])
    
    # Sensor to world frame
    
    img_World = RotMat@np.array([img_sens[0],img_sens[1],img_sens[2],np.ones(img_sens[0].shape)])
    # print(img_World.shape)
    
    x_img = img_World[0]
    y_img = img_World[1]
    z_img = img_World[2]
    
    z_n_valid_ind = (z_img < 0.1) & (z_img > -0.1)
    
    
    x_st = np.ceil((x_img - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_st = np.ceil((y_img - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    x_n_valid_ind = (x_st < MAP['sizex']) & (x_st >= 0)
    y_n_valid_ind = (y_st < MAP['sizey']) & (y_st >= 0)
    
    # print(x_n_valid_ind.shape,y_n_valid_ind.shape,z_n_valid_ind.shape)
    xyz_valid_ind_temp = x_n_valid_ind & y_n_valid_ind 
    xyz_valid_ind = xyz_valid_ind_temp & z_n_valid_ind
    
    x_n = x_st[xyz_valid_ind]
    y_n = y_st[xyz_valid_ind]
    # z_n = z_n[xyz_valid_ind]
    rgbu = rgbu[xyz_valid_ind].astype(int)
    rbgv = rbgv[xyz_valid_ind].astype(int)
    # print(rgbu.shape)
    # break
    
    
    MAP['map'][x_n,y_n] = imc[rbgv,rgbu]
    # if(img_num%100==0):
    #     plt.figure(2)   
    # # sigmoid_map = np.exp(MAP['map'])/(1+np.exp(MAP['map']))
    #     plt.imshow(MAP['map'].astype(int))  
    #     plt.savefig(path_figures + "textureMap/" + "texture" + str(img_num) + ".png")
 
plt.figure(1)   
# sigmoid_map = np.exp(MAP['map'])/(1+np.exp(MAP['map']))
plt.imshow(MAP['map'].astype(int))  
plt.savefig(path_figures + texture_fig)
plt.show() 
