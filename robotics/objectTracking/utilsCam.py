#%%
# utils file consisting of all the modules
import pickle
import sys
from transforms3d.euler import euler2mat, mat2euler, quat2euler
# import numpy as jnp
from pyquaternion import Quaternion
from jax import jacrev, jit
import jax.numpy as jnp
from tqdm import tqdm
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm,trange
import time 
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2mat, mat2euler
import jax
from jax import value_and_grad
from transforms3d.euler import quat2mat, mat2euler, quat2euler
#%%
#%%
def read_data(fname):
    """_summary_

    Args:
        fname (str): give ijnput file

    Returns:
        : dictionary type is returned
    """
    d = []

    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')  # need for python 3
        return d

def xyz2rpy(data):
    """give out the roll,pitch,yaw for xyz

    Args:
        data in numpy 2D array: _description_
    """
    x_Roll = []
    y_Pitch = []
    z_Yaw = []
    for i in range(data.shape[2]):
        
        R = data[:,:,i]
        
        x_roll,y_pitch,z_yaw = mat2euler(R,'sxyz')
        x_Roll.append(x_roll)
        y_Pitch.append(y_pitch)
        z_Yaw.append(z_yaw)
        
    return x_Roll,y_Pitch,z_Yaw

def scalingFacs():
    """This function gives out the scaling factors

    Returns:
        float: The return output1 is scaling factor for accelerator
               The return output1 is scaling factor for gyroscope
                
    """
    sensitivtyGyro = 3.33 # in units 
    sensitivtyAcc = 300
    Vref = 3300 # in units mv
    scalingFactAcc = Vref/(1023*sensitivtyAcc)

    scalingFactGyro = (jnp.pi*3300)/(180*1023*sensitivtyGyro) 
    # print(scalingFactAcc,":",scalingFactGyro)
    return scalingFactAcc, scalingFactGyro
# scalingFacs()


def quat2eul(data):
    data = jnp.array(data)
    x_Roll = []
    y_Pitch = []
    z_Yaw = []
    for i in range(data.shape[0]):
        
        Q = Quaternion(data[i])
        x_roll,y_pitch,z_yaw = quat2euler(Q,'sxyz')
        x_Roll.append(x_roll)
        y_Pitch.append(y_pitch)
        z_Yaw.append(z_yaw)
    return x_Roll,y_Pitch,z_Yaw

def pix2spheric(hFOV,vFOV,imgRow,imgCol):
    hFOV = 60
    vFOV = 45

    lam = np.linspace(hFOV/2,-hFOV/2,imgCol)
    phi = np.linspace(-vFOV/2,vFOV/2,imgRow)

    z = 1
    camLatLong = np.ones((imgRow,imgCol,3))
    camLatLong[:,0:320,0]=lam
    camLatLong[0:240,:,1]=np.array([phi,]*320).transpose()
    
    return camLatLong

def spheric2cart(imgRow,imgCol,camLatLong):
    camCart = np.ones((imgRow,imgCol,3))
    sinPhi = np.sin((np.pi/180)*camLatLong[:,:,1])
    cosPhi = np.cos((np.pi/180)*camLatLong[:,:,1])
    sinLam = np.sin((np.pi/180)*camLatLong[:,:,0])
    cosLam = np.cos((np.pi/180)*camLatLong[:,:,0])
    camCart[:,:,0] = cosPhi*cosLam
    camCart[:,:,1] = -cosPhi*sinLam
    camCart[:,:,2] = -sinPhi   

    return camCart

def cart2spheric(x,y,z):
    
    r = np.sqrt(x**2 + y**2 + z**2)
    lam = np.arcsin(-z/r)
    phi = np.arctan2(y, x)
    return lam, phi, r

