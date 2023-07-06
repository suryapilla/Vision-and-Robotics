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

def f(qt,tau_t,wx,wy,wz):
    
    temp_quat = jnp.array([0,0.5*tau_t*wx,0.5*tau_t*wy,0.5*tau_t*wz])
    qt_1 = quatMul(qt,quatExp(temp_quat))
    return qt_1
    
def motionModel(ts,W):
    """
    Calculate the motion model
    
    """
    q0 = jnp.array([1,0,0,0])
    qt = q0
    qt_1_arr = []
    qt_1_arr.append(qt)
    for i in trange(1,ts.shape[1]):
        tau_t = (ts[0][i]-ts[0][i-1])
        w_t = jnp.array([0,W[0,i],W[1,i],W[2,i]])
        p = 0.5*tau_t* w_t
        qt_1 = quatMul(qt,quatExp(p))
        qt_1_arr.append(qt_1)
        qt = qt_1
    return qt_1_arr
    
def observationModel(qt):
    """Calculation of Observation Model

    Args:
        qt nd array: _description_

    Returns:
        _type_: list of quaternions
    """
    a = jnp.array([0,0,0,1])
    qbs_t = []
    qt = jnp.array(qt)
    for i in trange(qt.shape[0]):
        Q = qt[i]
        qt_inver = quatInverse(Q)
        
        qbs_t.append(quatMul(quatMul(qt_inver,a),Q))
        
    return qbs_t

def logC(qt_1,qt,tau_t,wx_t,wy_t,wz_t):
    """

    Args:
        qt_1 (_type_): _description_
        qt (_type_): _description_
        tau_t (_type_): _description_
        wx (_type_): _description_
        wy (_type_): _description_
        wz (_type_): _description_

    Returns:
        _type_: _description_
    """
    # assert isinstance(qt_1,jnp.ndarray)
    qt_1 = jnp.array(qt_1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    # funI  = quatMul(quatInverse(qt_1),f(qt,tau_t,wx_t,wy_t,wz_t))
    # funI  = quatMul(quatInverse(qt_1),f_1)
    return jnp.square(quatNorm(2*quatlog(quatMul(quatInverse(qt_1),f(qt,tau_t,wx_t,wy_t,wz_t)))))

def costfn(Qt_1,ts,imuD):
    cost1 = 0
    
    # assert isinstance(Qt_1,jnp.ndarray)
    
    for i in trange(0,len(Qt_1)-1):
        
        qt_1 = jnp.array(Qt_1[i+1])
        # f_1 = f[i]
        qt = Qt_1[i]
        tau_t = (ts[0][i+1]-ts[0][i])
        cost1 = cost1 + logC(qt_1,qt,tau_t,imuD[3,i],imuD[4,i],imuD[5,i])  
        # cost1 += logC(qt_1,f_1)  
    
    cost2 = 0
    q_acc = jnp.array([0,0,0,1])
    
    for i in trange(1,len(Qt_1)):
        qt_i = Qt_1[i]
        qt_i_inver = quatInverse(Qt_1[i])
        ht_i = quatMul(quatMul(qt_i_inver,q_acc),qt_i)
        at_i = jnp.array([0,imuD[0,i],imuD[1,i],imuD[2,i]])
        cost2 = cost2 + jnp.square(quatNorm(at_i-ht_i))
    
    return 0.5*cost1  + 0.5*cost2


def gradCostFn():#ts,angVx,angVy,angVz,ax,ay,az):
    J = jit(jacrev(costfn))  
    return J



def quatConj(q):
    """_summary_

    Args:
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # assert isinstance(q,jnp.ndarray)
    # qConj = jnp.zeros(q.shape)
    # x = x.at[idx].set(y)
    qConj_v = q[1:4]
    qConj = jnp.array([q[0],-qConj_v[0],-qConj_v[1],-qConj_v[2]])
    # qConj[1:4] = -1*q[1:4]
    
    return qConj    

    
def quatNorm(q):
    """_summary_

    Args:
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    # assert isinstance(q,jnp.ndarray)
    # x = x.at[idx].set(y)
    qs = q[0]
    qv = q[1:4]
    return jnp.sqrt((qs*qs + qv.T@qv))

def quatInverse(q):
    """_summary_

    Args:
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    eps = 0.000001
    # assert isinstance(q,jnp.ndarray)
    
    qInv = quatConj(q)/((quatNorm(q)**2) + eps)
    return qInv

def quatlog(q):
    q = q + 0.00001
    """_summary_

    Args:
        q (_type_): _description_
    """
    # x = x.at[idx].set(y)
    eps = 0.000001
    # assert isinstance(q,jnp.ndarray)
    qs = q[0]
    # qv = jnp.array([q[1],q[2],q[3]])
    qv = q[1:4]
    qv_1 = (qv/(jnp.linalg.norm(qv) + eps))*jnp.arccos(qs/(quatNorm(q) + eps))
    
    logq = jnp.array([jnp.log(quatNorm(q)), qv_1[0], qv_1[1], qv_1[2]])
  
    return logq
# q = jnp.array([1,2,3,4])
# print(quatlog(q))
# q = Quaternion(q)
# print(Quaternion.log(q))

def quatExp(q):
    """_summary_

    Args:
        q (_type_): _description_

    Returns:
        _type_: _description_
    """
    eps = 0.000001
    # assert isinstance(q,jnp.ndarray)
    qs = q[0]
    qv = q[1:4]
    ConstE = (jnp.exp(qs)*jnp.sin(jnp.linalg.norm(qv)))/(jnp.linalg.norm(qv) + eps)
    qv_1 = ConstE*qv
    # p = jnp.zeros(q.shape)
    # p[0] = (jnp.exp(qs))*jnp.cos(jnp.linalg.norm(qv))
    p = jnp.array([(jnp.exp(qs))*jnp.cos(jnp.linalg.norm(qv)), qv_1[0], qv_1[1], qv_1[2]])

    return p
# q = jnp.array([1,2,3,4])
# print(quatExp(q))
# q = Quaternion(q)
# print(Quaternion.exp(q))
    
def quatMul(q,p):
    """find the quaternion multiplication

    Args:
        q (ndarray): _description_
        p (ndarray): _description_

    Returns:
        _type_: _description_
    """
    # assert isinstance(q,jnp.ndarray)
    # assert isinstance(p,jnp.ndarray)
    
    qs = q[0]
    qv = q[1:4]
    ps = p[0]
    pv = p[1:4]

    rv = qs*pv + ps*qv + jnp.cross(qv,pv)
    r = jnp.array([qs*ps - qv.T@pv , rv[0], rv[1], rv[2]])
    
    # pv = jnp.array([p[1],p[2],p[3]])
    # r = jnp.zeros(q.shape)
    # r[0] = qs*ps - (qv.T)@pv
    
    # r = r.at[0].set(qs*ps - (qv.T)@pv)
    # r = r.at[1].set(rv[0])
    # r = r.at[2].set(rv[1])
    # r = r.at[3].set(rv[2])

    return r


# q = jnp.array([1,2,3,4])
# p = jnp.array([23,4,5,6])
# print(quatMul(q,p))
# q = Quaternion(q)
# p = Quaternion(p)
# print(p*q)

