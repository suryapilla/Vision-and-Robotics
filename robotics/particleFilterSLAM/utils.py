import numpy as np
import matplotlib.pyplot as plt; plt.ion(), plt.gca()
from mpl_toolkits.mplot3d import Axes3D
import time
import random

def tic():
  return time.time()
def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
  return cpr


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1),dtype="int")
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))
    

def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])	
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
    x,y = bresenham2D(sx, sy, 500, 200)
  print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))

def test_mapCorrelation(lidar_ranges):
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = lidar_ranges

  # take valid indices
  indValid = np.logical_and((ranges < 30),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]

  # init MAP
  MAP = {}
  MAP['res']   = 0.05 #meters
  MAP['xmin']  = -20  #meters
  MAP['ymin']  = -20
  MAP['xmax']  =  20
  MAP['ymax']  =  20 
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8


  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)
  
  # convert position in the map frame here 
  Y = np.stack((xs0,ys0))
  
  # convert from meters to cells
  xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
  yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
  
  # build an arbitrary map 
  indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
  MAP['map'][xis[indGood[0]],yis[indGood[0]]]=1
      
  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

  x_range = np.arange(-0.2,0.2+0.05,0.05)
  y_range = np.arange(-0.2,0.2+0.05,0.05)


  
  print("Testing map_correlation with {}x{} cells".format(MAP['sizex'],MAP['sizey']))
  ts = tic()
  c = mapCorrelation(MAP['map'],x_im,y_im,Y,x_range,y_range)
  toc(ts,"Map Correlation")

  c_ex = np.array([[3,4,8,162,270,132,18,1,0],
		  [25  ,1   ,8   ,201  ,307 ,109 ,5  ,1   ,3],
		  [314 ,198 ,91  ,263  ,366 ,73  ,5  ,6   ,6],
		  [130 ,267 ,360 ,660  ,606 ,87  ,17 ,15  ,9],
		  [17  ,28  ,95  ,618  ,668 ,370 ,271,136 ,30],
		  [9   ,10  ,64  ,404  ,229 ,90  ,205,308 ,323],
		  [5   ,16  ,101 ,360  ,152 ,5   ,1  ,24  ,102],
		  [7   ,30  ,131 ,309  ,105 ,8   ,4  ,4   ,2],
		  [16  ,55  ,138 ,274  ,75  ,11  ,6  ,6   ,3]])
    
  if np.sum(c==c_ex) == np.size(c_ex):
    print("...Test passed.")
  else:
    print("...Test failed. Close figures to continue tests.")	

  #plot original lidar points
  fig1 = plt.figure(1)
  plt.plot(xs0,ys0,'.k')
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Laser reading")
  plt.axis('equal')
  
  #plot map
  fig2 = plt.figure(2)
  plt.imshow(MAP['map'],cmap="gray")
  plt.title("Occupancy grid map")
  plt.pause(0.001)
  
  
  #plot correlation
  # fig3 = plt.figure(3)
  # ax3 = fig3.add_subplot(projection='3d')
  # X, Y = np.meshgrid(np.arange(0,9), np.arange(0,9))
  # ax3.plot_surface(X,Y,c,linewidth=0,cmap=plt.cm.jet, antialiased=False,rstride=1, cstride=1)
  # plt.title("Correlation coefficient map")  
  # plt.pause(60)
  # plt.show()
  
  
def show_lidar(lidar_ranges):
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = lidar_ranges
  plt.figure(4)
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(10)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.pause(0.01)
  plt.show()
def get_lidar(lidar_ranges):
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = lidar_ranges

  # take valid indices
  indValid = np.logical_and((ranges < 30),(ranges> 0.1))
  ranges = ranges[indValid]
  angles = angles[indValid]
  
  # init MAP
  MAP = {}
  MAP['res']   = 0.05 #meters
  MAP['xmin']  = -20  #meters
  MAP['ymin']  = -20
  MAP['xmax']  =  20
  MAP['ymax']  =  20 
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
  
  # xy position in the sensor frame
  xs0 = ranges*np.cos(angles)
  ys0 = ranges*np.sin(angles)
  
def motionModel_nParticles(Particles_N, imu_angular_velocity, tau, encoder_cnts):
    
    sigma_v = 0.01
    sigma_w = 0.01
    
    wt_noise = np.random.normal(0,sigma_w,Particles_N.shape[0])
    imu_angular_velocity_N = imu_angular_velocity*np.ones(Particles_N.shape[0]) + wt_noise
    
    theta_N = imu_angular_velocity_N * tau
    
    vt_noise = np.random.normal(0,sigma_v,Particles_N.shape[0])
    linear_velocity = 0.0022*(encoder_cnts/tau)
    linear_velocity_N = linear_velocity*np.ones(Particles_N.shape[0]) + vt_noise
    
    theta_t_1_N = Particles_N[:,2] + theta_N
    
    # x_t_1_N = Particles_N[:,0] + tau*linear_velocity_N*np.cos(theta_t_1_N)
    # y_t_1_N = Particles_N[:,1] + tau*linear_velocity_N*np.sin(theta_t_1_N)
    
    x_t_1_N = Particles_N[:,0] + tau*linear_velocity*np.cos(theta_t_1_N) + np.random.normal(0,sigma_v,Particles_N.shape[0])
    y_t_1_N = Particles_N[:,1] + tau*linear_velocity*np.sin(theta_t_1_N) + np.random.normal(0,sigma_v,Particles_N.shape[0])
    theta_N = theta_t_1_N
    
    return x_t_1_N,y_t_1_N,theta_N
  
def updateStep(Particles_N,MAP,W0,lidar_ranges,lidar_angles,k_init):
  
  N_threshold = Particles_N.shape[0]/5
  
  x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
  y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
  corr_append = []
  for i in range(Particles_N.shape[0]):
    x = Particles_N[i,0]
    y = Particles_N[i,1]
    theta = Particles_N[i,2]
    
    RotMat = np.array([[np.cos(theta),-np.sin(theta),x],[np.sin(theta),np.cos(theta),y],[0,0,1]])
    xs0 = lidar_ranges*np.cos(lidar_angles)
    ys0 = lidar_ranges*np.sin(lidar_angles)
    
    LidWorld = RotMat@np.array([xs0,ys0,np.ones(xs0.shape)])
    
    # convert from meters to cells
    # LidWorld[0] = np.ceil((LidWorld[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    # LidWorld[1] = np.ceil((LidWorld[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    
    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)
    
    corr_i = mapCorrelation(MAP['map'], x_im, y_im, LidWorld, x_range, y_range)
    corr_i = corr_i/(np.sum(corr_i))
    corr_append.append(np.max(corr_i))
    
  W0 = W0*np.array(corr_append)
  
  if(np.sum(W0) == 0):
    W0 = np.ones((Particles_N.shape[0],1))[:,0]/Particles_N.shape[0]
    k = k_init
    N_eff = 1
  else:
    W0 = W0/np.sum(W0)
    k = np.argmax(W0)
    N_eff = 1 / np.sum(np.square(W0))
  k_init = k
  # Resampling
  
  if N_eff < N_threshold:
    Particles_N_new = np.array(random.choices(Particles_N,weights=W0,k=Particles_N.shape[0]))
    W0 = np.ones((Particles_N.shape[0],1))[:,0]/Particles_N.shape[0]
    print(N_eff)
  else:
    Particles_N_new = Particles_N
  # print(N_eff)
    
  return Particles_N[k,0],Particles_N[k,1],Particles_N[k,2],W0,Particles_N_new,k_init

      