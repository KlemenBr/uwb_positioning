#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


plt.figure(figsize=(12,12), dpi=300, layout='tight')

##
##
## Floorplan 0
##

anchors = np.asarray([[8.47, 11.87], [8.41, 7.85], [5.95, 7.45], [5.45, 6.71], [6.95, 0.05], [2.24, 0.05], [3.06, 8.33], [4.47, 11.21]])

df = pd.read_csv('../data_set/raw_data/environment0/walking_path.csv', sep=',', header=None, skiprows=1)
walking_path = df.values
print("Number of walking path points: " + str(walking_path.shape[0]))

img = mpimg.imread('../data_set/environment0/floorplan/floorplan.png')
img = np.flipud(img)

ax1 = plt.subplot(2,2,1)
plt.title('a)')
plt.imshow(img)
plt.scatter(walking_path[:,0]*200, walking_path[:,1]*200, s=8, c='blue', label='TAG position')
plt.scatter(anchors[:,0]*200, anchors[:,1]*200, marker='x', c='k', label='ANCHOR position')
plt.xlabel('Position x [m]')
plt.ylabel('Position y [m]')
plt.xlim(0.0, 1836)
plt.ylim(0.0, 2414)
plt.xticks([0,400,800,1200,1600], [0,2,4,6,8])
plt.yticks([0,400,800,1200,1600,2000,2400], [0,2,4,6,8,10,12])
#plt.legend()
#plt.savefig('../data_set/environment0/floorplan/floorplan_track.jpg', dpi=300, bbox_inches='tight')
#plt.show()


##
##
## Floorplan 1
##
anchors = np.asarray([[2.56, 0.09], [3.54, 2.48], [2.7, 4.65], [2.69, 5.49], [1.39, 5.88], [0.6, 1.73], [0.6, 3.43], [0.63, 4.77]])

df = pd.read_csv('../data_set/raw_data/environment1/walking_path.csv', sep=',', header=None, skiprows=1)
walking_path = df.values
print("Number of walking path points: " + str(walking_path.shape[0]))

img = mpimg.imread('../data_set/environment1/floorplan/floorplan.png')
img = np.flipud(img)

ax2 = plt.subplot(2,2,2)
plt.title('b)')
plt.imshow(img)
plt.scatter((walking_path[:,0]+0.2)*200, (walking_path[:,1]+0.32)*200, s=8, c='blue', label='TAG position')
plt.scatter((anchors[:,0]+0.2)*200, (anchors[:,1]+0.32)*200, marker='x', c='k', label='ANCHOR position')
plt.xlabel('Position x [m]')
plt.ylabel('Position y [m]')
plt.xlim(0.0, 800)
plt.ylim(0.0, 1420)
plt.xticks([0,400,800], [0,2,4])
plt.yticks([0,400,800,1200], [0,2,4,6])
#plt.legend()
#plt.savefig('../data_set/environment1/floorplan/floorplan_track.jpg', dpi=300, bbox_inches='tight')
#plt.show()


##
##
## Floorplan 2
##
anchors = np.asarray([[4.11, 4.25], [0.18, 11.12], [10.16, 7.45], [4.20, 6.55], [9.90, 10.10], [21.75,1.50], [14.08, 0.18], [11.36,0.18]])

df = pd.read_csv('../data_set/raw_data/environment2/walking_path.csv', sep=',', header=None, skiprows=1)
walking_path = df.values
print("Number of walking path points: " + str(walking_path.shape[0]))

img = mpimg.imread('../data_set/environment2/floorplan/floorplan.png')
img = np.flipud(img)
ax3 = plt.subplot(2,2,3)
plt.title('c)')
plt.imshow(img)
plt.scatter((walking_path[:,0]+0.2)*200, (walking_path[:,1]+0.2)*200, s=8, c='blue', label='TAG position')
plt.scatter((anchors[:,0]+0.2)*200, (anchors[:,1]+0.2)*200, marker='x', c='k', label='ANCHOR position')
plt.xlabel('Position x [m]')
plt.ylabel('Position y [m]')
plt.xlim(0.0, 4480)
plt.ylim(0.0, 2440)
plt.xticks([0,400,800,1200,1600,2000,2400,2800,3200,3600,4000], [0,2,4,6,8,10,12,14,16,18,20])
plt.yticks([0,400,800,1200,1600,2000,2400], [0,2,4,6,8,10,12])
#plt.legend()
#plt.savefig('../data_set/environment2/floorplan/floorplan_track.jpg', dpi=300, bbox_inches='tight')
#plt.show()



##
##
## Floorplan 3
##
anchors = np.asarray([[0.8, 10.04], [0.8, 5.73], [7.49, 3.97], [9.87, 3.4], [7.26, 11.88], [8.14, 8.24], [11.32, 11.88], [12.18, 8.07]])

df = pd.read_csv('../data_set/raw_data/environment3/walking_path.csv', sep=',', header=None, skiprows=1)
walking_path = df.values
print("Number of walking path points: " + str(walking_path.shape[0]))

img = mpimg.imread('../data_set/environment3/floorplan/floorplan.png')
img = np.flipud(img)
ax4 = plt.subplot(2,2,4)
plt.title('d)')
plt.imshow(img)
plt.scatter(walking_path[:,0]*200, walking_path[:,1]*200, s=8, c='blue', label='TAG position')
plt.scatter(anchors[:,0]*200, anchors[:,1]*200, marker='x', c='k', label='ANCHOR position')
plt.xlabel('Position x [m]')
plt.ylabel('Position y [m]')
plt.xlim(0.0, 3328)
plt.ylim(0.0, 2500)
plt.xticks([0,400,800,1200,1600,2000,2400,2800,3200], [0,2,4,6,8,10,12,14,16])
plt.yticks([0,400,800,1200,1600,2000,2400], [0,2,4,6,8,10,12])
#plt.legend()
plt.savefig('../data_set/technical_validation/floorplans.png', dpi=300, bbox_inches='tight')
plt.close()


