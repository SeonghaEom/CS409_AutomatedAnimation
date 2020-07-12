# Let`s import all packages that we may need:

import sys 
import json
import math
import numpy as np # linear algebra
from functools import reduce
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D



# ## for Deep-learing:
# import keras
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.utils import to_categorical
# from keras.optimizers import SGD 
# from keras.callbacks import EarlyStopping
# from keras.utils import np_utils
# import itertools
# from keras.layers import LSTM
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# from keras.layers import Dropout

pose_pos = ['nose',
'left_eye',
'right_eye',
'left_ear',
'right_ear',
'left_shoulder',
'right_shoulder',
'left_elbow',
'right_elbow',
'left_wrist',
'right_wrist',
'left_hip',
'right_hip',
'left_knee',
'right_knee',
'left_ankle',
'right_ankle']

def get_angle(vec1, vec2):
    unit_vector_1 = vec1 / np.linalg.norm(vec1)
    unit_vector_2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    if (angle > math.pi):
        angle = angle - math.pi
    return angle * 180 / math.pi

with open('../(G)I-DLE-02_matching.json') as json_file:
    json_string = json.load(json_file)
json_data = json.loads(json_string)
for i, each in enumerate(list(json_data.values())):
    for frame in list(each.values()):
        frame['frame'] = i
frames = list(map(lambda frame: list(frame.values()), list(json_data.values())))
# print (type(frames[0]))

concat = sum(frames, [])
filtered = list(filter(lambda json: json['id'] == 1, concat))
pose = []
frameid = []
for each in filtered:
    # print (list(filter(lambda key: key == 'frame' or key == 'pose_pos', each)))
    pose.append({key: val for key, val in each.items() if key == 'pose_pos'})
    # print (each['frame'])
    frameid.append(each['frame'])

pose_flatten = {}
for i in range(len(pose_pos)):
    pose_flatten[pose_pos[i]]= []
for pose_each in pose:
    for i in range(len(pose_pos)):
        pose_flatten[pose_pos[i]].append(pose_each['pose_pos'][i])

df = pd.DataFrame(data = pose_flatten, columns= pose_pos)
# df_idx = pd.DataFrame(data = frameid, columns = ['frame'])
df.insert(0, "frame", frameid, True)
# print(df.head())
# print (df.info())
print (df.describe())

## finding all columns that have nan:

droping_list_all=[]
for j in range(0,8):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        #print(df.iloc[:,j].unique())
print (droping_list_all)

df['center_shoulder'] = df.apply(lambda row: [(row.left_shoulder[0] + row.right_shoulder[0])/2, (row.left_shoulder[1]+row.right_shoulder[1])/2], axis=1)

df['vector'] = df.apply(lambda row: np.array(row.center_shoulder) - np.array(row.left_shoulder), axis=1)
df['vector2'] = df.apply(lambda row: np.array(row.left_elbow) - np.array(row.left_shoulder), axis=1)
df['vector3'] = df.apply(lambda row: np.array(row.left_wrist) - np.array(row.left_elbow), axis=1)
df['left_shoulder_angle'] = df.apply(lambda row: get_angle(row.vector, row.vector2), axis=1)
df['left_arm_angle'] = df.apply(lambda row: get_angle(-1 * row.vector2, row.vector3), axis=1)

df['vector4'] = df.apply(lambda row: np.array(row.center_shoulder) - np.array(row.right_shoulder), axis=1)
df['vector5'] = df.apply(lambda row: np.array(row.right_elbow) - np.array(row.right_shoulder), axis=1)
df['vector6'] = df.apply(lambda row: np.array(row.right_wrist) - np.array(row.right_elbow), axis=1)
df['right_shoulder_angle'] = df.apply(lambda row: get_angle(row.vector4, row.vector5), axis=1)
df['right_arm_angle'] = df.apply(lambda row: get_angle(-1 * row.vector5, row.vector6), axis=1)



df2 = df.filter(['frame', 'left_shoulder_angle', 'left_arm_angle', 'right_shoulder_angle', 'right_arm_angle'])
print (df.head())
print (df2.head())
# x = df2['left_shoulder_angle']
# y = df2['left_arm_angle']
# sns.scatterplot(x=x, y=y)
# plt.show()

## finding all columns that have nan:

df2.dropna(inplace=True)
droping_list_all=[]
for j in range(0,5):
    if not df2.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        print(df2.iloc[:,j].unique())

print (droping_list_all)
print (df2.info())
df2.to_csv('./angles.csv', sep=',', na_rep='NaN')


feature = df2[:] #df[:1000]
model = DBSCAN(eps=45, min_samples=30) #maximum 90 dots in 90frames
predict = pd.DataFrame(model.fit_predict(feature))
predict.columns=['predict']
predict['frame'] = df2['frame']
# r = pd.concat([feature, predict], axis=1)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(r['frame'], r['left_shoulder_angle'], r['left_arm_angle'])
# ax.set_xlabel('frame')
# ax.set_xlabel('left_shoulder_angle')
# ax.set_xlabel('left_arm_angle')
# plt.show()

print (set(predict['predict']))
predict.to_csv("./predict.csv", sep=',', na_rep='NaN')

