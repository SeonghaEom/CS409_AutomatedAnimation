# Let`s import all packages that we may need:

import sys 
import json
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


