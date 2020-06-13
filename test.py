from parse import Behavior
from parse import Human
from parse import Frame
from parse import Video
import numpy as np
import matplotlib.pyplot as plt
import copy

""" initialize behavior """
""" TODO: index uniquely assign """
clap = Behavior(0, [9,10])
raiseArm = Behavior(1, [0,1,4,7])


print (clap.akpnt)

""" initialize video """
video = Video('../(G)I-DLE-01_matching.json')
i = 0
# for i in range(0, 1500):
#     print (video.getCenter(i))
# print (video.getFormationChunk())
print(video.getFeetMov())

# print ("There are ", video.hmcnt, "people in this video")
# print ("There are ", len(video.frame.get(9).humans), "people in this frame")
# print ("There are ", video.frame.getLength(), "frames")

""" track frame 99 ~ 101 by clap behavior """
# print(video.Track(clap, 99, 101))
# plot = []
# x = []
# y = []
# y2 = []
# previous_form =[]
# formation = []
# for each in video.frame.instances:
#     previous_form = copy.copy(formation)
#     if (len(each.humans) == video.hmcnt):
#         print ("frame no ", each.id)
#         if (each.id > 5000):
#             x.append(each.id)
#         cores = []
#         formation = []
#         for hum in each.humans:
#             # calculate core pos
#             a = np.array(hum.pose_pos)
#             core = np.mean(a, axis=0)
#             # print (core)
#             cores.append((core, hum.id))
#         cores.sort(key = lambda element: element[0][0])
#         for ea in cores:
#             formation.append(ea[1])
#         if (each.id > 5000):
#             y.append(cores[video.hmcnt//2][1])
#         if (str(previous_form) == str(formation) and each.id > 5000):
#             y2.append(1)
#         elif (each.id > 5000):
#             y2.append(0)
#         print(cores)

# plt.scatter(x,y)
# plt.show()







