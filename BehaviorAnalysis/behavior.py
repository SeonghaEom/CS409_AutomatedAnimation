import numpy as np
from sklearn.cluster import KMeans

def getFootVector(video):
    foot = {}
    leftFootVector = {}
    rightFootVector = {}

    for h in range(video.hum_cnt):
        foot[h+1] = {}
    
    for each in video.frame.instances:
        for hum in each.humans:
            foot[hum.id][each.id] = [hum.pose_pos[15], hum.pose_pos[16]]
    
    for h in range(len(foot)):
        frame_list = foot[h+1].keys()
        
        for each in video.frame.instances:
            n_frame = video.frame.getNext(each.id)
            if n_frame and each.id in frame_list and n_frame.id in frame_list:
                left_vec = np.subtract(foot[h+1][n_frame.id][0], foot[h+1][each.id][0])
                right_vec = np.subtract(foot[h+1][n_frame.id][1], foot[h+1][each.id][1])

                if left_vec[0]>15: y_left = 1
                elif left_vec[0]<-15: y_left = -1
                else: y_left = 0

                if right_vec[0]>15: y_right = 1
                elif right_vec[0]<-15: y_right = -1
                else: y_right = 0
                
                if each.id not in leftFootVector.keys():
                    leftFootVector[each.id] = y_left
                else:
                    leftFootVector[each.id] += y_left
                if each.id not in rightFootVector.keys():
                    rightFootVector[each.id] = y_right
                else:
                    rightFootVector[each.id] += y_right
    

    leftFootVectorChunk = {}
    rightFootVectorChunk = {}
    prev_left = 0
    prev_right = 0
    first_left = 0
    first_right = 0
    frame_list = leftFootVector.keys()
    for each in video.frame.instances:
        if each.id not in frame_list: continue
        if leftFootVector[each.id] * prev_left < 0:
            leftFootVectorChunk[(first_left, each.id)] = prev_left
            prev_left = leftFootVector[each.id]
            first_left = each.id
        else:
            prev_left += leftFootVector[each.id]
        if rightFootVector[each.id] * prev_right < 0:
            rightFootVectorChunk[(first_right, each.id)] = prev_right
            prev_right = rightFootVector[each.id]
            first_right = each.id
        else:
            prev_right += rightFootVector[each.id]

    
    video.leftFootVectorChunk = leftFootVectorChunk
    video.rightFootVectorChunk = rightFootVectorChunk


def selectFootVectorChunk(video, interval):
    framesForLeftFoot = {}
    framesForRightFoot = {}
    for start, end in video.leftFootVectorChunk.keys():
        if end-start >= interval: framesForLeftFoot[(start, end)] = video.leftFootVectorChunk[(start, end)]

    for start, end in video.rightFootVectorChunk.keys():
        if end-start >= interval: framesForRightFoot[(start, end)] = video.rightFootVectorChunk[(start, end)]

    video.framesForLeftFoot = framesForLeftFoot
    video.framesForRightFoot = framesForRightFoot
    


def getLeftFeetMov(video):
    feet = {}

    i = 0
    for i in range(video.hum_cnt):
        feet[i+1] = {"left": [], "right": []}
    for each in video.frame.instances:
        for hum in each.humans:
            feet[hum.id]["left"].append(hum.pose_pos[15])
            feet[hum.id]["right"].append(hum.pose_pos[16])
            # print (feet[hum.id])
    grad = np.gradient(feet[1]["left"], axis=0)
    # print(grad)
    for i in range(2, video.hum_cnt):
        grad = np.concatenate((grad, np.gradient(feet[i]["left"], axis=0)), axis=0)
    # origin = [0], [0]
    # plt.quiver(*origin, grad[:,0], grad[:,1], angles='xy', scale_units='xy', scale=1)
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # plt.show()

    # # get num of clusters
    # distortions = []
    # K = range(1,20)
    # for k in K:
    #     kmeanModel = KMeans(n_clusters=k).fit(grad)
    #     kmeanModel.fit(grad)
    #     distortions.append(sum(np.min(cdist(grad, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / grad.shape[0])

    # # Plot the elbow
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()

    km = KMeans(n_clusters=9)
    km.fit(grad)
    # km.predict(np.gradient(feet[1]["left"], axis=0))
    # print (km.predict(np.gradient(feet[6]["left"], axis=0)))
    # print (km.predict(np.gradient(feet[3]["left"], axis=0)))
    # print (km.predict(np.gradient(feet[1]["left"], axis=0)))


    # df = pd.DataFrame(grad)
    # df['category'] = km.labels_
    # colormap = { 0: 'red', 1: 'green', 2: 'blue', 3:'yellow', 4:'purple', 5: 'orange', 6:'grey', 7:'pink', 8:'navy'}
    # colors = df.apply(lambda row: colormap[row.category], axis=1)
    # ax = df.plot(kind='scatter', x=0, y=1, alpha=0.1, s=300, c=colors, cmap=plt.cm.get_cmap('rainbow', 9))
    # plt.show()

    result = {}

    for i in range(video.hum_cnt):
        result[i+1] = km.predict(np.gradient(feet[i+1]["left"], axis=1))

    video.leftFeetMov = (result)
    return (result)



def getRightFeetMov(video):
    feet = {}

    i = 0
    for i in range(video.hum_cnt):
        feet[i+1] = {"left": [], "right": []}
    for each in video.frame.instances:
        for hum in each.humans:
            # print (hum.id)
            # print (hum.pose_pos[15])
            # print (hum.pose_pos[16])
            left_x_mov = float(hum.pose_pos[15][0]/(hum.box_pos[1] - hum.box_pos[0]))
            left_y_mov = float(hum.pose_pos[15][1]/(hum.box_pos[3] - hum.box_pos[2]))
            right_x_mov = float(hum.pose_pos[16][0]/(hum.box_pos[1] - hum.box_pos[0]))
            right_y_mov = float(hum.pose_pos[16][1]/(hum.box_pos[3] - hum.box_pos[2]))
            feet[hum.id]["left"].append(hum.pose_pos[15])
            feet[hum.id]["right"].append(hum.pose_pos[16])
            # print (feet[hum.id])
    grad = np.gradient(feet[1]["right"], axis=0)
    # print(grad)
    for i in range(2, video.hum_cnt):
        grad = np.concatenate((grad, np.gradient(feet[i]["right"], axis=0)), axis=0)
    # origin = [0], [0]
    # plt.quiver(*origin, grad[:,0], grad[:,1], angles='xy', scale_units='xy', scale=1)
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # plt.show()

    # get num of clusters
    # distortions = []
    # K = range(1,20)
    # for k in K:
    #     kmeanModel = KMeans(n_clusters=k).fit(grad)
    #     kmeanModel.fit(grad)
    #     distortions.append(sum(np.min(cdist(grad, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / grad.shape[0])

    # # Plot the elbow
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()

    km = KMeans(n_clusters=10)
    km.fit(grad)
    # km.predict(np.gradient(feet[1]["left"], axis=0))
    # print (km.predict(np.gradient(feet[6]["left"], axis=0)))
    # print (km.predict(np.gradient(feet[3]["left"], axis=0)))
    # print (km.predict(np.gradient(feet[1]["left"], axis=0)))
    result = {}
    for i in range(video.hum_cnt):
        result[i+1] = km.predict(np.gradient(feet[i+1]["right"], axis=0))
    
    video.rightFeetMov = (result)
    return (result)