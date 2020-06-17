import numpy as np
from sklearn.cluster import KMeans

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
        result[i+1] = km.predict(np.gradient(feet[i+1]["left"], axis=0))

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