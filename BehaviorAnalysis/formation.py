import numpy as np
import copy

def getCenterChunk(video):
    if not video.centerChunk:
        #video.center is empty, initlize it
        prev_center = -1
        first = 0
        for each in video.frame.instances:
            if (len(each.humans) == video.hum_cnt):
                cores = []
                for hum in each.humans:
                    # calculate core pos
                    a = np.array(hum.pose_pos)
                    core = np.mean(a, axis=0)
                    cores.append((core, hum.id))
                cores.sort(key = lambda element: element[0][0])
                center = cores[video.hum_cnt//2][1]
                each.center = center

                if prev_center != center:
                    video.centerChunk[(first, each.id)] = prev_center
                    prev_center = center
                    first = each.id

    return video.centerChunk

def getFormationChunk(video, interval):
    if not video.formationChunk:
        # formation chunk is empty
        previous_form =[]
        formation = []
        first = 0
        # last = 0
        for each in video.frame.instances:
            previous_form = copy.copy(formation)
            if (len(each.humans) == video.hum_cnt):
                cores = []
                formation = []
                for hum in each.humans:
                    # calculate core pos
                    a = np.array(hum.pose_pos)
                    core = np.mean(a, axis=0)
                    cores.append((core, hum.id))
                cores.sort(key = lambda element: element[0][0])
                for ea in cores:
                    formation.append(ea[1])
                each.formation = formation
                if str(previous_form) != str(formation):
                    video.formationChunk[(first, each.id)] = previous_form
                    first = each.id
            

    getDanceFormationKeys(video, interval)

    return video.formationChunk

def getDanceFormationKeys(video, interval):
    if len(video.danceFormationKeys) == 0:
        range_list = []
        formation = video.formationChunk
        for start, end in formation.keys():
            if start + interval < end:
                range_list.append((start, end))

        video.danceFormationKeys = range_list

    return video.danceFormationKeys