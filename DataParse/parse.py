import json
import numpy as np
import copy
import pprint
np.set_printoptions(threshold=np.inf)

class Video():
    def __init__(self, name):
        with open(name) as json_file:
            json_string = json.load(json_file)
        json_data = json.loads(json_string)
        hum_cnt = 0

        for i in range(len(json_data)):
            boxes = json_data[str(i)]
            humans = []
            for box_ind in boxes:
                each = Human(boxes[box_ind]['id'], i, boxes[box_ind]['pose_pos'], boxes[box_ind]['box_pos'])
                # print(boxes[box_ind]['pose_pos'])
                humans.append(each)
            Frame(i, humans)
            # print(len(humans))
            if (hum_cnt < len(boxes)):
                hum_cnt = len(boxes)
        # return Video(Frame, Behavior, hum_cnt)
        self.frame = Frame
        self.behavior = Behavior
        self.hum_cnt = hum_cnt
        self.centerChunk = {}
        self.formationChunk = {}
        self.danceFormationKeys = []
        self.footVector = {}

    def Track(self, bh, start, end):
        trk = []
        for frame_id in range(start, end):
            for human in Frame.get(frame_id).humans:
                li = [ human.pose_pos[anker] for anker in bh.akpnt]
                trk.append(json.dumps({'frame_id': human.frame_id, 'human_id': human.id, 'pose_pos': li}, sort_keys=True))
        return trk

    
    def getFeetMov(self):
        feet = {}
        i = 0
        for i in range(self.hum_cnt):
            feet[i+1] = {"left": [], "right": []}
        for each in self.frame.instances:
            for hum in each.humans:
                # print (hum.id)
                # print (hum.pose_pos[15])
                # print (hum.pose_pos[16])
                feet[hum.id]["left"].append(hum.pose_pos[15])
                feet[hum.id]["right"].append(hum.pose_pos[16])
        print (np.gradient(feet[1]["left"], axis = 1))
        
    
class Behavior():
    instances = []
    def __init__(self, id, akpnt):
        self.id = id
        self.akpnt = akpnt #[ankerpoint_idx]
        Behavior.instances.append(self)

    @classmethod
    def get(cls, id):
        for inst in cls.instances:
            if (inst.id == id):
                return inst

class Human():
    def __init__(self, id, frame_id, pose_pos, box_pos):
        self.id = id
        self.frame_id = frame_id
        self.pose_pos = pose_pos #[[x,y],..,]
        self.box_pos = box_pos
        self.is_center = False


class Frame():
    instances = []
    def __init__(self, id, humans):
        self.id = id
        self.humans = humans #[Human]
        self.center = -1
        self.formation = []
        #self.time
        Frame.instances.append(self)

    @classmethod
    def get(cls, id):
        for inst in cls.instances:
            if (inst.id == id):
                return inst

    @classmethod
    def getNext(cls, id):
        for inst in cls.instances:
            if (inst.id == id+1):
                return inst

    @classmethod
    def getLength(cls):
        return len(cls.instances)