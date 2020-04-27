
import json

class Video():
    def __init__(self, frames, behaviors, hum_cnt):
        self.frames = frames #[Frame]
        self.behaviors = behaviors #[Behavior]
        self.hum_cnt = hum_cnt #number of humans

class Behavior():
    def __init__(self, id, akpnt):
        self.id = id
        self.akpnt = akpnt #[ankerpoint_idx]

class Human():
    def __init__(self, id, frame_id, pose_pos, box_pos):
        self.id = id
        self.frame_id = frame_id
        self.pose_pos = pose_pos #[[x,y],..,]
        self.box_pos = box_pos

class Frame():
    def __init__(self, id, humans):
        self.id = id
        self.humans = humans #[Human]



print("generating json deserialization... ")
with open('../(G)I-DLE-02_matching.json') as json_file:
    json_string = json.load(json_file)

json_data = json.loads(json_string)

# print("frame index ranges  from 0 to ", len(json_data))
# frame_ind = input("enter frame index: ")
# print("frame index ranges  from 1 to ", len(json_data[frame_ind]))
# box_ind =  input("enter frame index: ")
# print(json_data[frame_ind][box_ind])
# print (json_data['0']['1'])

# print ("frame length: ", len(json_data))
# print (json_data['99']['1'])
# for i in range(len(json_data)):
#     each  = json_data[str(i)]
#     for j in json_data[str(i)]:
#         print (each[j])

frames = []
behaviors = []
hum_cnt = 0

for i in range(len(json_data)):
    boxes = json_data[str(i)]
    if (len(boxes) == 0):
        frames.append('empty_frame')
        continue
    humans = []
    for box_ind in boxes:
        each = Human(boxes[box_ind]['id'], i, boxes[box_ind]['pose_pos'], boxes[box_ind]['box_pos'])
        # print(boxes[box_ind]['pose_pos'])
        humans.append(each)
        # print(each)
    frame = Frame(i, humans)
    # print(frame.__dict__)
    frames.append(Frame(i, humans))
    # print(len(humans))
    if (hum_cnt < len(boxes)):
        hum_cnt = len(boxes)


behaviors.append(Behavior(0, [9,10]))

in_video = Video(frames, behaviors, hum_cnt)
# print("number of humans in this video: ", hum_cnt)
# print(frames[0].__dict__)
# print(frames[0].id)
# print(frames[9].humans[0].frame_id)
# print(frames[100].humans[0].pose_pos)
# print(frames[0].humans[0].box_pos)
