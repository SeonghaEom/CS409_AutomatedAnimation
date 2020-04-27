import numpy as np
import cv2
import math


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 50, (512, 512))

r = 30
x = []
step = []
upper = []
y = []
for n in range(45):
    x.append((256-r)+n+1)
    y.append(0)
    step.append(1)
    upper.append(-1)

t=0
while(t<1000):
    img = np.zeros((512,512,3), np.uint8)
    for i in range(45):
        if (r**2 > (x[i]-256)**2):
            y[i] = upper[i]*math.sqrt((r**2-((x[i]-256)**2)))+256
        else:
            y[i] = upper[i]*math.sqrt((((x[i]-256)**2)-r**2))+256
        start = (int(x[i]), int(y[i]))
        end = start

        frame = cv2.line(img, start, end, (255,255,255), 9)

        if (x[i] == 256-r) or (x[i] == 256+r):
            step[i] = step[i]*(-1)
            upper[i] = upper[i]*(-1)
        x[i] = x[i] + step[i]
    
    out.write(frame)

    t += 1
    # cv2.imshow('frame', frame)
    # cv2.waitKey()

out.release()
cv2.destroyAllWindows()