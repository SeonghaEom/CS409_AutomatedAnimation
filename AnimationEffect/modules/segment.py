import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from matplotlib import pyplot as plt
from gluoncv.utils.viz import get_color_pallete, cv_plot_image
import matplotlib.image as mpimg
from scipy import ndimage

import numpy as np
import cv2
import math
import sys
import time

def plot_color_mask(img, masks, color="random", alpha=0.5):
    """Visualize segmentation mask.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    masks : numpy.ndarray or mxnet.nd.NDArray
        Binary images with shape `N, H, W`.
    alpha : float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks

    """
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "pink": (255, 0, 255),
        "skyblue": (0, 255, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
    }

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(masks, mx.nd.NDArray):
        masks = masks.asnumpy()

    for mask in masks:
        if color == "random":
            color = np.random.random(3) * 255
        else: 
            color = np.array(colors[color])
        mask = np.repeat((mask > 0)[:, :, np.newaxis], repeats=3, axis=2)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)
    return img.astype('uint8')



def draw_segment_outline(model, ctx, frame, thickness=6, black=True, color="random", alpha=0.5):

    # Draw outline of segmentation of the given frame
    # thickness: thickness of outline
    # For black background, black=True
    # For original frame background, black=False

    img = test_transform(mx.nd.array(frame), ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    sx = ndimage.sobel(predict, axis=0, mode='constant')
    sy = ndimage.sobel(predict, axis=1, mode='constant')

    rsx = sx
    rsy = sy
    for j in range(thickness):
        rsx += np.roll(sx, j-int(thickness/2))
        rsy += np.roll(sy, j-int(thickness/2))

    outline = np.hypot(rsx, rsy)
    outline = np.expand_dims(outline, axis=0)

    if black:
        black = np.zeros_like(frame)
        output = plot_color_mask(mx.nd.array(black), outline, color=color, alpha=alpha)
    else:
        output = plot_color_mask(mx.nd.array(frame), outline, color=color, alpha=alpha)

    return output


def segment_mask(model, ctx, frame, inverse_mask=False, color="random", alpha=0.5):

    # mask segmentation of the given frame
    # For masked background, inverse_mask=True
    # For masked people, inverse_mask=False
    # alpha is related with opacity of mask
    # For transparent mask, alpha=0
    # For colored(non=transparent) mask, alpha=1

    img = test_transform(mx.nd.array(frame), ctx)
    output = model.predict(img)
    mask = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    if inverse_mask:
        mask = np.where(mask==0, 1, 0)
    
    mask = np.expand_dims(mask, axis=0)

    output = plot_color_mask(mx.nd.array(frame), mask, color=color, alpha=alpha)

    return output




# # using cpu
# ctx = mx.cpu(0)

# sys.path.append('/content/drive/My Drive/AutoAnimation')


# in_video_path = '../../Naver_video_02.mp4'
# out_video_path = '../../test_0614_8.mp4'

# cap = cv2.VideoCapture(in_video_path)

# width = int(cap.get(3))
# height = int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# print(width, height, fps)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(out_video_path, fourcc, fps, (1920, 1080))

# model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)

# start = time.time()
# print("Pose Detection Start!")

# i=0
# while cap.isOpened():
#     ret, frame = cap.read()

#     i += 1
#     if i<500: continue
#     elif i>530: break

#     img = draw_segment_outline(model, ctx, frame, thickness=6, black=True, color="pink")

#     if i<515:
#         img = segment_mask(model, ctx, frame, inverse_mask=True, color="black", alpha=1)
#     # else:
#     #     img = frame

    

    

#     # plt.imshow(img)
#     # plt.show()
#     print(i)
#     # break
    
#     out.write(img)

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# end = time.time()
# print("Pose Detectionn Done!")
# print("Estimation took {} sec" .format(end - start))