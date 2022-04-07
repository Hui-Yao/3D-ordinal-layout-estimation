import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2


color_dict = {0: [0, 0, 255], 1: [127, 255, 0], 2: [0, 255, 255], 3:[255, 255, 0],
                  4: [227, 23 ,13], 5: [218, 112, 214], 6: [255, 153, 87]}

color_key = color_dict.keys()
color_value = color_dict.values()

save_dir = f'/home/hui/pictures/paper_picture/semantic_smaple'          # seed=12~

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for key, value in zip(color_key, color_value):
    image_save_path = os.path.join(save_dir, 'semantic_'+ str(key) + '.png')

    pure_color = np.zeros((192, 256, 3))

    pure_color[:, :, 0] = value[0]
    pure_color[:, :, 1] = value[1]
    pure_color[:, :, 2] = value[2]

    pure_color = pure_color.astype(np.uint8)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(pure_color)
    plt.savefig(image_save_path)
    plt.show()





