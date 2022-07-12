import os
from os import listdir
from os import path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import functions

path = "test_images/"
save_path = "test_images_output/"
test_images = []
files = os.listdir(path)
for file in files:
    ext = os.path.splitext(file)[1]
    test_images.append(Image.open(os.path.join(path,file)))
for i in range (0,len(test_images)):
    test_images[i] = np.array(test_images[i])

for file in files:
    i=0
    plt.figure()
    # plt.imshow(process_image(test_images[i]))
    img = functions.process_image(test_images[i])
    img = Image.fromarray(img)
    img.save(save_path + "/" + file)
    i+=1