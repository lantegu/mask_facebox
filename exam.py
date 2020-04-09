
from skimage import io
import matplotlib.pyplot as plt
annotation = 'F:\FaceBoxes.PyTorch-master\data\MAFA\\train_annotations\\train_00000001.txt'
img_path = 'F:\FaceBoxes.PyTorch-master\data\MAFA\\train_images\\train_00000001.jpg'

with open(annotation,'rt') as f:
    index = list(map(int,f.readline().split()))
    xmin,ymin,xmax,ymax,gender = index[0],index[1],index[2],index[3],index[4]

img =io.imread(img_path)
plt.imshow(img)
plt.pause(1000)
