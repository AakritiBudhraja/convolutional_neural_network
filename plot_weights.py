import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import matplotlib.gridspec as gridspec


def process_filter_for_display(x):
	x-= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
	x += 0.5
	x = np.clip(x, 0, 1)
	x *= 255
	if x.shape[2] != 3:
		x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def save_filters(filters, img_width, img_height):
    margin = 5
    n = int(len(filters)**0.5)
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index < len(filters):
                img = filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
    cv2.imwrite('stitched_filters_%dx%d.jpg' % (n, n), stitched_filters)

filename = 'filter_conv1.txt'
x = np.loadtxt(filename)
x = np.reshape(x, [-1, 5,5,3])
plt.figure(1, figsize=(10,10))
#gs1 = gridspec.GridSpec(4, 8)
#gs1.update(wspace=0.025, hspace=0.0005)
plt.title('Plot of all filters of conv layer 1')
all_filters = np.empty([0,5,5,3])
#print(all_filters.shape)
num_filters = len(x)
n_columns = 8
n_rows = math.ceil(num_filters / n_columns) + 1
for i in range(num_filters):
	filter_image = process_filter_for_display(x[i])
	#print(filter_image.shape)
	plt.subplot(n_rows, n_columns, i+1)
	#ax1 = plt.subplot(gs1[i])
	plt.axis('off')
	#ax1.set_xticklabels([])
	#ax1.set_yticklabels([])
	#ax1.set_aspect('equal')
	plt.title('Filter: {0} '.format(str(i)))
	plt.imshow(filter_image)
	all_filters = np.append(all_filters,[filter_image], axis=0)
#plt.tight_layout()
plt.show()
#print(all_filters.shape)
save_filters(filter_image, 5, 5)