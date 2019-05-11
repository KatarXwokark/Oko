import tkinter as tk

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
from skimage.exposure import equalize_adapthist
from skimage.filters import frangi, threshold_triangle
from skimage.morphology import remove_small_objects

window = tk.Tk()
window.protocol("WM_DELETE_WINDOW", window.quit)
fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(15, 5))

img = cv2.imread("obrazki/images/02_dr.JPG", cv2.IMREAD_COLOR)
height, width = len(img), len(img[0])
p1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

green = img[:, :, 1]
green = equalize_adapthist(green)

lowpass = ndimage.gaussian_filter(green, 20)
green = green - lowpass

result = frangi(green, scale_range=(0, 6), scale_step=1)

thresh = threshold_triangle(result)
result_mask = result >= thresh

result_mask = remove_small_objects(result_mask, 50, 10)
p2.imshow(result_mask, cmap="gray")

manual_mask_read = cv2.imread("obrazki/manual1/02_dr.tif", cv2.IMREAD_GRAYSCALE)
p3.imshow(manual_mask_read, cmap="gray")

manual_mask = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        if manual_mask_read[i][j] > 128:
            manual_mask[i][j] = 1

# PRINT CONFUSION MATRIX
TP = 0
FP = 0
TN = 0
FN = 0
for i in range(height):
    for j in range(width):
        if result_mask[i][j] == 1:  # positive
            if manual_mask[i][j] == 1:
                TP += 1
            else:
                FP += 1
        else:  # negative
            if manual_mask[i][j] == 1:
                FN += 1
            else:
                TN += 1
print("sensitivity")
print(str(TP / (TP + FP) * 100) + " %")
print("specificity")
print(str(TN / (TN + FN) * 100) + " %")
print("accuracy")
print(str((TP + TN) / (TP + FP + TN + FN) * 100) + " %")

# SHOW WINDOW
figure_canvas_agg = FigureCanvasTkAgg(fig, master=window)
figure_canvas_agg.draw()
figure_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
tk.mainloop()
