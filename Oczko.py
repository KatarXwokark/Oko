import tkinter as tk

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

window = tk.Tk()
window.protocol("WM_DELETE_WINDOW", window.quit)
fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(15, 5))

img = cv2.imread("obrazki/images/02_dr.JPG", cv2.IMREAD_COLOR)
height, width = len(img), len(img[0])
p1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

pi = 0  # 0 #parametry do filtru Canny
ki = 10  # 10
pj = 15  # 15
kj = 40  # 25
result = np.zeros((height, width))
for i in range(pi, ki):
    for j in range(pj, kj):
        if i < j:
            print(i, j)
            tmp = cv2.GaussianBlur(img, (5, 5), 0)
            tmp = cv2.Canny(tmp, i, j)
            tmp = cv2.dilate(tmp, (10, 10), iterations=5)
            # tmp = cv2.erode(tmp, (10, 10), iterations=5)
            result += tmp
result /= (ki - pi) * (kj - pj)
result = cv2.dilate(result, (10, 10), iterations=5)
result = cv2.erode(result, (10, 10), iterations=5)
result = np.ones((height, width)) * 255 - result
p2.imshow(result, cmap=plt.cm.Greys_r)

manual_mask_read = np.ones((height, width)) * 255 - cv2.imread("obrazki/manual1/02_dr.tif", 0)
p3.imshow(manual_mask_read, cmap=plt.cm.Greys_r)

result_mask = np.zeros((height, width))
manual_mask = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        if result[i][j] < 128:
            result_mask[i][j] = 1
        if manual_mask_read[i][j] < 128:
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
