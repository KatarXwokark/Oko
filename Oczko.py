import tkinter as tk
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def line_equation(x1, y1, x2, y2):
    temp = np.zeros((1, 2))
    if x1 - x2 != 0:
        temp[0][0] = (y1 - y2) / (x1 - x2)
        temp[0][1] = (y2 * x1 - x2 * y1) / (x1 - x2)
    else:
        temp[0][0] = (y1 - y2) / 0.00001
        temp[0][1] = (y2 * x1 - x2 * y1) / 0.00001
    return temp

def main():
    window = tk.Tk()
    window.protocol("WM_DELETE_WINDOW", window.quit)
    fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(10, 5))
    img = cv2.imread("obrazki/images/02_dr.jpg", 0)
    p1.clear()
    p1.imshow(img, cmap=plt.cm.Greys_r)
    newimg = np.zeros((len(img),len(img[0])))
    pi = 0      #0 #parametry do filtru Canny
    ki = 10     #10
    pj = 15     #15
    kj = 40     #25
    for i in range(pi, ki):
        for j in range(pj, kj):
            if(i < j):
                print(i, j)
                tmp = cv2.GaussianBlur(img, (5, 5), 0)
                tmp = cv2.Canny(tmp, i, j)
                tmp = cv2.dilate(tmp, (10, 10), iterations=5)
                #tmp = cv2.erode(tmp, (10, 10), iterations=5)
                newimg += tmp
    newimg /= (ki - pi)*(kj - pj)
    newimg = cv2.dilate(newimg, (10, 10), iterations=5)
    newimg = cv2.erode(newimg, (10, 10), iterations=5)
    newimg = np.ones((len(img),len(img[0]))) * 255 - newimg
    p2.clear()
    p2.imshow(newimg, cmap=plt.cm.Greys_r)
    odp = np.ones((len(img),len(img[0]))) * 255 - cv2.imread("obrazki/manual1/02_dr.tif", 0)
    p3.clear()
    p3.imshow(odp, cmap=plt.cm.Greys_r)
    bitnewimg = np.zeros((len(img),len(img[0])))
    bitodp = np.zeros((len(img), len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(newimg[i][j] < 128):
                bitnewimg[i][j] = 1
            if (odp[i][j] < 128):
                bitodp[i][j] = 1
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(bitnewimg[i][j] == bitodp[i][j] and bitodp[i][j] == 1):
                TP += 1
            elif(bitnewimg[i][j] != bitodp[i][j] and bitodp[i][j] == 0):
                FP += 1
            elif(bitnewimg[i][j] == bitodp[i][j] and bitodp[i][j] == 0):
                TN += 1
            elif(bitnewimg[i][j] != bitodp[i][j] and bitodp[i][j] == 1):
                FN += 1
    print("sensitivity")
    print(str(TP / (TP + FP) * 100) + " %")
    print("specificity")
    print(str(TN / (TN + FN) * 100) + " %")
    print("accuracy")
    print(str((TP + TN) / (TP + FP + TN + FN) * 100) + " %")
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=window)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    tk.mainloop()

main()