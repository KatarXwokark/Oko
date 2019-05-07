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
    img = cv2.imread("obrazki/images/01_dr.jpg", 0)
    p1.clear()
    p1.imshow(img, cmap=plt.cm.Greys_r)
    newimg = np.zeros((len(img),len(img[0])))
    pi = 0      #0 #parametry do filtru Canny
    ki = 5     #10
    pj = 15     #15
    kj = 25     #25
    for i in range(pi, ki):
        for j in range(pj, kj):
            if(i < j):
                print(i, j)
                tmp = cv2.GaussianBlur(img, (5, 5), 0)
                tmp = cv2.Canny(tmp, i, j)
                #tmp = cv2.dilate(tmp, (10, 10), iterations=5)
                #tmp = cv2.erode(tmp, (10, 10), iterations=5)
                newimg += tmp
    newimg /= (ki - pi)*(kj - pj)
    newimg = cv2.dilate(newimg, (10, 10), iterations=5)
    #newimg = cv2.erode(newimg, (10, 10), iterations=5)
    newimg = np.ones((len(img),len(img[0]))) * 255 - newimg
    p2.clear()
    p2.imshow(newimg, cmap=plt.cm.Greys_r)
    p3.clear()
    p3.imshow(np.ones((len(img),len(img[0]))) * 255 - cv2.imread("obrazki/manual1/01_dr.tif", 0), cmap=plt.cm.Greys_r)
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=window)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    tk.mainloop()

main()