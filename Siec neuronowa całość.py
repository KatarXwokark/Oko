from sklearn.neural_network import MLPClassifier
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from sklearn.model_selection import KFold
from sklearn.externals import joblib

liczba_probek = 10000

oko = cv2.imread("obrazki/images/01_h.JPG", cv2.IMREAD_GRAYSCALE)
maska = cv2.imread("obrazki/manual1/01_h.tif", cv2.IMREAD_GRAYSCALE)
oko2 = cv2.imread("obrazki/images/02_h.JPG", cv2.IMREAD_GRAYSCALE)
maska2 = cv2.imread("obrazki/manual1/02_h.tif", cv2.IMREAD_GRAYSCALE)

oczy = []
maski = []

TP = 0
FP = 0
TN = 0
FN = 0

maska_otrz = np.zeros((len(oko2), len(oko2[0])))
clf = joblib.load('network.pkl')

start = time.time()

for i in range(2, len(oko2) - 2):
    print(i, (time.time() - start) / 60)
    for j in range(2, len(oko2[0]) - 2):
        tmpm = np.zeros((5, 5))
        for ii in range(-2, 3):
            for jj in range(-2, 3):
                tmpm[2 + ii][2 + jj] = oko2[i + ii][j + jj]
        X_tmp = []
        X_tmp.append(np.mean(tmpm))
        X_tmp.append(np.var(tmpm))
        _, tmp = cv2.threshold(tmpm, 128, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(tmp)
        huMoments = cv2.HuMoments(moments)
        for j in range(7):
            X_tmp.append(huMoments[j][0])
        odp = clf.predict([X_tmp])
        if odp == 1:  # positive
            if maska2[i][j] > 128:
                TP += 1
            else:
                FP += 1
        else:  # negative
            if maska2[i][j] > 128:
                FN += 1
            else:
                TN += 1
        maska_otrz[i][j] = odp

print("sensitivity")
print(str(TP / (TP + FP) * 100) + " %")
print("specificity")
print(str(TN / (TN + FN) * 100) + " %")
print("accuracy")
print(str((TP + TN) / (TP + FP + TN + FN) * 100) + " %")
fig, (p1, p2, p3) = plt.subplots(1, 3, figsize=(15, 5))
p1.clear
p1.imshow(oko2, cmap="gray")
p2.clear
p2.imshow(maska2, cmap="gray")
p3.clear
p3.imshow(maska_otrz, cmap="gray")
plt.show()
