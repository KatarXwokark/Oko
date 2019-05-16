from sklearn.neural_network import MLPClassifier
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import KFold

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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(100, 150, 200), random_state=1)

for i in range(liczba_probek): #generowanie próbek
    x = random.randrange(2, len(oko) - 3)
    y = random.randrange(2, len(oko[0]) - 3)
    tmp = np.zeros((5, 5))
    tmp2 = np.zeros((5, 5))
    for j in range(-2, 3):
        for k in range(-2, 3):
            tmp[2 + j][2 + k] = oko[x + j][y + k]
            tmp2[2 + j][2 + k] = maska[x + j][y + k]
    oczy.append(tmp)
    maski.append(tmp2)

X = []  #obliczanie miar statystycznych
y = []
for i in range(liczba_probek):
    X.append([])
    X[i].append(np.mean(oczy[i]))
    X[i].append(np.var(oczy[i]))
    _, tmp = cv2.threshold(oczy[i], 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(tmp)
    huMoments = cv2.HuMoments(moments)
    for j in range(7):
        X[i].append(huMoments[j][0])
    if(maski[i][2][2] > 128):
        y.append(1)
    else:
        y.append(0)

kfold = KFold(10, True, 1)
for train_index, test_index in kfold.split(X):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in train_index:
        X_train.append(X[i])
        y_train.append(y[i])
    for i in test_index:
        X_test.append(X[i])
        y_test.append(y[i])
    clf.fit(X_train, y_train)
    for i in range(len(test_index)):  #testowanie
        odp = clf.predict([X_test[i]])
        if odp == 1:  # positive
            if y_test[i] == 1:
                TP += 1
            else:
                FP += 1
        else:  # negative
            if y_test[i] == 1:
                FN += 1
            else:
                TN += 1
        #fig, (p1, p2) = plt.subplots(1, 2, figsize=(15, 5))
        #p1.clear
        #p1.imshow(odp[0], cmap="gray")
        #p2.clear
        #p2.imshow(odp[1], cmap="gray")
        #plt.show()

    #koncowe wyniki
    print("sensitivity")
    print(str(TP / (TP + FP) * 100) + " %")
    print("specificity")
    print(str(TN / (TN + FN) * 100) + " %")
    print("accuracy")
    print(str((TP + TN) / (TP + FP + TN + FN) * 100) + " %")
    TP = 0
    FP = 0
    TN = 0
    FN = 0
