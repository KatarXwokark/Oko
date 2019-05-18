import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

liczba_probek = 5000
liczba_testow = 1000

oko = cv2.imread("obrazki/images/13_h.jpg", cv2.IMREAD_GRAYSCALE)
maska = cv2.imread("obrazki/manual1/13_h.tif", cv2.IMREAD_GRAYSCALE)

oczy = []
maski = []

TP = 0
FP = 0
TN = 0
FN = 0

def metryka(a, b): #obliczenie metryki euklidesowej na miarach statystcznych
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 +
                   (a[3] - b[3]) ** 2 + (a[4] - b[4]) ** 2 + (a[5] - b[5]) ** 2 +
                   (a[6] - b[6]) ** 2 + (a[7] - b[7]) ** 2 + (a[8] - b[8]) ** 2)

def klas(j): #klasyfikacja zbioru testowego
    x_p = random.randrange(2, len(oko) - 3)
    y_p = random.randrange(2, len(oko[0]) - 3)
    oczy_test = np.zeros((5, 5))
    maski_test = np.zeros((5, 5))
    for j in range(-2, 3): #generowanie testownego fragmentu obrazka
        for k in range(-2, 3):
            oczy_test[2 + j][2 + k] = oko[x_p + j][y_p + k]
            maski_test[2 + j][2 + k] = maska[x_p + j][y_p + k]
    #obliczenie miar statystycznych
    miara_test = []
    miara_test.append(np.mean(oczy_test))
    miara_test.append(np.var(oczy_test))
    _, tmp = cv2.threshold(oczy_test, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(tmp)
    huMoments = cv2.HuMoments(moments)
    for i in range(7):
        miara_test.append(huMoments[j][0])
    #znajdowanie trzech nalbliższych sąsiadów (metoda kNN dla k = 3)
    _3NN = ()
    metryka0 = metryka(miary[0], miara_test)
    metryka1 = metryka(miary[1], miara_test)
    metryka2 = metryka(miary[2], miara_test)
    if metryka0 < metryka1 and metryka1 < metryka2:
        _3NN = [(metryka0, miary[0][9]), (metryka1, miary[1][9]), (metryka2, miary[2][9])]
    elif metryka0 < metryka2 and metryka2 < metryka1:
        _3NN = [(metryka0, miary[0][9]), (metryka2, miary[2][9]), (metryka1, miary[1][9])]
    elif metryka1 < metryka0 and metryka0 < metryka2:
        _3NN = [(metryka1, miary[1][9]), (metryka0, miary[0][9]), (metryka2, miary[2][9])]
    elif metryka1 < metryka2 and metryka2 < metryka0:
        _3NN = [(metryka1, miary[1][9]), (metryka2, miary[2][9]), (metryka0, miary[0][9])]
    elif metryka2 < metryka0 and metryka0 < metryka1:
        _3NN = [(metryka2, miary[2][9]), (metryka0, miary[0][9]), (metryka1, miary[1][9])]
    elif metryka2 < metryka1 and metryka1 < metryka0:
        _3NN = [(metryka2, miary[2][9]), (metryka1, miary[1][9]), (metryka0, miary[0][9])]
    for i in range(3, liczba_probek):
        met = metryka(miary[i], miara_test)
        if met < _3NN[0][0]:
            _3NN[2] = _3NN[1]
            _3NN[1] = _3NN[0]
            _3NN[0] = (met, miary[i][9])
        elif met < _3NN[1][0]:
            _3NN[2] = _3NN[1]
            _3NN[1] = (met, miary[i][9])
        elif met < _3NN[2][0]:
            _3NN[2] = (met, miary[i][9])
    if (_3NN[0][1] + _3NN[1][1] + _3NN[2][1] > 2):
        miara_test.append(1)
    else:
        miara_test.append(0)
    return (oczy_test, maski_test, miara_test[9])

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

miary = []  #obliczanie miar statystycznych
for i in range(liczba_probek):
    miary.append([])
    miary[i].append(np.mean(oczy[i]))
    miary[i].append(np.var(oczy[i]))
    _, tmp = cv2.threshold(oczy[i], 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(tmp)
    huMoments = cv2.HuMoments(moments)
    for j in range(7):
        miary[i].append(huMoments[j][0])
    if(maski[i][2][2] > 128):
        miary[i].append(1)
    else:
        miary[i].append(0)

for i in range(liczba_testow):  #testowanie
    odp = klas(i)
    if odp[2] == 1:  # positive
        if odp[1][2][2] > 128:
            TP += 1
        else:
            FP += 1
    else:  # negative
        if odp[1][2][2] > 128:
            FN += 1
        else:
            TN += 1
    # fig, (p1, p2) = plt.subplots(1, 2, figsize=(15, 5))
    # p1.clear
    # p1.imshow(odp[0], cmap="gray")
    # p2.clear
    # p2.imshow(odp[1], cmap="gray")
    # plt.show()

#koncowe wyniki
print("sensitivity")
print(str(TP / (TP + FP) * 100) + " %")
print("specificity")
print(str(TN / (TN + FN) * 100) + " %")
print("accuracy")
print(str((TP + TN) / (TP + FP + TN + FN) * 100) + " %")