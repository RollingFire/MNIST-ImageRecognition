'''
By Austin Dorsey
Started: 8/4/18
Last modified 8/15/18
Using the MNIST dataset and Scikit-learn, created an image recognition machine learning model that got a 97% f1 score.
'''

from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import math


def showImg(img):
    '''Displays square images'''
    shape = int(math.sqrt(len(img)))
    img = img.reshape(shape, shape)
    plt.imshow(img, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.show()


def showShapedImg(img):
    '''Displays preshaped images'''
    plt.imshow(img, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.show()

def getImgShifts(img, target):
    '''Takes an image and the target for the the image. Returns a list 
    of 4 of the same images that are shifted up, down, left, and right 
    along with the matching targets.'''
    shape = int(math.sqrt(len(img)))
    img = img.reshape(shape, shape)
    shifts = []
    shifts.append(np.roll(img,  1, axis=0).flatten())
    shifts.append(np.roll(img, -1, axis=0).flatten())
    shifts.append(np.roll(img,  1, axis=1).flatten())
    shifts.append(np.roll(img, -1, axis=1).flatten())
    targetList = [target]
    return shifts, targetList * 4


def getShiftedImgs(xTrain, yTrain):
    '''Takes many images and calls getImgShifts for each one and 
    returns the reslting image and target arrays for the whoe data set.'''
    xShiftsList = []
    yShiftsList = []
    for img, target in zip(xTrain, yTrain):
        imgs, targets = getImgShifts(img, target)
        xShiftsList += imgs
        yShiftsList += targets
    return np.array(xShiftsList, dtype='uint8'), np.array(yShiftsList)


def getSideRatio(img):
    '''Splits the image into left and right sides. Takes the sum of 
    each side. Returns the right/left ratio.'''
    shape = int(math.sqrt(len(img)))
    imgT = img.reshape(shape, shape).T
    pureImg = [x for x in imgT if np.sum(x) != 0]
    split = int(len(pureImg) / 2)
    if split * 2 != len(pureImg):
        pureImg = np.delete(pureImg, split, axis=0)
    right = pureImg[split:]
    left = pureImg[:split]
    return np.sum(right)/max(np.sum(left), 1)


def getSideRatios(x):
    '''Takes an array of images and calls getSideRatio for each. 
    Returns an array of the results.'''
    xRatiosList = []
    for img in x:
        xRatiosList.append(getSideRatio(img))
    return np.array(xRatiosList)


def batchSplit(x, y, splits=20):
    '''Splits x and y into the given number of groups. 
    If can not divide it evenly amongst all groups, then the 
    remainder will go in the last item in the list.'''
    xList, yList = [], []
    splitSize = int(len(x) / splits)
    for i in range(int(len(x) / splitSize)):
        if len(x) - splitSize * (i + 1)  < splitSize:
            xList.append(x[splitSize * i :])
            yList.append(y[splitSize * i :])
        else:
            xList.append(x[splitSize * i : min(splitSize * (i + 1), len(x))])
            yList.append(y[splitSize * i : min(splitSize * (i + 1), len(x))])
    return xList, yList


def splitTest(x, y, testSize=0.2, shuffle=True):
    '''Divides off the requested amount of data into a test set.
    Returns xTrain, yTrain, xTest, yTest'''
    shuffledInx = np.random.permutation(len(x))
    split = int(len(shuffledInx) * testSize)
    testIdx = shuffledInx[:split]
    xTrain, yTrain, xTest, yTest = [], [], [], []
    for i, _ in enumerate(x):
        if i in testIdx:
            xTest.append(x[i])
            yTest.append(y[i])
        else:
            xTrain.append(x[i])
            yTrain.append(y[i])
    return xTrain, yTrain, xTest, yTest
        
        
def plotLearningCurve(model, x, y, batches=2000):
    '''Plots the learning curve for the given model to see the bias 
    vs variance of the model.'''
    xTrain, yTrain, xVal, yVal = splitTest(x, y)
    xList, yList = batchSplit(xTrain, yTrain, splits=batches)
    workingX, workingY = [], []
    trainErrors, valErrors = [], []
    for i in range(len(xList)):
        workingX += xList[i]
        workingY += yList[i]
        model.fit(workingX, workingY)
        yTrainPredict = model.predict(workingX)
        yValPredict = model.predict(xVal)
        trainErrors.append(mean_squared_error(yTrainPredict, workingY))
        valErrors.append(mean_squared_error(yValPredict, yVal))
    plt.plot(np.sqrt(trainErrors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(valErrors), "b-", linewidth=3, label="val")
    plt.show()


def getExposure(img, target, pairs=2):
    '''Takes an image and shifts the exposure + and - 10 for each 
    pair. returns a list of the new amages and their copied targets.'''
    xFinal = []
    for i in range(pairs):
        a = np.copy(img)
        b = np.copy(img)
        change = (i + 1) * 10
        a[a > 0]
        a[a <= 255 - change] += change
        b[b >= change] -= change
        xFinal.append(a)
        xFinal.append(b)
    return xFinal, [target] * 2 * pairs


def getExposures(xTrain, yTrain, pairs=2):
    '''Takes an array of images and calls getExposure for each. 
    Returns an array of the results.'''
    xExposuresList = []
    yExposuresList = []
    for img, target in zip(xTrain, yTrain):
        imgs, targets = getExposure(img, target, pairs)
        xExposuresList += imgs
        yExposuresList += targets
    return np.array(xExposuresList, dtype='uint8'), np.array(yExposuresList)


def main():
    '''Collects the MNIST datase proforms data augmentation techniques on 
    it and trains a Random Forest model with the data. Tests model on the 
    MNIST test portion of its data, and returns the f1 score of the test.'''
    mnist = fetch_mldata('MNIST original')
    x, y = mnist["data"], mnist["target"]
    xTrain, xTest = x[:60000], x[60000:]
    yTrain, yTest = y[:60000], y[60000:]

    xExposures, yExposures = getExposures(xTrain, yTrain, pairs=4)
    xTrainFull = np.concatenate((xTrain, xExposures), axis=0)
    yTrainFull = np.concatenate((yTrain, yExposures), axis=0)

    xShifts, yShifts = getShiftedImgs(xTrainFull, yTrainFull)
    xTrainFull = np.concatenate((xTrainFull, xShifts), axis=0)
    yTrainFull = np.concatenate((yTrainFull, yShifts), axis=0)

    rndForestClf = RandomForestClassifier(random_state=24)
    rndForestClf.fit(xTrainFull, yTrainFull)
    xTestPredict = rndForestClf.predict(xTest)
    print(f1_score(yTest, xTestPredict, average='macro'))


if __name__ == '__main__':
    main()
