# I'm sorry if you need this read(

import numpy
import csv
import pandas as pd
import datetime
import matplotlib.pyplot as plt


from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.convolutional import Conv1D


posprice = "<OPEN>"
indata = 5
outdata = 3
sizeepochs = 100
batch = 512
prefixsaveweight = "convLSTM_"
istest = True
isloadweight = False
adddelta = 1

name = '/home/mrdarts/Mega/neuronslearn/data/EURUSD_010101_170828.csv'
nameweights = '/home/mrdarts/Mega/neuronslearn/programm/convLSTM2017823.h'

def add_in_end(begind, endd):
    alld = numpy.empty(len(begind)+len(endd)-1)
    alld[:] = numpy.NaN
    alld[-len(endd):] = endd
    return alld

def draw_plot(predd = None, needd = None, aldd = None):

    if predd :
        if aldd:
            predd = add_in_end(aldd, predd)
        plt.plot(predd, 'r')
    if needd:
        if aldd:
            needd = add_in_end(aldd, needd)
        plt.plot(needd, 'g')
    if aldd: plt.plot(aldd, 'b')
    plt.show()

def convert_for_nuerouns(olddata):

    newdata = []
    beforedata = olddata[0]
    for i in range(1, len(olddata)):
        newdata.append(olddata[i] - beforedata)
        beforedata = olddata[i]

    return newdata, olddata[0]

def convert_from_nuerouns(olddata, data):

    newdata = []
    newdata.append(data)
    for i in range(0, len(olddata)):
        newdata.append(olddata[i]+data)
        data = olddata[i] + data

    return newdata

def create_data(namefile):

    onedata = []
    xybegin = []

    with open(namefile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            onedata.append(row[posprice])

    df = pd.DataFrame(onedata)
    onedata = df.rolling(window=12, min_periods=0).mean().as_matrix()
    onedata = numpy.reshape(onedata, len(onedata))

    size = int(indata+outdata)
    print('size', size)
    print(onedata[:size])
    plt.plot(onedata[:size])
    plt.show()

    datax = []
    datay = []

    for i in range(0, len(onedata) - indata - outdata + 1):
        changx ,firtsx = convert_for_nuerouns(onedata[i:i + indata])
        changy, firtsy = convert_for_nuerouns(onedata[i + indata-1:i + indata + outdata])
        datax.append([[changx]])
        datay.append(changy)
        xybegin.append([firtsx, firtsy])

    datax = numpy.array(datax)
    datay = numpy.array(datay)

    #print(datax.shape, datay.shape)
    if ((len(datay) != len(datax) and (len(datay) != len(xybegin)))):
        print("wrong size datax, datay, xybegin")

    return datax, datay, xybegin


def create_model():

    model = Sequential()
    model.add(LSTM())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(outdata))
    #model.compile(loss='mean_squared_error', optimizer='adagrad')
    model.compile(loss='mse', optimizer='rmsprop')

    return model


def begin_train(namefile):

    trainx, trainy, fprice = create_data(namefile)
    testx, testy = trainx, trainy
    if(istest):
        sizetrin = int(len(trainx)*0.8)
        testx, testy = testx[sizetrin:], testy[sizetrin:]
        trainx, trainy = trainx[:sizetrin], trainy[:sizetrin]

    print('create model')
    model = create_model()

    if(isloadweight == True):
        model.load_weights(nameweights)

    print('begin fit')
    model.fit(trainx, trainy, epochs=sizeepochs, shuffle=True)
    now = datetime.datetime.now()
    nameweight = prefixsaveweight + str(now.year) + str(now.month) + str(now.day) + '.h'
    model.save_weights(nameweight)

    print('begin evaluate')
    testpredic = model.evaluate(testx, testy)

    print("Error test ", testpredic)
    pos = -1
    result = model.predict([testx])[pos]
    result = convert_from_nuerouns()
    draw_plot(predd=result, aldd=testx[pos], needd=testy[pos])


def neuron_test(namefile):
    print("create datas")
    pass



#create_model()
#datax, datay = create_data(name)
begin_train(name)
#neuron_test(name)


