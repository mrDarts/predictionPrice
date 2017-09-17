import numpy
import csv
import pandas as pd
import datetime
import matplotlib.pyplot as plt


from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.layers.convolutional import Conv1D


posprice = "<OPEN>"
indata = 512
outdata = 24
sizeepochs = 1
batch = 512
prefixsaveweight = "LSTM_"
istest = True
isloadweight = False
isdrawresulttrain = False
adddelta = 1

name = '/home/mrdarts/Mega/neuronslearn/data/h2014_2017.csv'
nameweights = '/home/mrdarts/Mega/neuronslearn/programm/predic_LSTM/convLSTM_201796.h'

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

    print("last data", onedata[-1])

    df = pd.DataFrame(onedata)
    onedata = df.rolling(window=12, min_periods=0).mean().as_matrix()
    onedata = numpy.reshape(onedata, len(onedata))

    # size = int(indata+outdata)
    # print('size', size)
    # print(onedata[:size])
    # plt.plot(onedata[:size])
    # plt.show()

    datax = []
    datay = []

    for i in range(0, len(onedata) - indata - outdata + 1):
        changx ,firtsx = convert_for_nuerouns(onedata[i:i + indata])
        changy, firtsy = convert_for_nuerouns(onedata[i + indata-1:i + indata + outdata])
        datax.append([changx])
        # datax.append(changx)
        datay.append(changy)
        xybegin.append([firtsx, firtsy])

    datax = numpy.array(datax)
    datay = numpy.array(datay)

    print(datax.shape, datay.shape)
    if ((len(datay) != len(datax) and (len(datay) != len(xybegin)))):
        print("wrong size datax, datay, xybegin")

    return datax, datay, xybegin


def create_model():

    model = Sequential()
    model.add(LSTM(512,dropout=0.4, recurrent_dropout=0.4, input_shape=(1, indata-1), activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(outdata, activation="linear"))

    model.compile(loss='mse', optimizer='rmsprop')

    return model


def begin_train(namefile):

    trainx, trainy, xybegin = create_data(namefile)
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
    print(trainx.shape, trainy.shape)
    model.fit(trainx, trainy, epochs=sizeepochs, shuffle=True)
    now = datetime.datetime.now()
    nameweight = prefixsaveweight + str(now.year) + str(now.month) + str(now.day) + '.h'
    model.save_weights(nameweight)

    print('begin evaluate')
    testpredic = model.evaluate(testx, testy)
    print("Error test ", testpredic)
    with open("error.txt", 'a') as errfile:
        errfile.write('{0}.{1}.{2} error {3}\n'.format(now.year, now.month, now.day, testpredic))


    if isdrawresulttrain:
        pos = 0
        drawdatain = testx[pos][0]
        forpredict = testx[pos][0]
        drawdataout = testy[pos]
        # print('type drawdatain ', type(drawdatain), drawdatain.shape)
        forpredict = numpy.reshape(forpredict, (1, 1, indata-1))
        result = model.predict(forpredict)[0]
        drawdatain = convert_from_nuerouns(drawdatain, xybegin[pos][0])
        drawdataout = convert_from_nuerouns(drawdataout, xybegin[pos][1])
        result = result.tolist()
        result = convert_from_nuerouns(result, drawdatain[-1])
        draw_plot(predd=result, aldd=drawdatain, needd=drawdataout)

def predict_test(namefile):
    testx, testy, xybegin = create_data(namefile)
    print('create model')
    model = create_model()
    model.load_weights(nameweights)
    pos = -1
    drawdatain = testx[pos][0]
    forpredict = testx[pos][0]
    drawdataout = testy[pos]
    # print('type drawdatain ', type(drawdatain), drawdatain.shape)
    forpredict = numpy.reshape(forpredict, (1, 1, indata - 1))
    result = model.predict(forpredict)[0]
    drawdatain = convert_from_nuerouns(drawdatain, xybegin[pos][0])
    drawdataout = convert_from_nuerouns(drawdataout, xybegin[pos][1])
    result = result.tolist()
    result = convert_from_nuerouns(result, drawdatain[-1])
    draw_plot(predd=result, aldd=drawdatain, needd=drawdataout)

def predict_last(namefile):
    print("create datas")
    onedata = []

    with open(namefile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            onedata.append(row[posprice])

    df = pd.DataFrame(onedata)
    onedata = df.rolling(window=12, min_periods=0).mean().as_matrix()
    onedata = numpy.reshape(onedata, len(onedata))

    forpredict,firstin = convert_for_nuerouns(onedata[-indata:])
    forpredict = numpy.array(forpredict)
    forpredict = numpy.reshape(forpredict, (1, 1, indata-1))

    print('create model')
    model = create_model()
    model.load_weights(nameweights)
    result = model.predict(forpredict)[0]
    result = result.tolist()
    result = convert_from_nuerouns(result, onedata[-1])
    onedata = onedata.tolist()
    draw_plot(predd=result, aldd=onedata)



# create_model()
# datax, datay, xybegin = create_data(name)
begin_train(name)
# predict_test(name)
# predict_last(name)
