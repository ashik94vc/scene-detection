import sys
import os
import pickle
from datetime import date

def enableConsoleOutput():
    sys.stdout = sys.__stdout__

def disableConsoleOutput():
    sys.stdout = open(os.devnull, 'w')

def __getFilePath(iternumber,date):
    filename = 'models/'+date_today[1]+'_'+date_today[0]+'_'+date_today[2]+'_'+str(iternumber)+'.model'
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.mkdir(directory)
    if os.path.exists(filename) and os.path.isfile(filename):
        return __getFilePath(iternumber+1,date)
    return filename

def saveModel(parameters):
    iternumber = 1
    date_today = str(date.today()).split('-')
    filename = __getFilePath(iternumber,date_today)
    fptr = open(filename,'wb')
    pickle.dump(parameters,fptr)
    fptr.close()

def loadModel(filepath):
    fptr = open(filepath,'rb')
    param = pickle.load(parameters,fptr)
    fptr.close()
    return param
