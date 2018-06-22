import io
import os
import codecs

def readFile(path):
#    inBody = False
#    lines = []
#    f = io.open(path, 'r', encoding = 'latin1')
##    f = io.open(path, 'r', encoding='base64')
#    for line in f:
#        if inBody:
#            lines.append(line)
#        elif line == '\n':
#            inBody = True
#    f.close()
    f = codecs.open(path, 'rb')
    lines = f.read()
    print(lines)
#    lines = pickle.load('path')
    return lines
    
data = readFile('tesseract_eng.traineddata')

    
    


























#import pickle
##import _pickle as cPickle
#
#def unpickle(file):
#    print(file)
#    fo = open(file, 'rb')
##    dict = cPickle.load(fo)
#    data = pickle.load(fo, encoding='bytes')
#    fo.close()
#    return dict
#
##d = unpickle("tesseract_eng_traineddata")
#    
#eng_model = unpickle('tesseract_eng_traineddata')
#
#
#
#


