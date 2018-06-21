import pickle
import _pickle as cPickle

def unpickle(file):
    print(file)
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

d = unpickle("tesseract_eng_traineddata")
    





