from sklearn.svm import SVC
import joblib
import glob
import os
from config import *
import numpy as np

def train_svm():
    pos_feat_path = r'/home/quocanhlee/Desktop/human-detector-master/data/features/pos'
    neg_feat_path = r'/home/quocanhlee/Desktop/human-detector-master/data/features/neg'

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print (np.array(fds).shape,len(labels))
    if clf_type == "LIN_SVM":
        clf = SVC(kernel= 'rbf')
        print ("Training a Linear SVM Classifier")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        md_path = os.path.join(model_path, 'svm.model')    
        joblib.dump(clf, md_path)
        print ("Classifier saved to {}".format(model_path))
        

train_svm()
