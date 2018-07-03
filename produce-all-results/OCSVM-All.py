import numpy as np
import sklearn
from sklearn import svm
import pickle
import sys

f = open('SVM-Res', 'w')
algo = ['GMM', 'SVM', 'KNN']
feats = ['GLCM', 'Gabor', 'LBP', ['GLCM','Gabor'], ['Gabor','LBP'], ['GLCM','LBP']]
params = [[0.1,0.1], [0.01,0.1], [0.01,0.01], [0.001,0.0001], [0.001,0.1], [0.001,0.001], [0.1,0.001], [0.2,0.1], [0.3,0.1], [0.2,0.01]]

for alg in (algo):
    if alg == 'SVM':
        for feat in (feats):
            if isinstance(feat, basestring):
                print ('loading single features')
                training_samples_features = np.load("train_samples_feats_matrix_" + feat + ".npy")
                valid_samples_features = np.load("valid_samples_feats_matrix_" + feat + ".npy")
                print('loaded')
                sys.stdout.flush()
               
            else: 
                print ('loading mixture features')
                training_samples_features = []
                valid_samples_features = []
                for subFeat in (feat):
                    training_samples_features.append(np.load("train_samples_feats_matrix_" + subFeat + ".npy"))
                    valid_samples_features.append(np.load("valid_samples_feats_matrix_" + subFeat + ".npy"))
                        
                training_samples_features = np.concatenate((training_samples_features), axis=1)
                valid_samples_features = np.concatenate((valid_samples_features), axis=1)
                print('loaded')
                sys.stdout.flush()
            
            for param in (params):
                   
                print ('SVM_'+ str(feat) + '_' + str(param))
                print ('training start')
                #Train a SVM  model from training data
                svm_model = svm.OneClassSVM(kernel='rbf', nu=param[0], gamma=param[1])
                svm_model.fit(training_samples_features)
                print ('nu: ', param[0], 'gamma: ', param[1])
                print ('training ended')
                sys.stdout.flush()
                #Save the model
                f1 = open('SVM_'+ str(feat) + '_' + str(param), 'wsb')
                pickle.dump(svm_model, f1)
                f1.close()
                
                #Predict labels of the new data points
                y_predict = svm_model.predict(valid_samples_features)
                #Predict novelty score of new data points
                y_predict_score = svm_model.decision_function(valid_samples_features)

                #Ground truth
                y1 = np.repeat(+1, 6000) #normal
                y2 = np.repeat(-1, 4000) #abnormal
                y = np.concatenate((y1,y2))
          
                f1_binary = sklearn.metrics.f1_score(y, y_predict, pos_label = -1, average = 'binary')
                f1_macro = sklearn.metrics.f1_score(y, y_predict, average = 'macro')      
                auc = sklearn.metrics.roc_auc_score(y, y_predict_score)
                Math_Cof = sklearn.metrics.matthews_corrcoef(y, y_predict) 
                kappa = sklearn.metrics.cohen_kappa_score(y, y_predict)
                f.write(" SVM: {0}, n_comp: {1}, Kappa: {2}, f1macro: {3}, f1binary: {4}, mathcof: {5}, auc: {6} ".format(feat, param, kappa, f1_macro, f1_binary, Math_Cof, auc)) 
 
                f.write("\n")
                       
f.close()                       
                    
       