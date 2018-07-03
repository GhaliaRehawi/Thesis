import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
import pickle
import sys

f = open('KNN-Res', 'w')
algo = ['GMM', 'SVM', 'KNN']
feats = ['GLCM', 'Gabor', 'LBP', ['GLCM','Gabor'],['Gabor','LBP'], ['GLCM','LBP']]

for alg in (algo):
    if alg == 'KNN':
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
            
            for param in (10, 50, 100, 150, 200, 300, 500):
                   
                print ('KNN_'+ str(feat) + '_' + str(param))
                print ('training start')
                #Train a KNN  model from training data
                knn_model = NearestNeighbors(param, algorithm = 'kd_tree', metric ='euclidean')
                knn_model.fit(training_samples_features) 
                print ('K: ', param)
                print ('training ended')
                sys.stdout.flush()
                #Save the model
                f1 = open('KNN_'+ str(feat) + '_' + str(param), 'wsb')
                pickle.dump(knn_model, f1)
                f1.close()
                
                #find k nearest points for each sample in the training dataset
                kth_dist, kth_ind = knn_model.kneighbors(training_samples_features)
                #1-D array contains distances of each data point to its kth nearest point 
                kth_nearest_dist = kth_dist[:,-1]

                #find k nearest points for each valid sample
                kth_dist2, kth_ind2 = knn_model.kneighbors(valid_samples_features)
                #1-D array contains distances of each data point to its kth nearest point
                kth_nearest_dist2 = kth_dist2[:,-1]
                labels = np.repeat(0,10000)
                #each validation data point whose distance to its kth nearest exceeds the 
                #threshold, which is np.amax(kth_nearest_dist) is novel (0)
                labels[kth_nearest_dist2 > np.mean(kth_nearest_dist)] = 1
                
    
                #Ground truth
                y1 = np.repeat(0, 6000) #normal
                y2 = np.repeat(1, 4000) #abnormal
                y = np.concatenate((y1,y2))
          
                f1_binary = sklearn.metrics.f1_score(y, labels, pos_label = 1, average = 'binary')
                f1_macro = sklearn.metrics.f1_score(y, labels, average = 'macro')      
                auc = sklearn.metrics.roc_auc_score(y, kth_nearest_dist2)
                Math_Cof = sklearn.metrics.matthews_corrcoef(y, labels) 
                kappa = sklearn.metrics.cohen_kappa_score(y, labels)
                f.write(" KNN: {0}, K: {1}, Kappa: {2}, f1macro: {3}, f1binary: {4}, mathcof: {5}, auc: {6} ".format(feat, param, kappa, f1_macro, f1_binary, Math_Cof, auc)) 
                 
                f1_thr = []
                fpr,tpr,thresh = sklearn.metrics.roc_curve(y, kth_nearest_dist2)
                f1_thr = np.zeros((len(thresh), 2))
                for t, thres in enumerate (thresh):
                    labels = np.repeat(0,10000)
                    labels[kth_nearest_dist2 > thres] = 1
                    f1_macro = sklearn.metrics.f1_score(y, labels, average = 'macro')
                    f1_thr[t] = [f1_macro, thres]
                max_idx = np.argmax(f1_thr[:, 0])
                best_f1_macro = f1_thr[max_idx, 0]
                best_threshold = f1_thr[max_idx, 1]

                f.write(" best_f1macro: {0}, threshold: {1} ".format(best_f1_macro, best_threshold))  
                f.write("\n")
                f.write("\n")
                       
f.close()                       
                    
       