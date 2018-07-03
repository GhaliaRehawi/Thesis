import numpy as np
import sklearn
from sklearn import mixture
import pickle
import sys

f = open('GMM-Res', 'w')
algo = ['GMM', 'SVM', 'KNN']
feats = ['GLCM', 'Gabor', 'LBP', ['GLCM','Gabor'], ['Gabor','LBP'], ['GLCM','LBP']]

for alg in (algo):
    if alg == 'GMM':
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
            
            for n_comp in (60, 100, 500,1000, 1500, 2000):
                
               print ('GMM_'+ str(feat) + '_' + str(n_comp))
               print ('training start')
                #Train a gaussian mixture model from training data
               gm_model = sklearn.mixture.GaussianMixture(n_components=n_comp, covariance_type='full', max_iter = 300)
               gm_model.fit(training_samples_features)
               print ('training ended')
                #Save the model
               f1 = open('GMM_'+ str(feat) + '_' + str(n_comp), 'wsb')
               pickle.dump(gm_model, f1)
               f1.close()
        
        #   print ('loading pickle')
        #       gm_model = pickle.load( open( 'GMM_'+ str(feat) + '_' + str(n_comp), "rb" ))
        #       print ('pickle loaded')
                sys.stdout.flush()
                
                #Predict labels of the new data points
                y_predict = gm_model.predict(valid_samples_features)
                #Predict novelty score of new data points
                y_predict_score = gm_model.score_samples(valid_samples_features)
                tmp = []
                x_predict_score = gm_model.score_samples(training_samples_features)
                novelty_thresh = np.amin(x_predict_score)

                # if data point belongs to the model then 1 else 0
                tmp = np.repeat(1, 10000)
                tmp[y_predict_score < novelty_thresh] = 0
        
                #Ground truth
                y1 = np.repeat(1, 6000) #normal
                y2 = np.repeat(0, 4000) #abnormal
                y = np.concatenate((y1,y2))
          
                f1_binary = sklearn.metrics.f1_score(y, tmp, pos_label = 0, average = 'binary')
                f1_macro = sklearn.metrics.f1_score(y, tmp, average = 'macro')      
                auc = sklearn.metrics.roc_auc_score(y, y_predict_score)
                Math_Cof = sklearn.metrics.matthews_corrcoef(y, tmp) 
                kappa = sklearn.metrics.cohen_kappa_score(y, tmp)
                f.write(" GMM: {0}, n_comp: {1}, Kappa: {2}, f1macro: {3}, f1binary: {4}, mathcof: {5}, auc: {6} ".format(feat, n_comp, kappa, f1_macro, f1_binary, Math_Cof, auc))
               
                f1_thr = []
                fpr,tpr,thresh = sklearn.metrics.roc_curve(y, y_predict_score)
                f1_thr = np.zeros((len(thresh), 2))
                for t, thres in enumerate (thresh):
                    tmp = np.repeat(1,10000)
                    tmp[y_predict_score < thres] = 0
                    f1_macro = sklearn.metrics.f1_score(y, tmp, average = 'macro')
                    f1_thr[t] = [f1_macro, thres]
                max_idx = np.argmax(f1_thr[:, 0])
                best_f1_macro = f1_thr[max_idx, 0]
                best_threshold = f1_thr[max_idx, 1]

                f.write(" f1macro: {0}, threshold: {1} ".format(best_f1_macro, best_threshold))  
                f.write("\n")
                       
f.close()                       
                    
       