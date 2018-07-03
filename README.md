# Thesis
### Purpose: 

The core purpose of this thesis is the comparison between the three most used **novelty detection** methods in the literature namely, **Gaussian mixture models**, **one-class support vectors machine** and **k-nearest neighbors**. Also different feature extraction methods are discussed and compared. The three methods are trained on the GAPs data set (The German Asphalt Pavement Distress), [Eisenbach et al., 2017](https://www.tu-ilmenau.de/fileadmin/media/neurob/publications/conferences_int/2017/Eisenbach-IJCNN-2017_Talk.pdf), using only the intact data with the goal of detecting distress in the images. GAPs is the first freely available pavement distress dataset providing high quality images recorded by a standardized process fulfilling German federal regulations. This will provide a base for a fair comparison of researches in this field.

### Feature Extraction methods used:
- Gabor Filter
- Gray-level co-occurrence matrix
- Local Binary Pattern

### Code directories:
The structure is divided as follows:
- Produce-all-results: where the actual training and evaluation of the three models occur (.py files)
- Notebooks-FeatureExtraction : The notebooks for generating features from training, validation and test datasets.
- Extracted Features: numpy arrays of all extracted features using the three aforementioned methods.
- All Notebooks: contain all experimental notebooks that was written during the thesis (Here you can execute the code to see ROC curves).

### Examples of generated ROC curves:
Performance of Gaussian mixture model using the three different feature extraction methods on Validation set
![image here](https://github.com/GhaliaRihawi/Thesis/blob/master/GMM_GLCM-vs-Gabor-vs-LBP_valid.png)

Performance of GMM, SVM, KNN using LBP as the feature extraction method on Validation set 
![image here](https://github.com/GhaliaRihawi/Thesis/blob/master/GMM_SVM_KNN_GLCM-LBP_validation.png)


