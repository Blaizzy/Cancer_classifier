import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from helper.Data import Dataset

'''Past Usage:
    Attributes 2 through 10 have been used to represent instances.
    Each instance has one of 2 possible classes: benign or malignant.

        1. Wolberg,~W.~H., \& Mangasarian,~O.~L. (1990). Multisurface method of
          pattern separation for medical diagnosis applied to breast cytology. In
          {\it Proceedings of the National Academy of Sciences}, {\it 87},
          9193--9196.
          -- Size of data set: only 369 instances (at that point in time)
          -- Collected classification results: 1 trial only
          -- Two pairs of parallel hyperplanes were found to be consistent with
             50% of the data
             -- Accuracy on remaining 50% of dataset: 93.5%
          -- Three pairs of parallel hyperplanes were found to be consistent with
             67% of data
             -- Accuracy on remaining 33% of dataset: 95.9%

    Relevant Information:

       Samples arrive periodically as Dr. Wolberg reports his clinical cases.
       The database therefore reflects this chronological grouping of the data.
       This grouping information appears immediately below, having been removed
       from the data itself:

         Group 1: 367 instances (January 1989)
         Group 2:  70 instances (October 1989)
         Group 3:  31 instances (February 1990)
         Group 4:  17 instances (April 1990)
         Group 5:  48 instances (August 1990)
         Group 6:  49 instances (Updated January 1991)
         Group 7:  31 instances (June 1991)
         Group 8:  86 instances (November 1991)
         -----------------------------------------
         Total:   699 points (as of the donated datbase on 15 July 1992)
         
    
    Number of Attributes: 10 plus the class attribute
    
    Attribute Information: (class attribute has been moved to last column)
    
       #  Attribute                     Domain
       -- -----------------------------------------
       1. Sample code number            id number
       2. Clump Thickness               1 - 10
       3. Uniformity of Cell Size       1 - 10
       4. Uniformity of Cell Shape      1 - 10
       5. Marginal Adhesion             1 - 10
       6. Single Epithelial Cell Size   1 - 10
       7. Bare Nuclei                   1 - 10
       8. Bland Chromatin               1 - 10
       9. Normal Nucleoli               1 - 10
      10. Mitoses                       1 - 10
      11. Class:                        (2 for benign, 4 for malignant)
    
    Missing attribute values: 16
    
       There are 16 instances in Groups 1 to 6 that contain a single missing 
       (i.e., unavailable) attribute value, now denoted by "?".  
    
    Class distribution:
     
       Benign: 458 (65.5%)
       Malignant: 241 (34.5%)'''


path = '../Cancer_classifier/data/breast_cancer_ dataset'




if __name__ == '__main__':
    # Get .txt file and create a Data frame
    data = Dataset(path)
    data =  data.load()

    data.to_csv(path_or_buf='../Cancer_classifier/data/cancer_dataset.csv', sep=',', encoding='utf-8')

    # Create ascatter diagram and a correlation matrix
    # of all atributes to see their relation
    scatter = Dataset.scatter_plot()
    Dataset.correlation_matrix(data)
   

    y = data['y']

    y = pd.get_dummies(y, drop_first=False)
    print(y.head())
    x = data[['Clump Thickness', 'Uniformity of Cell Size',
              'Uniformity of Cell Shape', 'Marginal Adhesion',
              'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
              'Normal Nucleoli', 'Mitoses']]

    X_train, X_test, y_train, y_test = train_test_split(x, y['4'], test_size=0.3, random_state=101)

    print(y_test.mean())
    print(y_train.mean())

    print(X_train.shape)
    print(y_train.shape)


    clf = LogisticRegression()
    model = clf.fit(X_train, y_train.values.ravel())
    acc = clf.score(X_train, y_train.values.ravel())
    print('Model Accuracy: ', round(acc*100,3),'%')

    pred = clf.predict(X_test)
    print(pred)
    Dataset.decode_preds(pred)





    # Testing our classifier on our Test set
    pred = clf.predict(X_test)
    # Accuracy
    # and a detailed Classification report
    acctest = clf.score(X_test,y_test)
    print('Model test Accuracy: ', round(accuracy_score(y_test,pred)*100,3), '%')
    print(classification_report(y_test, pred))

    # Now using a class method I created you reverse the One-hot enconding 
    # into Readable and understandable Diagnostics
    decode = Dataset.decode_preds(pred)

    # Here we take our Predictions and Probability and create a EXCEL spreadsheet
    #of our features X , our Diagnostic and Probability Distribuition
    label = pd.DataFrame({'Diagnostic': decode}, index=X_test.index)
    prob= pd.DataFrame(clf.predict_proba(X_test), columns=['Probabilty(Benign)','Probability(Malignant)'], index=X_test.index)
    result = X_test.join(label)
    result = result.join(prob['Probabilty(Benign)'])
    result.to_csv(path_or_buf='../Cancer_classifier/data/result_cancer_dataset.csv',
                      sep=',', encoding='utf-8')


    # Visualizing and understanding our models Accuracy
    #-  Confusion matrix will help us see how many samples we are misclassifying

    #There is room for improvement. 
    #I will iteratively improve this algorithm till 99%, so follow my Github profile to be updated.

     # Compute confusion matrix
     cnf_matrix = confusion_matrix(y_test, pred)
     np.set_printoptions(precision=2)

     classes = ['Benign', 'Malignant']

     # Plot non-normalized confusion matrix
     plt.figure()
     Dataset.plot_confusion_matrix(cnf_matrix, classes=classes,
                              title='Confusion matrix, without normalization')

     # Plot normalized confusion matrix
     plt.figure()
     Dataset.plot_confusion_matrix(cnf_matrix, classes= classes, normalize=True,
                              title='Normalized confusion matrix')

     plt.show()

     # Saving our model for future use
     from joblib import dump
     dump(clf, '../Cancer_classifier/data/WiscosinBreastCancerClf.joblib') 

