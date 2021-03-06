# Cancer_classifier

Github profile:  https://github.com/Blaizzy
Medium profile:  https://medium.com/@prince.canuma

    Dataset:
    [Wisconsin Breast Cancer Database](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names)
    This breast cancer database was obtained from the University of Wisconsin Hospitals, Madison from Dr William H. Wolberg on January 8, 1991.
    
    Citation:
     K. P. Bennett & O. L. Mangasarian: "Robust linear programming
     discrimination of two linearly inseparable sets", Optimization Methods
    and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers
    
# Class Dataset
I create a Class(Data.py) optimized for this Dataset
All the Data Preprocessing and Postprocessing is done automatically for you. 
You can contribute by forking this repo and extend the class to suit your needs.
    
Please don't forget to cite this repo. :+1:

## Methods

**load()**: Gets the text in the .txt file, creates a pandas Dataframe(DF),
copies it to the class variable df and returns DataFrame(DF).

## Class Methods

**scatter_plot()**: uses the copied pandas DF and creates a scatter plot with showing correlation and histogram
of all columns.

**df_scatter_plot(*DataFrame as args)**: receives a pandas DF and creates a scatter plot with showing correlation and histogram of all columns.

**class_distribution(*Dataframe as args)**: receives a pandas df and plots a the label distribution

**correlation_matrix()**: Uses the class copy of the DF and displays a correlation matrix between attributes.

**decode_preds(*predictions as args)**: recieves an array of predictions(0s or 1s)from the test set and returns the name of the
classes (Benign or Malignant)

**confusion_matrix(*true_labels, *predictions)**: recieves two arguments, the first is an array of the true labels
and the second are the predicited labels
    
    
# Data
![sample data](https://github.com/Blaizzy/Cancer_classifier/blob/Blaizzy-beta/img/Screenshot%20from%202019-02-03%2018-19-32.png)

# Classifier 

![pred](https://github.com/Blaizzy/Cancer_classifier/blob/Blaizzy-beta/img/precision_50%25.png)

**My classifier is only mislabeling 8 Benign cancer samples) out of 220 and mislabeling 7 (Malignant) cancer samples out of 219.**
There is room for improvement. 
I will iteratively improve this algorithm till 99%, so follow my Github profile to be updated.

You can download the model I created and use it on another dataset with the same distribution. link for [download](https://github.com/Blaizzy/Cancer_classifier/blob/Blaizzy-beta/models/saved_models/WiscosinBreastCancerClf.joblib)

You can run the classifier via this notebook [models/BreastCancer(Sklearn)](https://github.com/Blaizzy/Cancer_classifier/blob/Blaizzy-beta/models/BreastCancer(Sklearn).ipynb)

