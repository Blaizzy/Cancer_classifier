
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

class Dataset:

    '''This class is created and optimized by Prince Canuma for the  Wisconsin Breast Cancer Database (January 8, 1991)

    Github profile:  https://github.com/Blaizzy
    Medium profile:  https://medium.com/@prince.canuma


    Citation:
     K. P. Bennett & O. L. Mangasarian: "Robust linear programming
     discrimination of two linearly inseparable sets", Optimization Methods
    and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers
    ---------------- Methods ---------------------
    load(): Gets the text in the .txt file, creates a pandas Dataframe(DF),
    copies it to the class variable df and returns DataFrame(DF).

    ---------------- Class Methods----------------------
    scatter_plot(): uses the copied pandas DF and creates a scatter plot with showing correlation and histogram
    of all columns.

    df_scatter_plot(*DataFrame as args): receives a pandas DF and creates a scatter plot with showing correlation a
    nd histogram of all columns.

    class_distribution(*Dataframe as args): receives a pandas df and plots a the label distribution

    correlation_matrix(): Uses the class copy of the DF and displays a correlation matrix between attributes.

    decode_preds(*predictions as args): recieves an array of predictions(0s or 1s)from the test set and returns the name of the
    classes (Benign or Malignant)

    confusion_matrix(*true_labels, *predictions): recieves two arguments, the first is an array of the true labels
    and the second are the predicited label
    ----------------------------------------------------------

    '''
    df = 0

    def __init__(self, path):
        self.path = open(path, 'r')
        print('File opened successfully!!!')

    def load(self):
        text = []
        for i in self.path:
            str = i.strip('\n').split(',')
            text.append(str)



        # Read each collumn into an 1D array
        id = [text[i][0] for i in range(len(text))]
        ct = [text[i][1] for i in range(len(text))]
        ucsize = [text[i][2] for i in range(len(text))]
        ucshape = [text[i][3] for i in range(len(text))]
        ma = [text[i][4] for i in range(len(text))]
        secs = [text[i][5] for i in range(len(text))]
        bn = [text[i][6] for i in range(len(text))]
        bc = [text[i][7] for i in range(len(text))]
        nn = [text[i][8] for i in range(len(text))]
        mit = [text[i][9] for i in range(len(text))]
        label = [text[i][10] for i in range(len(text))]

        # Create Dataframe
        df = pd.DataFrame({'Clump Thickness': ct,
                           'Uniformity of Cell Size': ucsize,
                           'Uniformity of Cell Shape': ucshape,
                           'Marginal Adhesion': ma,
                           'Single Epithelial Cell Size': secs,
                           'Bare Nuclei': bn,
                           'Bland Chromatin': bc,
                           'Normal Nucleoli': nn,
                           'Mitoses': mit,
                           'y': label,
                           }, index=id)

        # Attributes range is (1-10)
        # Cleaning Missing values with the min value acceptable
        # Both 0 and 1 don't affect accuracy
        # from 2 to 10 it starts affecting accuracy
        df = df.replace(to_replace='?', value=1)
        # Standard Deviation & mean
        std = df["Bare Nuclei"].astype(int).std()
        mean = df["Bare Nuclei"].astype(int).mean()
        print('Standard deviation:',round(std,0),"\nMean:", round(mean,0))
        # Replacing 0 by standard deviation
        #df = df.replace(to_replace='?', value=int(round(std,0))) #has negligible difference in acc
        print('\n-------------------statistical details---------------\n')
        print(df.describe())
        print('\n-----------------sample of the Data-----------------')
        print(df.head())
        # copy data frame to class variable df.
        Dataset.df = df.astype(int)
        return df

    @classmethod
    def scatter_plot(cls):
        # collumn as numbers
        df_ = cls.df.copy(deep =True)
        new = ['ct', 'ucsize', 'ucshape', 'ma', 'secs', 'bn', 'bc', 'nn', 'mit', 'label']
        old = 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',\
        'Bare Nuclei', 'Bland Chromatin','Normal Nucleoli', 'Mitoses','y'
        df_.rename(columns = dict(zip(old, new)), inplace = True)
        scatter_matrix(df_.loc[:, 'ct':'mit'], diagonal='kde')
        plt.suptitle('Scatter Diagram')
        plt.tight_layout()
        plt.show()

    @classmethod
    def df_scatter_plot(cls, df):
        print(df.head())

        scatter_matrix(df.loc[:, 'Clump Thickness':'Mitoses'], diagonal='kde')
        # plt.tight_layout()
        plt.show()

    @classmethod
    def class_distribution(cls, df):
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.distplot(df['y'], bins=2)
        plt.title('Class Distribution')
        plt.show()



    @classmethod
    def correlation_matrix(cls):
        corr_matrix = cls.df.corr().round(2)
        sns.heatmap(data=corr_matrix, annot=True)
        plt.suptitle('Correlation Matrix')
        plt.show()


    @classmethod
    def decode_preds(cls, pred):
        preds = list()
        print('Classes Decoded succefully')
        for i in pred:

            if i == 0:
                preds.append('Benign')
            elif i == 1:
                preds.append('Malignant')
            elif i == 6:
                print('\n')
            else:
                continue
        preds = np.array(preds)
        print(preds[:5])
        #print(np.reshape(preds,(preds.shape)))
        return preds

    @classmethod
    def plot_confusion_matrix(cls,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
