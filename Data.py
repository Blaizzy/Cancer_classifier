
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

class Dataset:

    '''This class is created and optimized by Prince Canuma for the  Wisconsin Breast Cancer Database (January 8, 1991)
    citation:
     K. P. Bennett & O. L. Mangasarian: "Robust linear programming
     discrimination of two linearly inseparable sets", Optimization Methods
    and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers
    ---------------- Methods ---------------------
    load(): Gets the text in the .txt file, creates a pandas Dataframe, copies it to the class variable df \
    and returns DataFrame.

    ---------------- Class Methods
    scatterplot(): create a scatter plot with showing correlation and histogram of all columns

    df_scatter():


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

        #for i in range(len(text[0])):
         #   for j in range(len(text[0])):

        #id= []
        #for i in names:
         #   id.append(i)


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

        #def columns(no_columns):
            # as
         #   for

        #def dictionaries():
            #dicts

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


        df = df.replace(to_replace='?', value=0)
        std = df["Bare Nuclei"].astype(int).std()
        mean = df["Bare Nuclei"].astype(int).mean()
        print('Standard deviation:',round(std,0),"\nMean:", round(mean,0))
        df = df.replace(to_replace='?', value=int(round(std,0))) #has negligible difference in acc
        print(df.describe())
        print(df.head())

        Dataset.df =df
        return df

    @classmethod
    def scatter_plot(cls):
        print(cls.df.head())

        scatter_matrix(cls.df.loc[:, 'Clump Thickness':'Mitoses'], diagonal='kde')
        # plt.tight_layout()
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
        plt.show()

    @classmethod
    def correlation_matrix(cls, df):
        corr_matrix = df.corr().round(2)
        sns.heatmap(data=corr_matrix, annot=True)
        plt.show()

    @classmethod
    def two_var_corr(cls, df):
        # take a 5% sample as this is computationally expensive
        df_sample = df.sample(frac=0.05)

        # Pairwise plots
        sns.pairplot(df_sample, hue="y")
        plt.show()

        plt.figure(figsize=(20, 5))

        features = ['Uniformity of Cell Size', 'Mitoses']
        target = df['y']

        x = df[features[0]]
        y = target
        plt.scatter(x, y, marker='o')
        plt.title(features[0])
        plt.xlabel(features[0])
        plt.ylabel('y')
        plt.show()

    @classmethod
    def decode_preds(cls, pred):
        preds = list()
        print('Decoded classes:')
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
        print(np.reshape(preds,(1,210)))

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