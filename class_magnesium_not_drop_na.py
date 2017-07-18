import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().magic('matplotlib inline')

import os
from copy import deepcopy

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

#import sklearn
#from sklearn.decomposition import PCA
#from sklearn import preprocessing
#from sklearn.cluster import KMeans
#from sklearn.model_selection import cross_val_score

from tqdm import tnrange, tqdm_notebook

from matplotlib import colors as mcolors
import random
#from matplotlib.backends.backend_pdf import PdfPages

#import re

import itertools
from sklearn.utils.multiclass import check_classification_targets

#import winsound

colour = ['#fc977c', '#929292']

class Magnesium(object):
    def __init__(self, file_, model = None, fold = "rna-ion-step2/", with_groups = True, colours = ['#fc977c', '#929292'],
                choose_n = False, n_features = None):
        self.filename = file_.split('.csv')[0]
        if model is not None:
            self.model = model            
        else:
            self.model = RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='gini')
        self.colours = colours
        self.choose_n = choose_n
        self.n_features = n_features
        self.model_name = str(self.model).split('(')[0]
        self.trained_model = None
        self.data = pd.read_table(fold+file_).fillna(method = 'backfill', axis = 1)        
        self.data_numpy = np.matrix(self.data)
        self.features = list(self.data.columns)[1:]
        self.groups = self.data_numpy[:,:1]
        self.x = self.data_numpy[:, 1:-1]
        self.xt = None
        self.y = np.array(self.data_numpy[:,-1].flatten().tolist()[0])   
        self.y_pred = None
        self.y_prob = []
        self.indexes = []
        self.with_groups = with_groups
        self.feature_inds = None
        
        self.important_features = None
        self.train_score = []
        self.test_score= []
        self.test_roc_auc_score = []
        self.gridsearched_model = None
        self.tresholds = []
        self.prec_rec_data = {'precision':[], 'recall':[]}
       
    def choose_features(self, save_to_file = True, plot = False):
        classifier = self.model
        if self.with_groups:
            gss = StratifiedShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 0)
            splitted = gss.split(self.x, self.y, groups = self.groups)
        else:
            gss = StratifiedShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 0)
            splitted = gss.split(self.x, self.y)
        for train_index, test_index in splitted:        
            x_train = self.x[train_index]
            y_train = self.y[train_index]
            classifier.fit(x_train, y_train)
        model = SelectFromModel(classifier, prefit = True)
        
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.feature_inds = indices
        
        if not self.choose_n:
            self.xt = model.transform(self.x)
        elif self.n_features == 'all':
            self.xt = self.x
        else:
            self.choose_n_features(self.n_features)
        print("Feature reduction: %d -> %d" % (self.x.shape[1], self.xt.shape[1]))
        
        if save_to_file:
            file = open('Ranked features, %s.txt' % self.filename, 'w')
            for f in range(self.x.shape[1]):
                file.write("%d. %s (%f)\n" % (f + 1, self.features[indices[f]], importances[indices[f]]))
            file.close()
            

        if plot:
            std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis = 0)              
            ax = plt.figure(figsize = (15, 5)).add_subplot(111)
            ax.set_title(self.model_name + ".\n Feature importances", fontsize = 15)
            block = ax.bar(range(x_train.shape[1]), importances[indices], color = self.colours[0], yerr=std[indices], 
                           error_kw=dict(ecolor = self.colours[1]),
                           align="center", label = "importance")
            bord = ax.axvline(self.xt.shape[1], color='black', linestyle='--', alpha = 0.5)
            ax.set_xticks(list(plt.xticks()[0]) + [self.xt.shape[1]])
            ax.set_xlim([0, x_train.shape[1]])
            y_ticks = plt.yticks()[0]
            ax.set_ylim([0,max(y_ticks[:-1])])
            line = matplotlib.lines.Line2D([0,1], [1,1], color = self.colours[1])
            ax.legend([block, line, bord], ["importance","std", "edge"], fontsize = 12)
            plt.show()

        self.important_features = indices[:self.xt.shape[1]]
        return importances
            
    def choose_n_features(self, n):
        self.xt = self.x[:,self.feature_inds[:n]]
        
    def fit_predict(self, n_splits = 3, test_size = 0.2, plots = True, redundant = True, gridsearched = False, 
                    x = None, y = None):
        if gridsearched:
            self.trained_model = self.gridsearched_model
        else:
            self.trained_model = self.model
        
        if x is None:
            if redundant:
                x = self.xt
            else:
                x = self.x
            y = self.y
        else:
            x = x
            y = y
        #print(x.shape)
        gss = StratifiedShuffleSplit(n_splits = n_splits, test_size = 0.3, random_state = 0)
        if self.with_groups:
            splitted = gss.split(x, y, groups = self.groups)
        else:            
            splitted = gss.split(x, y) 
        if plots:
            ax = plt.figure(figsize = (10, 12)).add_subplot(211)
            learn = []
            learn_labels = []
    
        i = 0
#        pdf_pages = PdfPages('outputs/Misclassified/precision_recalls_%s.pdf' % (self.filename))
        for train_index, test_index in tqdm_notebook(splitted, desc = "Splits", leave = True):    
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
 #           print(y_train.shape)
            self.trained_model.fit(x_train, y_train)
            y_pred = self.trained_model.predict(x_test)
            y_prob = self.trained_model.predict_proba(x_test)[:, 1]
            self.train_score.append(self.trained_model.score(x_train, y_train))
            self.test_score.append(self.trained_model.score(x_test, y_test))
            self.test_roc_auc_score.append(roc_auc_score(y_test, y_prob))
            
            self.y_prob.append(y_prob)
            self.indexes.append(test_index)
            
            i = i + 1
#            self.prec_recall_pdf(y_test, y_prob, i, pdf_pages)
            
            if plots:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                learn = (ax.plot(fpr, tpr, color = self.colours[1], alpha=0.5, label = ''))
        self.y_data = [y_prob, test_index]
        print('Number of sites: ', np.sum(y_test == 1))
        print('Portion of sites: ', np.sum(y_test == 1)/y_test.shape[0])
        print('Average score: ', np.mean(self.test_score))
        print('Last score: ', self.test_score[-1])
#        pdf_pages.close()
        if plots:
            final = ax.plot(fpr, tpr, color = self.colours[0], label = 'final curve')
            labels, inds = np.unique(learn_labels, return_index = True)
            ax.legend(learn + final, ["Learning rates",'final curve'], loc = 4, fontsize = 12)
            ax.set_title(self.model_name + ". ROC curves.")

          #  ax = plt.figure(figsize = (10, 12)).add_subplot(312)
          #  ax.plot(list(range(len(self.train_score))), self.train_score, color = self.colours[1], label = 'Train accuracy')
          #  ax.plot(list(range(len(self.test_score))), self.test_score, color = self.colours[0], label = 'Test accuracy')
          #  ax.legend()
          #  ax.set_title(self.model_name + ' . Accuracy scores', fontsize = 12)

            pr = self.prec_recall(y_test, y_prob)
            self.plot_confusion_matrix(y_test, y_pred)
            plt.show()        
        return {'score': np.mean(self.test_score), 'roc_auc':[fpr, tpr], 'prec_rec':pr}
            
    def prec_recall(self,y_test, y_prob):
        precision, recall, treshold = precision_recall_curve(y_test,  y_prob)
        ax = plt.figure(figsize=(10, 12)).add_subplot(212)
        acc = average_precision_score(y_test, y_prob, average="micro")
        ax.scatter(recall, precision, color = self.colours[0])
        ax.plot(recall, precision, color = self.colours[0], lw=1, label=self.model_name + ' (area = {0:0.2f})'
                       ''.format(acc))
        ax.legend(fontsize = 12)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title("Presicion-recall")
        return [precision, recall, acc]
    
    
    def plot_confusion_matrix_(self, cm, normalize=False, title='Confusion matrix', cmap=plt.cm.YlGnBu):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        classes = ['Non-sites', 'Sites']
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, round(cm[i, j],2),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

                                                    
    def plot_confusion_matrix(self, y_test, y_pred):
    # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure().add_subplot(211)
        self.plot_confusion_matrix_(cnf_matrix, title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure().add_subplot(212)
        self.plot_confusion_matrix_(cnf_matrix, normalize=True,title='Normalized confusion matrix')

 #       plt.show()
        
    def prec_recall_pdf(self, y_test, y_prob, i, filename):
        precision, recall, treshold = precision_recall_curve(y_test,  y_prob)
        self.prec_rec_data['precision'].append(precision)
        self.prec_rec_data['recall'].append(recall)
        self.tresholds.append(treshold)
        
        plt.figure(figsize=(10, 5))
        acc = average_precision_score(y_test, y_prob, average="micro")
        plt.scatter(recall, precision, color="teal")
        plt.plot(recall, precision, color="teal", lw=1, label=self.model_name + ' (area = {0:0.2f})'
                       ''.format(acc))   
        plt.axvline(0.001, color='black', linestyle='--', alpha = 0.5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(fontsize = 12)
        plt.title('Iteration %d' % i)
        filename.savefig()  # saves the current figure into a pdf page
        plt.close()
        
    def compute(self, n_splits = 3, test_size = 0.3, plots = True, reduce_features = True,
                save_to_file = True, gridsearched = False):
        if reduce_features:
            self.choose_features(save_to_file)        
        res = self.fit_predict(n_splits, test_size, plots, reduce_features, gridsearched)
        return res
    
    def gridsearch(self, parametres):
        cv = GroupShuffleSplit(n_splits = 3, test_size = 0.7, random_state = 0)

        grid = GridSearchCV(self.model, param_grid = parametres, cv = cv.split(self.x,self.y, groups = self.groups), 
                            verbose = 10, n_jobs = 2)
        grid.fit(self.x, self.y)

        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
        with open('Parametres, '+self.model_name + '.txt', 'w') as out:
             out.write(str(grid.best_params_) + '\n' + str(grid.best_score_))
                
        self.gridsearched_model = grid.best_estimator_