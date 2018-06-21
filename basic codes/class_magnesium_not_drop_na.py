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

try:
    from tqdm import tnrange, tqdm_notebook
    tqdm = True
except ImportError:
    tqdm = False
    pass

from matplotlib import colors as mcolors
from matplotlib import cm
import random

import re

import itertools
from sklearn.utils.multiclass import check_classification_targets

from imblearn.under_sampling import RandomUnderSampler

from scipy.stats.kde import gaussian_kde
from scipy.optimize import brentq
from numpy import linspace

import pickle

from convert_to_roman import *

colour = ['#fc977c', '#929292']

class Magnesium(object):
    def __init__(self, file_, model = None, fold = "rna-ion-step2/", with_groups = True, colours = ['#fc977c', '#929292'],
                choose_n = False, n_features = None, name = ''):
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
        
         
 #       self.data = pd.read_table(fold+file_).fillna(method = 'backfill', axis = 0)  
  #      self.data = pd.read_table(fold+file_).fillna(value=0)
        self.data = pd.read_table(fold+file_)
        if ('DSSR' in self.data.columns):
            self.data.drop('DSSR', axis=1, inplace=True)            
        self.data = self.data.dropna()
 #       self.data.loc[self.data[150:355]>1]=1

        pairings = list(np.array([[''.join(['S', convert(i), j]) for i in range(1,30)] for j in  ['m2', 'm1', '', '1', '2']]).flatten())
        letters = ['W','H','S']
        pairings2 = np.array([[[[[''.join([i,j,k,l])] for i in ['t','c']] 
                       for k in letters[ind:]] for ind,j in enumerate(letters)] for l in ['m2', 'm1', '', '1', '2']]).flatten()
        pairings2 = [list(np.array(i).flatten()) for i in pairings2]
        for i in pairings2:
            pairings = pairings+i 
    
  #      for i in pairings:
   #         self.data.loc[self.data[i]>1, i]=1       

        self.y = deepcopy(np.array(np.matrix(self.data['mg']).flatten().tolist()[0])) 
  #      for i in self.data.columns[150:355]:
   #                 self.data.loc[self.data[i]>1]=0  
        self.data['mg'] = self.y
  #      if np.sum(self.data.isnull().any(axis=1)) > 0:
  #          self.data.fillna(method = 'pad', axis = 0, inplace = True)  
        self.data_numpy = np.matrix(self.data)
        self.features = list(self.data.columns)
        self.features.remove('pdb_chain')
        self.features.remove('mg')
        self.groups = [i.split('.cif1')[0] for i in self.data['pdb_chain'].values]        
        self.x = np.array(self.data[self.features])
        self.features.append('mg')
        self.xt = None
     #   self.y = np.array(self.data_numpy[:,-1].flatten().tolist()[0])   
        self.y_pred = []
        self.y_prob = []
        self.y_true = []
        self.indexes = []
        self.with_groups = with_groups
        self.feature_inds = None
        self.name = name
        
        self.important_features = None
        self.train_score = []
        self.test_score= []
        self.test_roc_auc_score = []
        self.gridsearched_model = None
        self.tresholds = []
        self.prec_rec_data = {'precision':[], 'recall':[]}
        
        self.cnf_plot = []
        self.cnf_normed_plot = []
        self.prec_recall_plot = []
        self.roc_auc_plot = []
        self.probability_density_plot = []
          
    def form_plot_string(self, type_of_plot, *args, **kwargs):
        arguments = ','.join([str(i.tolist()) if str(type(i)).split('.')[0] == "<class 'numpy" 
                              else ('"'+str(i)+'"' if type(i) == str else str(i)) for i in args])
        properties = ','.join(['='.join([str(name), str(value.tolist())]) if str(type(value)).split('.')[0] == "<class 'numpy"
                               else ('='.join([str(name), '"'+str(value)+'"']) if type(value) == str 
                                     else '='.join([str(name), str(value)]))  for name, value in kwargs.items()])
        return (type_of_plot+'('+arguments+','+properties+')') if (len(arguments) != 0 and len(properties) != 0) else (type_of_plot+'('+arguments+properties+')')
       
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
        
    def fit_predict(self, n_splits = 3, test_size = 0.2, plots = True, redundant = True, gridsearched = False, balanced = True,  
                    balanced_test = False, ratio = 'auto', x = None, y = None):
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
        gss = GroupShuffleSplit(n_splits = n_splits, test_size = 0.3, random_state = 0)
        rus = RandomUnderSampler(ratio = ratio, random_state=42)
        if self.with_groups:
            splitted = gss.split(x, y, groups = self.groups)
        else:            
            splitted = gss.split(x, y) 
    
        i = 0
        iterator = tqdm_notebook(splitted, desc = "Splits", leave = True) if tqdm else splitted            
        for train_index, test_index in iterator: 
            if balanced:            
                x_train, y_train = rus.fit_sample(x[train_index], y[train_index])
            else:
                x_train = x[train_index]
                y_train = y[train_index]
            if balanced_test:            
                x_test, y_test = rus.fit_sample(x[test_index], y[test_index])
            else:
                x_test = x[test_index]
                y_test = y[test_index]
            self.trained_model.fit(x_train, y_train)
     #       y_pred = self.trained_model.predict(x_test)
            y_prob = self.trained_model.predict_proba(x_test)[:, 1]
            self.train_score.append(self.trained_model.score(x_train, y_train))
            self.test_score.append(self.trained_model.score(x_test, y_test))
      #      self.test_roc_auc_score.append(roc_auc_score(y_test, y_prob))
            
            self.y_prob.append(y_prob)            
            self.y_true.append(y_test)
            self.indexes.append(test_index)
            treshold, _ = self.plot_probability_density(plots=False)
            y_pred = [1 if i>=treshold else 0 for i in y_prob]
            self.y_pred.append(y_pred)
            
            i = i + 1
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            if plots:
                self.roc_auc_plot.append(self.form_plot_string('plt.plot', fpr, tpr, color = self.colours[1], alpha=0.5))
        self.y_data = [y_prob, test_index]
   #     print('Number of sites: ', np.sum(y_test == 1))
        print('Portion of sites in test: ', np.sum(y_test == 1)/y_test.shape[0])
        print('Portion of sites in train: ', np.sum(y_train == 1)/y_train.shape[0])
  #      print('Average score: ', np.mean(self.test_score))
#        print('Last score: ', self.test_score[-1])
        if plots:
            self.roc_auc_plot = self.roc_auc_plot[:-1]
            self.roc_auc_plot.append(self.form_plot_string('plt.plot', fpr, tpr, color = self.colours[1], alpha=0.5, label = 'Learning rates'))
            self.roc_auc_plot.append(self.form_plot_string('plt.plot', fpr, tpr, color = self.colours[0], alpha=0.5, label = 'Final curve'))
            self.roc_auc_plot.append(self.form_plot_string('plt.legend', loc = 4, fontsize = 12))
            self.roc_auc_plot.append(self.form_plot_string('plt.title', self.model_name + ". ROC curves."))
            
        pr = self.prec_recall(y_test, y_prob, plots) 
        cnf = self.plot_confusion_matrix(y_test, y_pred, plots)   
        treshold, prob_dens = self.plot_probability_density()
        if (not os.path.isdir('trained_models')):
            os.mkdir('trained_models')
            
        model_name = '%s_depth=%d_leaves=%d_%s_validation'%(re.split("\.|\'", str(self.trained_model.__class__))[-2],
                                              self.trained_model.__dict__['max_depth'], 
                                              self.trained_model.__dict__['min_samples_leaf'],
                                              self.filename)
        with open("trained_models/"+model_name+".sav", 'wb') as file_to_save:
            pickle.dump(self.trained_model, file_to_save)
        
        return {'test score': np.mean(self.test_score), 'train score':np.mean(self.train_score), 'treshold':treshold, 'roc_auc':[fpr, tpr], 'prec_rec':pr, 'confusion': cnf,
               'plots':{'roc_auc': self.roc_auc_plot, 'prec_recall': pr[3],
                         'cnf_normed': cnf[3], 'cnf': cnf[2], 'prob_density': prob_dens}}

    def predict(self, x = None, y = None, model = None, file_ = None, plots = True):         
        if file_ is not None:
            data = np.matrix(pd.read_table(file_).fillna(method = 'backfill', axis = 1))   
            x = data[:, 1:-1] 
            y = np.array(data[:,-1].flatten().tolist()[0])
        trained_model = self.trained_model if model is None else model
        y_prob = trained_model.predict_proba(x)[:, 1]
        treshold, prob_dens = self.plot_probability_density(y_prob, y)
        y_pred = [1 if i>=treshold else 0 for i in y_prob]
        test_score = trained_model.score(x, y)
        test_roc_auc_score = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)        
        roc_auc_plot = [self.form_plot_string('plt.plot', fpr, tpr, color = self.colours[0], alpha=0.5, label = '')]
        roc_auc_plot.append(self.form_plot_string('plt.title', self.model_name + ". ROC curves."))
        pr = self.prec_recall(y, y_prob, plots) 
        cnf = self.plot_confusion_matrix(y, y_pred, plots)
        return {'x': x, 'y': y, 'probability': y_prob, 'prediction': y_pred, 'treshold': treshold, 'test_score':test_score, 'roc_auc':[fpr, tpr], 'prec_rec':pr[:-1], 'confusion': cnf, 
                'plots':{'roc_auc': roc_auc_plot, 'prec_recall': pr[3],
                         'cnf_normed': cnf[3], 'cnf': cnf[2], 'prob_density': prob_dens}}
  
        
    def prec_recall(self,y_test, y_prob, plots):
        precision, recall, treshold = precision_recall_curve(y_test,  y_prob)
        acc = average_precision_score(y_test, y_prob, average="micro")
        if plots:  
            prec_recall_plot = []
            prec_recall_plot.append(self.form_plot_string('plt.scatter', recall, precision, color = self.colours[0]))
            prec_recall_plot.append(self.form_plot_string('plt.plot', recall, precision, color = self.colours[0],
                                                          lw=1, label=self.model_name + ' (area = {0:0.2f})'''.format(acc)))
            prec_recall_plot.append(self.form_plot_string('plt.legend', fontsize = 12))
            prec_recall_plot.append(self.form_plot_string('plt.xlabel', 'Recall'))
            prec_recall_plot.append(self.form_plot_string('plt.ylabel', 'Precision'))
            prec_recall_plot.append(self.form_plot_string('plt.title', "Presicion-recall"))
        return [precision, recall, acc, prec_recall_plot]
    
    def plot_probability_density(self, y_prob = None, y = None, plots = True):
        if y_prob is None:
            y_prob, y = [self.y_prob[-1],self.y_true[-1]] 
        
        kde1 = gaussian_kde(y_prob[y == 1])
        kde2 = gaussian_kde(y_prob[y == 0])
        
        x1 = linspace(np.min(y_prob[y == 1]),np.max(y_prob[y == 1]),500)
        x2 = linspace(np.min(y_prob[y == 0]),np.max(y_prob[y == 0]),500)
        
        try:
            treshold = brentq(lambda x : kde1(x) - kde2(x), x2[np.argmax(kde1(x1))], x1[np.argmax(kde2(x2))])
        except ValueError:
            treshold = 0.5
            
        probability_density_plot = []
        if plots:
            probability_density_plot.append(self.form_plot_string('plt.fill_between',
                                                                       x1,kde1(x1),0, color='darkblue', alpha = 0.5, label = 'Sites'))
            probability_density_plot.append(self.form_plot_string('plt.fill_between',
                                                                      x2,kde2(x2), 0, color='darkgrey', alpha = 0.5, label = 'Non-sites'))
            probability_density_plot.append(self.form_plot_string('plt.axvline',
                                                                      x1[np.argmax(kde1(x1))], color='black', linestyle='--', alpha = 0.5))
            probability_density_plot.append(self.form_plot_string('plt.axvline',
                                                                      x2[np.argmax(kde2(x2))], color='black', linestyle='--', alpha = 0.5))
            probability_density_plot.append(self.form_plot_string('plt.axvline',
                                                                       treshold, color='black', linestyle='-.', alpha = 0.7, label = str(round(treshold,2)))) 
            probability_density_plot.append(self.form_plot_string('plt.xticks', [0, 0.2, 0.4, 0.6, 0.8, 1]))
            probability_density_plot.append(self.form_plot_string('plt.legend'))
            probability_density_plot.append(self.form_plot_string('plt.title', 'Probability Distributions'))
            probability_density_plot.append(self.form_plot_string('plt.xlabel', 'Probabilities'))
        return [treshold,probability_density_plot]

    
    def plot_confusion_matrix_(self, cm, normalize=False, title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        classes = ['Non-sites', 'Sites']
        general_plot_strings = []
        
        general_plot_strings.append(self.form_plot_string('plt.imshow', cm, interpolation='nearest', cmap="YlGnBu"))
        general_plot_strings.append(self.form_plot_string('plt.colorbar'))
        tick_marks = np.arange(len(classes))
        general_plot_strings.append(self.form_plot_string('plt.xticks', tick_marks, classes, rotation=45))
        general_plot_strings.append(self.form_plot_string('plt.yticks', tick_marks, classes))                               
        general_plot_strings.append(self.form_plot_string('plt.title', title))
        cm_not_normalized = deepcopy(cm)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = 2* cm_not_normalized.max() / 3.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            colour="white" if (cm_not_normalized[i, j] > thresh) else "black"
            general_plot_strings.append(self.form_plot_string('plt.text', j, i, round(cm[i, j],2), horizontalalignment="center", color=colour))
        general_plot_strings.append(self.form_plot_string('plt.tight_layout'))
        general_plot_strings.append(self.form_plot_string('plt.ylabel', 'True label'))
        general_plot_strings.append(self.form_plot_string('plt.xlabel', 'Predicted label'))
        return general_plot_strings

                                                    
    def plot_confusion_matrix(self, y_test, y_pred, plots):
    # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        cnf_normed = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)
        
        if plots:
            # Plot non-normalized confusion matrix
            cnf_plot = self.plot_confusion_matrix_(cnf_matrix, title='Confusion matrix, without normalization')

            # Plot normalized confusion matrix
            cnf_normed_plot = self.plot_confusion_matrix_(cnf_matrix, normalize=True,title='Normalized confusion matrix')
        
        return [cnf_matrix, cnf_normed, cnf_plot, cnf_normed_plot]

        
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
                                    
    def show_plots(self, plots):
        possible_plots = ['roc_auc', 'prec_recall', 'cnf', 'cnf_normed', 'prob_density']
        all_plots = [i for i in possible_plots if i in plots.keys()]
        n_plots = len(all_plots)
        subplots_params = [n_plots*100+11+itr for itr in range(n_plots)]
        fz = (14, 5*n_plots)
        if 'cnf' in all_plots:
            fz = (14, 4.5*(n_plots-1))
            i = (n_plots - 1)*100
            k = all_plots.index('cnf')
            subplots_params = [param-100 for param in subplots_params]
            subplots_params[k] += (n_plots + 9 - k) if k!=0 else (n_plots + 8 - k)
            subplots_params[k+1] += (n_plots + 9 - k) if k!=0 else (n_plots + 8 - k)
            for itr in range(k+2, n_plots):
                subplots_params[itr] -= 1
        fig = plt.figure(figsize=fz)
        
        for i,plot in enumerate(all_plots):
            ax = fig.add_subplot(subplots_params[i])
            [eval(plot_string) for plot_string in plots[plot] if plots[plot] != []]

        
    def compute(self, n_splits = 3, test_size = 0.3, plots = True, show_plots = True, reduce_features = False,
                save_to_file = False, gridsearched = False, balanced = True, balanced_test = False, ratio = 'auto'):
        if reduce_features:
            self.choose_features(save_to_file)     
        data = self.fit_predict(n_splits, test_size, plots, reduce_features, gridsearched, 
                                balanced, balanced_test, ratio = 'auto')
        if show_plots:
                self.show_plots(data['plots'])                
        return data
    
    def gridsearch(self, parametres, scoring_func, balanced = True, ratio = 'auto'):
        if balanced:
            rus = RandomUnderSampler(ratio = ratio, random_state=42)
            x, y = rus.fit_sample(self.x, self.y)
        else:
            x, y = self.x, self.y
        grid = GridSearchCV(self.model, param_grid = parametres, scoring = scoring_func,
                            cv = StratifiedShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 0),
                            verbose = 10, n_jobs = 2)
        grid.fit(self.x, self.y)

        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
        with open('outputs/Parametres, '+self.model_name + '.txt', 'w') as out:
             out.write(str(grid.best_params_) + '\n' + str(grid.best_score_))                
        self.gridsearched_model = grid.best_estimator_
        return grid.cv_results_
    
    
def plot_one_plot(plot_elements):
    [eval(plot_string) for plot_string in plot_elements]