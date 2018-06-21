from scipy.stats.kde import gaussian_kde
from scipy.optimize import brentq
from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def form_plot_string( type_of_plot, *args, **kwargs):
        arguments = ','.join([str(i.tolist()) if str(type(i)).split('.')[0] == "<class 'numpy" 
                              else ('"'+str(i)+'"' if type(i) == str else str(i)) for i in args])
        properties = ','.join(['='.join([str(name), str(value.tolist())]) if str(type(value)).split('.')[0] == "<class 'numpy"
                               else ('='.join([str(name), '"'+str(value)+'"']) if type(value) == str 
                                     else '='.join([str(name), str(value)]))  for name, value in kwargs.items()])
        return (type_of_plot+'('+arguments+','+properties+')') if (len(arguments) != 0 and len(properties) != 0) else (type_of_plot+'('+arguments+properties+')')

def plot_probability_density(y_prob, y):
        plots = True
               
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
            probability_density_plot.append(form_plot_string('plt.fill_between',
                                                                       x1,kde1(x1),0, color='darkblue', alpha = 0.5, label = 'Sites'))
            probability_density_plot.append(form_plot_string('plt.fill_between',
                                                                      x2,kde2(x2), 0, color='darkgrey', alpha = 0.5, label = 'Non-sites'))
            probability_density_plot.append(form_plot_string('plt.axvline',
                                                                      x1[np.argmax(kde1(x1))], color='black', linestyle='--', alpha = 0.5,label = str(round(x1[np.argmax(kde1(x1))], 2))))
            probability_density_plot.append(form_plot_string('plt.axvline',
                                                                      x2[np.argmax(kde2(x2))], color='black', linestyle='--', alpha = 0.5,label = str(round(x2[np.argmax(kde2(x2))], 2))))
            probability_density_plot.append(form_plot_string('plt.axvline',
                                                                       treshold, color='black', linestyle='-.', alpha = 0.7, label = str(round(treshold,2)))) 
            probability_density_plot.append(form_plot_string('plt.xticks', [0, 0.2, 0.4, 0.6, 0.8, 1]))
            probability_density_plot.append(form_plot_string('plt.legend'))
            probability_density_plot.append(form_plot_string('plt.title', 'Probability Distributions'))
            probability_density_plot.append(form_plot_string('plt.xlabel', 'Probabilities'))
        return probability_density_plot
    
def plot_one_plot(plot_data):
    [eval(plot) for plot in plot_data];