from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib notebook')
from scipy.interpolate import griddata

from ipywidgets import interact
import ipywidgets as widgets

def plot_gridsearch_results(grid_data, type_of_metric = 'auto'):
    @interact(showTrain = True)
    def plot_gridsearch_results_(showTrain):

        def plot_grid(x,y,z, name):
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')

            z_matrix = z.reshape(np.unique(x).shape[0], np.unique(y).shape[0])
            surf = ax.imshow(z_matrix, cmap = cm.Spectral, vmin=np.min(z), vmax=np.max(z))

            for i, j in itertools.product(range(z_matrix.shape[0]), range(z_matrix.shape[1])):
                ax.text(j, i, round(z_matrix[i, j],2),
                         horizontalalignment="center", color = 'black')

            plt.xticks(np.arange(np.unique(y).shape[0]), np.unique(y))
            plt.yticks(np.arange(np.unique(x).shape[0]), np.unique(x))
            plt.ylabel('Max depth')
            plt.xlabel('Min samples leaf')
            fig.colorbar(surf, shrink=0.5, aspect=10)
            plt.title(name, y = 1.12)


        x_test, y_test, z_test = [np.array(np.matrix(grid_data[[0,2,3]])[:,j].reshape(1,-1))[0] for j in [1,2,0]]
        grid_x, grid_y = np.mgrid[0:x_test.max():1000j, 0:y_test.max():2000j]

        grid_z_test = griddata(np.transpose(np.matrix([x_test,y_test])), z_test, (grid_x, grid_y), method='cubic')

        fig = plt.figure(figsize = (13,10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, grid_z_test, rstride=60, cstride = 30, cmap = cm.Spectral, 
                               vmin=np.min(z_test), vmax=np.max(z_test))
        ax.scatter(x_test,y_test,z_test)
        ax.text(x_test[int(round(x_test.shape[0])/2)], y_test[int(round(y_test.shape[0])/2)],
                z_test[int(round(z_test.shape[0])/2)],'Test')

        fig.colorbar(surf, shrink=0.5, aspect=5)

        x_train, y_train, z_train = [np.array(np.matrix(grid_data[[1,2,3]])[:,j].reshape(1,-1))[0] for j in [1,2,0]]
        if showTrain:            
            grid_z_train = griddata(np.transpose(np.matrix([x_train,y_train])), z_train, (grid_x, grid_y), method='cubic')

            ax.plot_surface(grid_x, grid_y, grid_z_train, rstride=60, cstride = 30, cmap = cm.Spectral, 
                                   vmin=np.min(z_train), vmax=np.max(z_train), alpha = 0.8)
            ax.plot_surface(grid_x, grid_y, grid_z_train, color = "black", alpha = 0.3)

            ax.scatter(x_train,y_train,z_train)
            ax.text(x_train[int(round(x_train.shape[0])/2)], y_train[int(round(y_train.shape[0])/2)],
                    z_train[int(round(z_train.shape[0])/2)],'Train')

        ax.set_xticks(x_test)
        ax.set_yticks(y_test)
        ax.set_xlim([x_test.min(), x_test.max()])
        ax.set_ylim([y_test.min(), y_test.max()])
        ax.set_xlabel('Max depth')
        ax.set_ylabel("Leaves")
        plt.ion()
        
        x_test, y_test, z_test = [np.array(np.matrix(grid_data[[0,2,3]])[:,j].reshape(1,-1))[0] for j in [1,2,0]]
        x_train, y_train, z_train = [np.array(np.matrix(grid_data[[1,2,3]])[:,j].reshape(1,-1))[0] for j in [1,2,0]]
        if type_of_metric == 'custom':
            tp_tests_mean, tp_trains_mean, fp_tests_mean, fp_trains_mean = [np.array(np.matrix(grid_data[[6,7,8,9]])[:,j].reshape(1,-1))[0] for j in [0,1,2,3]]        
        
        n_rows = 3 if type_of_metric == 'custom' else 1
        height = 15 if type_of_metric == 'custom' else 10
        fig = plt.figure(figsize = (13,height))
        ax = fig.add_subplot(n_rows,2,1)
        plot_grid(x_test, y_test, z_test, 'Test')
        ax = fig.add_subplot(n_rows,2,2)
        plot_grid(x_train, y_train, z_train, 'Train')
#        plt.subplots_adjust(wspace=0.4)

        if type_of_metric == 'custom':
            ax = fig.add_subplot(323)
            plot_grid(x_test, y_test, tp_tests_mean, 'TP Test')
            ax = fig.add_subplot(324)
            plot_grid(x_train, y_train, tp_trains_mean, 'TP Train')
            ax = fig.add_subplot(325)
            plot_grid(x_test, y_test, fp_tests_mean, 'FP Test')
            ax = fig.add_subplot(326)
            plot_grid(x_train, y_train, fp_trains_mean, 'FP Train')
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
        plt.ion()