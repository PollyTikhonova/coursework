from scipy.spatial import distance_matrix
import os
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pandas as pd
import numpy as np
from IPython.display import Image
import tqdm
from biopandas.pdb import PandasPdb
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from copy import deepcopy
import py3Dmol

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook
# %matplotlib inline

class Clustering(object):
    def __init__(self, fold, pdb, treshold=0.5, real=False):
        self.pdb = pdb
        self.dt = pd.read_csv(fold)
        self.dt['groups'] = [i.split('.cif1_')[0] for i in self.dt.pdb_chain]
        if not real:
            indexes = self.dt.loc[(self.dt.probability.values > treshold) & (self.dt.groups.values == pdb)].pdb_index.values
        else:
            indexes = self.dt.loc[(self.dt.mg.values ==1) & (self.dt.groups.values == pdb)].pdb_index.values
        self.ppdb = PandasPdb().fetch_pdb(pdb)
#         self.pdb_real = self.ppdb.df['ATOM']
        self.pdb_real = pd.concat([self.ppdb.df['ATOM'], self.ppdb.df['HETATM']])
        self.pdb_real = self.pdb_real.sort_values(by='atom_number')
        self.pdb_real.index = self.pdb_real.atom_number.values
#         print(self.pdb_real.atom_number.values[:20])
        self.sites = self.pdb_real.loc[indexes][[ 'x_coord', 'y_coord', 'z_coord']]
        self.resi = np.unique(self.pdb_real.loc[indexes][['residue_number']])
        self.ids = self.pdb_real.loc[indexes][['atom_number']].values
#         self.ids = indexes
        self.ids = np.array(self.ids).reshape(1,-1)[0].tolist()
#          self.n_clusters = self.sites.shape[0]/6
        self.n_clusters = self.sites.shape[0]/10
        self.true_predicted = None
        self.false_predicted = None
        self.not_predicted = None
        self.atoms = None
        self.full_atoms = None
        print('Sites:', self.sites.shape[0])
        del indexes

    def clusterize(self, n_clusters = None, eps=0.5, min_samples=5, dbscan = False, n_clusters_real = None):
        if dbscan:
            if n_clusters_real is None:
                n_clusters = self.n_clusters if n_clusters is None else self.sites.shape[0]/n_clusters 
            else:
                n_clusters = n_clusters_real
            kmeans = KMeans(n_clusters=int(n_clusters),random_state=0).fit(self.sites.values)
            mg_coords = kmeans.cluster_centers_
#             print(mg_coords[:,0:3:2])
            print('n_classes:', len(mg_coords))
            dbscan = DBSCAN(eps, min_samples).fit(mg_coords)
            mg_classes = dbscan.labels_
            print(np.unique(mg_classes))
            if -1 in mg_classes:
                mg_classes[mg_classes == -1] = np.arange(np.sum(mg_classes == -1))+np.unique(mg_classes).shape[0]-1
#             mg_coords = [(np.mean(self.sites.loc[mg_classes==i].values, axis=0)) for i in np.unique(mg_classes)]
            print(np.unique(mg_classes))
            mg_coords = [(np.mean(mg_coords[mg_classes==i], axis=0)) for i in np.unique(mg_classes)]
            mg_coords = np.round_(mg_coords, 3)  
            print(mg_coords.shape)
        else:
            if n_clusters_real is None:
                n_clusters = self.n_clusters if n_clusters is None else self.sites.shape[0]/n_clusters 
            else:
                n_clusters = n_clusters_real          
            kmeans = KMeans(n_clusters=int(n_clusters),random_state=0).fit(self.sites.values)
#             print(np.unique(kmeans.labels_, return_counts=True))
            mg_coords = kmeans.cluster_centers_
            mg_coords = np.round_(mg_coords, 3)
            dist = distance_matrix(mg_coords, mg_coords)
            dist = dist.reshape(1,-1)[0]            
            print(dist.reshape(1,-1).shape)
            print('MEAN: %.2f. MEDIAN: %.2f'%(np.mean(dist), np.median(dist)))
#             dist = dist[(dist<10) & (dist>0)]
            dist = dist[(dist>0)]
            print('MEAN: %.2f. MEDIAN: %.2f. MEANMEAN:%.2f. PERCENT:%.2f'%(
                    np.mean(dist), np.median(dist), np.mean([np.mean(dist), np.median(dist)]), dist.shape[0]/self.sites.shape[0]))
#             print(sorted(dist)[:20])
            plt.hist(dist, bins=40, color='darkmagenta')
        self.mg = self.pdb_real.loc[np.arange(len(mg_coords))+1]
        self.mg[[ 'x_coord', 'y_coord', 'z_coord']] = mg_coords
        self.mg[['atom_name', 'residue_name', 'element_symbol']] = np.repeat('MC', repeats = self.mg.shape[0]*3).reshape(self.mg.shape[0], 3)
        self.mg['residue_number'] = self.mg.atom_number.values
        self.mg.index = list(range(0, self.mg.shape[0]))
        del mg_coords
        
    def find_k(self, tresh = 0.9):
        self.coverege = [[0,0]]
        i = 1
        while (self.coverege[-1][1] < tresh) and (i<= self.sites.values.shape[0]):
#         for i in range(1, 20, 3):
            n_clusters = i         
            kmeans = KMeans(n_clusters=int(n_clusters),random_state=0).fit(self.sites.values)
    #             print(np.unique(kmeans.labels_, return_counts=True))
            mg_coords = kmeans.cluster_centers_
            mg_coords = np.round_(mg_coords, 3)            
            self.mg = self.pdb_real.loc[np.arange(len(mg_coords))+1]
            self.mg[[ 'x_coord', 'y_coord', 'z_coord']] = mg_coords
            self.mg[['atom_name', 'residue_name', 'element_symbol']] = np.repeat('MC', repeats = self.mg.shape[0]*3).reshape(self.mg.shape[0], 3)
            self.mg['residue_number'] = self.mg.atom_number.values
            self.mg.index = list(range(0, self.mg.shape[0]))
            self.coverege.append([i,self.count_coverege()])            
            i += 3
        i -= 4
        if i > 1:
            while self.coverege[-1][1] > tresh:
                n_clusters = i         
                kmeans = KMeans(n_clusters=int(n_clusters),random_state=0).fit(self.sites.values)
        #             print(np.unique(kmeans.labels_, return_counts=True))
                mg_coords = kmeans.cluster_centers_
                mg_coords = np.round_(mg_coords, 3)            
                self.mg = self.pdb_real.loc[np.arange(len(mg_coords))+1]
                self.mg[[ 'x_coord', 'y_coord', 'z_coord']] = mg_coords
                self.mg[['atom_name', 'residue_name', 'element_symbol']] = np.repeat('MC', repeats = self.mg.shape[0]*3).reshape(self.mg.shape[0], 3)
                self.mg['residue_number'] = self.mg.atom_number.values
                self.mg.index = list(range(0, self.mg.shape[0]))
                self.coverege.append([i,self.count_coverege()])            
                i -= 1      
        else:
            self.coverege.append([0,0]) 
        print(self.coverege)
        return self.coverege[-2][0]

        
        
    def delete_wrongs(self, mg=0):
#         for mg in mgs:
        self.percents = [] 
        self.full_atoms = []
        for i in range(self.mg.shape[0]):
            mg = self.mg.loc[i]
            ind = ((self.pdb_real['x_coord'].values >= mg['x_coord']-7) & (self.pdb_real['x_coord'].values <= mg['x_coord']+7)
                  & (self.pdb_real['y_coord'].values >= mg['y_coord']-7) & (self.pdb_real['y_coord'].values <= mg['y_coord']+7)
                  & (self.pdb_real['z_coord'].values >= mg['z_coord']-7) & (self.pdb_real['z_coord'].values <= mg['z_coord']+7)
                  & (self.pdb_real['element_symbol'].values == 'O') & (self.pdb_real['element_symbol'].values == 'N'))
            self.atoms = self.pdb_real.loc[ind]
#             print('atoms:', self.atoms.shape)
            dist = distance_matrix(self.atoms[[ 'x_coord', 'y_coord', 'z_coord']].values, 
                                   mg[[ 'x_coord', 'y_coord', 'z_coord']].values.reshape(1,-1)).reshape(1,-1)[0]
#             print(dist)
            self.atoms = self.atoms.loc[dist<=7]
            print('atoms:', self.atoms.shape)
            if self.atoms.shape[0] != 0:
                self.percents.append(len(list(set(self.atoms['atom_number'].values) - 
                                              set(self.ids)))/self.atoms.shape[0])
            else:
                self.percents.append('Fully wrong')
            self.full_atoms.append(self.atoms)        
        print(np.argsort(self.percents))
        print('Percents:', tuple(zip(np.argsort(self.percents), np.array(self.percents)[np.argsort(self.percents)])))
        
    def count_coverege(self):
        self.covered_inds = [] 
        for i in range(self.mg.shape[0]):
            mg = self.mg.loc[i]
            ind = ((self.pdb_real['x_coord'].values >= mg['x_coord']-7) & (self.pdb_real['x_coord'].values <= mg['x_coord']+7)
                  & (self.pdb_real['y_coord'].values >= mg['y_coord']-7) & (self.pdb_real['y_coord'].values <= mg['y_coord']+7)
                  & (self.pdb_real['z_coord'].values >= mg['z_coord']-7) & (self.pdb_real['z_coord'].values <= mg['z_coord']+7)
                  & (self.pdb_real['element_symbol'].values != 'C') & (self.pdb_real['element_symbol'].values != 'P'))
            self.atoms = self.pdb_real.loc[ind]
#             print('atoms:', self.atoms.shape)
            dist = distance_matrix(self.atoms[[ 'x_coord', 'y_coord', 'z_coord']].values, 
                                   mg[[ 'x_coord', 'y_coord', 'z_coord']].values.reshape(1,-1)).reshape(1,-1)[0]
#             print(dist)
            self.atoms = self.atoms.loc[dist<=7]
#             print('atoms:', self.atoms.shape)
            self.covered_inds = self.covered_inds + self.atoms['atom_number'].values.tolist()
#             print(len(self.covered_inds))
#             print(self.covered_inds)
        self.covered_inds = np.unique(self.covered_inds).tolist()
#         print('                 ',len(self.covered_inds))
        return len(set(self.ids) & set(self.covered_inds))/len(self.ids)
        
    def detect_atoms(self):
        indexes = self.dt.loc[(self.dt.mg.values ==1) & (self.dt.groups.values == self.pdb) 
                              & (self.dt.probability.values >= 0.5)].pdb_index.values
        self.true_predicted = self.pdb_real.loc[indexes]
        if (self.true_predicted.atom_number.values == indexes).all():
            print("OK")
        else:
            print('WRONG TRUE')
            print(self.true_predicted.atom_number.values)
            print(indexes)
        self.true_predicted[['atom_name', 'residue_name', 'element_symbol']] = np.repeat('TP', repeats = self.true_predicted.shape[0]*3).reshape(self.true_predicted.shape[0], 3)
         
        indexes = self.dt.loc[(self.dt.mg.values ==0) & (self.dt.groups.values == self.pdb) 
                      & (self.dt.probability.values >= 0.5)].pdb_index.values
        self.false_predicted = self.pdb_real.loc[indexes]
        if (self.false_predicted.atom_number.values == indexes).all():
            print("OK")
        else:
            print('WRONG false')
        self.false_predicted[['atom_name', 'residue_name', 'element_symbol']] = np.repeat('FP', repeats = self.false_predicted.shape[0]*3).reshape(self.false_predicted.shape[0], 3)
         
        indexes = self.dt.loc[(self.dt.mg.values ==1) & (self.dt.groups.values == self.pdb) 
                      & (self.dt.probability.values < 0.5)].pdb_index.values
        self.not_predicted = self.pdb_real.loc[indexes]
        if (self.not_predicted.atom_number.values == indexes).all():
            print("OK")
        else:
            print('WRONG not')
        self.not_predicted[['atom_name', 'residue_name', 'element_symbol']] = np.repeat('NP', repeats = self.not_predicted.shape[0]*3).reshape(self.not_predicted.shape[0], 3)

        
    def make_file(self, n=None):
        pdb_with_mg = deepcopy(self.ppdb)
        pdb_with_mg.df['ATOM'] = pdb_with_mg.df['ATOM'].append(self.mg)   
        if self.true_predicted is not None:
            pdb_with_mg.df['ATOM'] = pdb_with_mg.df['ATOM'].append(self.true_predicted) 
        if self.true_predicted is not None:
            pdb_with_mg.df['ATOM'] = pdb_with_mg.df['ATOM'].append(self.false_predicted)
        if self.true_predicted is not None:
            pdb_with_mg.df['ATOM'] = pdb_with_mg.df['ATOM'].append(self.not_predicted)
        if self.full_atoms is not None:
            n = np.argsort(self.percents)[-1] if n is None else n
            self.full_atoms[n][['atom_name', 'residue_name', 'element_symbol']] = np.repeat('AT', 
                                                                                    repeats = self.full_atoms[n].shape[0]*3).reshape(self.full_atoms[n].shape[0], 3)
            pdb_with_mg.df['ATOM'] = pdb_with_mg.df['ATOM'].append(self.full_atoms[n])
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        filename = 'outputs/%s_mg_coordinates.pdb'%self.pdb 
        pdb_with_mg.to_pdb(filename)
        del pdb_with_mg
        return filename
    
    def distances_from_real(self):
        real_mgs = self.pdb_real.loc[self.pdb_real.atom_name == 'MG']        
        dist = distance_matrix(real_mgs[[ 'x_coord', 'y_coord', 'z_coord']].values, 
                                   self.mg[[ 'x_coord', 'y_coord', 'z_coord']].values)
#         print(dist)
#         print(dist.shape)
        axis = 1 if real_mgs.shape[0] < self.mg.shape[0] else 0
        dist = np.min(dist, axis = axis)
#         print(dist)
#         print(dist.shape)
        return [np.mean(dist),  real_mgs.shape[0], self.mg.shape[0]]
   