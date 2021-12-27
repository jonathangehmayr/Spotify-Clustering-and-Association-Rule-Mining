import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  silhouette_score, confusion_matrix
import matplotlib.pyplot as plt
from tree_cluster_rules_retrieval import cluster_properties_report
from typing import Tuple,List
from pprint import pprint
import copy
import os


def load_data(n:int=1000)->pd.DataFrame:
    os.chdir('..')
    df=pd.read_csv('Spotify_dataset.csv')
    os.chdir('./Clustering')
    df=df.sample(n=n, random_state=69)
    return df

def preprocess_data(df:pd.DataFrame)->pd.DataFrame: 
    '''
    as the decaded attribute is string it is converted into int. the 
    uri, track and artist attributes are dropped
    '''
    decade=[]
    for item in df['decade'].iteritems():
        if item[1][:2]=='00' or item[1][:2]=='10':
            decade.append(int('20'+item[1][:2]))
        else:
            decade.append(int('19'+item[1][:2]))
    df['decade']=decade 
    df=df.drop(['uri','track','artist'], axis='columns')
    return df


def get_target(df:pd.DataFrame, target_attribute:str)->Tuple[pd.DataFrame, pd.Series]:
    '''
    splitting of dataframe into independent and dependent attributes
    '''
    target=df[target_attribute]
    df_=copy.copy(df)
    df_=df_.drop([target_attribute], axis='columns')
    return df_, target

def bin_target(target:pd.Series, n_bins:int=2)-> Tuple[pd.Series, dict]:
    '''
    binning of dependent variable in n_bins
    '''
    _,bins_=pd.cut(target,bins=n_bins,retbins=True)
    bins_=list(pd.Series(bins_,dtype=float).round(3))
    dict_labels={idx:'{} ({}, {})'.format(target.name,bins_[idx],bins_[idx+1])
            for idx,tresh in enumerate(bins_)
            if tresh!=bins_[-1]}
    target=pd.cut(target,bins=n_bins, labels=list(dict_labels.keys()))
    return target,dict_labels

def scale_data(df:pd.DataFrame)->pd.DataFrame:
    '''
    scaling with Standardscaler, so that the mean and std of an variable is 0 and 1 respectively
    '''
    features=df.columns
    scaler=StandardScaler()
    df_scaled= scaler.fit_transform(df)
    df_scaled=pd.DataFrame(df_scaled, columns=features)
    return df_scaled


def kmeans_clustering(df_scaled:pd.DataFrame,n_clusters:int=5)-> Tuple[pd.Series,KMeans]:
    '''
    clustering using kmeans algorithm
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=69).fit(df_scaled)
    labels = pd.Series(kmeans.labels_)
    return labels,kmeans

def hierarchical_clustering(df_scaled:pd.DataFrame,n_clusters:int=5)->Tuple[pd.Series,AgglomerativeClustering]:
    '''
    clustering using hierarchical clustering algorithm
    '''
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(df_scaled)
    labels = pd.Series(clustering.labels_)
    return labels,clustering

def perform_pca(df:pd.DataFrame,n_pc:int=3)->pd.DataFrame:
    '''
    principal component extraction
    '''
    pca = PCA(n_components=n_pc)
    #explained_variance=pca.explained_variance_ratio_  #AttributeError: 'PCA' object has no attribute 'explained_variance_ratio_'
    pc = pca.fit_transform(df)
    cols=['p{}'.format(val+1) for val in range(n_pc)]
    df_pc = pd.DataFrame(data = pc, columns = cols)
    return df_pc

def scatter(list_clusters:List[pd.DataFrame], three_d:bool=True)->None:
    '''
    plotting of 2d and 3d scatter plots of clustered data
    '''
    if three_d:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for idx, cluster in enumerate(list_clusters):
            cluster=cluster.reset_index(drop=True)
            ax.scatter(cluster['p1'], cluster['p2'],
                       cluster['p3'], label=cluster['cluster'][idx])
        plt.legend()
        plt.show()
    else:
        fig, ax = plt.subplots()
        for idx, cluster in enumerate(list_clusters):
            cluster=cluster.reset_index(drop=True)
            ax.scatter(cluster['p1'], cluster['p2'],
                       label=cluster['cluster'][idx])
        plt.legend()
        plt.show()

def get_cluster_class_mapping(y_true:pd.Series, y_pred:pd.Series,dict_labels:dict)->pd.Series:
    '''
    mapping of which clusters represent which classes. implementation
    assumes that number of clusters and classes are identical
    '''
    df_cm=cluster_confusion_matrix(y_true, y_pred, dict_labels)
    ser_cluster_class_mapping=pd.Series(df_cm.idxmax(axis='columns'),
                                        index=df_cm.index)
    return ser_cluster_class_mapping


def plot_clusters(df_pc:pd.DataFrame, labels:pd.Series, dict_labels:dict, ser_mapping:pd.Series, clustering:bool=True)->None:
    '''
    splitting clustered dataframe into list of dataframe, whereas each dataframe
    contains one cluster
    '''
    # map integers to actual classes
    df_pc['cluster'] = labels.apply(lambda x: dict_labels[x])
    
    # split dataframe according to clusters and save as dictionary  
    dict_clusters = {cluster:df_pc[df_pc['cluster'] == cluster]
                for _,cluster in dict_labels.items()}
    
    # in order to map clusters to classes the clustered data is ordered
    # according to which cluster represents wich class best, with ser_mapping
    if clustering:    
        list_clusters=[dict_clusters[idx] for idx in ser_mapping]
        scatter(list_clusters, three_d=True)
    else:
        list_classes=[dict_clusters[idx] for idx in ser_mapping.index]
        scatter(list_classes, three_d=True)
        

def silhouette_scores(df:pd.DataFrame, cluster_fun, n_clusters:tuple)->int:
    '''
    calculation of silhoutte coefficients for variable amount of clusters
    '''
    ser_sc=pd.Series(index=list(range(n_clusters[0],n_clusters[1])),dtype=float)
    for n in list(ser_sc.index):   
        y_pred, model=cluster_fun(df,n_clusters=n)
        sc = silhouette_score(df, y_pred)
        ser_sc[n]=sc
    return ser_sc.idxmax(), ser_sc.max().round(3)

    
def cluster_confusion_matrix(y_true:pd.Series, y_pred:pd.Series,dict_labels:dict)->pd.DataFrame:
    '''
    confusion matrix with custom x and y labels as dataframe
    '''
    cm=confusion_matrix(y_true, y_pred)
    cm=cm[:len(np.unique(y_true)),:len(np.unique(y_pred))]
    df_cm=pd.DataFrame(cm,
                   columns=['cluster {}'.format(i) for i in range(len(np.unique(y_pred)))], 
                   index=[dict_labels[i] for i in range(len(np.unique(y_true)))]) 
    return df_cm


def cluster_f1(y_true:pd.Series, y_pred:pd.Series, dict_labels:dict)->float:
    '''
    custom function to calculate the f1 score for clustering algorithms
    '''
    df_cm=cluster_confusion_matrix(y_true, y_pred,dict_labels)
    sum_clusters=df_cm.sum(axis='rows')
    sum_classes=df_cm.sum(axis='columns')
    
    list_f1=[]
    for i in range(len(df_cm.columns)):
        n_max_class=df_cm.iloc[:,i].max(axis='rows')
        max_idx=df_cm.iloc[:,i].idxmax(axis='rows')
        cluster_total=sum_clusters[i]
        class_total=sum_classes[max_idx]
        f1=(2*n_max_class)/(cluster_total+class_total)
        list_f1.append(f1)
    f1=np.mean(list_f1)
    return round(f1,ndigits=3)
 


def determine_candidate_target_attribute(df_scaled, sc_max, cluster_fun)->Tuple[str,pd.Series]:
    '''
    experimental approach to find the attribute that fits best as target
    to externally control if the clustering algorithm finds proper clusters
    '''
    list_attributes=list(df_scaled.columns)
    ser_clustering_scores=pd.Series(index=list_attributes)
    for target_attribute in list_attributes:
        
        df_scaled_, target=get_target(df_scaled, target_attribute)
        y_true,_=bin_target(target, n_bins=sc_max)
        
        y_pred, cluster_mdodel=cluster_fun(df_scaled_,n_clusters=sc_max)
        f1=cluster_f1(y_true, y_pred,_)  
        
        ser_clustering_scores[target_attribute]=f1
    target_attribute=ser_clustering_scores.idxmax(axis='columns')
    return target_attribute, ser_clustering_scores

def print_cluster_properties(list_clusters_rules:List[list])->None:
    '''
    print function for the characteristic rules of a cluster
    '''
    indent = '    '
    for idx,rule_list in enumerate(list_clusters_rules):
        print(indent+'Characteristic rules for cluster {}:'.format(idx))
        for rule in rule_list:
            print(2*indent+'rule: '+rule[0]+' applies for {} out of {} elements in cluster {}'.format(rule[1], rule[2], idx))
        print()


if __name__ == '__main__': 
    
    '''
    Cluster analysis according to following schema:
    1. Determination how many clusters describe the data best by calculation of silhouette coefficients using kmeans and hierarchical clustering.
    2. Exploration of which attribute is a possible target class candidate by iteratively using each attribute of the data as target class and store the F1-score.
    3. The run in 2. reaching the highest F1 score is further analyzed, by extracting distinct cluster rules. 
    4. Plotting of the samples in 3d space; firstly colored according to their true class and secondly to their belonging cluster of the best run in 2..
    '''
    
    df=load_data().reset_index(drop=True)
    df=preprocess_data(df)
    df_scaled=scale_data(df)
    
    
    '''
    KMeans analysis
    '''
    sc_max_kmeans, sc_kmeans=silhouette_scores(df_scaled, kmeans_clustering, n_clusters=(2,11))
    
    target_attribute_kmeans,ser_clustering_scores=determine_candidate_target_attribute(df_scaled,sc_max_kmeans, kmeans_clustering)

    df_scaled_kmeans, target_kmeans=get_target(df_scaled, target_attribute_kmeans)
    y_true_kmeans,_=bin_target(target_kmeans, n_bins=sc_max_kmeans)
    
    _, target_unscaled_kmeans=get_target(df, target_attribute_kmeans)
    _,dict_class_labels_kmeans=bin_target(target_unscaled_kmeans, n_bins=sc_max_kmeans)
    df=df.drop(target_attribute_kmeans, axis='columns')
    
    y_pred_kmeans, kmeans=kmeans_clustering(df_scaled_kmeans,n_clusters=sc_max_kmeans)
    dict_cluster_labels_kmeans={cluster:'cluster {}'.format(cluster) for cluster in set(list(y_pred_kmeans))}
    
    f1_kmeans=cluster_f1(y_true_kmeans, y_pred_kmeans, dict_class_labels_kmeans)
    df_cm_kmeans=cluster_confusion_matrix(y_true_kmeans, y_pred_kmeans, dict_class_labels_kmeans)
    
    ser_mapping_kmeans=get_cluster_class_mapping(y_true_kmeans, y_pred_kmeans, dict_class_labels_kmeans)
    
    df_pc_kmeans=perform_pca(df_scaled_kmeans)
    plot_clusters(df_pc_kmeans, y_true_kmeans, dict_class_labels_kmeans, ser_mapping_kmeans,clustering=False)

    plot_clusters(df_pc_kmeans, y_pred_kmeans, dict_cluster_labels_kmeans, ser_mapping_kmeans,clustering=True)
    
    ser_stacked_cluster_rules_kmeans,ser_cluster_rules_kmeans= cluster_properties_report(df, y_pred_kmeans)
    print_cluster_properties(ser_cluster_rules_kmeans)
    
    
    
    '''
    Hierarchical clustering analysis
    '''
    
    sc_max_hier, sc_hier=silhouette_scores(df_scaled, hierarchical_clustering, n_clusters=(2,11))
    
    target_attribute_hier,ser_clustering_scores=determine_candidate_target_attribute(df_scaled, sc_max_hier, hierarchical_clustering)

    df_scaled_hier, target_hier=get_target(df_scaled, target_attribute_hier)
    y_true_hier,_=bin_target(target_hier, n_bins=sc_max_hier)
    
    _, target_unscaled_hier=get_target(df, target_attribute_hier)
    _,dict_class_labels_hier=bin_target(target_unscaled_hier, n_bins=sc_max_hier)
    df=df.drop(target_attribute_hier, axis='columns')
    
    y_pred_hier, hier=hierarchical_clustering(df_scaled_hier,n_clusters=sc_max_hier)
    dict_cluster_labels_hier={cluster:'cluster {}'.format(cluster) for cluster in set(list(y_pred_hier))}
    
    f1_hier=cluster_f1(y_true_hier, y_pred_hier, dict_class_labels_hier)
    df_cm_hier=cluster_confusion_matrix(y_true_hier, y_pred_hier, dict_class_labels_hier)
    
    ser_mapping_hier=get_cluster_class_mapping(y_true_hier, y_pred_hier, dict_class_labels_hier)
    
    df_pc_hier=perform_pca(df_scaled_hier)
    plot_clusters(df_pc_hier, y_true_hier, dict_class_labels_hier, ser_mapping_hier,clustering=False)

    plot_clusters(df_pc_hier, y_pred_hier, dict_cluster_labels_hier, ser_mapping_hier,clustering=True)
    
    ser_stacked_cluster_rules_hier,ser_cluster_rules_hier= cluster_properties_report(df, y_pred_hier)
    print_cluster_properties(ser_cluster_rules_hier)
    
    
    
    

    #deleted unique rules
    #cluster correspondence is actually really interesting and should be properly coded
    #analysis with unique rules per cluster does not seem extremely good



