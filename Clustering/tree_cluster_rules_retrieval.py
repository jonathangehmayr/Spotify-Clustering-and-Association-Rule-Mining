from sklearn.tree import _tree, DecisionTreeClassifier
import pandas as pd
from itertools import groupby
from operator import itemgetter
from typing import Tuple,List

def extract_splitting_rules(clf: DecisionTreeClassifier, data: pd.DataFrame, feature_names: list, labels:list)-> pd.Series:
    '''
    inspired by https://gist.github.com/winsxx/5832a56fdcc554c27654f01946e1143b#file-rule_generation-py
    recursive retrieval of tree splitting rules 
    '''
    tree= clf.tree_
    clusters = clf.classes_
    ser_stacked_cluster_rules=pd.Series([[]]*len(clusters), index=clusters,dtype=object)
    ser_label_counts= pd.Series(labels).value_counts().sort_index() 

    def tree_splits(node_id:int=0, current_rule:list=[])->None:
        split_feature = tree.feature[node_id]
        # check if node is a leaf
        if split_feature != _tree.TREE_UNDEFINED: 
            name = feature_names[split_feature]
            threshold = tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, round(threshold,ndigits=3))]
            tree_splits(tree.children_left[node_id], left_rule)
            
            # right child
            right_rule = current_rule + ["({} > {})".format(name, round(threshold,ndigits=3))]
            tree_splits(tree.children_right[node_id], right_rule)
        
        # leaf
        else:
            dist = tree.value[node_id][0] # distribution of classes in leaf
            cluster_idx = dist.argmax() # getting dominant class in leaf
            n=int(dist[cluster_idx])

            cluster_rules=list(ser_stacked_cluster_rules[cluster_idx])
            cluster_rules.append((current_rule, n, ser_label_counts[cluster_idx]))
            ser_stacked_cluster_rules[cluster_idx] = cluster_rules
    
    tree_splits()
    return ser_stacked_cluster_rules

def cluster_properties_report(data: pd.DataFrame, clusters, pruning_level=0.01)->Tuple[pd.Series,pd.Series,pd.Series,DecisionTreeClassifier]:
    '''
    decision tree training with data labeled according to cluster and
    further extraction of splitting rules of tree to analyse the 
    properties of a cluster
    '''

    # fitting decission tree with data labeled according to clusters 
    clf = DecisionTreeClassifier(ccp_alpha=pruning_level, random_state=69) 
    clf.fit(data, clusters)
    
    # extract splitting rules from decision tree
    feature_names = data.columns
    ser_stacked_cluster_rules = extract_splitting_rules(clf, data, feature_names, clusters)
    
    #restructure rules for each cluster
    ser_cluster_rules=separate_cluster_rules(ser_stacked_cluster_rules)
    ser_cluster_rules=remove_duplicate_rules(ser_cluster_rules)
        
    return ser_stacked_cluster_rules,ser_cluster_rules

def separate_cluster_rules(ser_stacked_cluster_rules: pd.Series) -> pd.Series:
    '''
    split concatenated rules into single rules per cluster and
    save them as lists of tuples in form of: 
    (rule, number of elements of cluster that fulfil that rule, number of elements in cluster)
    '''
    ser_cluster_rules=pd.Series(index=ser_stacked_cluster_rules.index,dtype=object)
    for item in ser_stacked_cluster_rules.iteritems():
        idx=item[0]
        list_rules_data=item[1]
        ser_cluster_rules[idx]=[tuple([rule,rule_data[1],rule_data[2]]) 
                                      for rule_data in list_rules_data
                                      for rule in rule_data[0]]
    return ser_cluster_rules


def remove_duplicate_rules(ser_cluster_rules: pd.Series)-> pd.Series:
    '''
    if there are rule duplicates within a cluster
    the number of elements in the cluster fulfilling that rule are summed up
    and the duplicate is deleted
    '''
    for item in ser_cluster_rules.iteritems():
        idx=item[0]
        rule_tuples=item[1]
        
        # grouping of the rules if identical rules exist
        grouped_rule_tuples=[list(grouped_rule) for _,grouped_rule in groupby(sorted(rule_tuples,key=itemgetter(0)),itemgetter(0))]

        # if two rules in a cluster are identical, the number of elements belonging to them are summed
        rule_tuples=[]
        for group in grouped_rule_tuples:
            if len(group)>1:
                rule_tuples.append(tuple([group[0][0],                            #--> rule
                                          sum([element[1] for element in group]), #--> all elements of the cluster that fulfil that rule
                                          group[0][2]]))                          #--> all elements in the cluster
            else:
                rule_tuples.append(group[0])        
        ser_cluster_rules[idx]=rule_tuples  
    return sort_series(ser_cluster_rules)

def sort_series(ser:pd.Series)->pd.Series:
    '''
    sorting of pd.Series for which each element represents a list of tuples,
    based on the second element in each tuple
    '''
    for item in ser.iteritems():
        idx=item[0]
        l=item[1]
        ser[idx]=sorted(l, key=lambda tup: tup[1], reverse=True)
    return ser

