#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from scipy.stats import t
import textwrap

    
def load_rfs(path_to_radiomics, path_to_baseline,features_to_remove=[]):          # NOTE: HARD-CODED ERRORS REMOVED
    
    #Read the dataset :
    RF = pd.read_csv(path_to_radiomics)
    RF.dropna(inplace=True)
    cols = RF.columns
    RF = RF.drop(cols[range(7,29)],axis=1)
    RF = RF.drop(cols[[0,5,6]],axis=1)

    baseline = RF.loc[RF['Study']=='baseline'].reset_index(drop=True)
    radiomics_c2 = RF.loc[RF['Study']=='cycle2'].reset_index(drop=True)

    errorP = ['SAR_5SAR2_329007','SAR_5SAR2_338003','SAR_5SAR2_321014','SAR_5SAR2_511008']
    errorR = ['RLL','RLL','LML','right']

    indsB = [np.where((baseline.ID == errorP[i]) & (baseline.Location == errorR[i]))[0] for i in range(len(errorP))]
    indsB = np.concatenate(indsB).ravel()
    indsC = [np.where((radiomics_c2.ID == errorP[i]) & (radiomics_c2.Location == errorR[i]))[0] for i in range(len(errorP))]
    indsC = np.concatenate(indsC).ravel()

    baseline = baseline.drop(indsB,axis=0).reset_index(drop=True)
    radiomics_c2 = radiomics_c2.drop(indsC,axis=0).reset_index(drop=True)


    '''
    Both sanity checks are True:
    (baseline.ID == cycle2.ID).all() --> True
    (baseline.Location == cycle2.Location).all() --> True
    This means that the rows in baseline and cycle 2 correspond with one another.  :)
    '''
    
    morphs = ['whole lesion','lesion core','interior rim','exterior rim']

    wholeLesion = baseline.iloc[np.where(baseline.MorphRegion == morphs[0])[0],:].reset_index(drop=True)
    # wholeLesion.columns[4:] = wholeLesion.columns[4:] + '_' + morphs[0]
    lesionCore = baseline.iloc[np.where(baseline.MorphRegion == morphs[1])[0],4:].reset_index(drop=True)
    lesionCore.columns = lesionCore.columns + '_' + morphs[1]
    interiorRim = baseline.iloc[np.where(baseline.MorphRegion == morphs[2])[0],4:].reset_index(drop=True)
    interiorRim.columns = interiorRim.columns + '_' + morphs[2]
    exteriorRim = baseline.iloc[np.where(baseline.MorphRegion == morphs[3])[0],4:].reset_index(drop=True)
    exteriorRim.columns = exteriorRim.columns + '_' + morphs[3]

    radiomics_bl = pd.concat([wholeLesion,lesionCore,interiorRim,exteriorRim],axis=1)
    radiomics_bl = radiomics_bl.drop(features_to_remove,axis=1)
    
    # remove NaNs
    radiomics_bl = radiomics_bl.dropna(axis=0)
    # radiomics_c2 = radiomics_c2.iloc[radiomics_bl.index,:]

    ids = radiomics_bl.ID

    BL = pd.read_csv(path_to_baseline)
    inds = [np.where(BL.USUBJID == ids.iloc[i])[0] for i in range(len(ids))]
    inds = np.concatenate(inds).ravel()

    clinical = BL.iloc[inds,:].reset_index(drop=True)
    
    return radiomics_bl,radiomics_c2,clinical

def volume_change_filtering(radiomics_bl,radiomics_c2,threshold=3.5):
    
    # measurements
    baselineVolume = (1/1000) * radiomics_bl.original_shape_VoxelVolume.iloc[np.where(radiomics_bl.MorphRegion=='whole lesion')[0]]
    cycle2Volume = (1/1000) * radiomics_c2.original_shape_VoxelVolume.iloc[np.where(radiomics_c2.MorphRegion=='whole lesion')[0]].reset_index(drop=True)
    cycle2Volume = cycle2Volume[baselineVolume.index]
    # absolute volume change, percent volume change
    deltaV_abs = (cycle2Volume-baselineVolume)
    deltaV_perc = (cycle2Volume-baselineVolume)/baselineVolume*100
    

    inds_noOutliers = remove_outliers(baselineVolume,threshold)

    # plot 
    plot_volume_hist(baselineVolume[baselineVolume<250],'All Lesions')
    plot_volume_hist(baselineVolume[inds_noOutliers],'MAD Outlier Removal')
    
    return inds_noOutliers,baselineVolume,deltaV_abs,deltaV_perc

def plot_volume_hist(volDist,title='insert title here'):
    
    plt.hist(volDist,color='green')
    plt.xlabel('Baseline Lesion Volume (cc)')
    plt.ylabel('Number of Lesions')
    plt.title(title)
    plt.grid('both')
    plt.show()
        
    
def plot_response_dist(response_outcome,clinical,outcome='categorical',tr=50):
    '''
    How many data points in each category as determined by threshold?
    '''

    if outcome == 'categorical':
        threshs = range(1,101)
        rc = []
        sc = []
        pc = []
        re = []
        se = []
        pe = []

        respc = response_outcome[clinical.ARM==0]
        respe = response_outcome[clinical.ARM==1]

        for T in threshs:
            # control
            rc.append(np.sum(respc<=-T)/len(respc)*100)
            sc.append(np.sum(np.logical_and((respc>-T),(respc<=T)))/len(respc)*100)
            pc.append(np.sum(respc>T)/len(respc)*100)
            # experimental
            re.append(np.sum(respe<=-T)/len(respe)*100)
            se.append(np.sum(np.logical_and((respe>-T),(respe<=T)))/len(respe)*100)
            pe.append(np.sum(respe>T)/len(respe)*100)

        fig,axes = plt.subplots(nrows=1,ncols=2,sharey=True,figsize=(7,5))
        axes[0].plot(threshs,pc,label='progressor',color=(16/255,37/255,111/255),linewidth=3)    
        axes[0].plot(threshs,sc,label='stable',color=(157/255,72/255,33/255),linewidth=3)
        axes[0].plot(threshs,rc,label='responder',color=(30/255,101/255,36/255),linewidth=3)
        axes[0].set_ylabel('Lesions (%)')
        axes[0].set_xlabel('$T_r$ (%)')
        axes[0].set_ylim(0,100)
        axes[0].axvline(x=50,linewidth=3,color='red',linestyle='--')
        axes[0].grid(which='both')
        axes[1].plot(threshs,pe,label='progressor',color=(16/255,37/255,111/255),linewidth=3)    
        axes[1].plot(threshs,se,label='stable',color=(157/255,72/255,33/255),linewidth=3)
        axes[1].plot(threshs,re,label='responder',color=(30/255,101/255,36/255),linewidth=3)
        axes[1].set_ylim(0,100)
        axes[1].axvline(x=50,linewidth=3,color='red',linestyle='--')
        axes[1].grid(which='both')
        axes[1].set_xlabel('$T_r$ (%)')
        # plt.xlabel("$X_{axis}$")
        # plt.legend(loc='uppercentre',ncol=3)
        plt.legend(bbox_to_anchor=(1.02, 0.55), loc='best', borderaxespad=0)
    
    if outcome=='binary':
        threshs = range(1,101)
        rc = []
        pc = []
        re = []
        pe = []

        respc = response_outcome[clinical.ARM==0]
        respe = response_outcome[clinical.ARM==1]

        for T in threshs:
            # control
            rc.append(np.sum(respc<=T)/len(respc)*100)
            pc.append(np.sum(respc>T)/len(respc)*100)
            # experimental
            re.append(np.sum(respe<=T)/len(respe)*100)
            pe.append(np.sum(respe>T)/len(respe)*100)

        fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(7,6))
        axes[0].plot(threshs,rc,label='non-progressor',color=(30/255,101/255,36/255),linewidth=3)
        axes[0].plot(threshs,pc,label='progressor',color=(16/255,37/255,111/255),linewidth=3)   
        axes[0].set_ylabel('Lesions (%)')
        # axes[0].set_xlabel('$T_r$ (%)')
        axes[0].set_ylim(0,100)
        axes[0].axvline(x=tr,linewidth=3,color='red',linestyle='--')
        axes[0].grid(which='both')   
        axes[1].plot(threshs,re,label='non-progressor',color=(30/255,101/255,36/255),linewidth=3)
        axes[1].plot(threshs,pe,label='progressor',color=(16/255,37/255,111/255),linewidth=3) 
        axes[1].set_ylim(0,100)
        axes[1].axvline(x=50,linewidth=3,color='red',linestyle='--')
        axes[1].grid(which='both')
        axes[1].set_xlabel('$T_r$ (%)')
        axes[1].set_ylabel('Lesions (%)')
        # plt.xlabel("$X_{axis}$")
        # plt.legend(loc='uppercentre',ncol=3)
        plt.legend(bbox_to_anchor=(1.02, 0.55), loc='best', borderaxespad=0)
        
def simple_metric_comparison(radiomics_bl,baseline,deltaV_perc,inds_noOutliers,featuresOfInterest,tr=50):
    
    response_outcome = deltaV_perc[inds_noOutliers]
    response_features = radiomics_bl[inds_noOutliers][featuresOfInterest]
    response_features_all = radiomics_bl[:][featuresOfInterest]
    arm_inds = baseline.ARM[inds_noOutliers]

    plot_response_dist(response_outcome,baseline[inds_noOutliers],'binary',tr)  # plot distribution of response_outcome for given threshold of Tr

    l = len(response_outcome)


    # set response thresholds (user-input)
    responseInds = response_outcome > tr
    responseStr = responseInds.astype('str').reset_index(drop=True)
    responseStr[np.where(~responseInds)[0]] = 'non-progressive' # non-progressive
    responseStr[np.where(responseInds)[0]] = 'progressive'      # progressive


    ids = np.tile(radiomics_bl.ID[inds_noOutliers],(8,))
    drug = np.tile(arm_inds,(8,))
    metric = int(len(drug)/2) * ['Intensity'] + int(len(drug)/2) * ['Entropy']
    response = np.tile(responseStr,(8,))
    region = np.tile(l * ['whole lesion'] + l * ['lesion core'] + 
                              l * ['interior rim'] + l * ['exterior rim'], (2,))
    value = response_features.values.ravel('F')  

    simpleMetrics = pd.DataFrame([ids,drug,metric,response,region,value]).transpose()
    simpleMetrics.columns = ['ID','Arm','Metric','Response','Region','Value']

    baselineVolume = (1/1000) * radiomics_bl.original_shape_VoxelVolume.iloc[np.where(radiomics_bl.MorphRegion=='whole lesion')[0]]

    volMetrics = pd.DataFrame([baselineVolume[inds_noOutliers].values,responseStr]).transpose()
    volMetrics.columns = ['Volume (cc)','Response']
    volMetrics['Volume (cc)'] = volMetrics['Volume (cc)'].astype('float')
    volMetrics['Arm'] = arm_inds.values
    volMetrics.Arm[volMetrics.Arm==1] = 'Doxorubicin plus TH-302'
    volMetrics.Arm[volMetrics.Arm==0] = 'Doxorubicin Monotherapy'
    
    return response_features_all,simpleMetrics,volMetrics

def var_filter(df,thresh):
    
    var = df.var()
    cols = df.columns
    reduced_cols = cols[np.where(var>=thresh)]  

    return reduced_cols,df[reduced_cols]

def cluster_red(df,var_thresh=10,distance_thresh=0.5):
    
    # cols = df.columns
    # df = df[cols[4:]]    # exclude header/descriptive columns
    cols_varred, df_varred = var_filter(df,var_thresh)

    # obtain the linkages array
    corr = df_varred.corr()  # we can consider this as affinity matrix
    distances = 1 - corr.abs().values  # pairwise distnces

    distArray = ssd.squareform(distances)  # scipy converts matrix to 1d array
    # print(max(distArray))
    hier = hierarchy.linkage(distArray, method='average')  
    hier[hier<0] = 0

    fig = plt.gcf()
    fig.set_size_inches(18.5, 6)
    fig, ax = plt.subplots(figsize=(12, 6))

    hierarchy.dendrogram(hier, truncate_mode="level", p=30, color_threshold=distance_thresh,
                                no_labels=True,above_threshold_color='k')
    plt.axhline(y=1.5, color='r', linestyle='--')
    plt.ylabel('distance',fontsize=20)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    	label.set_fontsize(20)
    plt.axhline(y=distance_thresh, color='r',linestyle='--')
    plt.style.context('light_background')
    plt.show()

    cluster_labels = hierarchy.fcluster(hier, distance_thresh, criterion="distance")
    num = len(np.unique(cluster_labels))
    
    # print('Number of clusters: {}'.format(num))
    # print('Distance threshold: {}'.format(distance_thresh))

    kmeds = KMedoids(n_clusters=num,init='k-medoids++',max_iter=300,random_state=0)  # method='pam'
    kmeds.fit(corr)

    centers = kmeds.cluster_centers_
    feature_inds = np.where(centers==1)[1]
    cols_cluster = cols_varred[feature_inds]

    # define the feature df
    return df_varred[cols_cluster]

def remove_outliers(data, threshold=3.5):
        """
        Median Absolute Deviation (MAD) based outlier detection
        https://www.programcreek.com/python/?CodeExample=remove+outliers
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm#MAD
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        
        Returns Boolean vector for indexing non-outlier data points.
        
        """
        median = np.median(data)

        med_abs_deviation = np.median(abs(data - median))
        # scale constant 0.6745 
        modified_z_score = 0.6745 * (data - median)/ med_abs_deviation

        return modified_z_score <= threshold


def mixed_response(df,ids,tr):
    
    # define variables of interest
    resp = []
    prog = []
    mixed_resp = []
    only_resp = []
    only_prog = []
    ids_chg = []

    # check individual lesion responses for each patient
    for i in ids:
        
        inds = np.where(df.ID == i)[0]
        vals = df.deltaV[inds].values
        resp.append((vals<=tr).any())
        prog.append((vals>tr).any())
        ids_chg.append(i.replace('SAR_5SAR2','SARC'))

        mixed_resp.append(np.logical_and(resp[-1],prog[-1]))
        only_resp.append(np.logical_and(resp[-1],~prog[-1]))
        only_prog.append(np.logical_and(~resp[-1],prog[-1]))
        
    df_mixed = pd.DataFrame(data=[ids_chg,only_resp,only_prog,mixed_resp]).T
    df_mixed.columns = ['USUBJID','AR-volume-'+str(tr),'NR-volume-'+str(tr),'MR-volume-'+str(tr)]

    return df_mixed

def mixed_response_categorical(df,ids,tr):
    
    # define variables of interest
    mr = []

    # check individual lesion responses for each patient
    for i in ids:
        
        inds = np.where(df.ID == i)[0]
        vals = df.deltaV[inds].values
        resp = (vals<=tr).any()
        prog = (vals>tr).any()
        if np.logical_and(resp,prog):
            mr.append('MR')
        if np.logical_and(resp,~prog):
            mr.append('AR')
        if np.logical_and(~resp,prog):
            mr.append('NR')
        
    df_mixed = pd.DataFrame(data=[ids,mr]).T
    df_mixed.columns = ['USUBJID','Volume-'+str(tr)]

    return df_mixed


def define_pairs(select='categorical'):
    
    pairs = [
            [('whole lesion', 'responder'), ('whole lesion', 'progressor')],
            [('whole lesion', 'responder'), ('whole lesion', 'stable')],
            [('whole lesion', 'progressor'), ('whole lesion', 'stable')],

            [('lesion core', 'responder'), ('lesion core', 'progressor')],
            [('lesion core', 'responder'), ('lesion core', 'stable')],
            [('lesion core', 'progressor'), ('lesion core', 'stable')],
            
            [('interior rim', 'responder'), ('interior rim', 'progressor')],
            [('interior rim', 'responder'), ('interior rim', 'stable')],
            [('interior rim', 'progressor'), ('interior rim', 'stable')],
            
            [('exterior rim', 'responder'), ('exterior rim', 'progressor')],
            [('exterior rim', 'responder'), ('exterior rim', 'stable')],
            [('exterior rim', 'progressor'), ('exterior rim', 'stable')]
            ]
    
    if select == 'binary':
        pairs = [
                [('whole lesion', 'non-progressive'), ('whole lesion', 'progressive')],
                [('lesion core', 'non-progressive'), ('lesion core', 'progressive')],               
                [('interior rim', 'non-progressive'), ('interior rim', 'progressive')],
                [('exterior rim', 'non-progressive'), ('exterior rim', 'progressive')],
                ]
        
    if select == 'volume':
        pairs = [
            [('Doxorubicin plus TH-302', 'progressive'), ('Doxorubicin plus TH-302', 'non-progressive')],
            [('Doxorubicin Monotherapy', 'progressive'), ('Doxorubicin Monotherapy', 'non-progressive')]
            ]
        
    my_pal = {"progressive": '#001c7f',"non-progressive":'#12711c'}
    
    return pairs,my_pal
    
    
def define_outcomes(deltaV,T=50):
    
    volchgCat = np.zeros([len(deltaV),1])
    volchgCat[np.where(deltaV>+T)[0]] = 1
    volchgCat[np.where(deltaV<-T)[0]] = -1
    volchgCat = volchgCat.ravel()

    volchgBin = deltaV >= +T
    volchgBin = volchgBin.ravel()
    
    return volchgCat,volchgBin



def CCC(np_true, np_pred, feature_names):
    """Concordance correlation coefficient."""

    # Pearson product-moment correlation coefficients
    corr = np.array([np.corrcoef(np_true[:,i].astype(float),np_pred[:,i].astype(float))[0,1] for i in range(len(feature_names))])
    # Mean
    mean_true = np.mean(np_true,axis=0)
    mean_pred = np.mean(np_pred,axis=0)
    # Variance
    var_true = np.var(np_true,axis=0)
    var_pred = np.var(np_pred,axis=0)
    # Standard deviation
    sd_true = np.std(np_true,axis=0)
    sd_pred = np.std(np_pred,axis=0)
    # Calculate CCC
    numerator = 2 * np.multiply(np.multiply(corr,sd_true),sd_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator

def stability_feature_reduction(path_to_features):
    
    '''
    Description:
    Simulate inter-observer variability wrt contouring and eliminate "non-stable" features.

    Algorithm:
        • erode mask w/ 1 mm ball-shaped element
        • dilate mask w/ 1 mm ball-shaped element
        • calculate features using original mask, dilated mask and eroded mask;
        • calculate CCC for the different "observers";
        • compile list of features for which CCC is consistently above 0.8  :P

    '''
    
    # read csv of features as defined above (eroded, dilated and original mask)
    result = pd.read_csv(path_to_features)
    # check for nans -- unlikely but you never know  :)
    if result.isnull().values.any():
        print('remove nans')
       
    # separate into arrays
    feature_names = result.columns[29:]
    df_true = np.array(result[result.MorphRegion == 'whole lesion'].iloc[:,29:],dtype=float)
    df_obs1 = np.array(result[result.MorphRegion == 'erosion'].iloc[:,29:],dtype=float)
    df_obs2 = np.array(result[result.MorphRegion == 'dilation'].iloc[:,29:],dtype=float)

    ccc1 = CCC(df_true,df_obs1,feature_names)
    ccc2 = CCC(df_true,df_obs2,feature_names)

    # remove features where CCC<0.8 overall (take the midpoint between the two)
    features_to_remove = feature_names[np.where((ccc1+ccc2)/2<0.8)[0]]
    features_to_remove = [features_to_remove,features_to_remove+"_lesion core",
                          features_to_remove+"_interior rim",features_to_remove+"_exterior rim"]
    features_to_remove = [item for sublist in features_to_remove for item in sublist]

    return features_to_remove

def calc_conf_intervals(lst_item, confidence = 0.95):
    
    m = np.mean(lst_item)
    s = np.std(lst_item)
    dof = len(lst_item) - 1
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    
    return m-s*t_crit/np.sqrt(len(lst_item)), m+s*t_crit/np.sqrt(len(lst_item))

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    
