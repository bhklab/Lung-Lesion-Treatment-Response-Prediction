#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
# Change working directory to be whatever directory this script is in
import os
os.chdir(os.path.dirname(__file__))

# Import required libraries :
import scripts.functionals as f
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib

# import statannotations
import seaborn as sns
from statannotations.Annotator import Annotator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from statsmodels.stats.multitest import multipletests as mtc
from scipy.stats import wilcoxon
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
from sklearn_extra.cluster import KMedoids
matplotlib.rcParams.update({'font.size': 16})
import warnings
warnings.filterwarnings("ignore")  # "default" or "always" to turn back on
from itertools import combinations

# %% OPTIONAL (FOR PRESENTATIONS)

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

# %% FEATURE REDUCTION -- Eliminate features with CCC < 0.8 ("inter-observer" variability simulation)

'''
Description:
Simulate inter-observer variability wrt contouring and eliminate "non-stable" features.

Algorithm:
    • erode mask w/ 1 mm ball-shaped element
    • dilate mask w/ 1 mm ball-shaped element
    • calculate features using original mask, dilated mask and eroded mask;
    • calculate concordance correlation coefficient (CCC) for the different "observers";
    • compile list of features for which CCC is consistently above 0.8  :P
    
Function:
    stability_feature_reduction reads in a csv with features as defined above,
    calculates the CCC between features calculated using the original mask and 
    the eroded mask, as well as features calculated using the original mask and
    the dilated mask. Returns a list of features which have CCC < 0.8.
    
Input:
    • path_to_features   : path to csv containing features extracted from the
                           eroded, dilated and original masks
    
Output:
    • features_to_remove : features with CCC < 0.8

'''

path_to_features = 'data/stability-analysis.csv' 
features_to_remove = f.stability_feature_reduction(path_to_features)

del(path_to_features)

# %% CLEAN UPDATED DF WITH PERITUMORAL FEATURES

df = pd.read_csv('data/radiomics-peritumoral-allSTS.csv')

# strings_list = ['baseline','cycle2','1','2']
# lesion_names = [i.split('/')[-1].split('.')[0].split('-') for i in df.Mask]
# lesion_list = []

# for i in range(len(lesion_names)):
#     for j in range(len(lesion_names[i])):
#         temp = lesion_names[i][j]
#         if temp not in strings_list:
#             lesion_list.append(temp)

# df['Location'] = lesion_list
# df.to_csv('data/radiomics-peritumoral-allSTS.csv',index=False)

# %%

# separate into baseline and cycle2 radiomics dataframes
radiomics_bl = df[df.Study == 'baseline']
radiomics_c2 = df[df.Study == 'cycle2']
radiomics_c2 = radiomics_c2[radiomics_c2.MorphRegion == 'whole lesion'].reset_index(drop=True)

# remove missing values
radiomics_bl = radiomics_bl.dropna(axis=1, how='all')
# drop columns from index 4 to original_shape_Elongation
radiomics_bl.drop(radiomics_bl.columns[4:43], axis=1, inplace=True)
radiomics_c2.drop(radiomics_c2.columns[4:43], axis=1, inplace=True)

# for each unique ID and Location, check to be sure there are at least 4 rows
# if not, remove the rows with the ID and Location from the dataframe
uniqueIDs = radiomics_bl.ID.unique()

for i in uniqueIDs:
    uniqueLocs = radiomics_bl.Location[radiomics_bl.ID == i].unique()
    for j in uniqueLocs:
        if len(np.where(np.logical_and(radiomics_bl.ID == i,radiomics_bl.Location == j))[0]) < 5:
            print('Removing ID: {}, Location: {}'.format(i,j))
            radiomics_bl = radiomics_bl.drop(radiomics_bl[(radiomics_bl.ID == i) & (radiomics_bl.Location == j)].index)
            radiomics_c2 = radiomics_c2.drop(radiomics_c2[(radiomics_c2.ID == i) & (radiomics_c2.Location == j)].index)           

# using the MorphRegion column, separate the radiomics_bl dataframe into 5 dataframes
# each containing the radiomics features for a specific morphological region
whole_lesion = radiomics_bl[radiomics_bl.MorphRegion == 'whole lesion'].reset_index(drop=True)
exterior_rim = radiomics_bl[radiomics_bl.MorphRegion == 'exterior rim'].reset_index(drop=True)
interior_rim = radiomics_bl[radiomics_bl.MorphRegion == 'interior rim'].reset_index(drop=True)
lesion_core = radiomics_bl[radiomics_bl.MorphRegion == 'lesion core'].reset_index(drop=True)
peripheral_ring = radiomics_bl[radiomics_bl.MorphRegion == 'peripheral ring'].reset_index(drop=True)

# rename all columns in each dataframe to include the morphological region
# whole_lesion.rename(columns={col: col+'_whole_lesion' for col in whole_lesion.columns[4:]}, inplace=True)
exterior_rim.rename(columns={col: col+'_exterior rim' for col in exterior_rim.columns[4:]}, inplace=True)
interior_rim.rename(columns={col: col+'_interior rim' for col in interior_rim.columns[4:]}, inplace=True)
lesion_core.rename(columns={col: col+'_lesion core' for col in lesion_core.columns[4:]}, inplace=True)
peripheral_ring.rename(columns={col: col+'_peripheral ring' for col in peripheral_ring.columns[4:]}, inplace=True)
# %% SANITY CHECK

all_dfs = [whole_lesion,exterior_rim,interior_rim,lesion_core]  # ,peripheral_ring

# for all combinations of dataframes, check that the ID and Location columns are the same
comb = list(combinations(range(len(all_dfs)), 2))

for i in comb:
    print('Comparing {} and {}'.format(i[0],i[1]))
    print('ID: ',all_dfs[i[0]].ID.equals(all_dfs[i[1]].ID))
    print('Location: ',all_dfs[i[0]].Location.equals(all_dfs[i[1]].Location))
    print('--------------------')

# %% DROP SOME COLUMNS AND CONCATENATE DATAFRAMES

base_df = whole_lesion.drop(['MorphRegion'],axis=1)

for i in range(1,len(all_dfs)):
    base_df = pd.concat([base_df,all_dfs[i].drop(['ID','Location','Study','MorphRegion'],axis=1)],axis=1)

volume_metrics = whole_lesion[['ID', 'Location', 'original_shape_VoxelVolume']].merge(
    radiomics_c2[['ID', 'Location', 'original_shape_VoxelVolume']],
    on=['ID', 'Location'],
    suffixes=('_baseline', '_cycle2'),
    how='left'
)

# replace missing values in volume_metrics with 0 -- these are lesions that were not contoured in cycle 2 (because they disappeared)
volume_metrics.fillna(0, inplace=True)


baselineVolume = (1/1000.0) * volume_metrics['original_shape_VoxelVolume_baseline'].values
cycle2Volume = (1/1000.0) * volume_metrics['original_shape_VoxelVolume_cycle2'].values
deltaV_abs = cycle2Volume - baselineVolume
deltaV_perc = (deltaV_abs / baselineVolume) * 100

# %% LOAD CLINICAL DATA

clinical = pd.read_csv('data/SARC021_Baseline.csv')
# make a dictionary of the clinical data, specifically USUBJID and ARM
clinical_dict = dict(zip(clinical.USUBJID, clinical.ARM))
# apply the dictionary to the ID column in volume_metrics
volume_metrics['ARM'] = volume_metrics['ID'].map(clinical_dict)
baseline = volume_metrics[['ID','ARM']]
baseline.columns = ['USUBJID','ARM']

# %% READ AND ORGANIZE DATA

'''
Description:
Load radiomics and clinical features. Separate features from baseline and first
follow-up into two dataframes. Remove "non-stable" features from baseline 
dataframe per the list generated in the above cell. Isolate clinical features 
from only those patients for which a lung lesion was contoured (clinical
dataframe should have one row for each lesion, which means some redundancy).
    
Function:
    load_rfs loads radiomics and clinical features and stores them in analysis
    friendly format.
    
Input:
    • path_to_radiomics  : path to csv containing features extracted from the
                           baseline and first follow-up images
    • path to clinical   : path to csv containing baseline clinical information
                           for all patients (even those without lung lesions)
    • features_to_remove : list of features with CCC < 0.8
    
Output:
    • radiomics_bl       : radiomics features from baseline
    • radiomics_c2       : radiomics features from first follow-up (cycle 2)
    • baseline           : baseline clinical features / clinical information

'''

# path_to_radiomics = 'data/radiomics.csv'  
# # path_to_radiomics = 'data/radiomics-peritumoral-allSTS.csv'
# path_to_clinical = 'data/baseline.csv' 

# radiomics_bl,radiomics_c2,baseline = f.load_rfs(path_to_radiomics,path_to_clinical,features_to_remove)
# del(path_to_radiomics,path_to_clinical)

# %% DEFINE THE OUTCOME (VOLUME CHANGE)

'''
NOTE: beyond this, we have little use for the cycle2 radiomic features. Many of 
the images in cycle2 have rescale slope/intercept issues and as such, the 
features are pretty much garbage (hence only using volume). In future iterations, 
it may prove useful to perform this scaling however, so we can look at delta 
radiomics features.

Description:
Helper function to calculate outcome (volume change) and eliminate outliers in
terms of baseline volume. Method used: median absolute deviation (MAD). 
    
Function:
    volume_change_filtering reads in a csv with features as defined above,
    calculates the CCC between features calculated using the original mask and 
    the eroded mask, as well as features calculated using the original mask and
    the dilated mask. Returns a list of features which have CCC < 0.8.
    
Input:
    • radiomics_bl    : radiomics features from baseline
    • radiomics_c2    : radiomics features from first follow-up (cycle 2)
    • mod_Zscore      : modified Z-score cutoff value
    
Output:
    • inds_noOutliers : logical array (for indexing; True if lesion is not an outlier)
    • baselineVolume  : baseline volume of lesions (in cc/mL)
    • deltaV_abs      : absolute change in volume from baseline to first follow-up
    • deltaV_perc     : percent change in volume from baseline to first follow-up
        
'''

# inds_noOutliers,baselineVolume,deltaV_abs,deltaV_perc = f.volume_change_filtering(base_df,radiomics_c2,3)

# %% COMPARE SIMPLE METRICS FOR RESPONSE CATEGORIES

'''
Description:
Helper function to isolate metrics of interest to compare between response categories. 
    
Function:
    volume_change_filtering reads in a csv with features as defined above,
    calculates the CCC between features calculated using the original mask and 
    the eroded mask, as well as features calculated using the original mask and
    the dilated mask. Returns a list of features which have CCC < 0.8.
    
Input:
    • radiomics_bl       : radiomics features from baseline
    • radiomics_c2       : radiomics features from first follow-up (cycle 2)
    
Output:
    • inds_noOutliers : logical array (for indexing; True if lesion is not an outlier)
    • baselineVolume  : baseline volume of lesions (in cc/mL)
    • deltaV_abs      : absolute change in volume from baseline to first follow-up
    • deltaV_perc     : percent change in volume from baseline to first follow-up
        
'''

# featuresOfInterest = ['original_firstorder_Mean','original_firstorder_Mean_lesion core',
#                       'original_firstorder_Mean_interior rim','original_firstorder_Mean_exterior rim',
#                      'original_firstorder_Entropy','original_firstorder_Entropy_lesion core',
#                       'original_firstorder_Entropy_interior rim','original_firstorder_Entropy_exterior rim']
# response_features_all,simpleMetrics,volMetrics = f.simple_metric_comparison(radiomics_bl,baseline,deltaV_perc,inds_noOutliers,featuresOfInterest,tr=50)


# %% VOLUME COMPARISON

# pairs,my_pal =f.define_pairs('volume')

# hue_plot_params = {
#                     'data': volMetrics,
#                     'x': 'Arm',           # change to Response if not comparing arms
#                     'y': 'Volume (cc)',
#                     'hue' : 'Response',
#                     "palette": my_pal,
#                     "width" : 0.5,
#                     'boxprops':{'edgecolor':'white'},
#                     'medianprops':{'color':'white'},
#                     'whiskerprops':{'color':'white'},
#                     'capprops':{'color':'white'}
#                 }
# fig = plt.figure(figsize=(5,3),dpi=500, facecolor = 'black')

# ax = sns.boxplot(**hue_plot_params)
# f.wrap_labels(ax, 12)
# plt.legend(bbox_to_anchor=(1.05, 0.55), loc='best', borderaxespad=0)
# sns.despine(ax=ax, offset=10, trim=True)

# annotator = Annotator(ax, pairs, 
#                       pvalue_thresholds=[[1e-4, '****'], [1e-3, '***'], [1e-2, '**'], [0.05, '*'], [0.15, '.'], [1, 'ns']],
#                       **hue_plot_params)
# annotator.configure(pvalue_thresholds=[[1e-4, '****'], [1e-3, '***'], [1e-2, '**'], [0.05, '*'], [0.15, '.'], [1, 'ns']],
#                         test="Mann-Whitney",  # t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal
#                         loc = 'inside',
#                         line_height = 0.01,
#                         line_offset_to_group = 0,
#                         line_offset = 0,
#                         text_offset = 0,
#                         color = 'white',
#                         comparisons_correction="fdr_bh"
#                         ).apply_and_annotate()
# plt.ylabel(r'$V_{t=0} (cc)$')
# plt.savefig('results/volume-comparison.png',bbox_inches='tight',dpi=100)



# %% FEATURE REDUCTION I

radiomics_bl = base_df

df_cluster = f.cluster_red(radiomics_bl.iloc[:,4:])

# %% FEATURE REDUCTION II

# remove columns from df_cluster that are strongly correlated with other columns
# (i.e. columns that have a correlation coefficient > 0.8 with another column)
if 'original_shape_VoxelVolume' not in df_cluster.columns:
    df_cluster.insert(0,'original_shape_VoxelVolume',base_df.original_shape_VoxelVolume)

cor = df_cluster.corr(method='spearman')['original_shape_VoxelVolume']
cols_to_keep = cor[abs(cor) < 0.2].index
features_volcorr = df_cluster[cols_to_keep]
if 'original_shape_VoxelVolume' not in cols_to_keep:
    features_volcorr.insert(0,'original_shape_VoxelVolume',base_df.original_shape_VoxelVolume)


# %% ANALYSIS

df = features_volcorr.copy()
# remove any row that has original_shape_VoxelVolume > 16.05 cc
# inds = np.where(df.original_shape_VoxelVolume > 1605)[0]
# df = df.drop(inds)


# modifiable arguments (preset for analysis)
Tr = 50
arm = 0
max_features = 10
splits = 10
vol_low = 0
vol_high = 1e10 # in cc
stats_testing = True

# preset parameters
# deltaVcat,deltaVbin = f.define_outcomes(deltaV_perc,Tr)
deltaVbin = deltaV_perc < -Tr
print('--------------------')
print('Tr: {} %'.format(Tr))

index_choice = 'arm' #'subset+'
model_choice = 'kNN' # 'logistic', 'kNN', 'naivebayes'
dat = 'imaging'
target_choice = deltaVbin         # deltaVbin, deltaVcat, delatV_perc


print('Ta(low) = {} cc'.format(vol_low))
print('Ta(high) = {} cc'.format(vol_high))
print('Model used: {}'.format(model_choice))
if arm == 0:
    print('Trial Arm: Doxorubicin monotherapy')
else:
    print('Trial Arm: TH-302 plus Doxorubicin')
print('--------------------')


indices = {
            # 'subset'  : np.where(np.logical_and((baselineVolume[inds_noOutliers]<vol_high),(baselineVolume[inds_noOutliers]>vol_low)))[0],
            # 'subset+' : np.where(np.logical_and(np.logical_and((baselineVolume<vol_high),(baselineVolume>vol_low)),(baseline['ARM']==arm)))[0],
            'arm'     : np.where(baseline['ARM']==arm)[0],
            'all'     : range(len(target_choice))
          }

models = {
            'logistic'   : [LogisticRegression(random_state=1),{
                                                                'penalty'  : ['l1', 'l2', 'elasticnet', None],
                                                                'solver'   : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                                                                'tol'      : [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
                                                                'max_iter' : [50,100,150,200]
                                                                }],
            'naivebayes' : [GaussianNB(),{}],
            'kNN'        : [KNeighborsClassifier(),{
                                                        'n_neighbors' : [3,5,7,9,11],
                                                        'weights'     : ['uniform','distance'],
                                                        'metric'      : ['euclidean','manhattan','minkowski']
                                                        }],
            'svm'        : [SVC(random_state=1),{
                                                    'C'         : [0.1,1,10,100,1000],
                                                    'kernel'    : ['linear','poly','rbf','sigmoid'],
                                                    'degree'    : [2,3,4,5,6],
                                                    'gamma'     : ['scale','auto']
                                                    }],
            'randomforest' : [RandomForestClassifier(random_state=1),{
                                                                    'n_estimators' : [100,200,300,400,500],
                                                                    'criterion'    : ['gini','entropy'],
                                                                    'max_depth'    : [10,20,30,40,50],
                                                                    'max_features' : ['auto','sqrt','log2']
                                                                    }]
            # insert additional models with relevant hyperparameters here
         }

# initialize for results table
selected_features = []
auroc = []
auprc = []
neg_log_loss = []
mcc_lst = []
wilcoxp = []
wilcoxp.append(np.nan)
fdr = []


for i in range(1,max_features+1):

    # features and outcomes
    predictors = df.iloc[indices[index_choice],:].reset_index(drop=True)
    # remove any data points that may have missing values (sometimes core too small and nans live there instead of radiomic features)
    predInds = predictors[predictors.isnull().any(axis=1)].index
    targInds = predictors.index
    harmInds = [i for i in targInds if i not in predInds]
    # consolidate
    predictors = predictors.loc[harmInds,:]
    targets = target_choice[indices[index_choice]][harmInds] 

    # feature selection
    predictors.pop('original_shape_VoxelVolume')
    fs = SelectKBest(score_func=f_classif, k=i-1)
    mask = fs.fit(predictors, targets).get_support()
    predictors = predictors[predictors.columns[mask]]
    predictors['original_shape_VoxelVolume'] = df.original_shape_VoxelVolume.iloc[indices[index_choice]].reset_index(drop=True)
    
        
    if i == 1:
        print('Progressor Fraction: %.3f' % (sum(targets==1)/len(targets)))
        print('Stable Fraction: %.3f' % (sum(targets==0)/len(targets)))
        print('Total Lesions: %f' % (len(targets)))
        print('--------------------')    
        
    selected_features.append(predictors.columns)
    print('Features selected({}): {}'.format(len(predictors.columns),list(predictors.columns)))

    # modeling
    model = models[model_choice][0]
    params = models[model_choice][1]
    
    if model_choice != 'naivebayes':
        gs = GridSearchCV(model, params, cv=5, scoring='matthews_corrcoef',n_jobs=1)
        gs.fit(predictors,targets)
        print('Best Params: ',gs.best_params_)
        model = gs.best_estimator_
    
    
    scaler = preprocessing.StandardScaler()    
    clf = make_pipeline(scaler, model)
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=10, random_state=1)

   
    negLL = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='neg_log_loss', cv=cv, n_jobs=1)
    neg_log_loss.append(np.mean(negLL))
    
    auc = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='roc_auc_ovo', cv=cv, n_jobs=1)    
    auc_lower,auc_upper = f.calc_conf_intervals(auc)
    print('Average AUROC: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(auc),auc_lower,auc_upper))
    auroc.append(np.mean(auc))
    
    aps = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='average_precision', cv=cv, n_jobs=1)    
    aps_lower,aps_upper = f.calc_conf_intervals(aps)
    print('Average Precision: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(aps),aps_lower,aps_upper))
    auprc.append(np.mean(aps))
    
    mcc = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='matthews_corrcoef', cv=cv, n_jobs=1)    
    mcc_lower,mcc_upper = f.calc_conf_intervals(mcc)
    print('Average MCC: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(mcc),mcc_lower,mcc_upper))
    mcc_lst.append(np.mean(mcc))
    
    print('--------------------')
    print('Significance Testing')
    print('--------------------')
    
    if i == 1:
        avg_precision = aps
    
    if i > 1:
        print('Wilcoxon p-value: ',wilcoxon(avg_precision,aps)[1])
        wilcoxp.append(wilcoxon(avg_precision,aps)[1])
    
    scores_precision, perm_scores_precision, pvalue_precision = permutation_test_score(
        clf, predictors.values, targets.astype('int'), scoring="matthews_corrcoef", cv=cv, n_permutations=1000
    )
    print('p-value: {:.3f}'.format(pvalue_precision))
    print('FDR-corrected p-value: {:.3f}'.format(mtc(np.repeat(pvalue_precision,10))[1][0]))
    fdr.append(mtc(np.repeat(pvalue_precision,10))[1][0])

    print('--------------------')
    
results_df = pd.DataFrame([auroc,auprc,mcc_lst,neg_log_loss,wilcoxp,fdr]).T
results_df.columns = ['AUROC','AUPRC','MCC','NegLogLoss','Wilcoxon P-Value','FDR']
results_df.index = range(1,max_features+1)
print(results_df)

# %% EXTRA FUNCTION (TESTING)

def isolateData(df,inds,featuresofInterest):
    # isolate the data of interest
    predictors = df.copy().iloc[inds,:].reset_index(drop=True)
    # remove any data points that may have missing values (sometimes core too small and nans live there instead of radiomic features)
    predInds = predictors[predictors.isnull().any(axis=1)].index
    targInds = predictors.index
    harmInds = [i for i in targInds if i not in predInds]
    # consolidate
    predictors = predictors.loc[harmInds,:]
    outFeatures = predictors[features[feature_choice]]
    outTargets = target_choice[inds][harmInds] 
    
    return outFeatures,outTargets


# %% Testing specific features for different arms

import scripts.functionals2 as f2

# indicate model of interest
feature_choice = 'dox-prog33-all-naivebayes'
model_choice = 'naivebayes'
df = features_volcorr.copy()

Tr = 33
outcomes = deltaV_perc > +Tr

features = {
            'dox-all'  : ['wavelet-LHL_firstorder_Variance_exterior rim', 
                          'log-sigma-3-0-mm-3D_gldm_GrayLevelVariance_lesion core', 
                          'volume'],
            'dox-low'  : ['wavelet-LHL_firstorder_Variance_exterior rim', 
                          'log-sigma-3-0-mm-3D_gldm_GrayLevelVariance_lesion core', 
                          'log-sigma-3-0-mm-3D_firstorder_10Percentile_exterior rim', 
                          'volume'],
            'evo-all'  : ['logarithm_glcm_SumSquares_lesion core', 
                          'log-sigma-4-0-mm-3D_gldm_GrayLevelVariance_interior rim', 
                          'volume'],
            'evo-low'  : ['wavelet-LLH_firstorder_Mean_lesion core', 
                          'logarithm_firstorder_Mean_lesion core', 
                          'wavelet-HHL_firstorder_Variance_lesion core', 
                          'wavelet-LHL_glcm_ClusterTendency_lesion core', 
                          'volume'],
            'evo-high' : ['logarithm_glcm_SumSquares_lesion core', 
                          'wavelet-HHL_firstorder_Variance_lesion core', 
                          'wavelet-LHL_glcm_ClusterTendency_lesion core', 
                          'logarithm_firstorder_Minimum_interior rim', 
                          'volume'],

            'dox-resp33-all-logistic' : ['exponential_glszm_ZoneVariance', 
                                         'wavelet-HHH_firstorder_Minimum_interior rim', 
                                         'original_firstorder_MeanAbsoluteDeviation', 
                                        'logarithm_firstorder_Variance_lesion core', 
                                        'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 
                                        'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 
                                        'gradient_glcm_SumSquares', 
                                        'original_shape_VoxelVolume'],
            'dox-resp25-all-logistic' : ['exponential_glszm_ZoneVariance', 
                                'wavelet-HHH_firstorder_Minimum_interior rim', 
                                'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 
                                'logarithm_firstorder_Variance_lesion core', 
                                'square_firstorder_Variance_exterior rim', 
                                'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 
                                'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 
                                'gradient_glcm_SumSquares', 
                                'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_interior rim', 
                                'original_shape_VoxelVolume'],
            'dox-prog25-all-logistic' : ['original_glrlm_GrayLevelVariance_lesion core', 
                                'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 
                                'original_shape_VoxelVolume'],
            'dox-prog33-all-logistic' : ['original_glrlm_GrayLevelVariance_lesion core', 
                                         'original_shape_VoxelVolume'],
            'dox-prog33-all-naivebayes' : ['wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'squareroot_glcm_SumSquares_lesion core', 'original_shape_VoxelVolume']
            }

params = {
            'dox-all'  : {'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1},
            'dox-low'  : {'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1},
            'evo-all'  : {'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1},
            'evo-low'  : {'max_iter': 100, 'penalty': 'l2', 'solver': 'sag', 'tol': 0.0001, 'random_state': 1},
            'evo-high' : {'max_iter': 50, 'penalty': 'l2', 'solver': 'newton-cg', 'tol': 0.0001, 'random_state': 1},
            'dox-resp33-all-logistic' : {'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1},
            'dox-resp25-all-logistic' : {'max_iter': 200, 'penalty': 'l2', 'solver': 'lbfgs', 'tol': 0.0001, 'random_state': 1},
            'dox-prog25-all-logistic' :  {'max_iter': 50, 'penalty': 'l2', 'solver': 'lbfgs', 'tol': 0.0001, 'random_state': 1},
            'dox-prog33-all-logistic' :  {'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg', 'tol': 0.0001, 'random_state': 1}
            }

# models = {
#             'logistic'   : [LogisticRegression(random_state=1),{**params[feature_choice]}],
#             'naivebayes' : [GaussianNB(),{}],
#             'kNN'        : [KNeighborsClassifier(),{**params[feature_choice]}]
# }

# specify training data
arm = 0
vol_low = 0#16.05
vol_high = 1600000000 # in cc e-3
inds = np.where(np.logical_and(np.logical_and((baselineVolume<vol_high),(baselineVolume>vol_low)),(baseline['ARM']==arm)))[0]

# trainingFeatures,trainingTargets = isolateData(df,inds,features[feature_choice])
trainingFeatures = df[features[feature_choice]].iloc[inds,:]
trainingTargets = outcomes[inds]

# specify testing data
arm = 1
vol_low = 0#16.05
vol_high = 1600000000 # in cc e-3
inds = np.where(np.logical_and(np.logical_and((baselineVolume<vol_high),(baselineVolume>vol_low)),(baseline['ARM']==arm)))[0]

# testingFeatures,testingTargets = isolateData(df,inds,features[feature_choice])
testingFeatures = df[features[feature_choice]].iloc[inds]
testingTargets = outcomes[inds]

# instantiate the model object
# model = LogisticRegression(**params[feature_choice])
model = GaussianNB()
# model = models[model_choice][0]

model.fit(trainingFeatures,trainingTargets)

same_predictions = model.predict(trainingFeatures)
opp_predictions = model.predict(testingFeatures)  # testing model from one arm on lesions from other arm

print('Dox Model on Dox Lesions')
print('Predicted definitive and was definitive ',np.sum(np.logical_and(same_predictions,trainingTargets)))
print('Predicted definitive and NOT definitive',np.sum(np.logical_and(same_predictions,~trainingTargets)))
print('Predicted NOT definitive and definitive',np.sum(np.logical_and(~same_predictions,trainingTargets)))
print('Predicted NOT definitive and NOT definitive',np.sum(np.logical_and(~same_predictions,~trainingTargets)))
print('--------------------')
print('Dox Model on Dox+Evo Lesions')
print('Predicted definitive and was definitive ',np.sum(np.logical_and(opp_predictions,testingTargets)))
print('Predicted definitive and NOT definitive',np.sum(np.logical_and(opp_predictions,~testingTargets)))
print('Predicted NOT definitive and definitive',np.sum(np.logical_and(~opp_predictions,testingTargets)))
print('Predicted NOT definitive and NOT definitive',np.sum(np.logical_and(~opp_predictions,~testingTargets)))

#DOXlesionsOfInterest = radiomics_bl.copy().iloc[radiomics_bl.index[inds][np.where(np.logical_and(opp_predictions,~testingTargets))[0]]]
#DOXbaselineInfoOfInterest = baseline.copy().iloc[radiomics_bl.index[inds][np.where(np.logical_and(opp_predictions,~testingTargets))[0]]]

# %%
auprc_iter,mcc_iter,aps,mcc,y_real,y_proba = f2.draw_cv_pr_curve(clf, cv, predictors, targets, title='Cross Validated PR Curve')

print('Truth count: ',np.sum(y_real))
print('Predicted count: ',np.sum(y_proba>=0.5))
print('++ ',np.sum(np.logical_and(y_real,y_proba>=0.5)))
print('+- ',np.sum(np.logical_and(~y_real,y_proba>=0.5)))
print('-+ ',np.sum(np.logical_and(y_real,~(y_proba>=0.5))))
print('-- ',np.sum(np.logical_and(~y_real,~(y_proba>=0.5))))

# %%
