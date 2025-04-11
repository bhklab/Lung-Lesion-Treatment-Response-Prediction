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
# from statannotations.Annotator import Annotator
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
matplotlib.rcParams.update({'font.size': 16})
import warnings
warnings.filterwarnings("ignore")  # "default" or "always" to turn back on
from itertools import combinations
from sklearn.utils import parallel_backend
import textwrap
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc

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
Tr = -33
arm = 0
max_features = 2
splits = 10
vol_low = 0
vol_high = 1e10 # in cc
stats_testing = True

# preset parameters
# deltaVcat,deltaVbin = f.define_outcomes(deltaV_perc,Tr)
deltaVbin = deltaV_perc < Tr
print('--------------------')
print('Tr: {} %'.format(Tr))

index_choice = 'arm' #'subset+'
model_choice = 'naivebayes' # 'logistic', 'kNN', 'naivebayes', 'svm', 'randomforest'
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
    
    # with parallel_backend('threading', n_jobs=-1):
    #     scores_precision, perm_scores_precision, pvalue_precision = permutation_test_score(
    #         clf, predictors.values, targets.astype('int'), scoring="matthews_corrcoef", cv=cv, n_permutations=1000
    #     )
#     print('p-value: {:.3f}'.format(pvalue_precision))
#     print('FDR-corrected p-value: {:.3f}'.format(mtc(np.repeat(pvalue_precision,10))[1][0]))
#     fdr.append(mtc(np.repeat(pvalue_precision,10))[1][0])

#     print('--------------------')
    
# results_df = pd.DataFrame([auroc,auprc,mcc_lst,neg_log_loss,wilcoxp,fdr]).T
# results_df.columns = ['AUROC','AUPRC','MCC','NegLogLoss','Wilcoxon P-Value','FDR']
# results_df.index = range(1,max_features+1)
# print(results_df)

# Save the results to a log file
# log_file_path = 'results/analysis_results.log'
# with open(log_file_path, 'w') as log_file:
#     log_file.write(results_df.to_string())

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

# %% PREPROCESSING PMH DATA

pmh_data = pd.read_csv('/Users/caryngeady/Documents/GitHub/Mixed-Response-Work/spreadsheets/pyradiomics_output-PMH-5mm.csv')
hfx = pd.read_csv('/Users/caryngeady/Desktop/PMH-Sarcoma/FAZA-SARC-HFX.csv')

# locate the first index that contains 'original_shape'
start_ind = pmh_data.columns.get_loc('original_shape_Elongation')
pmh_radiomics = pmh_data.copy().iloc[:,start_ind:]
pmh_radiomics.insert(0,'USUBJID',pmh_data.ID)
pmh_radiomics.insert(1,'MorphRegion',pmh_data.MorphRegion)

# separate into different morphological regions
whole_lesion = pmh_radiomics[pmh_radiomics.MorphRegion == 'whole lesion'].reset_index(drop=True)
exterior_rim = pmh_radiomics[pmh_radiomics.MorphRegion == 'exterior rim'].reset_index(drop=True)
interior_rim = pmh_radiomics[pmh_radiomics.MorphRegion == 'interior rim'].reset_index(drop=True)
lesion_core = pmh_radiomics[pmh_radiomics.MorphRegion == 'lesion core'].reset_index(drop=True)
peripheral_ring = pmh_radiomics[pmh_radiomics.MorphRegion == 'peripheral ring'].reset_index(drop=True)

# rename all columns in each dataframe to include the morphological region
exterior_rim.rename(columns={col: col+'_exterior rim' for col in exterior_rim.columns[2:]}, inplace=True)
interior_rim.rename(columns={col: col+'_interior rim' for col in interior_rim.columns[2:]}, inplace=True)
lesion_core.rename(columns={col: col+'_lesion core' for col in lesion_core.columns[2:]}, inplace=True)
peripheral_ring.rename(columns={col: col+'_peripheral ring' for col in peripheral_ring.columns[2:]}, inplace=True)

# merge the dataframes
pmh_radiomics = whole_lesion.copy().drop('MorphRegion',axis=1)
pmh_radiomics = pd.concat([pmh_radiomics,exterior_rim.drop(['USUBJID','MorphRegion'],axis=1)],axis=1)
pmh_radiomics = pd.concat([pmh_radiomics,interior_rim.drop(['USUBJID','MorphRegion'],axis=1)],axis=1)
pmh_radiomics = pd.concat([pmh_radiomics,lesion_core.drop(['USUBJID','MorphRegion'],axis=1)],axis=1)
pmh_radiomics = pd.concat([pmh_radiomics,peripheral_ring.drop(['USUBJID','MorphRegion'],axis=1)],axis=1)

# isolate the data where HFX is available
hfx_data = hfx.copy().dropna()
pmh_sub = pmh_radiomics[pmh_radiomics.USUBJID.isin(hfx_data['Case #'])]

# %% Testing specific features for different arms

import scripts.functionals2 as f2

# indicate model of interest for Dox arm
feature_choice = 'rfc-prog33-7features'
df = features_volcorr.copy()
scaleFlag = False

Tr = 33
outcomes = deltaV_perc > +Tr

features = {
            # Response to Dox Model
            'knn-prog33-8features'  : ['wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'squareroot_glcm_SumSquares_lesion core', 'original_shape_VoxelVolume'],
            'knn-prog33-9features'  : ['wavelet-HHH_firstorder_Minimum_interior rim', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'squareroot_glcm_SumSquares_lesion core', 'original_shape_VoxelVolume'],
            'log-prog33-8features'  : ['wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'squareroot_glcm_SumSquares_lesion core', 'original_shape_VoxelVolume'],
            'log-prog33-9features'  : ['wavelet-HHH_firstorder_Minimum_interior rim', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'squareroot_glcm_SumSquares_lesion core', 'original_shape_VoxelVolume'],
            'rfc-resp25-6features'  : ['wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'square_firstorder_Variance_exterior rim', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'gradient_glcm_SumSquares', 'original_shape_VoxelVolume'],
            'rfc-prog25-7features'  : ['wavelet-HHH_firstorder_Minimum_interior rim', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'original_shape_VoxelVolume'],
            'rfc-prog33-7features'  : ['wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'squareroot_glcm_SumSquares_lesion core', 'original_shape_VoxelVolume'],
            'rfc-prog33-10features' : ['wavelet-HHH_firstorder_Minimum_interior rim', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'gradient_glcm_ClusterTendency_exterior rim', 'original_glrlm_GrayLevelVariance_lesion core', 'original_firstorder_Minimum_interior rim', 'square_glrlm_GrayLevelVariance_interior rim', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'squareroot_glcm_SumSquares_lesion core', 'original_shape_VoxelVolume'],
            # Response to Evo Model (Progression)
            'evo-knnr25-9featPROG'  : ['wavelet-HLH_glrlm_HighGrayLevelRunEmphasis_lesion core', 'wavelet-HHH_firstorder_Minimum_interior rim', 'original_glcm_SumSquares_interior rim', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'wavelet-LHH_gldm_HighGrayLevelEmphasis', 'logarithm_firstorder_Variance_lesion core', 'wavelet-LHH_gldm_LargeDependenceLowGrayLevelEmphasis_interior rim', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_interior rim', 'original_shape_VoxelVolume'],
            'evo-knnp50-6featPROG'  : ['wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_lesion core', 'gradient_glcm_ClusterTendency_exterior rim', 'gradient_gldm_HighGrayLevelEmphasis_exterior rim', 'gradient_glcm_SumSquares', 'wavelet-HLL_glrlm_HighGrayLevelRunEmphasis_interior rim', 'original_shape_VoxelVolume'],

            }

models = {
            'knn-prog33-8features'  : KNeighborsClassifier(**{'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'distance'}),
            'knn-prog33-9features'  : KNeighborsClassifier(**{'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'distance'}),
            'log-prog33-8features'  : LogisticRegression(**{'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1}),
            'log-prog33-9features'  : LogisticRegression(**{'max_iter': 50, 'penalty': 'l2', 'solver': 'lbfgs', 'tol': 0.0001, 'random_state': 1}),
            'rfc-resp25-6features'  : RandomForestClassifier(**{'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 300, 'random_state': 1}),
            'rfc-prog25-7features'  : RandomForestClassifier(**{'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 400, 'random_state': 1}),
            'rfc-prog33-7features'  : RandomForestClassifier(**{'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 1}),
            'rfc-prog33-10features' : RandomForestClassifier(**{'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 1}),
            # Response to Evo Model (Progression)
            'evo-knnr25-9featPROG'  : KNeighborsClassifier(**{'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'uniform'}),
            'evo-knnp50-6featPROG'  : KNeighborsClassifier(**{'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}),
            }


# specify training data (lesions that received doxorubicin monotherapy)
doxinds = baseline['ARM']==0
trainingFeatures = df[features[feature_choice]][doxinds]
trainingTargets = outcomes[doxinds]

# specify testing data (lesions that received TH-302 plus doxorubicin)
evoinds = baseline['ARM']==1
testingFeatures = df[features[feature_choice]][evoinds]
testingTargets = outcomes[evoinds]

if scaleFlag:
    scaler = preprocessing.StandardScaler()
    trainingFeatures = scaler.fit_transform(trainingFeatures)
    testingFeatures = scaler.fit_transform(testingFeatures)

# instantiate the model object
dox_model = models[feature_choice]
dox_model.fit(trainingFeatures,trainingTargets)

same_predictions = dox_model.predict(trainingFeatures)
opp_predictions = dox_model.predict(testingFeatures)  # testing model from one arm on lesions from other arm

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
# 

if 'prog' in feature_choice:
    outcome_of_interest = 'Progression'
    responding_lesions = np.where(np.logical_and(opp_predictions,~testingTargets))[0]
    non_responding_lesions = np.where(testingTargets)[0]
    evo_model_lesions = np.sort(np.concatenate((responding_lesions, non_responding_lesions)))

else:
    outcome_of_interest = 'Response'
    responding_lesions = np.where(np.logical_and(~opp_predictions,testingTargets))[0]
    non_responding_lesions = np.where(~testingTargets)[0]
    evo_model_lesions = np.sort(np.concatenate((responding_lesions, non_responding_lesions)))

# plotting parameters
plt.rcParams.update({
    "lines.color": "black",
    "patch.edgecolor": "black",
    "text.color": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "lightgray",
    "figure.facecolor": "white",
    "figure.edgecolor": "black",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "black",
    "font.size": 18,  # Increase font size
    "axes.titlesize": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        wrapped_text = '\n'.join(textwrap.wrap(text, width=width, break_long_words=break_long_words))
        labels.append(wrapped_text)
    ax.set_xticklabels(labels, rotation=0)

    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        wrapped_text = '\n'.join(textwrap.wrap(text, width=width, break_long_words=break_long_words))
        labels.append(wrapped_text)
    ax.set_yticklabels(labels, rotation=0)


training_data = [[np.sum(np.logical_and(same_predictions,trainingTargets)),np.sum(np.logical_and(~same_predictions,trainingTargets))],
                 [np.sum(np.logical_and(same_predictions,~trainingTargets)),np.sum(np.logical_and(~same_predictions,~trainingTargets))]]
testing_data = [[np.sum(np.logical_and(opp_predictions,testingTargets)),np.sum(np.logical_and(~opp_predictions,testingTargets))],
                [np.sum(np.logical_and(opp_predictions,~testingTargets)),np.sum(np.logical_and(~opp_predictions,~testingTargets))]]


# Create DataFrames for better visualization
training_df = pd.DataFrame(training_data, index=['Definitive '+outcome_of_interest, 'NOT Definitive '+outcome_of_interest], columns=['Definitive '+outcome_of_interest, 'NOT Definitive '+outcome_of_interest])
testing_df = pd.DataFrame(testing_data, index=['Definitive '+outcome_of_interest, 'NOT Definitive '+outcome_of_interest], columns=['Definitive '+outcome_of_interest, 'NOT Definitive '+outcome_of_interest])

# Plot the confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(training_df, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Dox Model Predictions on Dox Lesions')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
sns.heatmap(testing_df, annot=True, fmt='d', cmap='Greens', ax=axes[1])

axes[1].set_title('Dox Model Predictions on Dox+Evo Lesions')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].yaxis.set_tick_params(rotation=0)

# Apply wrapping to the axes labels
wrap_labels(axes[0], 10)
wrap_labels(axes[1], 10)

plt.tight_layout()
plt.show()

# %% EVO VALIDATION

# indicate model of interest for Evo arm
feature_choice = 'evo-knnp50-6featPROG'
df = features_volcorr.copy()
scaleFlag = False
hf_thresh = 0.3
Tr = 50
outcomes = deltaV_perc < +Tr


# specify the secondary training data (lesions that received TH-302 plus doxorubicin)
trainingEvoFeatures = df[features[feature_choice]].iloc[evoinds.index[evoinds][evo_model_lesions]]
trainingEvoTargets = outcomes[evoinds.index[evoinds][evo_model_lesions]]

evo_info = baseline.iloc[evoinds.index[evoinds][evo_model_lesions]]
evo_info.insert(2,'Response(PROG)',trainingEvoTargets)

# id patients with a mixed response (i.e. definitive and non-definitive lesions)
patients,numlesions = np.unique(evo_info.USUBJID,return_counts=True)
mixed_response = evo_info.copy()[evo_info.USUBJID.isin(patients[numlesions > 1])]
mixed_response.pop('ARM')
mixed_response = mixed_response.groupby('USUBJID').apply(lambda x: np.unique(x)>1)
mixed_response = evo_info.groupby('USUBJID')['Response(PROG)'].apply(lambda x: x.nunique() > 1)


# specify validation data (primary tumors with HFX data)
validationFeatures = pmh_sub[features[feature_choice]]
validationTargets = (hfx_data['HFX'] > hf_thresh).values

if scaleFlag:
    scaler = preprocessing.StandardScaler()
    trainingEvoFeatures = scaler.fit_transform(trainingFeatures)
    validationFeatures = scaler.fit_transform(pmh_sub[features[feature_choice]])

# instantiate the model object
evo_model = models[feature_choice]
evo_model.fit(trainingEvoFeatures,trainingEvoTargets)

same_predictions = evo_model.predict(trainingEvoFeatures)
opp_predictions = evo_model.predict(validationFeatures)  # testing model from one arm on lesions from other arm

# Report
print('Evo Model on Evo Lesions')
print('Predicted definitive and was definitive ',np.sum(np.logical_and(same_predictions,trainingEvoTargets)))
print('Predicted definitive and NOT definitive',np.sum(np.logical_and(same_predictions,~trainingEvoTargets)))
print('Predicted NOT definitive and definitive',np.sum(np.logical_and(~same_predictions,trainingEvoTargets)))
print('Predicted NOT definitive and NOT definitive',np.sum(np.logical_and(~same_predictions,~trainingEvoTargets)))
print('--------------------')
print('Evo Model on PMH Data')
print('Predicted definitive and was definitive ',np.sum(np.logical_and(opp_predictions,validationTargets)))
print('Predicted definitive and NOT definitive',np.sum(np.logical_and(opp_predictions,~validationTargets)))
print('Predicted NOT definitive and definitive',np.sum(np.logical_and(~opp_predictions,validationTargets)))
print('Predicted NOT definitive and NOT definitive',np.sum(np.logical_and(~opp_predictions,~validationTargets)))

training_data = [[np.sum(np.logical_and(same_predictions,trainingEvoTargets)),np.sum(np.logical_and(~same_predictions,trainingEvoTargets))],
                 [np.sum(np.logical_and(same_predictions,~trainingEvoTargets)),np.sum(np.logical_and(~same_predictions,~trainingEvoTargets))]]
testing_data = [[np.sum(np.logical_and(opp_predictions,validationTargets)),np.sum(np.logical_and(~opp_predictions,validationTargets))],
                [np.sum(np.logical_and(opp_predictions,~validationTargets)),np.sum(np.logical_and(~opp_predictions,~validationTargets))]]


# Create DataFrames for better visualization
training_df = pd.DataFrame(training_data, index=['Definitive '+outcome_of_interest, 'NOT Definitive '+outcome_of_interest], columns=['Definitive '+outcome_of_interest, 'NOT Definitive '+outcome_of_interest])
testing_df = pd.DataFrame(testing_data, index=['Hypoxic', 'NOT Hypoxic'], columns=['Hypoxic', 'NOT Hypoxic'])

# Plot the confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(training_df, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Evo Model Predictions on Evo Lesions')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
sns.heatmap(testing_df, annot=True, fmt='d', cmap='Greens', ax=axes[1])

axes[1].set_title('Evo Model Predictions on PMH Data')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].yaxis.set_tick_params(rotation=0)

# Apply wrapping to the axes labels
wrap_labels(axes[0], 10)
wrap_labels(axes[1], 10)

plt.tight_layout()
plt.show()

# %% PERFORMANCE METRICS AND PLOTS

# reset the plotting parameters
plt.rcParams.update(plt.rcParamsDefault)

# Calculate performance metrics
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, y_prob)
    auprc = average_precision_score(y, y_prob)
    return auroc, auprc, y_pred, y_prob

# Plot ROC and Precision-Recall curves
def plot_curves(y, y_prob, title_suffix):
    fpr, tpr, _ = roc_curve(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic ' + title_suffix)
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve ' + title_suffix)
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()

# Calculate and plot metrics for Dox model on Dox lesions
auroc_dox, auprc_dox, y_pred_dox, y_prob_dox = calculate_metrics(dox_model, trainingFeatures, trainingTargets)
print(f'Dox Model on Dox Lesions - AUROC: {auroc_dox:.2f}, AUPRC: {auprc_dox:.2f}')
plot_curves(trainingTargets, y_prob_dox, '(Dox Model on Dox Lesions)')

# Calculate and plot metrics for Dox model on Dox+Evo lesions
auroc_evo, auprc_evo, y_pred_evo, y_prob_evo = calculate_metrics(dox_model, testingFeatures, testingTargets)
print(f'Dox Model on Dox+Evo Lesions - AUROC: {auroc_evo:.2f}, AUPRC: {auprc_evo:.2f}')
plot_curves(testingTargets, y_prob_evo, '(Dox Model on Dox+Evo Lesions)')

# Calculate and plot metrics for Evo model on Evo lesions
auroc_evo_train, auprc_evo_train, y_pred_evo_train, y_prob_evo_train = calculate_metrics(evo_model, trainingEvoFeatures, trainingEvoTargets)
print(f'Evo Model on Evo Lesions - AUROC: {auroc_evo_train:.2f}, AUPRC: {auprc_evo_train:.2f}')
plot_curves(trainingEvoTargets, y_prob_evo_train, '(Evo Model on Evo Lesions)')

# Calculate and plot metrics for Evo model on PMH data
auroc_pmh, auprc_pmh, y_pred_pmh, y_prob_pmh = calculate_metrics(evo_model, validationFeatures, validationTargets)
print(f'Evo Model on PMH Data - AUROC: {auroc_pmh:.2f}, AUPRC: {auprc_pmh:.2f}')
plot_curves(validationTargets, y_prob_pmh, '(Evo Model on PMH Data)')

# %% TRAINING EVO MODEL

# import warnings
# warnings.filterwarnings("ignore")
# warnings.simplefilter(action='ignore', category=FutureWarning)

# import sys
# import warnings

# # Ignore all warnings
# warnings.filterwarnings("ignore")

# df = features_volcorr.copy()

# # modifiable arguments (preset for analysis)
# Tr = [-50,-33,-25,+25,+33,+50]
# arm = 1
# max_features = 10
# splits = 10
# index_choice = 'evo' #'subset+'
# model_choice = 'randomforest' # 'logistic', 'kNN', 'naivebayes', 'svm', 'randomforest'
# dat = 'imaging'

# # Open a file to write the log
# log_file = open('/Users/caryngeady/Documents/GitHub/Mixed-Response-Work/'+model_choice+'.log', 'w')
# original_stdout = sys.stdout
# # Redirect stdout to the log file
# sys.stdout = log_file

# for threshold in Tr:
#     # preset parameters
#     if threshold < 0:
#         deltaVbin = deltaV_perc < threshold
#     else:
#         deltaVbin = deltaV_perc > threshold

#     print('--------------------')
#     print('Tr: {} %'.format(threshold))

#     target_choice = deltaVbin         # deltaVbin, deltaVcat, delatV_perc

#     print('Model used: {}'.format(model_choice))
#     if arm == 0:
#         print('Trial Arm: Doxorubicin monotherapy')
#     else:
#         print('Trial Arm: TH-302 plus Doxorubicin')
#     print('--------------------')


#     indices = {
#                 # 'subset'  : np.where(np.logical_and((baselineVolume[inds_noOutliers]<vol_high),(baselineVolume[inds_noOutliers]>vol_low)))[0],
#                 # 'subset+' : np.where(np.logical_and(np.logical_and((baselineVolume<vol_high),(baselineVolume>vol_low)),(baseline['ARM']==arm)))[0],
#                 'arm'     : np.where(baseline['ARM']==arm)[0],
#                 'evo'     : np.where(baseline['ARM']==arm)[0][evo_model_lesions],
#                 'all'     : range(len(target_choice))
#             }

#     models = {
#                 'logistic'   : [LogisticRegression(random_state=1),{
#                                                                     'penalty'  : ['l1', 'l2', 'elasticnet', None],
#                                                                     'solver'   : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
#                                                                     'tol'      : [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
#                                                                     'max_iter' : [50,100,150,200]
#                                                                     }],
#                 'naivebayes' : [GaussianNB(),{}],
#                 'kNN'        : [KNeighborsClassifier(),{
#                                                             'n_neighbors' : [3,5,7,9,11],
#                                                             'weights'     : ['uniform','distance'],
#                                                             'metric'      : ['euclidean','manhattan','minkowski']
#                                                             }],
#                 'svm'        : [SVC(random_state=1),{
#                                                         'C'         : [0.1,1,10,100,1000],
#                                                         'kernel'    : ['linear','poly','rbf','sigmoid'],
#                                                         'degree'    : [2,3,4,5,6],
#                                                         'gamma'     : ['scale','auto']
#                                                         }],
#                 'randomforest' : [RandomForestClassifier(random_state=1),{
#                                                                         'n_estimators' : [100,200,300,400,500],
#                                                                         'criterion'    : ['gini','entropy'],
#                                                                         'max_depth'    : [10,20,30,40,50],
#                                                                         'max_features' : ['auto','sqrt','log2']
#                                                                         }]
#                 # insert additional models with relevant hyperparameters here
#             }

#     # initialize for results table
#     selected_features = []
#     auroc = []
#     auprc = []
#     neg_log_loss = []
#     mcc_lst = []
#     wilcoxp = []
#     wilcoxp.append(np.nan)
#     fdr = []


#     for i in range(1,max_features+1):

#         # features and outcomes
#         predictors = df.iloc[indices[index_choice],:].reset_index(drop=True)
#         # remove any data points that may have missing values (sometimes core too small and nans live there instead of radiomic features)
#         predInds = predictors[predictors.isnull().any(axis=1)].index
#         targInds = predictors.index
#         harmInds = [i for i in targInds if i not in predInds]
#         # consolidate
#         predictors = predictors.loc[harmInds,:]
#         targets = target_choice[indices[index_choice]][harmInds] 

#         # feature selection
#         predictors.pop('original_shape_VoxelVolume')
#         fs = SelectKBest(score_func=f_classif, k=i-1)
#         mask = fs.fit(predictors, targets).get_support()
#         predictors = predictors[predictors.columns[mask]]
#         predictors['original_shape_VoxelVolume'] = df.original_shape_VoxelVolume.iloc[indices[index_choice]].reset_index(drop=True)
        
            
#         if i == 1:
#             print('Progressor Fraction: %.3f' % (sum(targets==1)/len(targets)))
#             print('Stable Fraction: %.3f' % (sum(targets==0)/len(targets)))
#             print('Total Lesions: %f' % (len(targets)))
#             print('--------------------')    
            
#         selected_features.append(predictors.columns)
#         print('Features selected({}): {}'.format(len(predictors.columns),list(predictors.columns)))

#         # modeling
#         model = models[model_choice][0]
#         params = models[model_choice][1]
        
#         if model_choice != 'naivebayes':
#             gs = GridSearchCV(model, params, cv=5, scoring='matthews_corrcoef',n_jobs=1)
#             gs.fit(predictors,targets)
#             print('Best Params: ',gs.best_params_)
#             model = gs.best_estimator_
        
        
#         scaler = preprocessing.StandardScaler()    
#         clf = make_pipeline(scaler, model)
#         cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=10, random_state=1)

    
#         negLL = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='neg_log_loss', cv=cv, n_jobs=1)
#         neg_log_loss.append(np.mean(negLL))
        
#         auc = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='roc_auc_ovo', cv=cv, n_jobs=1)    
#         auc_lower,auc_upper = f.calc_conf_intervals(auc)
#         print('Average AUROC: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(auc),auc_lower,auc_upper))
#         auroc.append(np.mean(auc))
        
#         aps = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='average_precision', cv=cv, n_jobs=1)    
#         aps_lower,aps_upper = f.calc_conf_intervals(aps)
#         print('Average Precision: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(aps),aps_lower,aps_upper))
#         auprc.append(np.mean(aps))
        
#         mcc = cross_val_score(clf, predictors.values, targets.astype('int'), scoring='matthews_corrcoef', cv=cv, n_jobs=1)    
#         mcc_lower,mcc_upper = f.calc_conf_intervals(mcc)
#         print('Average MCC: {:.2f} (95% conf. int. [{:.2f},{:.2f}])'.format(np.mean(mcc),mcc_lower,mcc_upper))
#         mcc_lst.append(np.mean(mcc))
        
#         print('--------------------')
#         print('Significance Testing')
#         print('--------------------')
        
#         if i == 1:
#             avg_precision = aps
        
#         if i > 1:
#             print('Wilcoxon p-value: ',wilcoxon(avg_precision,aps)[1])
#             wilcoxp.append(wilcoxon(avg_precision,aps)[1])
        
#         with parallel_backend('loky', n_jobs=4):
#             scores_precision, perm_scores_precision, pvalue_precision = permutation_test_score(
#                 clf, predictors.values, targets.astype('int'), scoring="matthews_corrcoef", cv=cv, n_permutations=1000,n_jobs=4
#             )
#         print('p-value: {:.3f}'.format(pvalue_precision))
#         print('FDR-corrected p-value: {:.3f}'.format(mtc(np.repeat(pvalue_precision,10))[1][0]))
#         fdr.append(mtc(np.repeat(pvalue_precision,10))[1][0])

#         print('--------------------')
        
#     results_df = pd.DataFrame([auroc,auprc,mcc_lst,neg_log_loss,wilcoxp,fdr]).T
#     results_df.columns = ['AUROC','AUPRC','MCC','NegLogLoss','Wilcoxon P-Value','FDR']
#     results_df.index = range(1,max_features+1)
#     print(results_df)


# # Restore stdout to its original state
# sys.stdout = original_stdout
# # Close the log file
# log_file.close()
# print('Cell finished running')
# # %%
