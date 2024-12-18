#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

# if 'Unnamed: 0' in df.columns:
#     df = df.drop('Unnamed: 0',axis=1)

# if 'Location' in df.columns:
#     print('Location column already exists')  
#     # move 'Location column to index 1
#     cols = list(df.columns)
#     cols.insert(1, cols.pop(cols.index('Location')))
#     df = df.loc[:, cols]
# # lesion_names = [i.split('/')[-1].split('-')[0] for i in df.Mask]
# # df['Location'] = lesion_names
# df.to_csv('data/radiomics-peritumoral-allSTS.csv',index=False)

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

path_to_radiomics = 'data/radiomics.csv'  
# path_to_radiomics = 'data/radiomics-peritumoral-allSTS.csv'
path_to_clinical = 'data/baseline.csv' 

radiomics_bl,radiomics_c2,baseline = f.load_rfs(path_to_radiomics,path_to_clinical,features_to_remove)
del(path_to_radiomics,path_to_clinical)

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

inds_noOutliers,baselineVolume,deltaV_abs,deltaV_perc = f.volume_change_filtering(radiomics_bl,radiomics_c2,3)

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

featuresOfInterest = ['original_firstorder_Mean','original_firstorder_Mean_lesion core',
                      'original_firstorder_Mean_interior rim','original_firstorder_Mean_exterior rim',
                     'original_firstorder_Entropy','original_firstorder_Entropy_lesion core',
                      'original_firstorder_Entropy_interior rim','original_firstorder_Entropy_exterior rim']
response_features_all,simpleMetrics,volMetrics = f.simple_metric_comparison(radiomics_bl,baseline,deltaV_perc,inds_noOutliers,featuresOfInterest,tr=50)


# %% VOLUME COMPARISON

pairs,my_pal =f.define_pairs('volume')

hue_plot_params = {
                    'data': volMetrics,
                    'x': 'Arm',           # change to Response if not comparing arms
                    'y': 'Volume (cc)',
                    'hue' : 'Response',
                    "palette": my_pal,
                    "width" : 0.5,
                    'boxprops':{'edgecolor':'white'},
                    'medianprops':{'color':'white'},
                    'whiskerprops':{'color':'white'},
                    'capprops':{'color':'white'}
                }
fig = plt.figure(figsize=(5,3),dpi=500, facecolor = 'black')

ax = sns.boxplot(**hue_plot_params)
f.wrap_labels(ax, 12)
plt.legend(bbox_to_anchor=(1.05, 0.55), loc='best', borderaxespad=0)
sns.despine(ax=ax, offset=10, trim=True)

annotator = Annotator(ax, pairs, 
                      pvalue_thresholds=[[1e-4, '****'], [1e-3, '***'], [1e-2, '**'], [0.05, '*'], [0.15, '.'], [1, 'ns']],
                      **hue_plot_params)
annotator.configure(pvalue_thresholds=[[1e-4, '****'], [1e-3, '***'], [1e-2, '**'], [0.05, '*'], [0.15, '.'], [1, 'ns']],
                        test="Mann-Whitney",  # t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal
                        loc = 'inside',
                        line_height = 0.01,
                        line_offset_to_group = 0,
                        line_offset = 0,
                        text_offset = 0,
                        color = 'white',
                        comparisons_correction="fdr_bh"
                        ).apply_and_annotate()
plt.ylabel(r'$V_{t=0} (cc)$')
plt.savefig('results/volume-comparison.png',bbox_inches='tight',dpi=100)



# %% FEATURE REDUCTION

df_cluster = f.cluster_red(radiomics_bl.iloc[:,4:])

# assess correlation of all features with volume
corrVals = df_cluster.corrwith(baselineVolume,method='spearman')
# assess correlation between volume-independent features
df = df_cluster[corrVals.index[corrVals<=0.2]]

df['volume'] = baselineVolume.values
df = df.dropna(axis = 1, how = 'all')



# %% ANALYSIS

# modifiable arguments (preset for analysis)
Tr = 50
arm = 0
max_features = 3
splits = 10
vol_low = 0
vol_high = 1605 # in cc

# preset parameters
deltaVcat,deltaVbin = f.define_outcomes(deltaV_perc,Tr)
print('--------------------')
print('Tr: {} %'.format(Tr))

index_choice = 'subset+' #'subset+'
model_choice = 'logistic'
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
            'subset'  : np.where(np.logical_and((baselineVolume[inds_noOutliers]<vol_high),(baselineVolume[inds_noOutliers]>vol_low)))[0],
            'subset+' : np.where(np.logical_and(np.logical_and((baselineVolume<vol_high),(baselineVolume>vol_low)),(baseline['ARM']==arm)))[0],
            'arm'     : np.where(baseline['ARM']==arm)[0],
            'all'     : range(len(target_choice))
          }

models = {
            'logistic'   : [LogisticRegression(random_state=1),{
                                                                'penalty'  : ['l1', 'l2', 'elasticnet', None],
                                                                'solver'   : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                                                                'tol'      : [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
                                                                'max_iter' : [50,100,150,200]
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
    predictors.pop('volume')
    fs = SelectKBest(score_func=f_classif, k=i-1)
    mask = fs.fit(predictors, targets).get_support()
    predictors = predictors[predictors.columns[mask]]
    predictors['volume'] = df.volume.iloc[indices[index_choice]].reset_index(drop=True)
    
        
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
    
    # scores_precision, perm_scores_precision, pvalue_precision = permutation_test_score(
    #     clf, predictors.values, targets.astype('int'), scoring="matthews_corrcoef", cv=cv, n_permutations=1000
    # )
    # print('p-value: {:.3f}'.format(pvalue_precision))
    # print('FDR-corrected p-value: {:.3f}'.format(mtc(np.repeat(pvalue_precision,5))[1][0]))
    # fdr.append(mtc(np.repeat(pvalue_precision,10))[1][0])

    print('--------------------')
    
results_df = pd.DataFrame([auroc,auprc,mcc_lst,neg_log_loss,wilcoxp,fdr]).T
results_df.columns = ['AUROC','AUPRC','MCC','NegLogLoss','Wilcoxon P-Value','FDR']
results_df.index = range(1,max_features+1)
print(results_df)

# %% EXTRA PLOTS

test = pd.DataFrame(data=[avg_precision,aps]).T
test.columns=['Volume','Volume+Radiomics']

hue_plot_params = {
                    'data': test,
                    # 'x': 'Model',           # change to Response if not comparing arms
                    # 'y': 'AUPRC',
                    # 'hue' : 'Response',
                    "palette": "bright",
                    "width" : 0.5,
                    'boxprops':{'edgecolor':'white'},
                    'medianprops':{'color':'white'},
                    'whiskerprops':{'color':'white'},
                    'capprops':{'color':'white'}
                }
fig = plt.figure(figsize=(5,3),dpi=500, facecolor = 'black')

ax = sns.boxplot(**hue_plot_params)
plt.ylabel('AUPRC')
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

# %% plotting

# roc curve and roc auc on an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
 
# plot no skill and model roc curves
def plot_roc_curve(test_y, naive_probs, model_probs):
 # plot naive skill roc curve
 fpr, tpr, _ = roc_curve(test_y, naive_probs)
 pyplot.plot(fpr, tpr, linestyle='--', linewidth=4,color='red',label='No Skill')
 # plot model roc curve
 fpr, tpr, _ = roc_curve(test_y, model_probs)
 pyplot.plot(fpr, tpr, marker='.', linewidth=4,markersize=4,label='Logistic',color='blue')
 # axis labels
 pyplot.xlabel('False Positive Rate')
 pyplot.ylabel('True Positive Rate')
 # show the legend
 pyplot.legend()
 # show the plot
 plt.savefig('results/auroc.png',bbox_inches='tight',dpi=250)
 pyplot.show()
 
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.99, 0.01], random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill model, stratified random class predictions
model = DummyClassifier(strategy='stratified')
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
naive_probs = yhat[:, 1]
# calculate roc auc
roc_auc = roc_auc_score(testy, naive_probs)
print('No Skill ROC AUC %.3f' % roc_auc)
# skilled model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
model_probs = yhat[:, 1]
# calculate roc auc
roc_auc = roc_auc_score(testy, model_probs)
print('Logistic ROC AUC %.3f' % roc_auc)
# plot roc curves
plot_roc_curve(testy, naive_probs, model_probs)


# %%

# pr curve and pr auc on an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot
 
# plot no skill and model precision-recall curves
def plot_pr_curve(test_y, model_probs):
 # calculate the no skill line as the proportion of the positive class
 no_skill = len(test_y[test_y==1]) / len(test_y)
 # plot the no skill precision-recall curve
 pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', linewidth=4,color='red',label='No Skill')
 # plot model precision-recall curve
 precision, recall, _ = precision_recall_curve(testy, model_probs)
 pyplot.plot(recall, precision, marker='.', linewidth=4,markersize=4,label='Logistic',color='blue')
 # axis labels
 pyplot.xlabel('Recall')
 pyplot.ylabel('Precision')
 # show the legend
 pyplot.legend()
 # show the plot
 plt.savefig('results/auprc.png',bbox_inches='tight',dpi=250)
 pyplot.show()
 
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=1)
# split into train/test sets with same class ratio
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# no skill model, stratified random class predictions
model = DummyClassifier(strategy='stratified')
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
naive_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(testy, naive_probs)
auc_score = auc(recall, precision)
print('No Skill PR AUC: %.3f' % auc_score)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
yhat = model.predict_proba(testX)
model_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(testy, model_probs)
auc_score = auc(recall, precision)
print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(testy, model_probs)

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
feature_choice = 'evo-low'

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
                          'volume']
            }

params = {
            'dox-all'  : {'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1},
            'dox-low'  : {'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1},
            'evo-all'  : {'max_iter': 50, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001, 'random_state': 1},
            'evo-low'  : {'max_iter': 100, 'penalty': 'l2', 'solver': 'sag', 'tol': 0.0001, 'random_state': 1},
            'evo-high' : {'max_iter': 50, 'penalty': 'l2', 'solver': 'newton-cg', 'tol': 0.0001, 'random_state': 1}
            }

# specify training data
arm = 0
vol_low = 0#16.05
vol_high = 16.05 # in cc
inds = np.where(np.logical_and(np.logical_and((baselineVolume<vol_high),(baselineVolume>vol_low)),(baseline['ARM']==arm)))[0]

trainingFeatures,trainingTargets = isolateData(df,inds,features[feature_choice])

# specify testing data
arm = 1
vol_low = 0#16.05
vol_high = 16.05 # in cc
inds = np.where(np.logical_and(np.logical_and((baselineVolume<vol_high),(baselineVolume>vol_low)),(baseline['ARM']==arm)))[0]

testingFeatures,testingTargets = isolateData(df,inds,features[feature_choice])

# instantiate the model object
model = LogisticRegression(**params[feature_choice])

model.fit(trainingFeatures,trainingTargets)

same_predictions = model.predict(trainingFeatures)
opp_predictions = model.predict(testingFeatures)  # testing model from one arm on lesions from other arm

print(np.sum(np.logical_and(~same_predictions,trainingTargets)))
print(np.sum(np.logical_and(~opp_predictions,testingTargets)))

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
