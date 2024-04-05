#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Change working directory to be whatever directory this script is in
import os
os.chdir(os.path.dirname(__file__))

# Import required libraries :
import scripts.functionals as f
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib
# from sklearn.preprocessing import StandardScaler
import seaborn as sns
# import statannotations
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
    • radiomics_bl       : radiomics features from baseline
    • radiomics_c2       : radiomics features from first follow-up (cycle 2)
    
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
                    "width" : 0.6
                }
fig = plt.figure(figsize=(5,3),dpi=500, facecolor = 'white')

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
                        color = 'black',
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
max_features = 5
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
    
    scores_precision, perm_scores_precision, pvalue_precision = permutation_test_score(
        clf, predictors.values, targets.astype('int'), scoring="matthews_corrcoef", cv=cv, n_permutations=1000
    )
    print('p-value: {:.3f}'.format(pvalue_precision))
    print('FDR-corrected p-value: {:.3f}'.format(mtc(np.repeat(pvalue_precision,5))[1][0]))
    fdr.append(mtc(np.repeat(pvalue_precision,10))[1][0])

    print('--------------------')
    
results_df = pd.DataFrame([auroc,auprc,mcc_lst,neg_log_loss,wilcoxp,fdr]).T
results_df.columns = ['AUROC','AUPRC','MCC','NegLogLoss','Wilcoxon P-Value','FDR']
results_df.index = range(1,max_features+1)
print(results_df)


