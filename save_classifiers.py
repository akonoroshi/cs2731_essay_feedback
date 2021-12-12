from joblib import dump
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, DotProduct, WhiteKernel
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTENC

model = 'sbert'
model_suffix = ''
rubrics = ['d_Prompt', 'd_Thesis', 'd_Claims',
    'd_Evidence', 'd_Reasoning', 'd_Organization', 'd_Rebuttal', 'd_Precision', 'd_Fluency', 'd_Coventions']
train_X = np.load(f'./numpy_data/{model}{model_suffix}_train_X.npy')
val_X = np.load(f'./numpy_data/{model}{model_suffix}_val_X.npy')
test_X = np.load(f'./numpy_data/{model}{model_suffix}_test_X.npy')
#train_X = np.concatenate((train_X, val_X), axis=0)

for rubric in rubrics:
    train_y = np.load(f'./numpy_data/{rubric}_bin_train_y.npy')
    val_y = np.load(f'./numpy_data/{rubric}_bin_val_y.npy')
    test_y = np.load(f'./numpy_data/{rubric}_bin_test_y.npy')
    #train_y = np.concatenate((train_y, val_y), axis=0)

    sm = SMOTENC(categorical_features=list(range(11)), random_state=1, n_jobs=-1)
    train_X_sm, train_y_sm = sm.fit_resample(train_X, train_y)

    if rubric == 'd_Prompt':
        clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=800, ccp_alpha=0.01, min_samples_leaf=3, max_features=50)
    elif rubric == 'd_Thesis':
        #clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=400)
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=1000, early_stopping=True, categorical_features=list(range(11)), max_leaf_nodes=51, max_depth=20, n_iter_no_change=5)
    elif rubric == 'd_Claims':
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=1000, early_stopping=True, categorical_features=list(range(11)), max_depth=30, max_leaf_nodes=41)
    elif rubric == 'd_Reasoning':
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=1000, early_stopping=True, categorical_features=list(range(11)), max_leaf_nodes=40)
    elif rubric == 'd_Organization':
        clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=1150, ccp_alpha=0.01, min_samples_leaf=3, max_features=50, max_depth=30, criterion='gini', max_leaf_nodes=None, min_impurity_decrease=0, bootstrap=True)
    elif rubric == 'd_Rebuttal':
        clf = GaussianProcessClassifier(kernel=RationalQuadratic(), random_state=0, n_jobs=-1)
    elif rubric == 'd_Precision':
        clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=400, ccp_alpha=1e-4)
    elif rubric == 'd_Fluency':
        clf = GaussianProcessClassifier(kernel=DotProduct()+WhiteKernel(noise_level_bounds=(1e-12, 1e5)), random_state=0, n_jobs=-1)
    elif rubric == 'd_Coventions':
        clf = HistGradientBoostingClassifier(random_state=0, max_iter=1000, early_stopping=True, categorical_features=list(range(11)), max_leaf_nodes=45)
    else:
        clf = RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=500, max_depth=50)

    clf.fit(train_X_sm, train_y_sm)
    pred_y = clf.predict(test_X)
    print('Rubric:', rubric)
    print('Accuracy:', accuracy_score(test_y, pred_y))
    precision, recall, f1, _ = precision_recall_fscore_support(test_y, pred_y)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1, '\n')
    dump(clf, f'models/{rubric}.joblib')
