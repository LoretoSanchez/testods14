import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

SEED = 123
N_CV = 5

PARAM_GRID = {
    "n_estimators": [64, 86, 128, 186, 256],
    "max_depth": [2, 3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3], 
    "subsample": [0.3, 0.5, 0.6],
    "colsample_bytree": [0.1, 0.3, 0.5, 0.8],
    "gamma": [0, 0.05, 0.1],
    "reg_alpha": [0, 0.5, 1, 1.5],
}

def mape(y_true, y_pred):
    out = 100 * np.abs(y_true - y_pred) / y_true
    # out = np.round(100 * out.mean(), decimals=4)
    return out.mean()

mape_scorer = make_scorer(mape, greater_is_better=False)

def train_model(X, y, date):
    X_train = X.loc[X.index < date]
    y_train = y.loc[y.index < date]
    X_test = X.loc[X.index >= date]
    y_test = y.loc[y.index >= date]
    
    model = XGBRegressor(booster='gbtree', verbosity=1, random_state=SEED, n_jobs=-1, objective ='reg:squarederror')
    cv = TimeSeriesSplit(N_CV)
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_GRID,
        n_iter=500,
        n_jobs=-1, 
        cv=cv, 
        random_state=SEED,
        verbose=1,
        scoring="neg_mean_squared_error" #   # 'neg_mean_absolute_error' # 'r2'mape_scorer  # 
    )
    grid_search.fit(X_train, y_train)
    
    y_test_pred = grid_search.predict(X_test)
    y_train_pred = grid_search.predict(X_train)
    
    y_train_pred = pd.Series(y_train_pred, index=y_train.index)
    y_test_pred = pd.Series(y_test_pred, index=y_test.index)
    
    return y_train, y_train_pred, y_test, y_test_pred, grid_search

def print_metrics(y1, y2):
    print(f"mape: {mape(y1, y2)}")
    print(f"r2_score: {r2_score(y1, y2)}")
    print(f"mean_squared_error: {mean_squared_error(y1, y2 )}")
    print(f"root_mean_squared_error: {np.sqrt(mean_squared_error(y1, y2))}")

def plot_ts(y1, y2):
    plt.plot(y1, '-', label='true')
    plt.plot(y2, '-', label='predicted')
    plt.legend()
    plt.grid()
    

def get_feat_importance(model, td):
    #importances = pd.DataFrame({'feature': feats, 'importance':model.best_estimator_.feature_importances_})
    #importances = importances.merge(td['description'], how='left', left_on='feature', right_index=True)
    #importances['name'] = importances['feature'].astype(str) + ' - ' + importances['description'].astype(str)
    #importances.sort_values('importance', ascending=True, inplace=True)
    #importances.iloc[-20:,:].plot.barh(x='name', y='importance')
    
    imp = model.best_estimator_.get_booster().get_score(importance_type="gain")
    importances = pd.DataFrame.from_dict(imp, orient='index').reset_index()
    importances.columns = ["feature", "importance"]
    importances = importances.merge(td[['real_tag', 'name']], how='left', left_on='feature', right_index=True)
    importances['vis_name'] = importances['real_tag'].astype(str) + ' - ' + importances['name'].astype(str)
    importances.sort_values('importance', ascending=True, inplace=True)
    importances.iloc[-20:,:].plot.barh(x='vis_name', y='importance')

    return importances.feature.to_list()[::-1]


def plot_corr(data):
    plt.figure(figsize=(12, 12))
    sns.heatmap(data, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True))

    
def plot_heatmap(data):
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(10, 220, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(data, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f",)