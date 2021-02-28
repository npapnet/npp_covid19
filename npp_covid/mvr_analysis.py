#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from npp_covid.heroku import hag_create_df, prepare_cases

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

import  statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def visualize_correlations(data, figsize=(16,10), fontsize = 14):
    """produces a tringular heatmap with color coded 

    Args:
        data ([type]): [description]
    """    
    mask = np.zeros_like(data.corr())
    triangle_indices = np.triu_indices_from(mask)
    mask[triangle_indices]=1
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), mask =mask, annot=True, annot_kws={'size':14})
    sns.set_style('white')
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)

class ModelContainer():
    def __init__(self, features, target):
        self.features = features
        self.target = target
    

    def split_data(self, random_state=None, test_size=0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.features, self.target, 
            test_size=test_size, random_state=random_state)

    def prepare_model(self):
        self.regr = LinearRegression().fit(self.X_train, self.Y_train)
        # fitted_vals= self.regr.predict(self.X_train)
        return self.regr
    
    def prepare_OLSmodel(self):
        X_incl_const = sm.add_constant(self.X_train)
        self.OLS_results = sm.OLS(self.Y_train, X_incl_const).fit()
        return self.OLS_results

    def p_values(self):
        results = self.OLS_results

        #results params
        return pd.DataFrame({
                'coef':results.params,
                'p-value': np.round(results.pvalues,3)
            })

    def variance_inflation_factors(self):
        # from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_incl_const = sm.add_constant(self.X_train)
        vifs = []
        for i in range(X_incl_const.shape[1]):
            vifs.append( variance_inflation_factor(exog=X_incl_const.values, exog_idx=i))

        dvifs = pd.DataFrame({'coef_name':X_incl_const.columns, 'vif':np.round(vifs, decimals=2)})
        return dvifs

    def gen_paiplot(self, fname=None):    # %% save to disk pairplot
        sns_plot = sns.pairplot(self.features,  kind ='reg',plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.4}})
        sns_plot.savefig(fname)
    
    def plot_residuals(self):
        plt.figure()
        results = self.OLS_results
        tmp = (results.resid/self.Y_train).sort_index()
        tmp.plot()


        # graph of actual vs predicted prices
        Y_train= self.Y_train
        corr = Y_train.corr(results.fittedvalues)
        plt.figure()
        print("Corr between actual and predicted: {:.2f}".format(corr))
        plt.scatter(x=Y_train, y=results.fittedvalues, c='navy', alpha=0.6)
        plt.plot(Y_train, Y_train, c='cyan') # strain line
        plt.xlabel('actual  $y_{i}$', fontsize = 14)
        plt.ylabel('Predicted  $\hat{y}_{i}$', fontsize = 14)
        plt.title(f'Actual vs preidcted  : $y_i - \hat y_i$ (Corr: {corr:.2f})', fontsize=17)
        # plt.plot(Y_train, results.resid, '.')

        ## residulavs vs. Predicted values
        plt.figure()
        plt.scatter(x=results.fittedvalues, y= results.resid, c='blue', alpha=0.6)
        plt.xlabel('Predicted  $\hat{y}_{i}$', fontsize = 14)
        plt.ylabel('Residuals', fontsize = 14)
        plt.title(f'Residuals vs Fitted Values', fontsize=17)

        # distributon of residuals  - checking for nomrality
        plt.figure()
        resid_mean = round (results.resid.mean(),3)
        resid_skew = round (results.resid.skew(),3)
        sns.distplot(results.resid, color = 'navy')
        plt.title(f'price model: residuals Skew:({resid_skew}) Mean: ({resid_mean})')



    def plot_correlations(self):
        visualize_correlations(self.features)

    def mc_score(self, n = 100, random_state=None):
        """Perform monte carlo analysis for scoring of the data

        TODO: keep coefficients and scoring in order to perform analysis

        Args:
            n (int, optional): [description]. Defaults to 100.
            random_state ([type], optional): [description]. Defaults to None.
        """        
        def perform_single_sim(features, target):
            X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2)#, random_state=10)
            regr = LinearRegression().fit(X_train, Y_train)
            fitted_vals= regr.predict(X_train)

            # MSE = ((fitted_vals-target)**2).mean()[0]
            # RMSE = np.sqrt(MSE)
            # MSE = mean_squared_error(target, fitted_vals)
            # RMSE = np.sqrt(MSE)
            # print(MSE)
            # print(RMSE)
            return (regr.score(X_train, Y_train),regr.score(X_test, Y_test))
        res = np.zeros([100,2])      
        for i in range(n):
            res[i,:] = perform_single_sim(self.features, self.target)
        
        
        return pd.DataFrame(res, columns=['train', 'test'])


# %%
