#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# from npp_covid.io.heroku import hag_create_df
# from npp_covid.misc.smooth_funcs import prepare_cases_gaussian

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
import seaborn as sns

import  statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from npp_covid.CFClass import PLT_FIGSIZE


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
    """This is a Monte Carlo model container"""

    def __init__(self, features, target, metadata:dict=None):
        self.features = features
        self.target = target
    

    def split_data(self, random_state=None, test_size=0.2):
        """splits data into different states

        ## Splitting of the dataset
        The features are broken up into train and split at a ratio of 8:2.
        - $X_{train}$ 
        - $y_{train}$
        - $X_{test}$ 
        - $y_{test}$

        the train data are used for the derivation of the constants of the model, while the tests are reserved for 


        Args:
            random_state ([type], optional): random seed. Defaults to None.
            test_size (float, optional): fraction of data to be assigned to test. Defaults to 0.2.
        """        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.features, self.target, 
            test_size=test_size, random_state=random_state)

    def prepare_model(self):
        """Prepares a linear regression model 

        Returns:
            [type]: [description]
        """
        self.regr = LinearRegression().fit(self.X_train, self.Y_train)
        # fitted_vals= self.regr.predict(self.X_train)
        return self.regr
    
    def prepare_OLSmodel(self):
        """Prepare an Ordinary Least Squares model with **statsmodels.api**

        This has additional information for 
        - residuals 
        - p-values

        
        that can be used

        Returns:
            [type]: OLS results from the fitting.
        """
        X_incl_const = sm.add_constant(self.X_train)
        self.OLS_results = sm.OLS(self.Y_train, X_incl_const).fit()
        return self.OLS_results

    def p_values(self):
        """returns the p values from the OLS model 
        (requires OLS Model)

        Returns:
            [type]: [description]
        """
        results = self.OLS_results

        #results params
        return pd.DataFrame({
                'coef':results.params,
                'p-value': np.round(results.pvalues,3)
            })

    def variance_inflation_factors(self):
        """Returns the Variance inflation factors for the feature set

        Returns:
            [type]: [description]
        """        
        # from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_incl_const = sm.add_constant(self.X_train)
        vifs = []
        for i in range(X_incl_const.shape[1]):
            vifs.append( variance_inflation_factor(exog=X_incl_const.values, exog_idx=i))

        dvifs = pd.DataFrame({'coef_name':X_incl_const.columns, 'vif':np.round(vifs, decimals=2)})
        return dvifs

    def gen_paiplot(self, fname=None):    # %% save to disk pairplot
        """This is a function that generates  a pairplot and saves into a file
        Takes a long time to process. 

        Args:
            fname ([type], optional): [description]. Defaults to None.
        """
        sns_plot = sns.pairplot(self.features,  kind ='reg',plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.4}})
        sns_plot.savefig(fname)
    
    def plot_normalised_residuals(self):
        """Plot normalised residuals
        (requires OLS Model)

        $$resid_i = \\frac{y_i-\hat{y}_i}{y_i}$$
        """        
        results = self.OLS_results

        fig, axs = plt.subplots(1,1, figsize = PLT_FIGSIZE)

        tmp = (results.resid/self.Y_train).sort_index()
        axs.plot(tmp.index, tmp.values, '.')
        axs.set_ylabel('Normalised residuals')
        axs.grid()
        axs.legend()

    def plot_predicted_vs_target(self):
        """graph of actual (y_i) vs predicted prices ($\hat{y}_i$)
        (requires OLS Model)
        """        
        # 
        results = self.OLS_results
        Y_train= self.Y_train
        corr = Y_train.corr(results.fittedvalues)
        fig, axs = plt.subplots(1,1, figsize = PLT_FIGSIZE)
        print("Corr between actual and predicted: {:.2f}".format(corr))
        plt.scatter(x=Y_train, y=results.fittedvalues, c='navy', alpha=0.6)
        plt.plot(Y_train, Y_train, c='cyan') # strain line
        plt.xlabel('actual  $y_{i}$', fontsize = 14)
        plt.ylabel('Predicted  $\hat{y}_{i}$', fontsize = 14)
        plt.title(f'Actual vs preidcted  : $y_i - \hat y_i$ (Corr: {corr:.2f})', fontsize=17)
        # plt.plot(Y_train, results.resid, '.')
    
    def plot_residuals_vs_predicted(self):
        """graph of resid ($y_i-\hat{y}_i$) vs predicted prices ($\hat{y}_i$)
        (requires OLS Model)
        """            
        results = self.OLS_results
        ## residual vs vs. Predicted values
        fig, axs = plt.subplots(1,1, figsize = PLT_FIGSIZE)
        plt.scatter(x=results.fittedvalues, y= results.resid, c='blue', alpha=0.6)
        plt.xlabel('Predicted  $\hat{y}_{i}$', fontsize = 14)
        plt.ylabel('Residuals', fontsize = 14)
        plt.title(f'Residuals vs Fitted Values', fontsize=17)

    def plot_residuals_distribution(self):
        """Plot distribution of residual and reports skewness values
        (requires OLS Model)
        """
        results = self.OLS_results
        # distributon of residuals  - checking for normality
        resid_mean = round (results.resid.mean(),3)
        resid_skew = round (results.resid.skew(),3)
        fig, axs = plt.subplots(1,1, figsize = PLT_FIGSIZE)
        # sns.displot(results.resid, color = 'navy',kind='kde', ax=ax)
        sns.displot(results.resid, color = 'navy',kind='hist', kde=True, rug=True, ax=axs)
        plt.title(f'model: residuals Skew:({resid_skew}) Mean: ({resid_mean})')

    def plot_correlations(self):
        """plots a heatmap with the correlations of features.
        depends on:
        - visualize_correlations function
        """
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

# %%
