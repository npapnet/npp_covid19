
#%%
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pathlib

# from scipy.stats.stats import ks_2samp
from scipy import stats
import seaborn as sns
from matplotlib.colors import LogNorm

from scipy.optimize import curve_fit
from npp_covid.DataContainer import Data_Container

PLT_FIGSIZE=(16,9)

class CF_Data_Container(Data_Container):
    def generate_asdf(self):
        hag_d = self.data
        self.asdf = pd.DataFrame({
            'conf': hag_d.confirmed,
            'tests': hag_d.tests,
            'conf_ratio': hag_d.confirmed/hag_d.confirmed_sm,
            'pos_idx': hag_d.pr,
            'sm_confirmed': hag_d.confirmed_sm,
            'sm_tests': hag_d.tests_sm,
            'sm_pos_idx': hag_d.pr_sm
            }    )
        self.asdf.eval('ratio_pos_idx = pos_idx/sm_pos_idx', inplace=True)
        self.asdf.eval('ratio_tests=tests/sm_tests', inplace=True)

        return self.asdf         


#%% Plot classes

class CF_analysis_plots():
    def __init__(self, asdf, type_of_tests:str = None):
        """[summary]

        Args:
            asdf ([type]): dataseries
            type_of_tests (str): This should be either "total" or "RT-PCR"
        """        
        self._asdf= asdf
        try:
            assert type_of_tests in ['total', 'RT-PCR']
        except AssertionError as e:
            raise Exception('type_of_tests (str): This should be either "total" or "RT-PCR"')
        self._tests_str = type_of_tests
        
    def npr_vs_tests(self):
        
        plt.figure(figsize=PLT_FIGSIZE)
        plt.plot(self._asdf.tests, self._asdf.conf_ratio,'.')
        plt.xlabel('no of {} tests []'.format(self._tests_str))
        plt.ylabel('Confirmed normalised w.r.t. to smoothed confirmed $\\frac{actual}{smoothed}$')
        # plt.xlim([0,2])
        # plt.ylim([0,3])
        plt.grid()
        plt.title('Normalised confirmed cases vs {} tests'.format(self._tests_str))

    def pr_vs_spr(self):
        
        fig, axs = plt.subplots(1,1, figsize=PLT_FIGSIZE)
        axs.plot(self._asdf.index, self._asdf.pos_idx,'.', label = 'PR')
        axs.plot(self._asdf.index, self._asdf.sm_pos_idx,'-', label = 'SPR')
        axs.set_xlabel('date')
        axs.set_ylabel('positive rate $\\left[\\frac{conf}{1000 ~tests}\\right]$')
        axs.legend()
        # plt.xlim([0,2])
        # plt.ylim([0,3])
        axs.grid()
        title_str = 'Positive Rate (PR) and Smoothed Positive Rate (SPR)'
        title_str += "\n based on {} tests".format(self._tests_str)
        axs.set_title(title_str)
        
    def npr_vs_tr(self):
        fig, axs = plt.subplots(1,1, figsize=PLT_FIGSIZE)
        axs.plot(self._asdf.ratio_tests, self._asdf.ratio_pos_idx,'.')
        axs.set_xlabel(r'test ratio $\left[\frac{actual}{smoothed}\right]$')
        axs.set_ylabel('Normalised positive rate $\\left[\\frac{raw }{smoothed}\\right]$')
        axs.set_xlim([0,2])
        axs.set_ylim([0,3])
        axs.set_aspect('equal')
        axs.grid()
        axs.set_title('Normalised positive rate vs test ratio ({})'.format(self._tests_str))

    def npr_vs_tests(self):
        
        fig, axs = plt.subplots(1,1, figsize=PLT_FIGSIZE)
        axs.plot(self._asdf.tests, self._asdf.ratio_pos_idx,'.')
        axs.set_xlabel('no of {} tests []'.format(self._tests_str))
        axs.set_ylabel('Normalised positive rate $\\frac{actual}{smoothed}$')
        # plt.xlim([0,2])
        # plt.ylim([0,3])
        axs.grid()
        axs.set_title('Normalised Positive Rate vs no of {} tests'.format(self._tests_str))

    # def npr_vs_pcr(self):
    #     fig, axs = plt.subplots(1,1, figsize=PLT_FIGSIZE)
    #     axs.plot(hag_d.pcr_tests, asdf.ratio_pos_idx,'.')
    #     axs.set_xlabel('no of tests []')
    #     axs.set_ylabel('Normalised positive rate $\\frac{actual}{smoothed}$')
    #     # plt.xlim([0,2])
    #     plt.ylim([0,3])
    #     axs.grid()
    #     axs.set_title('Normalised Positive Rate vs no of RT-PCR tests')
    def npr_vs_invtests(self):
        plt.figure(figsize=PLT_FIGSIZE)
        plt.plot(1/self._asdf.tests, self._asdf.ratio_pos_idx,'.')
        plt.xlabel('test')
        plt.ylabel('Normalised positive rate $\\frac{actual}{smoothed}$')
        # plt.xlim([0,2])
        #plt.ylim([0,3])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid()
        plt.title('Normalised positive rate vs inverse of no. tests (loglog scale)  ')

    def plot_model_with_ci2(self, x_name:str, y_name:str, model_func, 
        alpha:float=0.01, whisker_plot:bool=True, 
        bins:Union[int,float,np.ndarray] = None,
        xlab:str=None, ylab:str=None)->dict:
        """ plots a fitted model to x_data and y_data and also plots confinence intervals and whisker plots
        
        this version uses the FinalPlotCalculations
        

        Args:
            x_name (str): x data column name
            y_name (str): y  data column name
            model_func ([type]): model to be fitted
            alpha (float, optional): alpha value for confidence level. Defaults to 0.01.
            whisker_plot (bool, optional): Whether to add whisker plot. Defaults to True.
            bins (Union[int,float,np.ndarray], optional): how to determine the bins . Defaults to None.
                - int: number or bins
                - float: multiple of max(x_data) 
                - nd.array: 
            xlab (str, optional): label for x-axis. Defaults to None.
            ylab (str, optional): label for y-axis. Defaults to None.

        Returns:
            CFCalcForCI_Plots: returns the calculations object.
        """
        # TODO: change x_name to be able to receive str and pd.Series
        # initialising
        x_data = self._asdf[x_name].values 
        y_data = self._asdf[y_name].values
        if bins is None:
            bins = np.linspace(0, 1.1*np.max(x_data), num=10)
        if isinstance(bins, int):
            bins = np.linspace(0, 1.1*np.max(x_data), num=bins)
        # performing calculations ====================================================
        fpc = CFCalcForCI_Plots(x_data=x_data, y_data=y_data)
        dict_params = fpc.calc_params(model_func=model_func)
        dict_whisker = fpc.calc_whisker_plot(bins=bins)
        dict_cis = fpc.calc_cis(num_points=101,alpha=alpha)

        # plottting ==================================================================
        fig, axs = plt.subplots(1,1,figsize=PLT_FIGSIZE)
        if whisker_plot :
            flier_kwargs = dict(marker = 'o', markerfacecolor = 'silver',
                                markersize = 3, alpha=0.7)
            line_kwargs = dict(color = 'k', linewidth = 1)
            bp = plt.boxplot( list(dict_whisker['y_bin_data'].values()), positions=list(dict_whisker['y_bin_data'].keys()) ,
                            patch_artist=True,
                            capprops = line_kwargs,
                            boxprops = dict(color = 'k', linewidth = 1, facecolor='white', alpha=0.5),
                            whiskerprops = line_kwargs,
                            medianprops = line_kwargs,
                            flierprops = flier_kwargs,
                            widths = dict_whisker['binsize']/3,
                            manage_ticks = False)

        # plotting the model
        plt.plot(dict_cis['x'], dict_cis['y'], 'black', label='model')
        # plotting the confidence intervals
        plt.fill_between(dict_cis['x'], dict_cis['y_upper'], dict_cis['y_lower'],
                        color = 'black', alpha = 0.15)

        plt.plot(fpc.x_data, fpc.y_data, '.', label='data', alpha=0.5)
        parameters = dict_params.get('parameters')
        sigma_ab= dict_params.get('sigma')
        tval= dict_cis.get('t-statistic')
        # text_res = "Best fit parameters:\na0 = {:.3g} $\\pm$ {:.3g} \nb = {:.3g} $\\pm$ {:.3g}".format(parameters[0],sigma_ab[0]*tval, parameters[1],sigma_ab[1]*tval)
        text_res = "Best fit parameters:\na0 = {:.3g} $\\pm$ {:.3g} ".format(parameters[0],sigma_ab[0]*tval)
        for k in range(1, parameters.shape[0]):
            text_res += "\na{} = {:.3g} $\\pm$ {:.3g}".format(k,parameters[k],sigma_ab[k]*tval)
        t = plt.text(0.8*fpc.x_data.max(), 0.8*fpc.y_data.max(), text_res)
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid()
        plt.legend()
        # plt.axis('equal')
        # plt.show()
        return fpc

# %%

class CFCalcForCI_Plots():
    '''this is a complementary class  to CF calculation for the final plot
    the reasoning that this is used to calculate and contain the data for the plot
    and also to use the coefficients and model for the comparison plots.

    Example usage:
    - fpc = CFCalcForCI_Plots(x_data=asdf['tests'], y_data=asdf['ratio_pos_idx'])
    - print(fpc.calc_params(model_func=hyperbolic))
    - dic_whisk=fpc.calc_whisker_plot(bins = np.linspace(0, 4e5, 11))
    - print(fpc.calc_cis())
    '''
    def __init__(self, x_data, y_data) -> None:
        """Initialisation with data points

        Args:
            x_data ([type]): x-axis data
            y_data ([type]): y-axis data
        """
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)

    def calc_params(self, model_func):
        """Parameter calculation

        Args:
            model_func (function): a function to be fitted with the model 

        Returns:
            dict: {'parameters': ``nd.array``, 'cov':``nd.array``, 'sigma':``nd.array``, 'model_name':``str``}
        """
        self.model_func = model_func
        self.parameters, self.covariance = curve_fit(model_func, self.x_data , self.y_data)
        self.sigma_ab = np.sqrt(np.diagonal(self.covariance))
        return {'parameters':self.parameters, 'cov':self.covariance, 'sigma':self.sigma_ab, 'model_name':model_func.__name__}
        # the covariance can be used for parameter estimation 
        # e.g. for parameter 1, the values are between:
        #           paramaters[0]- covariance[0,0]**2*tval 
        # and 
        #           paramaters[0]+ covariance[0,0]**2*tval 
        # (tval: is the student-t value for the dof and confidence level)
        # see
        # - https://education.molssi.org/python-data-analysis/03-data-fitting/index.html
        # - https://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
        # - https://stackoverflow.com/questions/39434402/how-to-get-confidence-intervals-from-curve-fit

    def calc_whisker_plot(self, bins):
        """calculates the whisker plot data.

        Args:
            bins ([type]): [description]

        Returns:
            [type]: [description]
        """

        binsize = np.diff(bins).min()
        x_bins = pd.cut(self.x_data, bins)
        bins_centers =(bins[:-1] + bins[1:])/2
        x_bc = (bins_centers[x_bins.codes]) #.astype(dtype='int')
        x_bcu = np.unique(x_bc)
        y_binned = dict()
        for k in x_bcu :
            y_binned[k] = list(self.y_data[x_bc==k])
        return {
            'binsize': binsize,
            'x_bins_all': x_bins,
            'x_bin_centers': bins_centers,
            'y_bin_data': y_binned 
        }

    def calc_cis(self, num_points:int=101, alpha:float=0.01):
        """Calculates the data series for the ci plot

        Args:
            num_points (int, optional): number of equispaced datapoints. Defaults to 100.
        """        
        hires_x = np.linspace(self.x_data.min(), self.x_data.max(), num_points)
        hires_y = self.model_func(hires_x, *self.parameters)
        dof = len(self.y_data)-len(self.parameters)
        tval = stats.t.ppf(1.0-alpha/2., dof)
        bound_upper = self.model_func(hires_x, *(self.parameters + self.sigma_ab*tval))
        bound_lower = self.model_func(hires_x, *(self.parameters - self.sigma_ab*tval))
        self.dict_cis = {
            'x': hires_x,
            'y': hires_y,
            'dof': dof,
            't-statistic': tval,
            'y_upper': bound_upper,
            'y_lower':bound_lower
        }
        return self.dict_cis
