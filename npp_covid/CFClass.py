
#%%
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

PLT_FIGSIZE=(16,9)
class Data_Container():
    
    def __init__(self, datadir):
        self._datadir = datadir
    
    def load_rawdatafile(self, fname):
        self._df = pd.read_excel(pathlib.Path(self._datadir,fname), index_col='date', parse_dates=True)

    def set_tests(self, tests_func:str=None):
        ''' Sets the tests (whether its pcr_tests+rapid_Tests or just pcr_tests)
        '''
        self._tests_func  = tests_func
        self._df.eval(tests_func, inplace=True)

    def preprocessing(self, cleanup_func, smooth_func):
        """performs complete preprocessing on a dataset"""
        dfc = self.data_cleanup(cleanup_func)
        dfcp = self.process_smoothed_data(dfc, smooth_func=smooth_func)
        dfcp = dfcp [ np.isfinite(dfcp).all(1)]

        self.hag_d = dfcp
        return dfcp

    def data_cleanup(self, cleanup_func)->pd.DataFrame:
        tmp_df = cleanup_func(self._df.copy())
        return tmp_df

    def process_smoothed_data(self, hag_d, smooth_func= None)->pd.DataFrame:
        ''' function for producing the smoothed data set. 

        Returns a DataFrame containing:
        - index: a DateTimeIndex
        - confirmed: confirmed cases per day
        - deaths:  death cases per day
        - rapid_tests:  rapid tests per day
        - pcr_tests:  pcr tests per day
        
        calculates and adds:
        - confirmed_smoothed: the smoothed confirmed cases using the smooth function.
        - tests_sm: the smoothed tests  cases using the smooth function.
        - conf_per_1000t_act: the **actual** positivity ratio expressed in confirmed cases per 1000 tests 
        - conf_per_1000t_sm: the **smoothed**  positivity ratio expressed in confirmed cases per 1000 tests 

        '''
        # =================================
        #  Add calculated values
        hag_d['confirmed_smoothed'] = smooth_func(hag_d.confirmed)
        hag_d['tests_sm' ]= smooth_func(hag_d.tests)

        # confirmed 
        hag_d['conf_per_1000t_act']=hag_d['confirmed']/hag_d['tests']*1000
        # smoothed
        hag_d['conf_per_1000t_sm']=hag_d['confirmed_smoothed']/hag_d['tests_sm']*1000
        
        hag_d.dropna(inplace=True)
        if hag_d.isnull().any().any():
            print( hag_d.isnull().any())
            raise ValueError("Need to remove NaN's")
        return hag_d.copy()
    
    def process_weekly_Series(self, data_series_name:str) -> pd.DataFrame:
        ''' Function that produces data for a boxplot.

        this is more convenient because it takes and returns any series with an DateTimeIndex
        '''
        ds = self.hag_d[data_series_name]
        name_str = data_series_name # snap
        df = ds.to_frame()
        df['weekday'] = df.index.to_series().dt.dayofweek
        first_mon_index = np.argmax(df.weekday==0)
        df = df.iloc[first_mon_index:]
        df['week_no'] = (df.index[:] - df.index[0]).days//7
        grouped = df.loc[:,[name_str, 'week_no']].groupby(['week_no']).mean()
        df['weekly_average'] = grouped.loc[df.week_no.values,name_str].values
        df['normalized_{}'.format(name_str)] = df[name_str]/df['weekly_average']
        return df

    def save_to_excel_processed_file(self,fname):
        try:
            self.hag_d.to_excel(pathlib.Path(RAW_DATA_DIRNAME, fname))
        except Exception as err:
            print("Could not save to file")
            print(type(err))
            print(err)
    def sum_sqr_error_sm(self, col1_name:str="confirmed", col2_name:str="confirmed_smoothed"):
        """calculates and returns the sum of sqruare error of smoothing and raw data
    
        Args:
            df_m (pd.DataFrame): Data frame
            col1_name (str, optional): [description]. Defaults to "confirmed".
            col2_name (str, optional): [description]. Defaults to "confirmed_smoothed".
    
        Returns:
            [type]: [description]
        """
        df_m = self.hag_d
        return np.sqrt(((df_m[col1_name] - df_m[col2_name])**2).sum()/df_m.shape[0])

    def get_weekday_samples(self, data_series_name, wkday1:int, wkday2:int):
        
        df_w = self.process_weekly_Series(data_series_name)
        def select_weekday_data(df_w:pd.DataFrame, weekday:int)->pd.DataFrame:
            return df_w[df_w['weekday']==weekday]
        d1 = select_weekday_data(df_w=df_w, weekday=wkday1)
        d2 = select_weekday_data(df_w=df_w, weekday=wkday2)
        return d1.set_index('week_no').iloc[:,0], d2.set_index('week_no').iloc[:,0], data_series_name

    def calculate_ks_2samp_for_week(self, data_series_name):
        arr_res = np.zeros((7,7))
        for k in range(7):
            for l in range(k,7):
                d1,d2,name_str = self.get_weekday_samples(data_series_name, wkday1=k, wkday2=l)
                # ks_res= stats.mannwhitneyu (d1,d2)  # mann whitney non parametric
                # ks_res= stats.epps_singleton_2samp (d1,d2)  # epps singleton test
                ks_res= stats.ks_2samp (d1,d2)  # Kolmogorov-Smirnov 2sample test
                arr_res[k,l] = ks_res.pvalue
                
                # print(f"{k},{l}:  {ks_res.pvalue:.4f}")
        return arr_res
   
class CF_Data_Container(Data_Container):
    def generate_asdf(self):
        hag_d = self.hag_d
        self.asdf = pd.DataFrame({
            'conf': hag_d.confirmed,
            'tests': hag_d.tests,
            'conf_ratio': hag_d.confirmed/hag_d.confirmed_smoothed,
            'pos_idx': hag_d.confirmed/hag_d.tests*1000,
            'sm_confirmed': hag_d.confirmed_smoothed,
            'sm_tests': hag_d.tests_sm,
            'sm_pos_idx': hag_d.conf_per_1000t_sm
            }    )
        self.asdf.eval('ratio_pos_idx = pos_idx/sm_pos_idx', inplace=True)
        self.asdf.eval('ratio_tests=tests/sm_tests', inplace=True)

        return self.asdf         
class CF_plots_class():
    ''' This is a class that uses the Data Container object to plot information
    '''
    def __init__(self, dc:Data_Container):
        self._dc = dc
        
    def plot_confirmed(self):
        hag_d = self._dc.hag_d
        hag_d['confirmed'].plot(lw=0, marker='.')    
        hag_d['confirmed_smoothed'].plot()

    def weekday_boxplot(self, data_series_name:str, save_to_file:bool=False):
        """ Function that produces a boxplot and saves the data

        Args:
            ds (pd.Series): series with index a DateTimeIndex.
            save_to_file (bool, optional): set to true to save into a file. Defaults to False.
        """
        WEEKDAYS = ("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
        
        name_str = data_series_name
        df = self._dc.process_weekly_Series(data_series_name=data_series_name)
        if save_to_file:
            df.to_excel(pathlib.Path(RAW_DATA_DIRNAME, PROC_FNAME.replace('.', '_boxplot.')))
        # plotting
        fig,axs=plt.subplots(1,1,figsize=PLT_FIGSIZE)
        df.boxplot(column='normalized_{}'.format(name_str), by = 'weekday', ax =axs, fontsize=16,
            color =  dict(boxes="black", whiskers="black", medians="green", caps="Gray"), grid=True,
            boxprops = dict(color = 'k', linewidth = 1, facecolor='white', alpha=1),
            patch_artist=True
            )
        axs.set_ylabel('{} normalised wrt to  week average'.format(name_str),fontsize=16)
        axs.set_xlabel('Weekday',fontsize=16)
        axs.set_title('{} normalised wrt to week average vs. weekday ({} weeks) '.format(name_str, df['week_no'].max()), fontsize=20)
        plt.xticks(list(range(1,8)), WEEKDAYS)

    def plot_tests_pertype_log_scale(self):
        hag_dcp = self._dc.hag_d
        fig,axs = plt.subplots(1,1, figsize=PLT_FIGSIZE)
        hag_dcp['pcr_tests'].plot(lw=0, marker='.', logy=True)
        hag_dcp['rapid_tests'].plot(lw=0, marker='.', logy=True)
        plt.ylabel('no of performed tests  in log scale' )
        plt.grid()
        plt.title('Rapid and PCR tests in log scale' )

    def check_smoothing_linearity_confirmed(self):

        fig,axs = plt.subplots(2,1, figsize=(10,10))
        plt.title('confirmed vs smoothed confirmed \n(check smoothing linearity)')
        self._dc.hag_d.plot.scatter(x='confirmed',y='confirmed_smoothed', logy=False, logx=False, xlim=[1, 1e4], ax=axs[0])
        plt.axis('equal')
        plt.grid()
        self._dc.hag_d.plot.scatter(x='confirmed',y='confirmed_smoothed', logy=True, logx=True, xlim=[1, 1e4], ylim=[1, 1e4], ax=axs[1])
        plt.axis('equal')
        plt.grid()


    def plot_figure_showing_problem(self):
        fig, axs = plt.subplots (2,1, figsize=PLT_FIGSIZE,sharex=True)
        hag_d= self._dc.hag_d
        hag_d.confirmed.plot(ax=axs[0])
        axs[0].set_ylabel('PCR tests')
        axs[0].grid()
        hag_d.tests.plot(ax=axs[1])
        axs[1].set_ylabel('PCR tests')
        axs[1].grid()


    def plot_figure_smoothing(self):
        hag_d= self._dc.hag_d 
        fig, axs = plt.subplots (2,1, figsize=PLT_FIGSIZE,sharex=True)
        hag_d.confirmed.plot(ax=axs[0], lw=0, marker='.')
        hag_d.confirmed_smoothed.plot(ax=axs[0])
        axs[0].set_ylabel('confirmed cases')
        axs[0].grid()
        hag_d.tests.plot(ax=axs[1], lw=0, marker='.')
        hag_d.tests_sm.plot(ax=axs[1])
        axs[1].set_ylabel('PCR tests')
        axs[1].grid()
        axs[1].set_ylim([0,(hag_d.tests.max()//5000+1)*5000])    

    def get_columns(self):
        return self._dc.hag_d.columns

    def visualize_heatmap_ks(self, data_series_name, figsize=(16,10), fontsize = 14):
        """produces a tringular heatmap with color coded 

        Args:
            data ([type]): [description]
        """    
        data = self._dc.calculate_ks_2samp_for_week(data_series_name=data_series_name).T
        mask = np.zeros_like(data)
        triangle_indices = np.triu_indices_from(mask)
        mask[triangle_indices]=1
        plt.figure(figsize=figsize)
        sns.heatmap(data, mask =mask, annot=True, annot_kws={'size':14}, norm=LogNorm())
        sns.set_style('white')
        WEEKDAYS = ("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
        posx, textvals = plt.xticks()
        posy, textvals = plt.yticks()
        plt.xticks(posx, WEEKDAYS, ha='center', fontsize = fontsize)
        plt.yticks(posy, WEEKDAYS, va='center', fontsize = fontsize)    

class CF_analysis_plots():
    def __init__(self, asdf):
        self._asdf= asdf
        
    def npr_vs_tests(self):
        
        plt.figure(figsize=PLT_FIGSIZE)
        plt.plot(self._asdf.tests, self._asdf.conf_ratio,'.')
        plt.xlabel('no of tests []')
        plt.ylabel('Confirmed normalised w.r.t. to smoothed confirmed $\\frac{actual}{smoothed}$')
        # plt.xlim([0,2])
        # plt.ylim([0,3])
        plt.grid()
        plt.title('Normalised confirmed cases vs tests ')

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
        axs.set_title('Positive Rate (PR) and Smoothed Positive Rate (SPR)')
    
    def npr_vs_tr(self):
        fig, axs = plt.subplots(1,1, figsize=PLT_FIGSIZE)
        axs.plot(asdf.ratio_tests, asdf.ratio_pos_idx,'.')
        axs.set_xlabel(r'test ratio $\left[\frac{actual}{smoothed}\right]$')
        axs.set_ylabel('Normalised positive rate $\\left[\\frac{raw }{smoothed}\\right]$')
        axs.set_xlim([0,2])
        axs.set_ylim([0,3])
        axs.set_aspect('equal')
        axs.grid()
        axs.set_title('Normalised positive rate vs test ratio')

    def npr_vs_tests(self):
        
        fig, axs = plt.subplots(1,1, figsize=PLT_FIGSIZE)
        axs.plot(self._asdf.tests, self._asdf.ratio_pos_idx,'.')
        axs.set_xlabel('no of tests []')
        axs.set_ylabel('Normalised positive rate $\\frac{actual}{smoothed}$')
        # plt.xlim([0,2])
        # plt.ylim([0,3])
        axs.grid()
        axs.set_title('Normalised Positive Rate vs no of total tests')

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


    def plot_model_with_ci(self, x_name:str, y_name:str, model_func, alpha=0.01, whisker_plot:bool=True, bins = np.arange(0, 50000, 2000),
        xlab:str=None, ylab:str=None):
        ''' plots a fitted model to x_data and y_data and also plots confinence intervals and whisker plots
        '''
        x_data = self._asdf[x_name].values 
        y_data = self._asdf[y_name].values
        asdf = self._asdf
        parameters, covariance = curve_fit(model_func, x_data , y_data)
        sigma_ab = np.sqrt(np.diagonal(covariance))
        # the covariance can be used for parameter estimation 
        # e.g. for parameter 1, the values are between:
        #           paramaters[0]- covariance[0,0]**2*tval 
        # and 
        #           paramaters[0]+ covariance[0,0]**2*tval 
        # (tval: is the student-t value for the dof and confidence level)
        # see
        # - https://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
        # - https://stackoverflow.com/questions/39434402/how-to-get-confidence-intervals-from-curve-fit

        fig, axs = plt.subplots(1,1,figsize=PLT_FIGSIZE)
        if whisker_plot :
            binsize = np.diff(bins).min()
            x_bins = pd.cut(x_data, bins)
            bins_centers =(bins[:-1] + bins[1:])/2
            x_bc = (bins_centers[x_bins.codes]) #.astype(dtype='int')
            x_bcu = np.unique(x_bc)
            y_binned = []
            for k in x_bcu :
                y_binned.append(list(y_data[x_bc==k]))

            flier_kwargs = dict(marker = 'o', markerfacecolor = 'silver',
                                markersize = 3, alpha=0.7)
            line_kwargs = dict(color = 'k', linewidth = 1)
            bp = plt.boxplot( y_binned, positions=x_bcu ,
                            patch_artist=True,
                            capprops = line_kwargs,
                            boxprops = dict(color = 'k', linewidth = 1, facecolor='white', alpha=0.5),
                            whiskerprops = line_kwargs,
                            medianprops = line_kwargs,
                            flierprops = flier_kwargs,
                            widths = binsize/3,
                            manage_ticks = False)

        # plotting the model
        hires_x = np.linspace(x_data.min(), x_data.max(), 100)
        plt.plot(hires_x, model_func(hires_x, *parameters), 'black', label='model')
        dof = len(y_data)-len(parameters)
        tval = stats.t.ppf(1.0-alpha/2., dof)
        bound_upper = model_func(hires_x, *(parameters + sigma_ab*tval))
        bound_lower = model_func(hires_x, *(parameters - sigma_ab*tval))
        # plotting the confidence intervals
        plt.fill_between(hires_x, bound_lower, bound_upper,
                        color = 'black', alpha = 0.15)

        plt.plot(x_data, y_data, '.', label='data', alpha=0.5)
        text_res = "Best fit parameters:\na = {:.3g} $\\pm$ {:.3g} \nb = {:.3g} $\\pm$ {:.3g}".format(parameters[0],sigma_ab[0]*tval, parameters[1],sigma_ab[1]*tval)
        t = plt.text(0.8*x_data.max(), 0.8*y_data.max(), text_res)
        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.grid()
        plt.legend()
        # plt.axis('equal')

