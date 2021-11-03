
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

PLT_FIGSIZE=(16,9)
class Data_Container():
    """This is a basic DataContainer that is used as a parent  for other classes 

    - CF_Data_Container
    - MVR_Data_Container
    """
    
    def __init__(self, datadir):
        self._datadir = datadir
    
    def load_rawdatafile(self, fname):
        """This works with Heroku (assumes 4 columns
        - confirmed
        - deaths
        - pcr_tests
        - rapid_tests)

        Args:
            fname ([type]): [description]
        """
        self._df = pd.read_excel(pathlib.Path(self._datadir,fname), index_col='date', parse_dates=True)

    def set_df(self, df:pd.DataFrame):
        self._df = df

    def set_tests(self, tests_func:str=None):
        ''' Sets the tests (whether its pcr_tests+rapid_Tests or just pcr_tests)
        '''
        self._tests_func  = tests_func
        self._df.eval(tests_func, inplace=True)

    def preprocessing(self, cleanup_func, smooth_func)->pd.DataFrame:
        """performs complete preprocessing on a dataset

        Args:
            cleanup_func ([type]): [description]
            smooth_func ([type]): [description]

        Returns:
            pd.DataFrame: Dataframe after cleaning and smoothing.
        """
        if cleanup_func is None:
            cleanup_func = lambda x: x.copy()
        

        dfc = self.data_cleanup(cleanup_func)
        dfcp = self.process_smoothed_data(dfc, smooth_func=smooth_func)
        dfcp = dfcp [ np.isfinite(dfcp).all(1)]

        self.hag_d = dfcp
        return dfcp

    def get_tests_average(self):
        return self.hag_d['tests'].mean()

    def data_cleanup(self, cleanup_func)->pd.DataFrame:
        """Function that performs the cleanup. uses a user supplied function

        This is normally called from preprocessing()

        Args:
            cleanup_func ([type]): [description]

        Returns:
            pd.DataFrame: [description]
        """
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
        """Saves to files

        Args:
            fname ([type]): [description]
        """
        try:
            self.hag_d.to_excel(pathlib.Path(self._datadir, fname))
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
        """Function that 

        Args:
            data_series_name ([type]): [description]
            wkday1 (int): [description]
            wkday2 (int): [description]

        Returns:
            [type]: [description]
        """
        
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
   
