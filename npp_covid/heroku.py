import requests
import numpy as np
import pandas as pd

#%%
def hag_get_tests():
    ''' download tests data 
    
    USes the https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests to 

    other potential sources
    # https://ourworldindata.org/coronavirus-testing
 
    # https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests

    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/total-tests")
    dictr = response.json()
    df = pd.json_normalize (dictr['total_tests'])
    df['tests']= pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    return df

def ha_get_all():
    ''' download from heroku a response with date, confirmed deaths
    
    USes the https://covid-19-greece.herokuapp.com/all

    other potential sources
    # https://ourworldindata.org/coronavirus-testing
 
    # https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests

    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/all")
    dictr = response.json()
    df = pd.json_normalize (dictr['cases'])
    # df['tests']= pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    return df


def hag_create_df():
    ''' collects data from 
    - date, confirmed, deaths
    - rapid-tests, tests  (PCR)
    '''

    df_cases = ha_get_all()
    df_tests = hag_get_tests()

    df = df_cases.set_index('date').join(df_tests.set_index('date'))
    df['rapid-tests']=pd.to_numeric(df['rapid-tests'].fillna(0), downcast='integer')
    df['tests']=pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    return df


def hag_get_intensive_care():
    ''' downloads history of intensive-care 
    
    USes the https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/ to 

    other potential sources
    # https://ourworldindata.org/coronavirus-testing
 
    # https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests

    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/intensive-care")
    dictr = response.json()
    df = pd.json_normalize (dictr['cases'])
    df['intensive_care']= pd.to_numeric(df['intensive_care'].fillna(0), downcast='integer')
    return df



def prepare_cases_gaussian(new_cases , 
    cutoff=25, 
    periods_count=14, 
    stds=4):
    """[summary]  This is probably broken now
    
    This is a function I copy pasted for the smoothing of the original Rt
    calculation. 
    Args:
        new_cases ([type]): contains the data to be smoothed
        cutoff (int, optional): This is a quantity that is used to remove low initial values (useful for Rt). Defaults to 25.
        periods_count (int, optional): Window size . Defaults to 14.
        stds (int, optional): Determines the weights for each sample in the gaussian window. Defaults to 4.

    Returns:
        [type]: smoothed dataset truncated to cutoff.


    IT is based on the gaussian window. to better understand how this works look at 
    [scipy.signal.windows.gaussian](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html)

    or run the following code   
    > from scipy import signal
    > import matplotlib.pyplot as plt
    > no_periods = 51 # change to 15 also
    > plt.plot(signal.windows.gaussian(no_periods, std=7), label = 'weights for std=7')
    > plt.plot( signal.windows.gaussian(no_periods, std=4), label = 'weights for std=4')
    > plt.plot( signal.windows.gaussian(no_periods, std=2), label = 'weights for std=2')
    > plt.legend()
    > plt.title(r"Gaussian window ($\sigma$=4 and 7)")
    > plt.ylim(0,1)
    > plt.title(r"Gaussian window ($\sigma$=4 and 7)")
    > plt.ylabel("Amplitude")
    > plt.xlabel("Sample")

    TODO: update 
    TODO: the original variable is not returned. 

    TODO: remove this function from this package. use a simpler version

    """

    
    smoothed = new_cases.rolling(periods_count,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=stds).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    # original = new_cases.loc[smoothed.index]
    
    return  smoothed

def hag_get_vaccinations():
    ''' download vaccination data data 
    
    Uses the https://covid-19-greece.herokuapp.com/docs/#

    The data come in regions, so this function first groups by date the data to provide the total number of vaccinations
    There are alos other data available
    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/vaccinations-per-region-history")
    dictr = response.json()
    df = pd.json_normalize (dictr['vaccinations-history'])
    # df['tests']= pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    
    # group by date to provide cumulative results. 
    dn = df.drop(columns = ['area_en', 'area_gr', 'dailydose1', 'dailydose2', 'daydiff', 'daytotal']) 
    # 
    dfsum = dn.groupby('referencedate').sum()

    return dfsum