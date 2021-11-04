import numpy as np
import pandas as pd

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


#%%
def smooth_func_factory(periods_count:int=15, stds:float=4, init_val:int=100, window_type='gaussian'):
    """Factory function. produces a smooth function parametrically

    Args:
        periods_count (int, optional): periods . Defaults to 15.
        stds (float, optional): stds for the gaussian. Defaults to 4.
        init_val (int, optional): Threshold value of the smoothed to keep. Defaults to 100.
    """

    def smooth_f_g(new_cases:pd.Series)-> pd.Series:
        """smooth function. smooths and returns the first value past a number

        Args:
            new_cases (pd.Series): [description]
            periods_count (int, optional): [description]. Defaults to 15.
            stds (int, optional): [description]. Defaults to 4.
            init_val (int, optional): [description]. Defaults to 25.

        Returns:
            pd.Series: [description]
        """    
        '''smooth function'''
        smoothed = new_cases.rolling(window=periods_count,
            win_type='gaussian', min_periods=1,
            center=True).mean(std=stds).round()

        init_index  = np.argmax(smoothed>init_val)
        return smoothed[init_index:]

    def smooth_f_tri(new_cases:pd.Series)-> pd.Series:
        """smooth function. smooths and returns the first value past a number

        Args:
            new_cases (pd.Series): [description]
            periods_count (int, optional): [description]. Defaults to 15.
            stds (int, optional): [description]. Defaults to 4.
            init_val (int, optional): [description]. Defaults to 25.

        Returns:
            pd.Series: [description]
        """    
        '''smooth function'''
        smoothed = new_cases.rolling(window=periods_count,
            win_type='triang', min_periods=1,
            center=True).mean().round()

        init_index  = np.argmax(smoothed>init_val)
        return smoothed[init_index:]

    def smooth_f_owid(new_cases:pd.Series)-> pd.Series:
        """smooth function. smooths and returns the first value past a number

        Args:
            new_cases (pd.Series): [description]
            periods_count (int, optional): [description]. Defaults to 15.
            stds (int, optional): [description]. Defaults to 4.
            init_val (int, optional): [description]. Defaults to 25.

        Returns:
            pd.Series: [description]
        """    
        '''smooth function'''
        smoothed = new_cases.rolling(window=7,
            center=False).mean().round()

        init_index  = np.argmax(smoothed>init_val)
        return smoothed[init_index:]


    f_dict = {'gaussian': smooth_f_g, 
        'tri': smooth_f_tri,
        'owid': smooth_f_owid}
    
    return f_dict.get(window_type, smooth_f_g)



