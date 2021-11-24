#%% support R_t file
import pandas as pd
import numpy as np
from scipy import stats as sps
from scipy.interpolate import interp1d

from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
import matplotlib.pyplot as plt

#%%

class Rt_Calculator():
    # R_T_MAX = 6
    # r_t_num = 20
    # #GAMMA = 1/7
    # posteriors_p = 0.90
    # posteriors_sigma = 0.15

    def __init__(self, confirmed:pd.Series, is_smoothed=True):
        self._orig_backup = confirmed
        self.is_smoothed = is_smoothed
        if self.is_smoothed:
            self.smoothed = confirmed
        else:
            self.smooth_clean()


    def smooth_clean(self, cutoff = 20, smooth_periods_count=14,smooth_stds=4):
        '''Process for cleaning and smoothing data
        '''
        self.raw, self.smoothed = prepare_cases(self._orig_backup, is_cases_cumulative=False, cutoff=cutoff, 
            periods_count=smooth_periods_count, stds=smooth_stds)

    def process(self, 
        R_T_MAX = 6, r_t_num = 20
        , GAMMA = 1/7
        , posteriors_p = 0.90, posteriors_sigma = 0.15):
        """Main processing function that produces the result"""

        self.processing_params = {
                'R_T_MAX': R_T_MAX ,
                'r_t_num': r_t_num,
                'GAMMA' : GAMMA, 
                'posteriors_p' : posteriors_p,
                'posteriors_sigma' : posteriors_sigma
        }
        # Note that we're fixing sigma to a value just for the example
        self.posteriors, self.log_likelihood = get_posteriors(self.smoothed, 
            sigma=self.processing_params.get('posteriors_sigma') ,
            R_T_MAX=self.processing_params.get('R_T_MAX'), 
            r_t_num=self.processing_params.get('r_t_num'), 
            GAMMA=self.processing_params.get('GAMMA'))
        self.hdis = highest_density_interval(self.posteriors, p=self.processing_params.get('posteriors_p') )
        most_likely = self.posteriors.idxmax().rename('ML')
        self.result = pd.concat([most_likely, self.hdis], axis=1)
        return self.result

    def get_df_cases(self):
        """This is a convenience function for the plot_Rt

        Returns:
            [type]: [description]
        """
        return pd.DataFrame({'original':self.raw, 'smoothed':self.smoothed})



#%%
def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])

def prepare_cases(cases:pd.Series, is_cases_cumulative:bool=False, cutoff=25, periods_count=7, stds=2):
    """[summary]

    Args:
        cases (pd.Series): these are the cumulative cases with a DateTimeIndex
        is_cases_cumulative (bool): Whether the cases are reported as cumulative or daily values.
        cutoff (int, optional): When to start considering the data (a minimum threshold is required). Defaults to 25.
        periods_count (int, optional): Window for smoothing. Defaults to 7.
        stds (int, optional): Standard deviation for gaussian smoothing. Defaults to 2.

    Returns:
        [type]: [description]
    """    
    if is_cases_cumulative:
        new_cases = cases.diff()
    else:
        new_cases = cases

    smoothed = new_cases.rolling(periods_count,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=stds).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed

def get_posteriors(sr, sigma=0.15, R_T_MAX = 12, r_t_num = 100, GAMMA=1/7):
    r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*r_t_num+1)

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood

#%%
def plot_rt(result, state_name, Rt_plot_start_date:str="2020-08-01" , Rt_plot_end_date:str=None , **kwargs):
    # initialise
    # Rt_plot_start_date = "2020-08-01" if Rt_plot_start_date is None else Rt_plot_start_date
    figsize=kwargs.get('figsize',(1200/72,400/72))
    ylim = kwargs.get('ylim',(0,3))
    dfCases = kwargs.get('df_cases',None)

    # main 
    if dfCases is None: 
        fig, axs = plt.subplots(nrows=1, ncols=1,figsize=figsize)
        ax = axs
    else:
        fig, axs = plt.subplots(nrows=2, ncols=1,figsize=figsize)

        ax0 = dfCases.plot( ax =axs[0], marker='o' )
        if Rt_plot_end_date is None:
            ax0.set_xlim(pd.Timestamp(Rt_plot_start_date), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
        else:
            ax0.set_xlim(pd.Timestamp(Rt_plot_start_date), pd.Timestamp(Rt_plot_end_date))
        ax = axs[1]

    ax.set_title(f"{state_name}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side

    Low_str = result.columns[['Low' in k for k in result.columns]][0]
    High_str = result.columns[['High' in k for k in result.columns]][0]

    lowfn = interp1d(date2num(index),
                     result[Low_str].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result[High_str].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    # extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
    extended = pd.date_range(start=pd.Timestamp(Rt_plot_start_date),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25)
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())


    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(ylim)
    if Rt_plot_end_date is None:
        ax.set_xlim(pd.Timestamp(Rt_plot_start_date), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
    else:
        ax.set_xlim(pd.Timestamp(Rt_plot_start_date), pd.Timestamp(Rt_plot_end_date))
    # ax.set_xlim(pd.Timestamp(Rt_plot_start_date), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
    fig.set_facecolor('w')

    ax.set_title(f'Real-time $R_t$ for {state_name}')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())    
    if len(extended)>90:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    


def plot_rt_4oop(rtc, state_name:str, 
    Rt_plot_start_date:str="2020-08-01", Rt_plot_end_date:str=None , plot_cases:bool=False, **kwargs):
    """Function that plots both the confirmed cases and the Rt for the data 
 
    Args:
        rtc ([type]): [description]
        state_name ([type]): [description]
        Rt_plot_start_date (str, optional): [description]. Defaults to "2020-08-01".
        Rt_plot_end_date ([type], optional): [description]. Defaults to None.
    """
    if plot_cases:
        kwargs['df_cases'] = rtc.get_df_cases()
    plot_rt(rtc.result, state_name, Rt_plot_start_date=Rt_plot_start_date ,Rt_plot_end_date=Rt_plot_end_date , **kwargs)
