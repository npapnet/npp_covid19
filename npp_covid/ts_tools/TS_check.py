#%%
import pandas as pd
import numpy as np

def get_no_days(ds:pd.Series)-> int:
    """gets number of days in time series based on last and first day

    Args:
        ds (pd.Series): [description]

    Returns:
        int: [description]
    """    
    return (ds.index[-1]-ds.index[0]).days+1

def is_contiguous(ds:pd.Series)-> bool:
    """checks if dataTime index is contiguous

    Args:
        ds (pd.Series): [description]

    Returns:
        bool: [description]
    """    
    clean = ds.dropna()
    is_cont = True
    # if clean.shape != ds.shape:
    #     is_cont = False
    if clean.shape != gen_continuous_DTindex(clean).shape:
        is_cont = False
    return is_cont

def gen_continuous_DTindex(ds:pd.Series)->pd.DatetimeIndex:
    """Generates a continuous DateTimeindex which 
    - starts from the first date of the ds.index
    - ends   on the final date of the ds.index
    checks that the index is of type pd.DateTimendex

    Args:
        ds (pd.Series): pandas.Series with index a pd.DatetimeIndex

    Returns:
        pd.DatetimeIndex: [description]
    """
    assert is_index_date_time(ds)
    new_index = pd.date_range(start=ds.index[0], end =ds.index[-1])
    return new_index


def is_index_date_time(ds:pd.Series)->bool:
    """checks whether the index of the time Series is of type DatetimeIndex

    Args:
        ds (pd.Series): [description]

    Returns:
        bool: [description]
    """    
    return  isinstance(ds.index, pd.DatetimeIndex)
# %%
class TS_Index_Checker():
    def __init__(self, ds:pd.Series) -> None:
        self._ds = ds
        self._clean = ds.dropna()
        self._analyze()
        # Initial checks
        if not is_index_date_time(ds):
            raise(TypeError("Index type is not pd.DatetimeIndex"))
    
    @property
    def ds(self):
        """returns a copy  original data series 
        #TODO: clear up what happens if the self._ds is passed on.

        Returns:
            [type]: [description]
        """
        return self._ds[:]

    @property
    def clean_largest(self)->pd.Series:
        """returns the largest continuous segment:
        - without na
        - TODO: without negative values

        Returns:
            [type]: [description]
        """        
        l_indx = find_largest_contiguous_segment(self._clean)
        return self._clean.iloc[l_indx[0]:l_indx[1]]

    def ds_frac(self, indxs):
        return get_ds_perc(self.ds, indxs)
 
    def _analyze(self)-> dict:
        """Returns a dictionary with some statistics for the time-series
        

        Returns:
            dict: {
                'length': original length.
                'na': number of na in the time series
                'largest_segment' : refers to the indexes of the **clean** ds
            }
        """        
        self.metadata = {}
        self.metadata['length'] = len(self._ds)
        self.metadata['na'] = self._ds.isna().sum()
        self.metadata['negative'] = (self.ds<0).sum()
        self.metadata['largest_segment'] = find_largest_contiguous_segment(self._clean)
        return self.metadata
    
    def get_largest_continuous_segment(self):
        inds = find_largest_contiguous_segment(self._clean)
        new_ds = self._clean.iloc[inds[0]:inds[1]]
        return new_ds
        
    def is_suitable(self):
        return self.is_contiguous(self.ds)
    
def find_contiguous_segments(ds:pd.Series)->list:
    """REturns a list of list with the indexes of contiguous segments
    All segments are returned

    Args:
        ds (pd.Series): ds with na removed

    Returns:
        list: list of [start_index, end_index] 
    """    
    day_diff = (ds.index[1:]-ds.index[:-1]).days.values
    ioi = np.where(day_diff>1)[0] #indexes of interest
    
    segment_indxs = []
    tmp = [0]
    if ioi.size>0:
        for indx in ioi:
            tmp.append(indx)
            segment_indxs.append(tmp)
            tmp = [indx+1]
    tmp.append(len(ds)-1)
    segment_indxs.append(tmp)
    
    return segment_indxs
    
def find_largest_contiguous_segment(ds:pd.Series)->list:
    """returns the indexes of the largest continuous index

    Args:
        ds (pd.Series): requires a time series with na removed.

    Returns:
        list: the start and end index of the **clean** ds 
    """    
    if ds.isna().any():
        raise ValueError(f'The time series should not contain na. There are {ds.isna().sum()} NaNs in the ds')
    all_segs = find_contiguous_segments(ds)
    index_len = 0
    max_len = 0 
    for k in range(len(all_segs)):
        seg = all_segs[k]
        seg_len = seg[1]-seg[0]+1
        if (seg_len>max_len):
            max_len = seg_len
            index_len = k

    return all_segs[index_len]    

def get_ds_perc(ds:pd.Series, indxs:list=[0,1])->pd.Series:
    """return data series as percentile

    Args:
        ds (pd.Series): data series
        indxs (list with two elemetns, optional): fractional indexes. Defaults to [0,1].

    Returns:
        [type]: [description]
    """    
    # ind_start = ds.index[int(np.round(indxs[0]*ds.size))]
    # ind_end =  ds.index[int(np.round(indxs[1]*ds.size))]
    ind_start = ds.index[int(ds.size // (1/indxs[0]) ) ]
    ind_end = ds.index[int( ds.size // (1/indxs[1]) ) ]
    return ds[ind_start:ind_end]



# %%
def convert_neg_to_na(ds:pd.Series, winsize:int=7)->pd.Series:
    """This function generates a new data series and 
    whenever a negative number appears it 
    replaces the negative AND the subsequent winsize number with nan
    
    This is to be applied on a raw time series. This is to further clean smoothed data series
    from entries that have dependence on negative numbers.

    Args:
        ds (pd.Series): [description]
        winsize (int, optional): [description]. Defaults to 7.

    Returns:
        pd.Series: [description]
    """    
    ds_copy = ds[:]
    neg_indx = np.where(ds_copy<0)[0]
    ds_len = ds_copy.size
    for k in range(len(neg_indx)):
        ind = neg_indx[k]
        for l in range(winsize):
            ind2 = min([ind+l, ds_len-1])
            ds_copy.iloc[ind2] = np.nan
    return ds_copy

def gen_ext_index(ds:pd.Series, win_size:int=7):
    """generates an extended data_range index 
    based on the smoothed data,  that starts at 
    winsize days earlier that the ds
    TODO: 

    Args:
        ds (pd.Series): [description]
        winsize (int): window size (Defaults to 7)

    Returns:
        [type]: extended index by 7 days.
    """        
    return pd.date_range(start=ds.index[0]- dtm.timedelta(days=win_size-1), end=ds.index[-1],freq='D')

# %%

if __name__ =="__main__":
    def create_ds(no_periods=5):
        indx = pd.date_range(start='2021-12-01', periods=no_periods)
        ds = pd.Series(range(no_periods), index = indx)
        return ds
    no_periods=15
    ds = create_ds(no_periods)
    ds2 = ds.replace(4,  np.nan)    
    oa = TS_Index_Checker(ds=ds2)
    
    
    