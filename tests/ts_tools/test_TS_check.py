#%%
import pytest
import pandas as pd
import numpy as np

import npp_covid.ts_tools.TS_check as tsc

def create_ds(no_periods=5):
    indx = pd.date_range(start='2021-12-01', periods=no_periods)
    ds = pd.Series(range(no_periods), index = indx)
    return ds


def test_get_days():
    indx = pd.date_range(start='2021-12-01', end ='2021-12-05')
    ds = pd.Series(0, index = indx)
    assert tsc.get_no_days(ds) == 5

def test_get_days2():
    no_periods=5
    ds = create_ds(no_periods)
    assert tsc.get_no_days(ds) == no_periods


def test_is_contiguous_length():
    """checks if a ds is an NA in the middle is contiguous (based on length)
    """    
    no_periods=5
    ds = create_ds(no_periods)
    ds2 = ds.replace(2,  np.nan)
    assert (tsc.is_contiguous(ds2) == False)

def test_is_contiguous_edge():
    """checks if a ds is an NA on the edge  is contiguous (based on length)
    """    
    no_periods=5
    ds = create_ds(no_periods)
    ds2 = ds.replace(0,  np.nan)
    assert tsc.is_contiguous(ds2) == True
    ds2 = ds.replace(no_periods-1,  np.nan)
    assert tsc.is_contiguous(ds) == True

def test_is_index_date_time():
    no_periods=5
    ds = create_ds(no_periods)
    assert tsc.is_index_date_time(ds)
    
def test_find_contiguous_segments():
    no_periods=15
    ds = create_ds(no_periods)
    ds2 = ds.replace(4,  np.nan).dropna()

    assert (tsc.find_contiguous_segments(ds2) == [[0, 3], [4, 13]])
    assert (tsc.find_contiguous_segments(ds) == [[0, 14]])


def test_find_largest_contiguous_segment():
    no_periods=15
    ds = create_ds(no_periods)
    assert (tsc.find_largest_contiguous_segment(ds)== [0, 14])
    
    ds2 = ds.replace(4,  np.nan).dropna()
    assert (tsc.find_largest_contiguous_segment(ds2) == [4, 13])

def test_analyze():
    no_periods=15
    ds = create_ds(no_periods)
    
    oa = tsc.TS_Index_Checker(ds=ds)
    assert (oa._analyze() == {'length':15, 'na':0,
                'largest_segment': [0,14],
                'negative' : 0})

    ds2 = ds.replace(4,  np.nan)    
    oa = tsc.TS_Index_Checker(ds=ds2)
    assert (oa._analyze() == {'length':15, 'na':1,
                'largest_segment': [4,13],
                'negative' : 0})
    
    ds2a= ds2.dropna()
    oa = tsc.TS_Index_Checker(ds=ds2a)
    assert (oa._analyze() == {'length':14, 'na':0,
                'largest_segment': [4,13],
                'negative' : 0})
  

def test_analyze_end():
    """tests analyze function of TS_Index_Checker
    """    
    no_periods=15
    ds = create_ds(no_periods)
    
    oa = tsc.TS_Index_Checker(ds=ds)
    assert (oa._analyze() == {'length':15, 'na':0,
                'largest_segment': [0,14],
                'negative' : 0})

    ds2 = ds.replace(14,  np.nan)    
    oa = tsc.TS_Index_Checker(ds=ds2)
    assert (oa._analyze() == {'length':15, 'na':1,
                'largest_segment': [0,13],
                'negative' : 0})

    ds2a= ds2.dropna()
    oa = tsc.TS_Index_Checker(ds=ds2a)
    assert (oa._analyze() == {'length':14, 'na':0,
                'largest_segment': [0,13],
                'negative' : 0})


def test_analyze_negative():
    """tests analyze function of TS_Index_Checker for negative data
    """
    no_periods=15
    ds = create_ds(no_periods)
    
    oa = tsc.TS_Index_Checker(ds=ds)

    ds2 = ds.replace(14,  -1)    
    oa = tsc.TS_Index_Checker(ds=ds2)
    assert (oa._analyze() == {'length':15, 'na':0,
                'largest_segment': [0,14],
                'negative' : 1})

    ds2a= ds2.replace(10,  -1) 
    oa = tsc.TS_Index_Checker(ds=ds2a)
    assert (oa._analyze() == {'length':15, 'na':0,
                'largest_segment': [0,14],
                'negative' : 2})

#%%
def test_convert_neg_na():
    no_periods=15
    ds = create_ds(no_periods)
    ds2 = ds.replace(1,  -1)
    winsize=2
    ds3 = ds.replace(1,np.nan)
    ds3 = ds3.replace(2,np.nan)
    actual = tsc.convert_neg_to_na(ds2,winsize=winsize)
    assert ds3.equals(actual)
        
        
        
    
def test_convert_neg_na2():
    """test conversion of negative to nan with two different locations
    one in the middle and one right at the end (winsize extends beyond the sizes)
    """    
    no_periods=15
    ds = create_ds(no_periods)
    ds2 = ds.replace(1,  -1)
    ds2.replace(14,  -1, inplace=True)
    winsize=2
    ds3 = ds.replace(1,np.nan)
    ds3.replace(2,np.nan, inplace=True)
    ds3.replace(14,np.nan, inplace=True)

    actual = tsc.convert_neg_to_na(ds2,winsize=winsize)
    assert ds3.equals(actual)