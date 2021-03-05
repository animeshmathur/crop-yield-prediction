import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox

# Method to remove crops with zero production
def remove_zero_production_crops(df):
    '''
    Method to remove crops with zero production
    
    Parameters:
    df - Dataframe which should have 'Crop' and 'Production' columns.
    '''
    crop_net_production = df.groupby('Crop')['Production'].sum().sort_values()
    crops_with_zero_production = crop_net_production.loc[crop_net_production == 0].index
    df = df.loc[~df['Crop'].isin(crops_with_zero_production)].reset_index(drop=True)
    return df

# Method to remove records for 'other...' crops
def remove_other_crops(df):
    '''
    Method to remove records for 'other...' crops
    
    Parameters:
    df - Dataframe which should have 'Crop' column.
    '''
    other_crops = [crop for crop in df['Crop'].unique() if 'other' in crop.lower()]
    df = df.loc[~df['Crop'].isin(other_crops)].reset_index(drop=True)
    return df

# Clusterer for Crops
def identify_crop_types(df, clusterer):
    '''
    Returns dataframe with column 'Crop_Type' identified from values of 'Crop' and 'Season' for each row.
    
    Parameters:
    df - Dataframe which should have 'Crop' and 'Season' columns.
    clusterer - Clustering model for Crop and Season.
    '''
    
    df['Crop_Type'] = clusterer.predict(pd.get_dummies(df[['Crop', 'Season']].copy()))
    
    crop_type_map = {}
    
    for i, crop_type in enumerate(df['Crop_Type'].unique()):
        crop_type_map[i] = f'C{i}'
        
    df['Crop_Type'] = df['Crop_Type'].map(crop_type_map).copy()
    return df

# Clusterer for Lat-Long
def identify_geo_region(df, clusterer):
    '''
    Returns dataframe with column 'Geo_Region' identified from values of 'Lat' and 'Long' for each row.
    
    Parameters:
    df - Data frame which should have 'Lat' and 'Long' columns.
    clusterer - Clustering model for Lat and Long.
    '''
    df['Geo_Region'] = clusterer.predict(df[['Lat', 'Long']])
    
    region_map = {}
    
    for i, region in enumerate(df['Geo_Region'].unique()):
        region_map[i] = f'R{i}'
        
    df['Geo_Region'] = df['Geo_Region'].map(region_map).copy()
    
    return df

# Returns Weight of Evidence table for Crop column of given target
def crop_woe(data, target):
    '''
    Returns Weight of Evidence table for Crop column of given target
    
    Parameters:
    data - Data frame which should have Crop column.
    target - Name of target.
    '''
    
    df_woe = data.copy()
    SumY = df_woe.groupby('Crop')[target].sum()
    
    # % of observations falling in each category
    obs_per = (df_woe.groupby('Crop')[target].count() / df_woe.shape[0]) * 100
    
    # % of target in each category
    Y_per = (SumY / SumY.sum()) * 100
    
    # Weight of Evidence (WoE) = log (% of Y in each category / % obs. for each category)
    woe = np.log((Y_per / obs_per))
    
    # Information value
    information_value = np.sum((Y_per - obs_per) * woe)
    
    print(f"Information Value: {information_value}")
    
    return woe

def convert_to_yield(data, to_boxcox=False):
    '''
    Returns column 'Yield' obtained from Production and Area.
    
    If to_boxcox = False:
        Yield = Production / Area
    
    If to_boxcox = True:
        Yield = boxcox(Production/Target) 
        In this case, lambda value estimated for boxcox transform is returned along with new dataframe.
    
    Parameters:
        data - Data frame which should have 'Production' and 'Area' columns.
        to_boxcox (default: False) - Returns Yield with boxcox transformation along with lambda.
    
    Returns:
        New dataframe with column Yield. If to_boxcox=True, also returns lambda for transformation.
    '''
    
    df = data.copy()
    
    df['Yield'] = df['Production'] / df ['Area']
    
    if to_boxcox == True:
        df['Yield'], lmda = boxcox(df['Yield'] + 1) # Add +1 to avoid 0
        return df, lmda
    
    return df