import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Standard scaling is for features with gaussian-like distribution or lots of outliers
# MinMax scaling is for features with different distribution and few outliers


def create_scaler(std=True, minmax=False, borne_inf=0, borne_sup=1):
    """Create sklearn scaler object, either Standard or Minmax scaler.
    
    :param std: bool - if True then creates standard scaler
    :param minmax: bool - if True then creates minmax scaler
    :param borne_inf: int - lower limit for minmax scaler, should be lower or equal as borne_sup
    :param borne_sup: int - upper limit for minmax scaler
    """
    assert borne_inf <= borne_sup
    if minmax: 
        return MinMaxScaler(feature_range=(borne_inf, borne_sup))
    elif std: 
        return StandardScaler()


def create_pipeline(steps): 
    """Create sklearn pipeline object instance initialized with a non empty list of pipeline steps."""
    return Pipeline(steps)


def add_scaler_to_pipeline(pipeline=None, message=None, position=None, std=True, minmax=False, **kwargs):
    """Create sklearn scaler and add it to existing pipeline.
    
    :param pipeline: sklearn pipeline object 
    :param message:  str - message that will caracterize the scaler in pipeline steps list
    :param position: int - index in the pipeline steps list at which insert scaler
    :param std: bool - if True then creates standard scaler
    :param minmax: bool - if True then creates minmax scaler
    :param **kwargs: positional arguments of create_scaler function, id est borne_inf & borne_sup
    """
    descr = message or ('minmax_scaler' if minmax else 'standard_scaler')
    scaler_step = (descr, create_scaler(std, minmax, **kwargs))
    steps = pipeline.steps if pipeline else []
    if position:
        steps = steps[:position] + [scaler_step] + steps[position:]
    else: 
        steps.append(scaler_step)
    if pipeline: 
        pipeline.steps = steps
    else:
        pipeline = create_pipeline(steps)
    return pipeline


def scale_numerical_dataframe(df, scaler):
    """Scale columns of the pandas dataframe using the selected scaling method.
    
    :param df: pandas.DataFrame - dataframe with ONLY numerical columns
    :param scaler: sklearn.preprocessing.Scaler - either StandardScaler or MinMaxScaler
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    for column in df.columns:
        col = df[column]
        df[column] = scaler.fit_transform(np.array(col).reshape(-1,1))
    return df


def apply_scaling(df, to_standardize, to_minmax, borne_inf=0, borne_sup=1):
    """Standardize only the columns of the dataframe specified in the column list.
    
    :param df: pandas.DataFrame
    :param to_standardize: str list - list of column names, 
        standing for NUMERICAL columns only, to which standard scaler will be applied
    :param to_minmax: str list - idem but for minmax scaler
    :param borne_inf: int - lower limit for minmax scaler
    :param borne_sup: int - upper limit for minmax scaler
    """
    df_std = df[to_standardize]
    scaler_std = create_scaler()
    df_std = scale_numerical_dataframe(df_std, scaler_std)
    for standardized_column in to_standardize:
        df[standardized_column] = df_std[standardized_column]
    
    df_minmax = df[to_minmax]
    scaler_minmax = create_scaler(minmax=True, borne_inf=borne_inf, borne_sup=borne_sup)
    df_minmax = scale_numerical_dataframe(df_minmax, scaler_minmax)
    for minmaxed_column in to_minmax:
        df[minmaxed_column] = df_minmax[minmaxed_column]
    
    return df

# Function apply_scaling call example below:
# df = apply_scaling(df, to_standardize, to_minmax, borne_inf, borne_sup)

# Other possibilities : standardize features with possible outliers then minmax scale all the numerical features to [0,1]