import os
import sys
import numpy
import pandas
import config
# from datetime import timedelta
#from sklearn.ensemble import IsolationForest
#from sklearn_pandas import DataFrameMapper



def hole_tagging(df_input, values_col_name, start_time=None, end_time=None, freq_default="15Min"):
    """
    we tag as hole the missing events in the continuous time-series (no timestamp and no variable values)

    INPUTS:
    :param end_time:
    :param start_time:
    :param df_input: dataframe                                                        [pandas.DataFrame]
    :param values_col_name: the name of the dataframe column with values              [string]
    :param freq_default: the default frequency of the signals. Default value = "1T"   [string]

    :return: new df with holes tagged in a column called "values_col_name_Tag"
    NOTE: Tagging column values will be "Hole" OR "Regular". In this context, "Regular" means that they are not "Holes".

    ============================================================
    """
    # instancing DataFrame
    # df_process = pandas.DataFrame()
    log = config.log
    column_tag_name = values_col_name + "_tag"

    try:
        # accessing minimum and maximum index values
        if not start_time:
            start_time = df_input.index.min()
        if not end_time:
            end_time = df_input.index.max()
        # accessing time series frequency. If infer is not possible, use default one
        if len(df_input) > 1:
            freq_df = pandas.infer_freq(df_input.index)
        else:
            freq_df = freq_default

        if not freq_df:
            freq_df = freq_default

        # create a complete date_index
        date_index_new = pandas.date_range(start_time, end_time, freq=freq_df)

        # check if the input dataset is already complete, otherwise reoirder df by index and exit
        if len(date_index_new) == len(df_input.index):
            df_process = (df_input[[values_col_name]].sort_index(axis=0)).copy()
            df_process[column_tag_name] = "Regular"
            return df_process

        # create a df with values set to = "hole"
        df_process = df_input[[values_col_name]].reindex(date_index_new, copy=True, fill_value="hole")
        df_process[column_tag_name] = "Regular"
        df_process.loc[df_process[values_col_name] == "hole", column_tag_name] = "Hole"
        # ##print("Tagging done")
        df_process[values_col_name].replace(to_replace="hole", value=numpy.nan, inplace=True)
#

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log.error("FR: some unexpected error occurred in script {1}: {0}".format(err, fname))
        log.error(exc_type, fname, exc_tb.tb_lineno)  # print: error class, file name, row number
        return "Problems_check_data"

    # example of usage:
    # df_tagged = hole_tagging(df_hole_testing, values_col_name="some_value_name")

    return df_process


def NaN_tagging(df_input, values_col_name):
    """
    we tag as NaN the isnull() events

    INPUTS:
    :param df_input: dataframe                                                        [pandas.DataFrame]
    :param values_col_name: the name of the dataframe column with values              [string]

    :return: new df with NaN tagged in a column called "values_col_name_tag"
    NOTE: Tagging column values will be "is_NaN" OR something else. In this context, "is_NaN"
    measn pandas[element] -> isnull().

    ============================================================
    """
    # instancing DataFrame
    # df_process = pandas.DataFrame()

    log = config.log

    try:
        df_process = df_input.copy()   # copying the input dataframe, to be sure not to have problems with address
        column_tag_name = values_col_name + "_tag"

#        df_process.loc[df_process[values_col_name].isnull(), column_tag_name] = "is_NaN"
        df_process.loc[(df_process[values_col_name].isnull()) & (df_process[column_tag_name] == "Regular"), column_tag_name] = "is_NaN"
        # print("Tagging done")

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log.error("FR: some unexpected error occurred in script {1}: {0}".format(err, fname))
        log.error(exc_type, fname, exc_tb.tb_lineno)  # print: error class, file name, row number
        return "Problems_check_data"

    # example of usage:
    # df_tagged = NaN_tagging(df_NaN_testing, values_col_name="some_value_name")

    return df_process


def threshold_tagging(df_input, values_col_name, lw_trsh=-numpy.inf, up_trsh=numpy.inf):
    """
    we tag above or below certain thresholds events

    INPUTS:
    :param df_input: dataframe                                                        [pandas.DataFrame]
    :param values_col_name: the name of the dataframe column with values              [string]
    :param lw_trsh: lower threshold (default=-inf)                                    [float]
    :param up_trsh: upper threshold (default=inf)                                     [float]

    :return: new df with above/below thresholds tagged in a column called "values_col_name_tag"
    NOTE: Tagging column values will be "Ab_Th", "Be_Th" OR something else.

    ============================================================
    """
    # instancing DataFrame
    # df_process = pandas.DataFrame()
    log = config.log
    try:
        df_process = df_input.copy()   # copying the input dataframe, to be sure not to have problems with address
        column_tag_name = values_col_name + "_tag"

        df_process.loc[(df_process[values_col_name] < lw_trsh) & (df_process[column_tag_name] == "Regular"), column_tag_name] = "Be_Th"
        df_process.loc[(df_process[values_col_name] > up_trsh) & (df_process[column_tag_name] == "Regular"), column_tag_name] = "Ab_Th"
        # print("Tagging done")

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log.error("FR: some unexpected error occurred in script {1}: {0}".format(err, fname))
        log.error(exc_type, fname, exc_tb.tb_lineno)  # print: error class, file name, row number
        return "Problems_check_data"

    # example of usage:
    # df_tagged = threshold_tagging(df_thre_testing, values_col_name="some_value_name", lw_trsh=2, up_trsh=4)

    return df_process


def spikes_tagging(df_input, values_col_name, tags_col_name, der_trsh, spike_delay):  # values_col_name = 'measure_value'
    """
    Tag of spike (or notch) values based on derivative with previous and next samples
    :param df_input: dataframe to be analyzed (pandas.DataFrame)
    :param values_col_name: name of the columns to be analyzed (string)
    :param tags_col_name: name of the column with values tag (string)
    :param der_trsh: threshold of derivative (float): basic: set to the maximum value of the variable, 
    that is saying that a variable cannot vary more than the maximum. 
    For irradiance, you could use 600 W/m2 (about a half of the typical maximum)
    For the temperature, you could use: 60 celsius degrees for the atmospheric temperature
    :param spike_delay: max number of values to identify a spike (int)
    :return: dataframe with updated tags columns
    """
    log = config.log
    df = df_input.copy()
    try:
        log.debug('Note: Spike delay ({0}) is not considered in this function'.format(spike_delay))
        for var in df.columns:
            if '_tag' not in var:
                df['diff'] = df[var].diff()
                df['diff2'] = df[var].diff(-1)
                with numpy.errstate(invalid='ignore'):
                    df = df.assign(diff_sgn=numpy.sign(df['diff']))
                    df = df.assign(diff2_sgn=numpy.sign(df['diff2']))
                # df['diff_sgn'] = numpy.sign(df['diff'])
                # df['diff2_sgn'] = numpy.sign(df['diff2'])
                df_spk = df[(df['diff'].abs() > der_trsh) & (df['diff2'].abs() > der_trsh) & (df['diff_sgn'] != df['diff2_sgn'])]
                df_spk[tags_col_name] = ['Spike'] * len(df_spk)
                df_spk = df_spk.drop([var, 'diff', 'diff2', 'diff_sgn', 'diff2_sgn'], axis=1)
                df.update(df_spk)
                df = df.drop(['diff', 'diff2', 'diff_sgn', 'diff2_sgn'], axis=1)
    except Exception as e:
        log.error('Unable to complete spike data tagging for column {0}: {1}'.format(values_col_name, e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log.error(exc_type, fname, exc_tb.tb_lineno)
        # print: error class, file name, row number
    return df


def frozen_tagging(df_input, values_col_name, tags_col_name, taxonomy, frozen_zero=True, tolerance=0.01):
    """
    Tag of constant values as "Frozen"
    :param df_input: dataframe to be analyzed (pandas.DataFrame)
    :param values_col_name: name of the column to be analyzed (string)
    :param tags_col_name: name of the column with values tag (string)
    :param frozen_zero: if True values constant to 0 will be tagged as "Frozen" (Boolean)
    :param tolerance: pu tolerance on zero values (float)
    :param taxonomy: taxonomy df (pandas.DataFrame)
    :return: dataframe with updated tags columns
    """

    log = config.log
    df = df_input.copy()
    try:
        for var in df.columns:
            if '_tag' not in var:
                up_trsh = float(taxonomy[taxonomy['Signal_Name_from_cloud'] == var]['Upper_limit'])
                df['diff'] = df[var].diff()
                # using diff(15), constant values periods with less than 15 samples will not be tagged as frozen
                if frozen_zero:
                    df_frz = df[df['diff'] == 0].copy()
                else:
                    df_frz = df[(df['diff'] == 0) & (df[var] >= tolerance*up_trsh)].copy()
                df_frz.loc[:, tags_col_name] = 'Frozen'
                df_frz = df_frz.drop([var, 'diff'], axis=1)
                df.update(df_frz)
                df = df.drop('diff', axis=1)
    except Exception as e:
        log.error('Unable to complete frozen data tagging for column {0}: {1}'.format(values_col_name, e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log.error(exc_type, fname, exc_tb.tb_lineno)

    return df


def median_outliers_tagging(df_input, values_col_name, m=3):
    """
    we tag above or below certain thresholds events

    INPUTS:
    :param df_input: dataframe                                                        [pandas.DataFrame]
    :param values_col_name: the name of the dataframe column with values              [string]
    :param m: how many times far apart from the median we consider a datum an outlier  [int]

    :return: new df with outliers tagged in a column called "values_col_name_tag"
    NOTE: Tagging column values will be "Median_Outlier" OR something else.

    ============================================================
    """
    # instancing DataFrame
    # df_process = pandas.DataFrame()
    log = config.log
    try:
        df_process = df_input.copy()  # copying the input dataframe, to be sure not to have problems with address
        column_tag_name = values_col_name + "_tag"

        # we want to calculate the median and MAD only for the "Regular" events, so we do slicing
        df_rel_temp = (df_process[[values_col_name, column_tag_name]]).copy()
        df_reliable = df_rel_temp.loc[df_rel_temp[column_tag_name] == "Regular"]
        var_median = df_reliable[values_col_name].median()
        var_MAD = df_reliable[values_col_name].mad()

        # print("var_median =", var_median)
        # print("(m * var_MAD)", (m * var_MAD))

        df_process.loc[((abs(df_process[values_col_name] - var_median) > (m * var_MAD)) & (
            df_process[column_tag_name] == "Regular")), column_tag_name] = "MedianOut"

    except Exception as err:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        log.error("FR: some unexpected error occurred in script {1}: {0}".format(err, fname))
        log.error(exc_type, fname, exc_tb.tb_lineno)  # print: error class, file name, row number
        return "Problems_check_data"

    # example of usage:
    # df_tagged = outliers_tagging_as_MatLab(df_outliers_testing, values_col_name="value")

    return df_process


# def isolated_outliers_tagging(df_input, df_mapper, values_col_name, outliers_fraction=0.01, debug_mode=False):
#     """
#     ============================================================
#     This function performs an automatic outliers removal assuming a fixed outliers_fraction = (default = 0.01)
#     INPUTS:
#     :param df_input: dataframe                                                                       [pandas dataframe]
#     :param df_mapper: a df mapper                                     [sklearn_pandas.dataframe_mapper.DataFrameMapper]
#     :param outliers_fraction: the apriori threshold outliers_fraction (a priori estimation of n(outliers))      [float]
#     :param debug_mode: the debug mode                                                                         [boolean]
#     :param values_col_name                                                                                     [string]
#     OUTPUTS:
#     :return: df_with_outliers_tag, Pandas dataframe with outliers tagged ONLY for Regular subsample. Isolated= outlier


#     #example of usage:
#     #outlier automatic detection and removal
#     from sklearn_pandas import DataFrameMapper
#     from matplotlib import pyplot as plt

#     feature_1 = "Eta_TOT"
#     feature_2 = 'Q_TOT'
#     to_tag_outlier_mapper = DataFrameMapper([('{0}'.format(feature_1), None), (['{0}'.format(feature_2)], None)])
#     #OBS: could be done with every number of dimension,from 1D to n-D. The idea is get isolated samples

#     df_with_outliers_tag = automatic_outlier_removal(df_G3_for_fit, to_tag_outlier_mapper, values_col_name = "Q_TOT")

#     ============================================================
#     """

#     try:
#         df_process = df_input.copy()  # copying the input dataframe, to be sure not to have problems with address
#         column_tag_name = values_col_name + "_tag"
#         # df_process['index_num'] = range(0, len(df_process))
#         # df_process.set_index('index_num', inplace=True)

#         df_reliable = (df_process.loc[df_process[column_tag_name] == "Regular"]).copy()

#         if len(df_reliable) == 0:
#             print("No regular data found, outliers not tagged")
#             return df_input
#         if debug_mode:
#             print("*_-_*_-_*\n\nAutomatic outliers removal assuming a fixed outliers_fraction (default = 0.01)")

#         # fixing seed to 42, to be able to repeat the outputs. For the specific task, should not be a problem
#         # if the seed is let fixed.
#         rng = numpy.random.RandomState(42)
#         # define classifier (I define a list, to be ready to add additional classifiers)
#         classifiers = {"Isolation Forest": IsolationForest(max_samples=len(df_reliable),
#                                                            contamination=outliers_fraction,
#                                                            random_state=rng, behaviour='new')}

#         # dataset formatting using sklearn-pandas package DataFrameMapper
#         to_tag_outlier = numpy.round(df_mapper.fit_transform(df_reliable[["{}".format(values_col_name)]].copy()), 2)
#         if debug_mode:
#             print("print out to_tag_outlier")
#             print(to_tag_outlier)

#         # use classifiers to learn outliers only in the reliable dataset where tag = Regular
#         # The classification will be then applied to the dataset column_tag_name where the tag is regular
#         for i, (clf_name, clf) in enumerate(classifiers.items()):
#             # fit the data and tag outliers
#             if debug_mode:
#                 print("Data classification using Outlier classifier = {0}".format(clf_name))
#             clf.fit(to_tag_outlier)

#             df_reliable["temp_tag"] = clf.predict(to_tag_outlier)
#             if debug_mode:
#                 print(df_reliable["temp_tag"])

#         # creating dataframe with outliers tagged
#         if debug_mode:
#             print("Setting flag")
#             print("df_reliable")
#             print(df_reliable)
#             print("df_process before")
#             print(df_process)

#         # merge df_process and df_reliable over index
#         df_process = pandas.concat([df_process, df_reliable[["temp_tag"]]], axis=1)
#         if debug_mode:
#             print("df_process after")
#             print(df_process)

#         df_process.loc[(df_process['temp_tag'] == -1) & (
#             df_process[column_tag_name] == "Regular"), column_tag_name] = "Isolated"

#         df_process.drop(columns=['temp_tag'], inplace=True)

#         if debug_mode:
#             print(df_process.info())

#         del df_reliable

#     except Exception as err:
#         exc_type, exc_obj, exc_tb = sys.exc_info()
#         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#         print("FR: some unexpected error occurred in script {1}: {0}".format(err, fname))
#         print(exc_type, fname, exc_tb.tb_lineno)  # print: error class, file name, row number
#         return None

#     return df_process


# def outliers_tagging(df_input, values_col_name, df_mapper=DataFrameMapper([('', None)]),
#                      tag_mode='Isolation', debug_mode=False):
#     """
#     Main function for outiers tagging
#     :param df_input: dataframe                                                        [pandas.DataFrame]
#     :param values_col_name: the name of the dataframe column with values              [string]
#     :param df_mapper: df mapper                                     [sklearn_pandas.dataframe_mapper.DataFrameMapper]
#     :param tag_mode: 'Isolation' or 'Median' mode for outliers tagging              [string]
#     :param debug_mode: the debug mode                                                                         [boolean]
#     :return: df_with_outliers_tag, Pandas dataframe with outliers tagged ONLY for Regular subsample. Isolated= outlier
#     """
#     try:
#         df_process = df_input.copy()  # copying the input dataframe, to be sure not to have problems with address
#         print('Start outiers tagging --> Mode: {}'.format(tag_mode))
#         if tag_mode == 'Isolation':
#             df_with_outliers_tag = isolated_outliers_tagging(df_process, df_mapper, values_col_name,
#                                                              debug_mode=debug_mode)
#             # isolation forest results consitency check
#             reg_number = len(df_process[df_process[values_col_name + '_tag']=='Regular'])
#             out_number = len(df_with_outliers_tag[df_with_outliers_tag[values_col_name + '_tag']=='Isolated'])
#             if reg_number == out_number:
#                 print('Warning: outliers tagging with Isolation Forest mode failed for signal {}, '
#                       'Median outliers tagging will be processed'.format(values_col_name))
#                 df_with_outliers_tag = outliers_tagging_as_MatLab_on_reliable_df(df_process,
#                                                                                  values_col_name=values_col_name)
#             return df_with_outliers_tag
#         elif tag_mode == 'Median':
#             df_with_outliers_tag = outliers_tagging_as_MatLab_on_reliable_df(df_process,
#                                                                              values_col_name=values_col_name)
#             return df_with_outliers_tag
#         else:
#             print("{} not valid tagging mode, try 'Isolation' or 'Median'".format(tag_mode))

#     except Exception as err:
#         exc_type, exc_obj, exc_tb = sys.exc_info()
#         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#         print("FR: some unexpected error occurred in script {1}: {0}".format(err, fname))
#         print(exc_type, fname, exc_tb.tb_lineno)  # print: error class, file name, row number
#         return None





def rolling_spike_tagging(df_raw, var_col, tag_col, N=3, Q=3, sigma=1.5):
  """
  we tag the spikes as local outliers wrt moving average, but averages from bots sides
  (different from rolling mean in standard pandas applications)
      INPUTS:
      :param df_raw: the input dataframe
      :param df_raw: the input dataframe
      :param df_raw: the input dataframe
      :param N: N=3  #N (Lagging): steps in the past T-1, T-2
      :param Q: Q=3  #Q (leading): steps in the future T+1,T+2, T+3
      :param sigma: how many sigma-far we consider the spike

      :return: new df with holes tagged in a column called "values_col_name_Tag"

  """
  try:

    T = N+Q+1   # To take into account T0, also!
    df_tagged =  df_raw.copy()
    
    
    df_tagged["bothrolling"] = df_tagged[[var_col]].rolling(T, min_periods=1).mean().shift(-Q)\
        .fillna(df_tagged[[var_col]][::-1].rolling(T, min_periods=Q).mean().shift(-N)[::-1])
    
    df_tagged["rolstd"] = df_tagged[[var_col]].rolling(T, min_periods=1).std().shift(-Q)\
        .fillna(df_tagged[[var_col]][::-1].rolling(T, min_periods=Q).std().shift(-N)[::-1])
    
    #print("here 3")
    df_tagged["UCL"] = df_tagged["bothrolling"] + sigma*df_tagged["rolstd"] 
    df_tagged["LCL"] = df_tagged["bothrolling"] - sigma*df_tagged["rolstd"] 
    
    #df_tagged[tag_col].loc[df_tagged[var_col] < df_tagged['LCL'] ] = -10  #"negspike"
    #df_tagged[tag_col].loc[df_tagged[var_col] > df_tagged['UCL'] ] = 180 #"posspike"
    
    df_tagged.loc[df_tagged[var_col] < df_tagged['LCL'] , tag_col] = "negspike" #-10  #"negspike"
    df_tagged.loc[df_tagged[var_col] > df_tagged['UCL'] , tag_col] = "posspike" #180 #"posspike"
    
    #df_tagged.loc[: , ["bothrolling", "new_cases"] ].plot(figsize=(12,12))
    
    df_complete = df_tagged.loc[: , [var_col, tag_col] ]


  except Exception as err:
    print(err)

  return df_complete

# example using covid data
do_example = 0
if do_example:
    
    import pandas as pd
    #pd.options.mode.chained_assignment = None
    pd.options.mode.chained_assignment = 'warn'

    df_prov = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv")
    df_LI_raw = df_prov[df_prov["denominazione_provincia"]  == "Livorno"  ]
    df_LI_raw["new_cases"] = df_LI_raw.totale_casi.diff()
    #print(df_LI_raw.tail(3))
    df_LI = df_LI_raw.loc[: , ["data", "new_cases"] ]
      
    df_LI["data"] = pd.to_datetime(df_LI.data)
    df_LI.set_index("data", inplace=True)
    df_LI["new_cases_tagging"] = 0
      
    df_test = rolling_spike_tagging(df_LI, var_col="new_cases", tag_col="new_cases_tagging")
    print(df_test.tail())
    print(df_test.info())
      
    print(df_test.new_cases_tagging.unique())
    df_test.loc[: , [ "new_cases", "new_cases_tagging"] ].plot(figsize=(12,12))













# Quality Check
# I want a test dataframe with some NaNs, some holes, some spikes, some out of thresholds 
# and some outliers so to create a quality_check_chain with a report

test = 1
if test:

    # test df
    import pandas as pd
    import numpy as np
    
    naive_times = pd.date_range(start='2015', end='2016', freq='15Min')
    #mu, sigma = 0, 0.1 # mean and standard deviation
    mu, sigma = 600, 25 # mean and standard deviation
    s = np.random.normal(mu, sigma, len(naive_times))
    
    df = pd.DataFrame({'Signal': s}, index=naive_times)
    df.index.rename("Datetime", inplace=True)
    
    df_mod = df.copy()
    # Holes in the timeseries
    df_mod.drop(pd.Timestamp('2015-01-01 01:00:00'), axis=0, inplace=True)
    df_mod.drop(pd.Timestamp('2015-02-01 01:00:00'), axis=0, inplace=True)
    df_mod.drop(pd.Timestamp('2015-03-01 01:00:00'), axis=0, inplace=True)
    # THRESHOLD crossed
    # setting some elements above threshold 1200
    df_mod.iloc[1011] = df_mod.iloc[2011] = df_mod.iloc[3031] = 1300
    # setting some elements below threshold 0
    df_mod.iloc[1022] = df_mod.iloc[2042] = df_mod.iloc[3052] = -25
    
    #NaN
    # setting some elements to NaN
    df_mod.iloc[23] = df_mod.iloc[53] = df_mod.iloc[73] = np.NaN
    
    # Outliers
    # setting some elements to ouliers
    df_mod.iloc[6064] = df_mod.iloc[6084] = df_mod.iloc[6104] = 1000
    df_mod.iloc[6075] = df_mod.iloc[6155] = df_mod.iloc[6175] = 10
    
    # spikes
    df_mod.iloc[65] = 900
    df_mod.iloc[66] = 950
    df_mod.iloc[67] = 850
    
    
    
    values_col_name = "Signal"
    df_mod["Signal_tag"] = "Regular"
    print(df_mod.Signal_tag.unique())
    
    df_ht = hole_tagging(df_mod, values_col_name, freq_default = "15Min")
    print(df_ht.Signal_tag.unique())
    print(df_ht[df_ht["Signal_tag"]!="Regular"])
    
    df_SP = rolling_spike_tagging(df_ht, values_col_name, values_col_name + "_tag", sigma = 3) 
    print(df_SP.Signal_tag.unique())
    print(df_SP[df_SP["Signal_tag"]!="Regular"])

    
    df_TT = threshold_tagging(df_SP, values_col_name, lw_trsh=0, up_trsh=1200)
    print(df_TT.Signal_tag.unique())
    print(df_TT[df_TT["Signal_tag"]!="Regular"])
    
    
    df_NN = NaN_tagging(df_TT, values_col_name)
    print(df_NN.Signal_tag.unique())
    print(df_NN[df_NN["Signal_tag"]!="Regular"])
    
        
    
    df_final = median_outliers_tagging(df_NN, values_col_name, m=5)
    
    print(df_final.Signal_tag.unique())
    print(df_final[df_final["Signal_tag"]!="Regular"])
    
    categories = {"Regular": 0, 
              "Hole": -333,
              "is_NaN": -999,
              "MedianOut": 111,
              "Ab_Th": 555,
              "Be_Th": -555,
              "negspike": -1100,
              "posspike": 1100}
    

    qc = round( df_final["Signal_tag"].value_counts()["Regular"] / len(df_final), 3)
    print("quality check: regular data are {} on the total dataset".format(qc))

    df_final.replace(categories, inplace=True)
    #df_final.plot()
    #df_final[df_final.Signal_tag!="Regular"  ].plot()
    #df_final[ df_final["Signal_tag"] != 0  ].plot()
    

    ax = df_final["Signal"].plot(color="DarkBlue", label="Regular", ls="", marker = ".")

    if 0:
        for cat in df_final.Signal_tag.unique():
            print(cat)
            if cat != 0:
                df_final[ df_final["Signal_tag"] == cat ]["Signal"].plot(label= cat, ax=ax, ls="", marker = "*")
   

    if 1:
        #df_final[ df_final["Signal_tag"] == -333 ]["Signal_tag"].plot(color="Black", label="Hole", ax=ax, ls="", marker = "*")
        df_final[ df_final["Signal_tag"] != 0 ]["Signal"].plot(color="DarkRed", label="Other Not - regular", ax=ax, ls="", marker = "*")
        df_final[ df_final["Signal_tag"] == -999 ]["Signal_tag"].plot(color="Red", label="is_NaN", ax=ax, ls="", marker = "*")
        df_final[ df_final["Signal_tag"] == 111 ]["Signal"].plot(color="DarkOrange", label="MedianOut", ax=ax, ls="", marker = "*")
        df_final[ df_final["Signal_tag"] == 555 ]["Signal"].plot(color="DarkGreen", label="Ab_Th", ax=ax, ls="", marker = "*")
        df_final[ df_final["Signal_tag"] == -555 ]["Signal"].plot(color="DarkGreen", label="Be_Th", ax=ax, ls="", marker = "*")
        #df_final[ df_final["Signal_tag"] == -1100 ]["Signal"].plot(color="DarkPurple", label="negspike", ax=ax, ls="", marker = "*")
        #df_final[ df_final["Signal_tag"] == 1100 ]["Signal"].plot(color="DarkPurple", label="posspike", ax=ax, ls="", marker = "*")

    ax.legend()
    ax.plot()
    
    print("END test")
    #df_sp = spikes_tagging(df_mod, values_col_name, tags_col_name, der_trsh, spike_delay)
    #print(df_sp.Signal_tag.unique())
    #print(df_sp[df_sp["Signal_tag"]!="Regular"])











