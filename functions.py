import itertools
import os
import pickle
import pandas as pd
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import csv 
from sklearn.calibration import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LassoCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
import joblib


def process_data(excel_path: str, sample_number: int):
    
    """Process data from an Excel file and combine it with the categoric encoded variables.

    This function reads data from an Excel file specified by the 'excel_path' argument,
    extracts information related to a particular sample identified by 'sample_number',
    and combines it with encoded variables from another sheet in the same Excel file.

    Args:
        excel_path (str): The file path to the Excel file containing the data.
        sample_number (int): The identifier of the sample to be processed, but 
        only the number of the sample as the "SF_" is already included inside the function. 

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data, where each row
        corresponds to a specific sample, and columns represent different features
        extracted from the Excel file and encoded variables. The columns format is 
        "offline feature"_"day".

    Example:
        sample = process_data(excel_path = "Clean_data_(SF)\\New Data.xlsx", sample_number = 0)
    """

    sheet_name = "SF_" + str(sample_number)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    new_df = pd.DataFrame()
    codes = "Variable_Encoding"
    df_encoded = pd.read_excel(excel_path, sheet_name=codes)
    new_df_encoded = pd.DataFrame()

    for column in df.columns:
        if column != 'Days':
            for day in df['Days']: 
                new_column_names = f"{column}_{day}"
                new_value = df.loc[df['Days'] == day, column].values[0]
                new_df.loc[sheet_name, new_column_names] = new_value

    new_df = new_df.drop(columns=["Mel_0", "Mel_1_2", "Lip_0", "Lip_1_2"])

    for column in df_encoded.columns:
        if column != 'ID':
            new_value = df_encoded.loc[df_encoded['ID'] == sheet_name, column].values[0]
            new_df_encoded.loc[sheet_name, column] = new_value

    sf_combined = pd.concat((new_df, new_df_encoded), axis=1)
    return sf_combined

def process_multiple_samples(excel_path: str, num_samples: int):

    """Process data for multiple samples from an Excel file and concatenate them into a single DataFrame.

    This function processes data for a specified number of samples from an Excel file,
    combines each sample into a DataFrame, and concatenates them into a single DataFrame.

    Args:
        excel_path (str): The file path to the Excel file containing the data.
        num_samples (int): The number of samples to be processed.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed data for all samples,
        where each row corresponds to a specific sample, and columns represent different features
        extracted from the Excel file corresponding to the values of the offline features in each day of
        the process and categoric encoded variables.

    Example:
        df = process_multiple_samples(excel_path = "Clean_data_(SF)\\New Data.xlsx" ,num_samples = 36)
    """

    all_samples = []
    
    for i in range(1, num_samples + 1):
        sample = process_data(excel_path, i)
        all_samples.append(sample)
    
    df_final = pd.concat(all_samples, axis=0)
    return df_final

def get_differences_between_days(excel_path,sample_number):
    sheet_name = "SF_" + str(sample_number)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    new_df = pd.DataFrame()
    for column in df.columns:
        if column != 'Days' and column != "Bio":
            for i in range(len(df["Days"])-1):
                day_a = df["Days"].iloc[-(i+1)]
                day_b = df["Days"].iloc[-(i+2)] 
                new_column_name = f"diff_{column}_{day_b}_to_{day_a}"
                first_value = df[column].iloc[-(i+1)]
                second_value = df[column].iloc[-(i+2)]
                new_value = first_value - second_value
                new_df.loc[sheet_name, new_column_name] = new_value
    
    new_df = new_df.drop(columns=['diff_Mel_1_2_to_4', 'diff_Mel_0_to_1_2','diff_Lip_1_2_to_4', 'diff_Lip_0_to_1_2'])
    return new_df

def multiple_get_differences_between_days(excel_path: str, num_samples: int):
    all_samples = []
    
    for i in range(1, num_samples + 1):
        sample = get_differences_between_days(excel_path, i)
        all_samples.append(sample)
    
    df_final = pd.concat(all_samples, axis=0)
    return df_final

def mean_sd_for_columns(dataframe: pd.DataFrame, columns: list = None, rows: list = None):
    
    """Calculate the mean and standard deviation for specific columns and rows in a DataFrame.

    This function calculates the mean and standard deviation for specified columns and rows
    in a DataFrame. If columns are not specified, it calculates for all columns in the DataFrame.
    If rows are not specified, it calculates for all rows in the DataFrame.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        columns (list, optional): A list of column names for which to calculate the mean and standard deviation.
            Defaults to None, which means all columns in the DataFrame are used if no columns are specified.
        rows (list, optional): A list of row labels for which to calculate the mean and standard deviation.
            Defaults to None, which means all rows in the DataFrame are used if no rows are specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the mean and standard deviation for each specified column,
        where each row represents a feature, and columns have the 'Mean' and 'Standard Deviation'.

    Example:
        mean_sd_for_columns(df,["Sugar_4","NaNO3_4"],["SF_15","SF_17"])
    """

    if columns is None:
        columns = dataframe.columns  
    
    if rows is None:
        df_selected = dataframe[columns]

    else:
        df_selected = dataframe.loc[rows, columns]

    df_numeric = df_selected.apply(pd.to_numeric, errors='coerce')
    means = df_numeric.mean()
    sd = df_numeric.std()
    
    return pd.DataFrame({'Mean': means, 'Standard Deviation': sd})
    
def carbon_nitrogen_ratio(dataframe: pd.DataFrame, days: list, rows: list = None):

    """Calculate the carbon and nitrogen ratio for specific days and samples in a DataFrame.

    This function calculates the carbon-to-nitrogen (C/N) ratio for each specified day and sample 
    in a given DataFrame. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.

    Args:
        dataframe (DataFrame): The DataFrame containing the data.
        days (list): A list of days for which the C/N ratio is calculated.
            If one wants to calculate the C/N ratio for day "1_2" the list should contain the days in str form,
            otherwise it can be passed as a list of int.
        rows (list, optional): A list of row labels (samples) for which the C/N ratio will be calculated. 
            Defaults to None, which means all rows in the DataFrame are used if no rows are specified.

    Returns:
        DataFrame: A DataFrame containing the calculated C/N ratios for each specified day, indexed by the row labels.

    Example:
        carbon_nitrogen_ratio(df,[4,7],["SF_1","SF_2"])
        carbon_nitrogen_ratio(df,["1_2","4","7"],["SF_1","SF_2"])
    """    

    ratio_values = {}

    if rows is None:
        rows = dataframe.index
    
    for row in rows:
        sample = dataframe.loc[row]  

        for day in days:
            value_carbon = sample[f"Sugar_{day}"]
            value_nitrogen = sample[f'NaNO3_{day}']
            value = value_carbon / value_nitrogen
            ratio_values.setdefault(row, {})[f'C/N_{day}'] = value
    
    return pd.DataFrame(ratio_values).T

def biomass_specific_growth_rate(dataframe: pd.DataFrame, start_day: Union[int, str], end_day: int, rows: list = None):
    
    """Calculates the specific growth rate for specific rows (samples).
    
    This function calculates the specific growth rate for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        start_day (int or str): The first day where there is biomass measurements.
            If the start day is "1_2" it should be in str format, otherwise int format.
        end_day (int): The last day where there is biomass measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the growth rate.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the specific growth rate for each specified row,
        where each row represents a sample, and the column is labeled 'GrowthRate'.
    
    Example:    
        biomass_specific_growth_rate(df,"1_2",7,["SF_1","SF_2"])
    """

    if start_day  == "1_2":
        start_day = 1.5
        time = (end_day - start_day) * 24
        start_day = "1_2"
    if end_day == "1_2":
        end_day = 1.5
        time = (end_day - start_day) * 24
        end_day = "1_2"
    else:
        time = (end_day - start_day) * 24


    growth_rates = {}

    if rows is None:
        rows = dataframe.index
    
    for row_idx in rows:  
        x_0 = dataframe.at[row_idx, f'Bio_{start_day}']
        if x_0 == 0.0:
            x_0 = 1
        x_t = dataframe.at[row_idx, f'Bio_{end_day}']
        growth_rate = np.log(x_t / x_0) / time
        growth_rates[row_idx] = growth_rate
    
    return pd.DataFrame(growth_rates.values(), index=growth_rates.keys(), columns=['GrowthRate'])

def carbon_and_nitrogen_consumption_rate(dataframe: pd.DataFrame, start_day: Union[int, str], end_day: int, rows: list = None):
        
    """Calculates the substrate consupation rate for specific rows (samples).
    
    This function calculates the carbon and nitrogen consumption rate for each specified sample 
    in a given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        start_day (int or str): The first day where there is substrates measurements.
            If the start day is "1_2" it should be in str format, otherwise int format.
        end_day (int): The last day where there is substrate measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the consumption rate.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the consumption rate for each specified row,
        where each row represents a sample, and there's two column one for each substrate.
    
    Example:    
        carbon_and_nitrogen_consumption_rate(df,0,18,["SF_4","SF_5"])   
    """
    if start_day  == "1_2":
        start_day = 1.5
        time = (end_day - start_day) * 24
        start_day = "1_2"
    if end_day == "1_2":
        end_day = 1.5
        time = (end_day - start_day) * 24
        end_day = "1_2"
    else:
        time = (end_day - start_day) * 24

    consumption_rates = {"CarbonConsumptionRate": [], "NitrogenConsumptionRate": []}
    
    if rows is None:
        rows = dataframe.index
    
    for row_idx in rows:  
        sample = dataframe.loc[row_idx] 
        carbon_0 = sample["Sugar_{}".format(start_day)]
        nitrogen_0 = sample['NaNO3_{}'.format(start_day)]
        carbon_7 = sample["Sugar_{}".format(end_day)]
        nitrogen_7 = sample['NaNO3_{}'.format(end_day)]

        carbon_consumption_rate = (carbon_7 - carbon_0) / time
        nitrogen_consumption_rate = (nitrogen_7 - nitrogen_0) / time

        consumption_rates["CarbonConsumptionRate"].append(carbon_consumption_rate)
        consumption_rates["NitrogenConsumptionRate"].append(nitrogen_consumption_rate)

    return pd.DataFrame(consumption_rates, index=rows)

def biomass_substrate_yield(dataframe: pd.DataFrame, start_day: Union[int, str], end_day: int, rows: list = None):
        
    """Calculates the biomass substrate yield for specific rows (samples).
    
    This function calculates the biomass substrate yield for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        start_day (int or str): The first day where there is sugar measurements.
            If the start day is "1_2" it should be in str format, otherwise int format.
        end_day (int): The last day where there is biomass and sugar measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the biomass substrate yield.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the biomass substrate yield for each specified row,
        where each row represents a sample, and the column is labeled 'Yield'.
    
    Example:    
        biomass_substrate_yield(df,0,18,["SF_4","SF_5"])
    """
    
    yields = {}

    if rows is None:
        rows = dataframe.index
    
    for row_idx in rows:  
        x_t = dataframe[f"Bio_{end_day}"].loc[row_idx]
        s_i = dataframe[f"Sugar_{start_day}"].loc[row_idx]
        s_o = dataframe[f"Sugar_{end_day}"].loc[row_idx]
        if s_o == 0 and s_i == 0:
            b_s_yield = 0
        else:
            b_s_yield = x_t/(s_i - s_o)

        yields[row_idx] = b_s_yield
    
    return pd.DataFrame(yields.values(), index=yields.keys(), columns=['Yield'])

def specific_rate_product_formation(dataframe: pd.DataFrame, time: list, end_day: int, rows: list = None):
         
    """Calculates the specific rate of product formation for specific rows (samples).
    
    This function calculates the specific rate of product formation for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        time (list): List of time intervals in days from day 0 to the last day of production.
            Instead of "1_2" in the list, one should put 1.5.
        end_day (int): The last day where there is product measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the specific rate of product formation.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the specific rate of product formation for each specified row,
        where each row represents a sample, and the column is labeled 'Production_rate'.
    
    Example:    
        specific_rate_product_formation(df,[0,1.5,4,7,10,14,18],end_day=18,rows=["SF_4","SF_5"])
    """
    
    production_rates = {}
    times = [t * 24 for t in time]
    if rows is None:
        rows = dataframe.index

    for row_idx in rows:
        biomass = dataframe[f"Bio_{end_day}"].loc[row_idx]
        product = dataframe[f"Mel_{end_day}"].loc[row_idx]
        time_interval = pd.Series(times).diff().mean() 

        production_rate = (1/biomass)*(product/time_interval)
        production_rates[row_idx] = production_rate

    return pd.DataFrame(production_rates.values(), index=production_rates.keys(), columns=['Production_rate'])

def specific_rate_of_substrate_consumption(dataframe: pd.DataFrame, time: list, substrate: str, end_day: Union[int, str], rows: list = None):
             
    """Calculates the specific rate of substrate consumption for specific rows (samples).
    
    This function calculates the specific rate of substrate consumption for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        time (list): List of time intervals in days from day 0 to the last day of production.
            Instead of "1_2" in the list, one should put 1.5.
        substrate (str): The substrate for which the calculation should be made ("Sugar" or "NaNO3")
        end_day (int): The last day where there is substrate measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the specific rate of substrate consumption.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the specific rate of substrate consumtpion for each specified row,
        where each row represents a sample, and the column is labeled 'SpecificRateOf{substrate}Consumption'.
    
    Example:    
        specific_rate_of_substrate_consumption(df,[0,1.5,4,7,10,14,18], substrate= "Sugar", end_day=18,rows=["SF_4","SF_5"])
    """

    times = [t * 24 for t in time]

    if rows is None:
        rows = dataframe.index

    substrate_consumptions = {}

    for row_idx in rows:
        biomass = dataframe[f"Bio_{end_day}"].loc[row_idx]
        substrates = []
        sample = dataframe.loc[row_idx] 

        for day in time:
            if day == 1.5:
                day = "1_2"
            
            substrate_values = sample[f"{substrate}_{day}"]
            substrates.append(substrate_values)     

        time_interval = pd.Series(times).diff().mean()
        substrate_interval = pd.Series(substrates).diff().mean()

        substrate_consumption = (1/biomass)*(substrate_interval/time_interval)
        substrate_consumptions[row_idx] = substrate_consumption
        
    return pd.DataFrame (substrate_consumptions.values(), index=substrate_consumptions.keys(), columns=[f"SpecificRateOf{substrate}Consumption"])

def volumetric_productivity(dataframe: pd.DataFrame, time: list, end_day: int, rows: list = None):
 
    """Calculates the volumetric productivity for specific rows (samples).
    
    This function calculates the volumetric productivity for each specified sample in a
    given Dataframe. If the rows are not specified, it calculates for all the samples present
    in the DataFrame.
    
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        time (list): List of time intervals in days from day 0 to the last day of production.
            Instead of "1_2" in the list, one should put 1.5.
        end_day (int): The last day where there is product measurements.
        rows (list, optional): A list of row labels (samples) for which to calculate the volumetric productivity.
            Defaults to None, which means all rows in the DataFrame are used if not specified.

    Returns:
        pandas.DataFrame: A DataFrame containing the volumetric productivity for each specified row,
        where each row represents a sample, and the column is labeled 'volumetric_productivity'.
    
    Example:    
    volumetric_productivity(df,[0,1.5,4,7,10,14,18], end_day=18, rows=["SF_4","SF_5"])
    """
    times = [t * 24 for t in time]

    if rows is None:
        rows = dataframe.index

    productivity = {}
    
    for row_idx in rows: 

        product = dataframe[f"Mel_{end_day}"].loc[row_idx]
        time_interval = pd.Series(times).diff().mean()   

        v_productivity = (product/time_interval)
        productivity[row_idx] = (v_productivity)

    return pd.DataFrame(productivity.values(), index=productivity.keys(), columns = ["volumetric_productivity"])

def plots (dataframe: pd.DataFrame, compound: str, name: str, save: str, rows: list = None):
    
    """Plot the values of a compound over different days for multiple samples. 

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        compound (str): The name of the compound to plot as it appears in the dataframe and without the day.
        name (str): The name of the compound (used for labeling the y-axis).
        save (str): Whether to save the plot. Provide 'y' to save, 'n' otherwise.
        rows (list, optional): A list of row labels (samples) for which to plot the data.
            Defaults to None, in which case all rows in the DataFrame are used.

    Returns: 
        None
    
    Example:
        plots(df, "Bio", "Biomass", save="n", rows=['SF_19', 'SF_20', 'SF_21'])
    """  

    days = [0, "1_2", 4, 7, 10, 14, 18]
    all_sample_values = []

    if rows is None:
        rows = dataframe.index
        labels = dataframe.index
    else:
        labels = [str(row) for row in rows]

    if compound == "Mel" or "Lip":
        days = [4,7,10,14,18]

    for row in rows:
        sample = dataframe.loc[row]
        sample_values = []  
        sample_days = []
        for day in days:
            value = sample[f"{compound}_{day}"]
            if not np.isnan(value):
                sample_values.append(value)
                sample_days.append(day)
        all_sample_values.append((row, sample_days.copy(), sample_values.copy()))

    plt.figure()

    for sample_values in all_sample_values:
        labels, sample_days, values = sample_values
        plt.plot(sample_days, values, 'o-', label=labels)

    plt.xlabel("Days")
    plt.ylabel(f"{name} g/L")
    # plt.legend()
    plt.tight_layout() 
    if save == "y":
        plt.savefig(compound, dpi=300)
    plt.show()

def bar_charts(dataframe: pd.DataFrame, feature: str, name: str, save: str, rows: list = None):
   
    """
    Create bar charts for a specific feature across multiple samples.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        feature (str): The feature to create bar charts for. 
            Should include the name of the compound and the day, just like it is present in the DataFrame.
        name (str): The name of the feature (used for labeling the plot).
        save (str): Whether to save the plot. Provide 'y' to save, 'n' otherwise.
        rows (list, optional): A list of row labels (samples) for which to create the bar charts.
            Defaults to None, in which case all rows in the DataFrame are used.

    Returns:
        None

    Example:
        bar_charts(df, "Sugar_7", "Sugar Concentration", save="n", rows=['SF_19', 'SF_20', 'SF_21'])
    """

    labels = []
    values = []

    if rows is None:
        rows = dataframe.index
    
    for row in rows:
        sample = dataframe.loc[row]
        label = row
        labels.append(label)

        value = sample[feature]
        values.append(value)
    
    plt.bar(labels, values, color='lightblue')
    plt.ylabel('g/L')
    plt.title(name)
    if save == "y":
        plt.savefig(feature, dpi=300) 
    plt.show()

def multiple_bar_charts(dataframe: pd.DataFrame, feature_1: str, feature_2:str, save: str, name1: str, name2: str, tittle: str, rows: list = None):
    
    """
    Create multiple bar charts comparing two features across multiple samples.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        feature_1 (str): The first feature to compare.
            Should include the name of the compound and the day, just like it is present in the DataFrame.
        feature_2 (str): The second feature to compare.
            Should include the name of the compound and the day, just like it is present in the DataFrame.
        save (str): Whether to save the plot. Provide 'y' to save, 'n' otherwise.
        name1 (str): The name of the first feature (used for labeling the plot).
        name2 (str): The name of the second feature (used for labeling the plot).
        title (str): The title of the plot.
        rows (list, optional): A list of row labels (samples) for which to create the bar charts.
            Defaults to None, in which case all rows in the DataFrame are used.


    Returns:
        None

    Example:
        multiple_bar_charts(df, "Sugar_7", "NaNO3_7", save="n", name1="Sugar Concentration", name2="NaNO3 Concentration", title="Comparison of Sugar and NaNO3 Concentration at day 7", rows=['SF_1', 'SF_2', 'SF_3'])
    """
    from matplotlib.patches import Patch

    labels = []
    values1 = []
    values2 = []
    
    if rows is None:
        rows = dataframe.index

    for row in rows:
        sample = dataframe.loc[row]
        label = row
        value1 = sample[feature_1]
        value2 = sample[feature_2]
        values1.append(value1)
        values2.append(value2)
        labels.append(label)

    bar_width = 0.35
    bar_positions1 = np.arange(len(labels)) + 1
    bar_positions2 = bar_positions1 + bar_width

    plt.bar(bar_positions1, values1, width=bar_width, color='lightblue', label=feature_1)
    plt.bar(bar_positions2, values2, width=bar_width, color='lightcoral', label=feature_2)


    legend_handles = [
        Patch(color='lightblue', label=f'{name1}'),
        Patch(color='lightcoral', label=f'{name2}')
    ]
    plt.ylabel('g/L')
    plt.title(tittle)
    plt.xticks(bar_positions1 + bar_width / 2, labels)
    plt.legend(handles=legend_handles)
    plt.tight_layout()
    if save == "y": 
        plt.savefig(feature_1 + feature_2, dpi=300)
    plt.show()

def shapiro_wilk_test (dataframe: pd.DataFrame):
    """
    Perform Shapiro-Wilk test for normality on each variable in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the Shapiro-Wilk test for each variable, including the 
        value of the statistic and the p-value.

    Example:
        shapiro_wilk_test(df, ['SF_19', 'SF_20', 'SF_21'])
    """
    from scipy.stats import shapiro

    results = {'Variable': [], 'Statistic': [], 'P-value': []}

    for column in dataframe.columns:
        column_data = dataframe[column].replace(0,np.nan).dropna().to_numpy()
        if len(column_data) > 2 and pd.api.types.is_numeric_dtype(dataframe[column]): 
            statistic, p_value = shapiro(column_data)
            results['Variable'].append(column)
            results['Statistic'].append(statistic)
            results['P-value'].append(p_value)
        else:
            results['Variable'].append(column)
            results['Statistic'].append(float('nan'))
            results['P-value'].append(float('nan'))

    return pd.DataFrame(results)

def boxplots(dataframe: pd.DataFrame, *args: str,name: str = None, save: str = None, rows: list = None):
    """
    Generate a boxplot for specified columns in the dataframe, optionally for specified rows.

    Args:
    -----
        `name` (str): The name for the x-axis of the plot.
        `dataframe` (pd.DataFrame): The DataFrame containing the data.
        `save` (str): Whether to save the plot. Provide 'y' to save, 'n' otherwise.
        `*args` (str): Variable-length argument list of features to be plotted.
        `rows` (list, optional): A list of row labels (samples) for plotting. 
            Defaults to None, in which case all rows are used.

    Returns:
    --------
        None

    Example:
    --------
        boxplots("Sugar several days",df, save = "n", "Sugar_10","Sugar_14","Sugar_18", rows= "SF_1", "SF_2", "SF_3", "SF_4", "SF_5", "SF_6", "SF_7", "SF_8", "SF_9")   
    """

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    if rows is None:
        dataframe_used = dataframe
    else:
        dataframe_used = dataframe.iloc[[dataframe.index.get_loc(row) for row in rows]].copy()
           
    mdf = pd.melt(dataframe_used, value_vars=args)
    
    sns.boxplot(x="variable", y="value", data=mdf, width=0.2)
    sns.swarmplot(x='variable', y='value', data=mdf, color="yellow", size=3)
    plt.xlabel(name)
    plt.ylabel("g/L")
    plt.xticks(rotation=10) 
    if save == "y":
         plt.savefig(f"boxplot_{name}", dpi=300)
    plt.title(f"Boxplot with Standard Deviations for {name}")
    plt.show()

def scatterplot_TargetvsVariable (dataframe: pd.DataFrame, variable: str, target: str, title: str, xlabel: str, ylabel: str, save:str, rows: list = None):
    """
    Generate a scatter plot of one variable against a target variable, optionally for specified rows.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        variable (str): The name of the variable to be plotted on the x-axis.
        target (str): The name of the target variable to be plotted on the y-axis.
        title (str): The title of the plot also used for saving.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        save (str): Whether to save the plot. Provide 'y' to save, 'n' otherwise.
        rows (list, optional): A list of row labels (samples) for plotting. 
            Defaults to None, in which case all rows are used.

    Returns:
        None
    
    Example:
        scatterplot_TargetvsVariable(df,"Sugar_4","Mel_7","Sugar vs Mel","Sugar at day 4", "Mel at day 7", save="n", rows="SF_1", "SF_2", "SF_3", "SF_4", "SF_5", "SF_6", "SF_7", "SF_8", "SF_9")

    """
    if rows is None:
        dataframe_used = dataframe
    else:
        dataframe_used = dataframe.iloc[[dataframe.index.get_loc(row) for row in rows]].copy()

    variable1 = dataframe_used[variable]

    target1 = dataframe_used[target]

    sns.scatterplot(x=variable1, y=target1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout() 
    if save == "y":
        plt.savefig(f"scatterplot_{title}", dpi=300)
    plt.show()

def corr_variables(dataframe: pd.DataFrame, categorical_features: str, annot: str, save: str, rows: list = None):
    """
    Generate a heatmap of correlation matrix between numeric variables in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        categorical_features (str): Whether to drop categorical features from the DataFrame.
                                    Pass "y" to drop them, otherwise, leave it blank.
        annot (str): Whether to annotate the heatmap with correlation values.
                     Pass "y" to annotate, otherwise, leave it blank.
        save (str): Whether to save the heatmap as an image.
                    Pass "y" to save, otherwise, leave it blank.
        rows (list, optional): A list of row labels (samples) for the heatmap. 
            Defaults to None, in which case all rows are used.

    Returns:
        None
    
    Example:
        corr_variables(df,"n","n","n", rows= None)   
    """

    if categorical_features == "y":
        dataframe = dataframe.drop(["Glucose", "Glycerol", "CW", "No carbon source used", "SBO", "RO", "WFO"], axis=1)
    
    if rows is None:
        dataframe_used = dataframe
    else:
        dataframe_used = dataframe.iloc[[dataframe.index.get_loc(row) for row in rows]].copy()
    
    new_data = {} 
    for column in dataframe_used.columns:
        if dataframe_used[column].count() > len(dataframe_used) / 2 and pd.api.types.is_numeric_dtype(dataframe_used[column]):
            new_data[column] = dataframe_used[column]

    cleaned_dataframe = pd.DataFrame(new_data)
 
    corr_matrix = cleaned_dataframe.corr("pearson")
    corr_matrix_filtered = corr_matrix.loc[:, (corr_matrix != 0).any(axis=0)]
    corr_matrix_filtered = corr_matrix_filtered[(corr_matrix_filtered.T != 0).any(axis=1)]
    corr_matrix_filtered = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
    names = corr_matrix_filtered

    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix_filtered, cmap='coolwarm')
    if annot == "y":
        for i in range(len(names)):
            for j in range(len(names)):
                plt.text(j + 0.5, i + 0.5, f"{corr_matrix_filtered.iloc[i, j]:.2f}",
                        ha='center', va='center', color='black', fontsize=8) 
    plt.xticks(np.arange(0.5, len(names) + 0.5), names, rotation=45, ha='right',fontsize=8)
    plt.yticks(np.arange(0.5, len(names) + 0.5), names, rotation=0,fontsize=8)
    plt.tight_layout() 
    if save == "y":
        plt.savefig('Corr_variable.png', dpi=300)
    plt.show()

def corr_shakeflasks(dataframe: pd.DataFrame, categorical_features: str, annot: str, save: str, rows: list = None):
    
    """
    Generate a heatmap of correlation matrix for shake flasks data.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        categorical_features (str): Whether to drop categorical features from the DataFrame.
                                    Pass "y" to drop them, otherwise, leave it blank.
        annot (str): Whether to annotate the heatmap with correlation values.
                     Pass "y" to annotate, otherwise, leave it blank.
        save (str): Whether to save the heatmap as an image.
                    Pass "y" to save, otherwise, leave it blank.
        rows (list, optional): A list of row labels (samples) for the heatmap. 
            Defaults to None, in which case all rows are used.

    Returns:
        None
    
    Example:
        corr_shakeflasks(df,"n","y","n",None)
    """

    if categorical_features == "y":
        dataframe = dataframe.drop(["Glucose", "Glycerol", "CW", "No carbon source used", "SBO", "RO", "WFO"], axis=1)
    
    if rows is None:
        dataframe_used = dataframe
    else:
        dataframe_used = dataframe.iloc[[dataframe.index.get_loc(row) for row in rows]].copy()
    
    new_data = {} 
    for column in dataframe_used.columns:
        if dataframe_used[column].count() > len(dataframe_used) / 2 and pd.api.types.is_numeric_dtype(dataframe_used[column]):
            new_data[column] = dataframe_used[column]

    cleaned_dataframe = pd.DataFrame(new_data)
    sf_df = cleaned_dataframe.T

    corr_matrix = sf_df.corr("pearson").abs()
    corr_matrix_filtered = corr_matrix.loc[:, (corr_matrix != 0).any(axis=0)]
    corr_matrix_filtered = corr_matrix_filtered[(corr_matrix_filtered.T != 0).any(axis=1)]
    corr_matrix_filtered = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
    names = corr_matrix_filtered

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix_filtered, cmap='coolwarm')
    if annot == "y":
        for i in range(len(names)):
            for j in range(len(names)):
                plt.text(j + 0.5, i + 0.5, f"{corr_matrix_filtered.iloc[i, j]:.2f}",
                    ha='center', va='center', color='black', fontsize=8)

    plt.xticks(np.arange(0.5, len(names) + 0.5), names, rotation=45, ha='right',fontsize=8)
    plt.yticks(np.arange(0.5, len(names) + 0.5), names, rotation=0,fontsize=8)
    plt.tight_layout() 
    if save == "y":
        plt.savefig('Corr_shakeflasks.png', dpi=300)
    plt.show()

def scatterplot_TargetvsVariables(dataframe: pd.DataFrame, target: str, x_variable1: str, x_variable2: str, xlabel: str, ylabel: str, label1: str, label2: str, save: str, rows: list = None):
    """
    Generate a scatter plot of one target variable against two predictor variables, optionally for specific samples.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        target (str): The name of the target variable to be plotted on the y-axis.
        x_variable1 (str): The name of the first predictor variable to be plotted on the x-axis.
        x_variable2 (str): The name of the second predictor variable to be plotted on the x-axis.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        label1 (str): The label for the first predictor variable on the legend.
        label2 (str): The label for the second predictor variable on the legend.
        save (str): Whether to save the plot. Provide 'y' to save, 'n' otherwise.
        rows (list, optional): A list of row labels (samples) for plotting. 
            Defaults to None, in which case all rows are used.

    Returns:
        None
    
    Example:
        scatterplot_TargetvsVariables(df, "Mel_7", "Sugar_4", "NaNO3_1_2", "Sugar at day 4", "Mel after 7 days (g/L)", "Carbon", "Nitrogen", save="n", rows=None)
    """


    if rows is None:
        dataframe_used = dataframe
    else:
        dataframe_used = dataframe.iloc[[dataframe.index.get_loc(row) for row in rows]].copy()

    variable1 = dataframe_used[x_variable1]
    
    variable2 = dataframe_used[x_variable2]
    
    target1 = dataframe_used[target]

    sns.scatterplot(x=variable1, y=target1, label=label1, color="blue")
    sns.scatterplot(x=variable2, y=target1, label=label2, color="red")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

    plt.tight_layout()
    if save == "y":
        plt.savefig(f"scatterplot_{x_variable1}_{x_variable2}", dpi=300)
    plt.show()

def check (dataframe: pd.DataFrame, columns: list = None, rows: list = None):
    """
    Calculate the percentage of non-null values in each column and row of a DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        columns (list, optional): A list of column names to include in the calculation.
            Defaults to None, in which case all columns are used.
        rows (list, optional): A list of row indices or labels to include in the calculation.
            Defaults to None, in which case all rows are used.

    Returns:
        column_percentages (pd.Series): A Series containing the percentage of non-null values in each column.
        row_percentages (pd.Series): A Series containing the percentage of non-null values in each row.

    Example:
        column_percentages, row_percentages = check(df, columns=['Sugar_0', 'NaNO3_4'], rows=[SF_1, SF_2, SF_3])
    """

    if columns:
        dataframe = dataframe[columns]

    if rows:
        dataframe = dataframe.loc[rows]

    column_percentages = (dataframe.notnull().sum() / len(dataframe)) * 100

    row_percentages = (dataframe.notnull().sum(axis=1) / len(dataframe.columns)) * 100

    return column_percentages, row_percentages

def list_experiment(start: int, end: int):
    """
    Generate a list of experiment names in the format "SF_i", where i ranges from start to end.

    Args:
        start (int): The starting index of the experiment.
        end (int): The ending index of the experiment.

    Returns:
        list: A list of experiment names.

    Example:
        >>> list_experiment(1, 5)
        ['SF_1', 'SF_2', 'SF_3', 'SF_4', 'SF_5']
    """
    experiment = []
    for i in range(start, (end + 1)):
        e = f"SF_{i}"
        experiment.append(e)
    return experiment

def counts (dataframe: pd.DataFrame):
    """
    Count the total number of non-null values in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to count non-null values from.

    Returns:
        int: The total number of non-null values in the DataFrame.

    Example:
        >>> counts(df)
        1000
    """
    counts = 0
    for column in dataframe.columns:
        counted = dataframe[column].count()
        counts += counted
    return counts

def fetch_random_values(dataframe: pd.DataFrame, num_samples:int, save: str = None):
    """
    Fetch random values from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to fetch random values from.
        num_samples (int): The number of random values to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing random values sampled from the input DataFrame.

    Example:
        >>> fetch_random_values(df, 100)
               0
        10  Sugar
        3      Lip
        ...   ...
    """

#     class_1 = []
#     class_2 = []
#     class_3 = []

# for id in TRAIN_Y.index:
#     v = TRAIN_Y.loc[id][0]
#     if v == 0:
#         class_1.append(id)
#     elif v == 1:
#         class_2.append(id)
#     elif v == 2:
#         class_3.append(id)
# one = np.random.choice(class_1,size=4,replace=True)
# two = np.random.choice(class_2,size=3,replace=True)
# three = np.random.choice(class_3,size=3,replace=True)
# print(one, two, three)

    random_rows = np.random.choice(dataframe.index, size=num_samples, replace=True)
    random_columns = np.random.choice(dataframe.columns, size=num_samples, replace=True)
    df = pd.DataFrame(random_columns,random_rows)
    if save == "y":
        df.to_excel("values_to_remove_before_check.xlsx")
    return df

def remove_nonexistent_values_and_duplicates(dataframe: pd.DataFrame, dataframe2: pd.DataFrame, save:str = None):
    """
    Remove rows from dataframe2 where the corresponding value does not exist or is NaN in dataframe.
    Remove duplicate rows from dataframe2.

    Args:
        dataframe (pd.DataFrame): The DataFrame to check for values.
        dataframe2 (pd.DataFrame): The DataFrame containing values to check.
        save (str, optional): Whether to save the filtered DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame with rows removed.

    Example:
        >>> remove_nonexistent_values_and_duplicates(df, df2, save="y")
    """
    rows_to_remove = []
    
    for index, row in dataframe2.iterrows():
        row_index = row.iloc[0] 
        column_name = row.iloc[1] 
        if np.isnan(dataframe.loc[row_index, column_name]):
            rows_to_remove.append(index)    

   
    df2_filtered = dataframe2.drop(index=rows_to_remove)
    
    df2_filtered = df2_filtered.drop_duplicates()
    
    if save == "y":
        df2_filtered.to_csv("values_to_remove_after_check.csv",index="")
    return df2_filtered

def remove_values_df (dataframe: pd.DataFrame, dataframe2: pd.DataFrame, save:str = None):
    """
    Replace values in dataframe based on coordinates specified in dataframe2 with NaN.

    Args:
        dataframe (pd.DataFrame): The DataFrame to modify.
        dataframe2 (pd.DataFrame): The DataFrame containing coordinates for replacement.
        save (str, optional): Whether to save the modified DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: The modified DataFrame with values replaced by NaN.

    Example:
        >>> remove_values_df(df, df2, save="y")
    """
    df_copy = dataframe.copy()

    for _, row in dataframe2.iterrows():
        row_index = row.iloc[0]    
        column_name = row.iloc[1]

        df_copy.loc[row_index, column_name] = np.nan  

    if save == "y":
        df_copy.to_csv("df_with_5%_MV.csv",index_label="ID")

    return df_copy

def remove_x_percent_df (dataframe,num_samples,keep):
    """
    Replace a certain percentage of values in the DataFrame with NaN based on random coordinates,
    removing nonexistent values and duplicates in the process.

    Args:
        dataframe (pd.DataFrame): The DataFrame to modify.
        num_samples (int): The number of random values to fetch for replacement.
        keep (str): Whether to save the modified DataFrame after processing. 
                    "y" to save, any other string to not save.

    Returns:
        None

    Example:
        >>> remove_x_percent_df(df, 100, keep="y")
    """
    fetch_random_values(dataframe,num_samples,save="y")
    coordinates = pd.read_excel("Clean_data_(SF)\\Generated_df\\MV\\values_to_remove_before_check_try.xlsx")
    remove_nonexistent_values_and_duplicates(dataframe,coordinates,save="y")
    coordinates = pd.read_csv("Clean_data_(SF)\\Generated_df\\MV\\values_to_remove_after_check_try.csv")
    remove_values_df(dataframe,coordinates,save=keep)

def generate_y(dataframe:pd.DataFrame,dataframe2:pd.DataFrame):
    """
    Generate a list of values from dataframe based on coordinates specified in dataframe2.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing values.
        dataframe2 (pd.DataFrame): The DataFrame containing coordinates for value extraction.

    Returns:
        list: A list of values extracted from dataframe based on coordinates in dataframe2.

    Example:
        >>> generate_y(df, df2)
    """
    list = []
    for _, row in dataframe2.iterrows():
        row_index = row.iloc[0]
        column_name = row.iloc[1]
        value = dataframe.loc[row_index, column_name]
        list.append(value)
    return list

def bar_for_metrics(dictionary: dict, first_label: str, second_label:str, y_label:str, save: str= None,):
    labels = list(dictionary.keys())
    first_values = [value[0] for value in dictionary.values()]
    second_values = [value[1] for value in dictionary.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_height = 0.35
    index = range(len(labels))
    first_bars = ax.barh(index, first_values, bar_height, label=first_label, color='lightblue')
    second_bars = ax.barh([i + bar_height for i in index], second_values, bar_height, label=second_label, color='lightcoral')

    ax.set_ylabel(y_label)
    ax.set_xlabel('Error')
    ax.set_title(f'{first_label} and {second_label} for Different {y_label}')
    ax.set_yticks([i + bar_height / 2 for i in index])
    ax.set_yticklabels(labels)
    ax.legend()

    for bars in [first_bars, second_bars]:
        for bar in bars:
            width = bar.get_width()
            ax.annotate('{}'.format(round(width, 2)),
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha='left', va='center')
    if save == "y":
         plt.savefig("results.png", dpi=300)
    plt.tight_layout()
    plt.show()

def get_best_param(dict: dict):
    # best_value = 0
    best_value = float('inf')
    for key, value in dict.items():
        if isinstance(value, list) and len(value) > 0:
            if value[0] < best_value:
                best_value = value[0]
                best_param = key
    return best_param, best_value

def percentage_histogram_MV (true: list, predict: list, title:str,save:str = None):
    values = []
    for (i,v) in zip(true,predict):
        actual = i
        predicted = v
        if np.isclose(actual,0,1,1e-3):
            values.append(round((np.abs(actual-predicted))*100,0))
        else:
            values.append(round((np.abs((actual-predicted)/actual))*100,0))
    sns.displot (data=values, kde=True)
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Percentage (%)")
    plt.tight_layout() 
    if save == "y":
        plt.savefig(f"percentange_in_histogram_for_{title}", dpi=300)
    plt.show()

experiments_and_days = {
    'experiment1': ['SF_1', 'SF_2', 'SF_3', 'SF_4', 'SF_5', 'SF_6', 'SF_7', 'SF_8', 'SF_9', 'SF_10', 'SF_11', 'SF_12', 'SF_13', 'SF_14', 'SF_15', 'SF_16', 'SF_17', 'SF_18'],
    'experiment2': ['SF_19', 'SF_20', 'SF_21', 'SF_22', 'SF_23', 'SF_24', 'SF_25', 'SF_26', 'SF_27', 'SF_28'],
    'experiment3': ['SF_29', 'SF_30', 'SF_31', 'SF_32', 'SF_33', 'SF_34', 'SF_35', 'SF_36','SF_47'],
    'experiment4': ["SF_37","SF_38","SF_39","SF_40","SF_41","SF_42","SF_43","SF_44","SF_45","SF_46"],
    'sf_1to16': ['SF_1', 'SF_2', 'SF_3', 'SF_4', 'SF_5', 'SF_6', 'SF_7', 'SF_8', 'SF_9', 'SF_10', 'SF_11', 'SF_12', 'SF_13', 'SF_14', 'SF_15', 'SF_16'],
    'sf_19_26_27': ["SF_19", "SF_26", "SF_27"],
    'sf_20to22_24to25_28to36': ["SF_20", "SF_22", "SF_24", "SF_25", "SF_28", 'SF_29', 'SF_30', 'SF_31', 'SF_32', 'SF_33', 'SF_34', 'SF_35', 'SF_36'],
    'sf_1to2_4to16': ['SF_1', 'SF_2', 'SF_4', 'SF_5', 'SF_6', 'SF_7', 'SF_8', 'SF_9', 'SF_10', 'SF_11', 'SF_12', 'SF_13', 'SF_14', 'SF_15', 'SF_16'],
    'sf_19_23_27': ["SF_19", "SF_23", "SF_27"],
    'sf_20to22_24to26_28to36': ["SF_20", "SF_22", "SF_24", "SF_25", "SF_26", "SF_28", 'SF_29', 'SF_30', 'SF_31', 'SF_32', 'SF_33', 'SF_34', 'SF_35', 'SF_36'],
    'sf_19to22_24to25_27to36': ['SF_19', 'SF_20', 'SF_21', 'SF_22', 'SF_24', 'SF_25', 'SF_27', 'SF_28', 'SF_29', 'SF_30', 'SF_31', 'SF_32', 'SF_33', 'SF_34', 'SF_35', 'SF_36'],
    'days_0to18x': [0, "1_2", 4, 7, 10, 18],
    'days_0to10x': [0, "1_2", 4, 7, 10],
    'days_0to7x': [0, "1_2", 4, 7],
    'days_0to4x': [0, "1_2", 4],
    'days_0to18y': [0, 1.5, 4, 7, 10, 18],
    'days_0to10y': [0, 1.5, 4, 7, 10],
    'days_0to7y': [0, 1.5, 4, 7],
    'days_0to4y': [0, 1.5, 4]
}

coordinates_MV = {
    "SF_19": ["Sugar_7","Sugar_10","NaNO3_10"],
    "SF_20": ["Sugar_10","NaNO3_10"],
    "SF_21": ["Sugar_10","NaNO3_10"],
    "SF_22": ["Sugar_10","NaNO3_10"],
    "SF_23": ["Sugar_1_2","Sugar_7","Sugar_10","NaNO3_0","NaNO3_1_2","NaNO3_4","NaNO3_7","NaNO3_10"],
    "SF_24": ["Sugar_10","NaNO3_10"],
    "SF_25": ["Sugar_10","NaNO3_10"],
    "SF_26": ["Sugar_10","NaNO3_7","NaNO3_10"],
    "SF_27": ["Sugar_7","Sugar_10","NaNO3_10"],
    "SF_28": ["Sugar_10","NaNO3_10"],
    "SF_32": ["Mel_4","Lip_4"],
}

RFR_param = {
    "n_estimators": [1,10,100,1000],
    "criterion": ("squared_error","absolute_error"),
    "max_depth": [None,2],
    "min_samples_split": [2,3,5,7],
    "max_features":("sqrt","log2"),
    "bootstrap":[True,False]
}

RFC_param = {
    "n_estimators": [1,10,100,1000],
    "criterion": ("gini","entropy"),
    "max_depth": [None,2],
    "min_samples_split": [2,3,5,7],
    "max_features":("sqrt","log2"),
    "bootstrap":[True,False]
}

KNR_param = {
    "n_neighbors": [2,3,5,7],
    "weights": ["uniform","distance"],
    "algorithm": ["auto","ball_tree","brute","kd_tree"],
    "metric": ["minkowski","euclidean","manhattan"],
}

BR_param = {
    "tol": [1e-3,1e-6,1],
    "alpha_init": [None,1e-6, 1e-9, 1e-3],
    "lambda_init": [None,1e-6, 1e-9, 1e-3],
}

DT_param = {
    "criterion": ("squared_error","absolute_error"),
    "splitter": ["best","random"],
    "max_depth": [None,2,5],
    "min_samples_split": [2,3,5,7],
    "max_features":("sqrt","log2"),
}

SVR_param = {
    "poly": {
        "kernel": ["poly"],
        "degree": [2, 3, 4],
        'epsilon': [0.001, 0.01, 0.1, 1, 2, 4],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    },
    "rbf": {
        "kernel": ["rbf"],
        'epsilon': [0.001, 0.01, 0.1, 1, 2, 4],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    },
    "sigmoid": {
        "kernel": ["sigmoid"],
        'epsilon': [0.001, 0.01, 0.1, 1, 2, 4],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    },  
    "linear": {
        "kernel": ["linear"],
        'epsilon': [0.001, 0.01, 0.1, 1, 2, 4],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    }
}  

SVC_param = {
    "poly": {
        "kernel": ["poly"],
        "degree": [2, 3, 4],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    },
    "rbf": {
        "kernel": ["rbf"],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    },
    "sigmoid": {
        "kernel": ["sigmoid"],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    },
    "linear": {
        "kernel": ["linear"],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "auto", "scale"], 
        "C": [0.1, 1, 10, 20, 50, 75, 100]
    }
}  

def train_test(rows_to_drop: list, target: pd.DataFrame, case : int, mode:str, df: pd.DataFrame, size: int):
    
    x = df.drop(rows_to_drop,axis=1)    

    y = target

    random_rows = np.random.choice(x.index, size=size, replace=False)
    x_train = x.drop(random_rows)
    y_train = y.drop(random_rows)

    x_test = x.loc[random_rows]
    
    y_test = y.loc[random_rows]

    x_train.to_csv(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{mode}\\Case_{case}\\x_train.csv")
    y_train.to_csv(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{mode}\\Case_{case}\\y_train.csv")
    x_test.to_csv(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{mode}\\Case_{case}\\x_test.csv")
    y_test.to_csv(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{mode}\\Case_{case}\\y_test.csv")


#
#
#
#####Feature engeneering functions

##PCA functions used

def create_pca_features(n_components:int,train_x_scaled:pd.DataFrame,columns,train_x):

    pca = PCA(n_components=n_components)
    pca.fit_transform(train_x_scaled)
    components = pca.components_
    components = np.mean(components, axis=0)
    features_subset = train_x.drop(columns[components <= 0], axis=1)
    return features_subset

def pca_explained_variance(n_components:int,train_x_scaled:pd.DataFrame):
    pca = PCA(n_components=n_components)
    pca.fit_transform(train_x_scaled)
    exp_var_pca = sum(pca.explained_variance_ratio_)

    print(exp_var_pca)

def plot_of_the_cumulative_sum_of_eigenvalues(train_x_scaled: pd.DataFrame,  type_of_experience: str = None, number_of_case: int = None,save: str = None,):
    pca = PCA()

    pca.fit_transform(train_x_scaled)

    exp_var_pca = pca.explained_variance_ratio_

    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\pca_explained_variance.png",dpi=300)


### ANOVA functions

def create_anova_features(k_features:int, train_x: pd.DataFrame, train_y:pd.DataFrame, train_x_columns ):
    selector_anova = SelectKBest(score_func=f_classif, k=k_features)

    X_anova_array = selector_anova.fit_transform(pd.DataFrame.to_numpy(train_x), np.ravel( pd.DataFrame.to_numpy(train_y)))

    feature_cols_anova = train_x_columns[selector_anova.get_support()].tolist()
    features_anova = pd.DataFrame(X_anova_array, columns=feature_cols_anova)
    return features_anova

def plot_ANOVA_F_values(train_x: pd.DataFrame, train_y_array, train_x_columns, type_of_experience: str = None, number_of_case: int = None, save: str = None,):
    fig, (ax1) = plt.subplots(1,figsize=(14, 6))
    f_values, _ = f_classif(train_x, train_y_array)
    ax1.bar(train_x_columns, f_values)
    ax1.set_title('ANOVA F-values')
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_ylabel('F-value')
    plt.tight_layout()
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\ANOVA_f_score.png",dpi=300)


### Lasso functions
def create_lasso_features(cv:int,max_iter:int,train_x_scaled,train_y_array,train_x_columns,train_x,random_state=42):
    lasso = LassoCV(cv=cv,max_iter=max_iter,random_state=random_state)
    lasso.fit(train_x_scaled, train_y_array)

    selected_columns = train_x_columns[np.abs(lasso.coef_) > 1e-9]
    features_lasso = train_x[selected_columns.tolist()]
    return features_lasso

def plot_lasso_coef_values(cv, max_iter,train_x_scaled,train_y_array,train_x_columns,type_of_experience: str = None, number_of_case: int = None, save: str = None,):
    lasso = LassoCV(cv=cv,max_iter=max_iter)
    lasso.fit(train_x_scaled, train_y_array)
    plt.figure(figsize=(18, 6))
    plt.bar(train_x_columns, np.abs(lasso.coef_))
    plt.title('Lasso Coefficients')
    plt.xticks(rotation=90)
    plt.ylabel('Coefficient')
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\lasso_coeficient.png",dpi=300)

###Correlations functions
def create_correlation_features(train_x,correlation_threshold):
    corr_matrix = train_x.corr()

    correlation_threshold=correlation_threshold

    correlation_matrix = train_x.corr(numeric_only=True).abs()

    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper.columns:
        for idx, val in enumerate(upper[column]):
            if val > correlation_threshold:
                if column not in to_drop and upper.index[idx] not in to_drop:
                    to_drop.append(column)

    features_correlation = train_x.drop(columns=to_drop)
    return features_correlation


### RFE functions

def create_RFE_features(n_features_to_select,train_x_scaled, train_y_scaled_array,train_x):
    estimator = SVR(kernel='linear')
    selector = RFE(estimator, n_features_to_select=n_features_to_select)

    selector = selector.fit(train_x_scaled, train_y_scaled_array)

    features_column = selector.get_feature_names_out()

    features_rfe = train_x[features_column.tolist()]

    return features_rfe

def plot_RFE_ranking(train_x_columns,n_features_to_select,train_x_scaled, train_y_scaled_array,type_of_experience: str = None, number_of_case: int = None, save: str = None,):

    estimator = SVR(kernel='linear')
    selector = RFE(estimator, n_features_to_select=n_features_to_select)

    selector = selector.fit(train_x_scaled, train_y_scaled_array)
    plt.figure(figsize=(18, 6))
    plt.bar(train_x_columns, selector.ranking_)
    plt.title('RFE Rankings')
    plt.xticks(rotation=90)
    plt.ylabel('Ranking')
    if save == "y":
        plt.savefig(f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{type_of_experience}\\Case_{number_of_case}\\Feature engineering\\RFE_rankings.png",dpi=300)



##### Regression



### Cross validation


def get_y_pred_cv(x,y,model,cv):
    y_pred = cross_val_predict(estimator=model,X=x,y=y,cv=cv)
    return y_pred

def get_scores(y_true,y_pred,metric):
    score = metric(y_true,y_pred)
    return score.mean()

# Before grid search

def evaluate_model_cv_unscaled(features, scoring, train_x, train_y_array, model, cv):
    results_cv = {}
    for subset in features:
        features_selected = features[subset]
        x = train_x[features_selected.columns]
        y_pred = get_y_pred_cv(x=x, y=train_y_array, model=model,cv=cv)

        for score in scoring:
            value_cv = get_scores(y_true=train_y_array, y_pred=y_pred, metric=scoring[score])
            result_key = f"{subset}_{score}"
            results_cv[result_key] = value_cv
    return results_cv

def evaluate_model_cv_scaled(features, cv, train_x_scaled, train_y_scaled_array, train_y_array, model, scoring, scaler_for_y, train_y_index, train_y_columns):
    results_cv = {}
    for subset in features:
        features_selected = features[subset]
        x = train_x_scaled[features_selected.columns]
        y_pred_scaled = get_y_pred_cv(x=x, y=train_y_scaled_array, model=model, cv=cv)
        y_pred_scaled = pd.DataFrame(y_pred_scaled, index=train_y_index, columns=train_y_columns)
        y_pred = pd.DataFrame(scaler_for_y.inverse_transform(y_pred_scaled), index=train_y_index, columns=train_y_columns)
        y_pred = y_pred.values.ravel()

        for score in scoring:
            value_cv = get_scores(y_true=train_y_array, y_pred=y_pred, metric=scoring[score])
            result_key = f"{subset}_{score}"
            results_cv[result_key] = value_cv
    
    return results_cv

# After grid search

def get_RF_cv_results(subset_key, features, cv, scoring, train_x, train_y_array, param_dict, results_dict):
    params = param_dict[subset_key]
    n_estimators = params['n_estimators']
    criterion = params['criterion']
    max_depth = params['max_depth']
    min_samples_split = params['min_samples_split']
    max_features = params['max_features']
    bootstrap = params['bootstrap']
    
    subset = features[subset_key]
    x = train_x[subset.columns]
    y_pred = get_y_pred_cv(x=x, y=train_y_array, cv=cv, model=RandomForestRegressor(n_estimators=n_estimators,
                                                                       criterion=criterion,
                                                                       max_depth=max_depth,
                                                                       min_samples_split=min_samples_split,
                                                                       max_features=max_features,
                                                                       bootstrap=bootstrap))
    for score in scoring:
        value_cv = get_scores(y_true=train_y_array, y_pred=y_pred, metric=scoring[score])
        result_key = f"{subset_key}_{score}"
        results_dict[result_key] = value_cv


def get_SVR_cv_results(subset_key, features, cv, scoring, train_x_scaled, train_y_scaled_array,train_y_array, param_dict, scaler_y, train_y_index, train_y_columns,results_dict):
    params = param_dict[subset_key]
    kernel = params['kernel']
    gamma = params['gamma']
    C = params['C']
    epsilon = params['epsilon']
    degree = params.get('degree', None)

    subset = features[subset_key]
    x = train_x_scaled[subset.columns]
    if kernel == "poly":
        y_pred_scaled = get_y_pred_cv(x=x, y=train_y_scaled_array,cv=cv, model=SVR(kernel=kernel,
                                                                degree=degree,
                                                                gamma=gamma,
                                                                C=C,
                                                                epsilon=epsilon))
    else:
        y_pred_scaled = get_y_pred_cv(x=x, y=train_y_scaled_array,cv=cv, model=SVR(kernel=kernel,
                                                                gamma=gamma,
                                                                C=C,
                                                                epsilon=epsilon))
    
    y_pred_scaled = pd.DataFrame(y_pred_scaled, index=train_y_index, columns=train_y_columns)
    y_pred = pd.DataFrame(scaler_y.inverse_transform(y_pred_scaled), index=train_y_index, columns=train_y_columns)
    y_pred = y_pred.values.ravel()

    for score in scoring:
        value_cv = get_scores(y_true=train_y_array, y_pred=y_pred, metric=scoring[score])
        result_key = f"{subset_key}_{score}"
        results_dict[result_key] = value_cv



### Grid Search

#Grid Search functions
def sanitize_args_dict(args_dict):
    return "_".join(str(v) for v in args_dict.values())

def save_checkpoint(store, filename='store_checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(store, f)

def load_checkpoint(filename='store_checkpoint.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return {}
    
def stage_experiments(dict_params:dict):
    #get list of lengths of each value associated with a key
    lens = [len(x) for x in dict_params.values()]
    #get a list of arrays with a random permutation for each value 
    permutes = [np.random.permutation(l) for l in lens]
    # make all possible combinations for the indexs
    combinations = list(itertools.product(*permutes))
    return combinations

def save_model(model,mode,case,algorithm,subset,args):
    joblib.dump(model,f"C:\\Users\\Admin\\Desktop\\Tese\\47\\Experiences\\{mode}\\Case_{case}\\GridSearch\\{algorithm}\\{subset}_{args}.pkl")

def load_model(file_path):
    loaded_model = joblib.load(file_path)
    return loaded_model

def search_ML(dict_param, algorithm, x_val_train, y_val_train, x_val_test, y_val_test, train_y_index, train_y_columns, scaler, train_y_as_array, scoring, mode, subset, case,regressor, scale=None,  checkpoint_file='checkpoint.pkl', arg_to_skip=None):
    grid = stage_experiments(dict_param)
    store = load_checkpoint(checkpoint_file)
    print("Initial checkpoint data:", store)
    count = 0

    for experiment in grid:
        args = [v[experiment[i]] for i, v in enumerate(dict_param.values())]
        count += 1
        print(f"Experiment {count}/{len(grid)}: {args}")
        args_dict = {k: args[i] for i, k in enumerate(dict_param)}

        if str(args_dict) in store:
            print(f"Skipping experiment {count}/{len(grid)} (already completed)")
            continue
        else:
            try:
                model = algorithm(**args_dict)
                model.fit(x_val_train,y_val_train)
                args_str = sanitize_args_dict(args_dict)
                save_model(model=model, mode=mode, case=case, algorithm=regressor, subset=subset, args=args_str)
                y_val_test_pred = model.predict(x_val_test) 
                

                if scale == "y":
                    y_val_test_pred_scaled = pd.DataFrame(y_val_test_pred, index=train_y_index, columns=train_y_columns)
                    y_val_test_pred = pd.DataFrame(scaler.inverse_transform(y_val_test_pred_scaled), index=train_y_index, columns=train_y_columns)
                    y_val_test_pred = y_val_test_pred.values.ravel()

                for metric in scoring:
                    value = get_scores(y_true=y_val_test, y_pred=y_val_test_pred, metric=scoring[metric])
                    if str(args_dict) not in store:
                        store[str(args_dict)] = []
                    store[str(args_dict)].append(value)
                print(f"Results for experiment {count}: {store[str(args_dict)]}")
                save_checkpoint(store, checkpoint_file)

            except Exception as e:
                store[str(args_dict)] = []
                save_checkpoint(store, checkpoint_file)
                print(f"Error encountered with args: {args_dict}")
                print(f"Exception: {e}")
                continue
    
    print("Final store contents:", store)
    return store


def transform_csv_into_dict(path: str) -> dict:
    data_dict = {}

    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            if len(row) == 2:
                # Convert the parameter string to a dictionary
                param_dict = ast.literal_eval(row[0])
                
                # Convert any list in the parameter dictionary to a tuple
                for key, value in param_dict.items():
                    if isinstance(value, list):
                        param_dict[key] = tuple(value)
                
                # Convert the scores string to a list of floats
                scores = ast.literal_eval(row[1])
                
                # Convert the param_dict to a tuple of tuples
                key = tuple(sorted(param_dict.items()))

                # Store in the dictionary
                data_dict[key] = scores

    return data_dict


### Testing

# Before grid search

def train_evaluate_model(classifier,train_x,train_y,test_x):
    classifier.fit(train_x,train_y)
    y_pred_train = classifier.predict(train_x)
    y_pred_test = classifier.predict(test_x)

    return y_pred_test, y_pred_train

def get_scores_testing(y_true_test,y_pred_test,y_true_train,y_pred_train,metric):
    score_test = metric(y_true_test,y_pred_test)
    score_train = metric(y_true_train,y_pred_train)
    return score_test,score_train

def testing_results_before_gs_unscaled(features,scoring,test_x,model,train_y_array,test_y_array,y_pred_model_test,y_pred_model_train,results_test_model,results_train_model):

    for subset in features:
        result_key2 = f"{subset}" 
        features_selected = features[subset]
        test_x_subset = test_x[features_selected.columns]
        y_pred_test, y_pred_train = train_evaluate_model(classifier=model,
                                                    train_x=features[subset],
                                                    train_y=train_y_array,
                                                    test_x=test_x_subset)
        y_pred_model_test[result_key2] = y_pred_test
        y_pred_model_train[result_key2] = y_pred_train

        for score in scoring:
            value_test,value_train = get_scores_testing(y_true_test=test_y_array,
                                                    y_pred_test=y_pred_test,
                                                    y_true_train=train_y_array,
                                                    y_pred_train=y_pred_train,
                                                    metric=scoring[score])    
            result_key1 = f"{subset}_{score}"
            results_test_model[result_key1] = value_test
            results_train_model[result_key1] = value_train

def testing_results_before_gs_scaled(features,scoring,test_x_scaled,train_x_scaled,model,train_y_scaled_array,test_y_array,train_y_array,train_y_index,train_y_columns,test_y_index,test_y_columns,scaler_y,y_pred_model_test,y_pred_model_train,results_test_model,results_train_model):
    for subset in features:
        result_key2 = f"{subset}"       
        features_selected = features[subset]
        train_x_subset_scaled = train_x_scaled[features_selected.columns]
        test_x_subset_scaled = test_x_scaled[features_selected.columns]
        y_pred_test_scaled, y_pred_train_scaled = train_evaluate_model(classifier=model,
                                                     train_x=train_x_subset_scaled,
                                                     train_y= train_y_scaled_array,
                                                     test_x=test_x_subset_scaled)
    
        y_pred_train_scaled = pd.DataFrame(y_pred_train_scaled,index=train_y_index,columns=train_y_columns)
        y_pred_train = pd.DataFrame(scaler_y.inverse_transform(y_pred_train_scaled),index=train_y_index,columns=train_y_columns)
        y_pred_train = y_pred_train.values.ravel()
        y_pred_test_scaled = pd.DataFrame(y_pred_test_scaled,index=test_y_index,columns=test_y_columns)
        y_pred_test = pd.DataFrame(scaler_y.inverse_transform(y_pred_test_scaled),index=test_y_index,columns=test_y_columns)
        y_pred_test = y_pred_test.values.ravel()
        y_pred_model_train[result_key2] = y_pred_train
        y_pred_model_test[result_key2] = y_pred_test

        for score in scoring:
            value_test,value_train = get_scores_testing(y_true_test=test_y_array,
                                                    y_pred_test=y_pred_test,
                                                    y_true_train=train_y_array,
                                                    y_pred_train=y_pred_train,
                                                    metric=scoring[score])
            result_key1 = f"{subset}_{score}"
            results_test_model[result_key1] = value_test
            results_train_model[result_key1] = value_train

# Atfer grid search

def get_RF_test_results(subset_key, features, scoring, train_y_array, test_x, test_y_array, param_dict, results_test_dict, results_train_dict, y_pred_test_dict, y_pred_train_dict):
    
    params = param_dict[subset_key]
    n_estimators = params['n_estimators']
    criterion = params['criterion']
    max_depth = params['max_depth']
    min_samples_split = params['min_samples_split']
    max_features = params['max_features']
    bootstrap = params['bootstrap']
    
    subset = features[subset_key]
    test_x_subset = test_x[subset.columns]
    y_pred_test, y_pred_train = train_evaluate_model(classifier=RandomForestRegressor(n_estimators=n_estimators,
                                                                                      criterion=criterion,
                                                                                      max_depth=max_depth,
                                                                                      min_samples_split=min_samples_split,
                                                                                      max_features=max_features,
                                                                                      bootstrap=bootstrap),
                                                     train_x=subset,
                                                     train_y=train_y_array,
                                                     test_x=test_x_subset)
    y_pred_test_dict[subset_key] = y_pred_test
    y_pred_train_dict[subset_key] = y_pred_train

    for score in scoring:
        value_test, value_train = get_scores_testing(y_true_test=test_y_array,
                                                     y_pred_test=y_pred_test,
                                                     y_true_train=train_y_array,
                                                     y_pred_train=y_pred_train,
                                                     metric=scoring[score])    
        result_key = f"{subset_key}_{score}"
        results_test_dict[result_key] = value_test
        results_train_dict[result_key] = value_train


def get_SVR_test_results(subset_key, features, scoring, train_x_scaled, train_y_scaled_array, train_y_array, test_x_scaled, test_y_array, param_dict, scaler_y, train_y_index, train_y_columns, test_y_index, test_y_columns, results_test_dict, results_train_dict, y_pred_test_dict, y_pred_train_dict):
    params = param_dict[subset_key]
    kernel = params['kernel']
    gamma = params['gamma']
    C = params['C']
    epsilon = params['epsilon']
    degree = params.get('degree', None)

    subset = features[subset_key]
    train_x_subset_scaled = train_x_scaled[subset.columns]
    test_x_subset_scaled = test_x_scaled[subset.columns]
    
    if kernel == "poly":
        y_pred_test_scaled, y_pred_train_scaled = train_evaluate_model(classifier=SVR(kernel=kernel,
                                                                        gamma=gamma,
                                                                        C=C,
                                                                        degree=degree,
                                                                        epsilon=epsilon),
                                                        train_x=train_x_subset_scaled,
                                                        train_y=train_y_scaled_array,
                                                        test_x=test_x_subset_scaled)
    else:
        y_pred_test_scaled, y_pred_train_scaled = train_evaluate_model(classifier=SVR(kernel=kernel,
                                                                        gamma=gamma,
                                                                        epsilon=epsilon,
                                                                        C=C,),
                                                        train_x=train_x_subset_scaled,
                                                        train_y=train_y_scaled_array,
                                                        test_x=test_x_subset_scaled)           

    y_pred_train_scaled = pd.DataFrame(y_pred_train_scaled, index=train_y_index, columns=train_y_columns)
    y_pred_train = pd.DataFrame(scaler_y.inverse_transform(y_pred_train_scaled), index=train_y_index, columns=train_y_columns)
    y_pred_train = y_pred_train.values.ravel()
    y_pred_test_scaled = pd.DataFrame(y_pred_test_scaled, index=test_y_index, columns=test_y_columns)
    y_pred_test = pd.DataFrame(scaler_y.inverse_transform(y_pred_test_scaled), index=test_y_index, columns=test_y_columns)
    y_pred_test = y_pred_test.values.ravel()
    
    y_pred_train_dict[subset_key] = y_pred_train
    y_pred_test_dict[subset_key] = y_pred_test

    for score in scoring:
        value_test, value_train = get_scores_testing(y_true_test=test_y_array,
                                                     y_pred_test=y_pred_test,
                                                     y_true_train=train_y_array,
                                                     y_pred_train=y_pred_train,
                                                     metric=scoring[score])
        result_key = f"{subset_key}_{score}"
        results_test_dict[result_key] = value_test
        results_train_dict[result_key] = value_train


### Plots for testing

def plot_y_true_pred(true_y: pd.DataFrame, target: str, pred_y:dict, model: str, save: str = None):
    y_true_labels = true_y.index.tolist()
    y_true_values = true_y[target].values

    plt.figure(figsize=(10, 6))

    plt.scatter(y_true_labels, y_true_values,color="red", label='Experimental Values',marker='D',s=80)

    colors = ['b', 'g', 'c', 'm', 'y', 'orange']

    for i, (key, y_pred) in enumerate(pred_y.items()):
        plt.scatter(y_true_labels, y_pred, color=colors[i % len(colors)],label=key)

    plt.ylim(bottom=0)


    # plt.title(f'Experimental vs Predicted Values for different feature subsets using {model}')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Samples',fontsize = 15)
    plt.ylabel('MELs concentration values (g/L)', fontsize = 15)
    plt.legend(fontsize = 10)
    plt.grid(False)
    if save == "y":
         plt.savefig(f"plot_y_pred_and_true_for_{model}.png", dpi=300)
    plt.show()

def percentage_histogram(true_y: pd.DataFrame, target: str, pred_y: dict, model: str, save: str = None):
    y_true_values = true_y[target].values

    num_subplots = len(pred_y)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (key, y_pred) in enumerate(pred_y.items()):
        values = []
        for actual, predicted in zip(y_true_values, y_pred):
            if np.isclose(actual, 0, 1, 1e-3):
                values.append(round((np.abs(actual - predicted)) * 100, 0))
            else:
                values.append(round((np.abs((actual - predicted) / actual)) * 100, 0))
        
        sns.histplot(values, kde=True, ax=axes[idx])
        axes[idx].set_title(f'{model} - {key}')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xlabel('Percentage (%)')
    
    plt.tight_layout()
    if save == "y":
        plt.savefig(f"percentage_histogram_for_{model}.png", dpi=300)
    plt.show()

def residual_histogram(true_y: pd.DataFrame, target: str, pred_y: dict, model: str, save: str = None, xlim=None, ylim=None):
    y_true_values = true_y[target].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (key, y_pred) in enumerate(pred_y.items()):
        residuals = y_true_values - y_pred
        
        sns.histplot(residuals, kde=True, ax=axes[idx])
        axes[idx].set_title(f'{model} - {key}')
        axes[idx].set_ylabel('Count')
        axes[idx].set_xlabel('Residuals')
        
        # Set x and y limits if specified
        if xlim:
            axes[idx].set_xlim(xlim)
        if ylim:
            axes[idx].set_ylim(ylim)

    plt.tight_layout()
    if save == "y":
        plt.savefig(f"residual_histogram_for_{model}.png", dpi=300)
    plt.show()





#### Classification

def evaluate_model_cv_unscaled_classification(features, metrics, train_x, train_y_array, model, cv, multiclass= False):
    results_cv = {}
    conf_matrices_cv = {}
    for subset in features:
        features_selected = features[subset]
        x = train_x[features_selected.columns]
        y_pred = get_y_pred_cv(x=x, y=train_y_array, model=model, cv=cv)
        conf_matrices_cv[subset] = confusion_matrix(train_y_array, y_pred)

        for metric in metrics:
            if multiclass and metric != "Accuracy":
                result = metrics[metric](train_y_array,y_pred,average="macro")
            else:
                result = metrics[metric](train_y_array,y_pred)
                
            result_key = f"{subset}_{metric}"
            results_cv[result_key] = result
    return results_cv, conf_matrices_cv

def evaluate_model_cv_scaled_classification(features, cv, train_x_scaled, train_y_array, model, metrics, multiclass = False):
    results_cv = {}
    conf_matrices_cv = {}
    for subset in features:
        features_selected = features[subset]
        x = train_x_scaled[features_selected.columns]
        y_pred = get_y_pred_cv(x=x, y=train_y_array, model=model, cv=cv)

        conf_matrices_cv[subset] = confusion_matrix(train_y_array, y_pred)

        for metric in metrics:
            if multiclass and metric != "Accuracy":
                result = metrics[metric](train_y_array,y_pred,average="macro")
            else:
                result = metrics[metric](train_y_array,y_pred)

            result_key = f"{subset}_{metric}"
            results_cv[result_key] = result
    return results_cv, conf_matrices_cv

def results_to_table_image(RF_results, ANN_results, SVC_results, file_name, save=False):
    # Create DataFrames for each model
    RF_df = pd.DataFrame(list(RF_results.items()), columns=['Feature_Metric', 'Score']).assign(Model='RF')
    ANN_df = pd.DataFrame(list(ANN_results.items()), columns=['Feature_Metric', 'Score']).assign(Model='ANN')
    SVC_df = pd.DataFrame(list(SVC_results.items()), columns=['Feature_Metric', 'Score']).assign(Model='SVC')

    # Split 'Feature_Metric' into 'Feature Subset' and 'Metric'
    for df in [RF_df, ANN_df, SVC_df]:
        df[['Feature Subset', 'Metric']] = df['Feature_Metric'].str.rsplit('_', n=1, expand=True)
        df.drop(columns=['Feature_Metric'], inplace=True)

    # Merge DataFrames
    merged_df = pd.merge(RF_df, ANN_df, on=['Feature Subset', 'Metric'], suffixes=('_RF', '_ANN'))
    merged_df = pd.merge(merged_df, SVC_df, on=['Feature Subset', 'Metric'])
    merged_df.rename(columns={'Score': 'Score_SVC'}, inplace=True)

    final_df = merged_df[['Feature Subset', 'Metric', 'Score_RF', 'Score_ANN', 'Score_SVC']]
    # Format the scores
    final_df.loc[:, ['Score_RF', 'Score_ANN', 'Score_SVC']] = final_df[['Score_RF', 'Score_ANN', 'Score_SVC']].applymap(lambda x: f"{x:.3f}")

    # Function to save DataFrame as an image
    def save_df_as_image(df, file_name):
        fig, ax = plt.subplots(figsize=(15, 8))  # set size frame
        ax.axis('tight')
        ax.axis('off')
        
        the_table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.2, 1.2)  # adjust size
        
        if save:
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1, dpi=300)

    save_df_as_image(final_df, file_name)
    
    plt.show()

def plot_confusion_matrices(conf_matrices, classifier_name, save=False, multiclass = False):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle(f'Confusion Matrices for {classifier_name}', fontsize=16)
    
    for ax, (subset, cm) in zip(axes, conf_matrices.items()):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(title=subset, ylabel='True label', xlabel='Predicted label')
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize = 15)
        if multiclass:
            ax.set_xticks([0,1,2])
            ax.set_yticks([0,1,2])
        else:
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.set_title(subset, fontsize=15)
        if save:

            plt.savefig( f"Confusion matrix for {classifier_name}", bbox_inches='tight', pad_inches=0.1,dpi=300)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def testing_classification_before_gs_unscaled(features, metrics, test_x, train_x, train_y_array, test_y_array, classifiers,multiclass = False):
    results_test = {}
    results_train = {}
    y_pred_test = {}
    y_pred_train = {}
    conf_matrices_test = {}

    for model_name, model in classifiers.items():
        for subset in features:
            result_key2 = f"{subset}"       
            features_selected = features[subset]
            test_x_subset = test_x[features_selected.columns]
            y_pred_test_subset, y_pred_train_subset = train_evaluate_model(
                classifier=model,
                train_x=train_x[features_selected.columns],
                train_y=train_y_array,
                test_x=test_x_subset
            )

            y_pred_test[f"{model_name}_{result_key2}"] = y_pred_test_subset
            y_pred_train[f"{model_name}_{result_key2}"] = y_pred_train_subset
            conf_matrices_test[f"{model_name}_{subset}"] = confusion_matrix(test_y_array, y_pred_test_subset)

            for metric in metrics:
                if multiclass and metric != "Accuracy":
                    value_train = metrics[metric](train_y_array,y_pred_train_subset,average="macro")
                    value_test = metrics[metric](test_y_array,y_pred_test_subset,average="macro")

                else:
                    value_train = metrics[metric](train_y_array,y_pred_train_subset)
                    value_test = metrics[metric](test_y_array,y_pred_test_subset)

                result_key1 = f"{subset}_{metric}"
                results_test[result_key1] = value_test
                results_train[result_key1] = value_train

    return results_test, results_train, y_pred_test, y_pred_train, conf_matrices_test

def testing_classification_before_gs_scaled(features, metrics, test_x_scaled, train_x_scaled, train_y_array, test_y_array, classifiers,multiclass=False):
    results_test = {}
    results_train = {}
    y_pred_test = {}
    y_pred_train = {}
    conf_matrices_test = {}

    for model_name, model in classifiers.items():
        for subset in features:
            result_key2 = f"{subset}"       
            features_selected = features[subset]
            train_x_subset_scaled = train_x_scaled[features_selected.columns]
            test_x_subset_scaled = test_x_scaled[features_selected.columns]
            y_pred_test_scaled, y_pred_train_scaled = train_evaluate_model(
                classifier=model,
                train_x=train_x_subset_scaled,
                train_y=train_y_array,
                test_x=test_x_subset_scaled
            )

            y_pred_test[f"{model_name}_{result_key2}"] = y_pred_test_scaled
            y_pred_train[f"{model_name}_{result_key2}"] = y_pred_train_scaled
            conf_matrices_test[f"{model_name}_{subset}"] = confusion_matrix(test_y_array, y_pred_test_scaled)

            for metric in metrics:
                if multiclass and metric != "Accuracy":
                    value_train = metrics[metric](train_y_array,y_pred_train_scaled,average="macro")
                    value_test = metrics[metric](test_y_array,y_pred_test_scaled,average="macro")

                else:
                    value_train = metrics[metric](train_y_array,y_pred_train_scaled)
                    value_test = metrics[metric](test_y_array,y_pred_test_scaled)

                result_key1 = f"{subset}_{metric}"
                results_test[result_key1] = value_test
                results_train[result_key1] = value_train

    return results_test, results_train, y_pred_test, y_pred_train, conf_matrices_test

def plot_confusion_matrix(cm, classifier_name, subset='Confusion Matrix', multiclass=False, save=False):

    """
    Plots a single confusion matrix for a classifier.
    
    Parameters:
    - cm: The confusion matrix (2D array or matrix).
    - classifier_name: A string representing the classifier's name.
    - subset: A string indicating the name of the dataset (e.g., 'Train', 'Test').
    - multiclass: Boolean flag, if True assumes multi-class classification, otherwise binary classification.
    - save: Boolean flag, if True saves the plot as a file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Display the confusion matrix as an image
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    ax.figure.colorbar(im, ax=ax)
    
    # Set axis labels and title
    ax.set(title=f'{subset} - {classifier_name}', ylabel='True label', xlabel='Predicted label')
    
    # Determine format for the values in the confusion matrix (integers)
    fmt = 'd'
    thresh = cm.max() / 2.0
    
    # Add the text annotations inside the matrix cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=15)
    
    # Adjust ticks based on whether it's a multi-class or binary classification
    if multiclass:
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
    else:
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    
    # Optionally save the figure
    if save:
        plt.savefig(f'{classifier_name}_{subset}_confusion_matrix.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    # Display the plot
    plt.tight_layout()
    plt.show()



    