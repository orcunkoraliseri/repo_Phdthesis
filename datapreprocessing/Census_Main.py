import pandas as pd
def remove_trailing_zero(value):
    str_value = str(value)
    if str_value.endswith('.0'):
        return str_value[:-2]
    return str_value

def convert_if_integer(value):
    try:
        if value.is_integer():
            return int(value)
    except AttributeError:
        pass
    return value

def convert_dtyes(df):
    # Loop through each column and try to convert it to integer
    for column in df.columns:
        try:
            df[column] = df[column].astype(int)
        except ValueError:
            raise ValueError(f"Column '{column}' contains NaN values and cannot be converted to integer.")

    df = df.applymap(convert_if_integer)
    df = df.applymap(remove_trailing_zero)

    return df

def merge(df1_path, df2_path, refCol,  out_csv=None):
    import pandas as pd

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    df1 = convert_dtyes(df1)
    df2 = convert_dtyes(df2)

    # Merge DataFrames using the 'id' column
    '''
    "inner": This is the intersection of keys from both dataframes, which is the default option. It only keeps rows where the key exists in both dataframes.
    "outer": This is the union of keys from both dataframes. It keeps all rows, filling in NaN where there are missing matches.
    "left": This uses only keys from the left dataframe. It keeps all rows from the left dataframe, but fills in NaN where there are missing matches in the right dataframe.
    "right": This uses only keys from the right dataframe. It keeps all rows from the right dataframe, but fills in NaN where there are missing matches in the left dataframe.
    '''
    df = pd.merge(df1, df2, on=refCol, how='outer') # no inner, outer, right
    #some columns are repeated, it can be detected using 'Accommodation ID' columns
    df.drop_duplicates(keep='first', inplace=True)
    '''housing and occupant datasets do not completely match each other due to missingness, thus, NaN rows are deleted
        so auto completion are needed
    '''
    df = df.dropna()
    df = df[['Residential_ID', 'Family_ID', 'Occupant_ID_in_HH',
             'Region', 'Number Family Members',
             'Gender', 'Age Classes', 'Marital Status', 'Education Degree',
             'Employment status', 'Kinship Relationship',
             'Job Type', 'Family_Typology_Simple', 'Citizenship', 'Economic Sector, Profession',
             'Room Count', 'Internet Access',
             'Mobile Phone Ownership', 'Car Ownership', 'Home Ownership', "Full_Part_time", "Permanent/fixed",
             "House Area",
             ]]

    # Rename column from "Family_ID" to "Household_ID"
    df.rename(columns={'Family_ID': 'Household_ID'}, inplace=True)

    #OUTPUT
    df.to_csv(out_csv, index=None)
    print('merge function: writing as .csv is done')

def merge_preprocessing(input_path, output_path_csv=None, to_csv=False,):
    from preProcessing_Func import preprocessing_df as preDF

    import pandas as pd

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    # PREPROCESS 1: dimension reduction
    preDF.merge_categories(df, "Education Degree", [0, 1, ], 1)
    preDF.merge_categories(df, "Education Degree", [2,3,4], 2)
    preDF.merge_categories(df, "Education Degree", [5, 6, 7], 3)

    preDF.merge_categories(df, "Family Typology", [1,2], 2)

    preDF.merge_categories(df, "Employment status", [2, 3], 3)
    preDF.merge_categories(df, "Employment status", [1], 2)

    preDF.merge_categories(df, "Age Classes", [1, 2,3], 1)
    preDF.merge_categories(df, "Age Classes", [4,5], 2)
    preDF.merge_categories(df, "Age Classes", [6,7], 3)
    preDF.merge_categories(df, "Age Classes", [8,9], 4)
    preDF.merge_categories(df, "Age Classes", [10, 11], 5)
    preDF.merge_categories(df, "Age Classes", [12, 13, 14], 6)

    # PREPROCESS 2: outlier cleaning
    df = preDF.filter_dataframe(df, "Room Count", 6)

    # PREPROCESS 3: drop unnecessary or complex columns
    df = df.drop(["Administrative Region", "Room Count"], axis=1)

    # PREPROCESS 4: manual imputation
    from Census_Functions import occ_in_HH_completion
    df = occ_in_HH_completion(df)

    preDF.merge_categories(df, "Occupant ID in HH", [1, 1.0], 1.0)

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

def impute_KNN_evaluate(input_path, output_path_csv=None, to_csv=False):
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.impute import KNNImputer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    # Filter the data according to 'Occupant ID in HH'
    filter_mask = df['Occupant ID in HH'] == 1
    filtered_df = df[filter_mask]

    filtered_df = filtered_df.sample(frac=0.1, replace=True, random_state=1)

    # Detect the columns to be imputed
    impute_columns = filtered_df.columns[filtered_df.isnull().any()].tolist()

    # Split the filtered data
    train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42)

    # Save the true values for later comparison
    true_values = test_df[impute_columns].copy()

    # Introduce missingness in the test data
    nan_mask = np.random.choice(test_df.index, replace=False, size=int(test_df.shape[0] * 0.5))
    test_df.loc[nan_mask, impute_columns] = np.nan

    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=3))
    ])

    # Apply pipeline to the training and test data
    imputed_train_df = pd.DataFrame(pipeline.fit_transform(train_df[impute_columns]), columns=impute_columns)
    imputed_test_df = pd.DataFrame(pipeline.transform(test_df[impute_columns]), columns=impute_columns)

    # Calculate the accuracy for each column, this is modified specifically for imputation process
    scores  = {}
    for column in impute_columns:
        mask = np.logical_not(np.isnan(true_values[column]))
        mask = mask.reset_index(drop=True)  # Reset the index of the mask
        scores[column] = accuracy_score(true_values[column].reset_index(drop=True)[mask],
                                                 np.round(imputed_test_df[column].reset_index(drop=True)[mask]))

    # Insert imputed values back into the original DataFrame
    df = df.reset_index(drop=True)
    df.loc[filter_mask, impute_columns] = pd.concat([imputed_train_df, imputed_test_df], ignore_index=True)

    print(scores)
    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

def impute_RF_evaluate(input_path, output_path_csv=None, to_csv=False, scaling=False):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import MinMaxScaler

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    # Filter the data according to 'Occupant ID in HH'
    filter_mask = df['Occupant ID in HH'] == 1
    filtered_df = df[filter_mask]

    filtered_df = filtered_df.sample(frac=0.1, replace=True, random_state=1)

    # Detect the columns to be imputed
    impute_columns = filtered_df.columns[filtered_df.isnull().any()].tolist()

    # Split the filtered data
    train_df, test_df = train_test_split(filtered_df, test_size=0.2, random_state=42)

    # Save the true values for later comparison
    true_values = test_df[impute_columns].copy()

    # Introduce missingness in the test data
    nan_mask = np.random.choice(test_df.index, replace=False, size=int(test_df.shape[0] * 0.5))
    test_df.loc[nan_mask, impute_columns] = np.nan

    # Create a RandomForestRegressor model for each column with missing data and use it to impute missing values
    for column in impute_columns:
        rf = RandomForestRegressor()
        if scaling:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(train_df[train_df[column].notna()].drop(impute_columns, axis=1))
            X_impute_train = scaler.transform(train_df[train_df[column].isna()].drop(impute_columns, axis=1))
            X_impute_test = scaler.transform(test_df[test_df[column].isna()].drop(impute_columns, axis=1))
        else:
            X_train = train_df[train_df[column].notna()].drop(impute_columns, axis=1)
            X_impute_train = train_df[train_df[column].isna()].drop(impute_columns, axis=1)
            X_impute_test = test_df[test_df[column].isna()].drop(impute_columns, axis=1)

        y_train = train_df[train_df[column].notna()][column]
        rf.fit(X_train, y_train)

        # Apply the imputation model to the missing values in the training data
        train_df.loc[train_df[column].isna(), column] = rf.predict(X_impute_train)

        # Apply the imputation model to the missing values in the test data
        test_df.loc[test_df[column].isna(), column] = rf.predict(X_impute_test)

    # Calculate the RMSE for each column
    scores = {}
    for column in impute_columns:
        mask = np.logical_not(np.isnan(true_values[column]))
        mask = mask.reset_index(drop=True)  # Reset the index of the mask
        scores[column] = mean_squared_error(true_values[column].reset_index(drop=True)[mask],
                                            test_df[column].reset_index(drop=True)[mask],
                                            squared=False)

    # Insert imputed values back into the original DataFrame
    df = df.reset_index(drop=True)
    df.loc[filter_mask, impute_columns] = pd.concat([train_df[impute_columns], test_df[impute_columns]], ignore_index=True)

    print(scores)
    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

##########################################

def finalStep(input_path, output_path=None, output_path_csv=None, to_csv=False, to_feather=False):
    '''
    if missing_rate is less than 1 percent
    if there are still missing values after imputation process, do not force yourself drop it down baby!!
    '''
    import pandas as pd

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    # RE-arrange column values OCCUPANCY SCHEDULES, WORK PACKAGE 3-A for 1st step of thesis
    from Census_Functions import changeNames
    df = changeNames(df)

    # RE-arrange column values OCCUPANCY SCHEDULES, WORK PACKAGE 3-A for 1st step of thesis
    from Census_Functions import re_arrangeColsOCCschedule, cont_to_cat
    df = re_arrangeColsOCCschedule(df)
    df = cont_to_cat(df, "Work Hours", slices=10, labeling='max')

    # RE-arrange column values OCCUPANCY SCHEDULES, WORK PACKAGE 3-A for 1st step of thesis
    from Census_Functions import smoothenCols
    df = smoothenCols(df)

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

    if to_feather == True:
        df.reset_index(drop=True, inplace=True) # Reset the index and convert it to a column
        df.to_feather(output_path)
        print('editing function: writing as .ftr is done')

def finalStep_activity(input_path, output_path=None, output_path_csv=None, to_csv=False, to_feather=False):
    '''
    if missing_rate is less than 1 percent
    if there are still missing values after imputation process, do not force yourself drop it down baby!!
    '''
    import pandas as pd

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    # RE-arrange column values OCCUPANCY SCHEDULES, WORK PACKAGE 3-A for 1st step of thesis
    from Census_Functions import changeNames
    df = changeNames(df)

    # RE-arrange column values OCCUPANCY SCHEDULES, WORK PACKAGE 3-A for 1st step of thesis
    from Census_Functions import re_arrangeColsOCCschedule
    df = re_arrangeColsOCCschedule(df)

    df = df.drop(columns=["Room Count"])

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

    if to_feather == True:
        df.reset_index(drop=True, inplace=True) # Reset the index and convert it to a column
        df.to_feather(output_path)
        print('editing function: writing as .ftr is done')

##########################################
#census main completion
def finalStep_occ_completion(input_path, output_path=None, output_path_csv=None, to_csv=False, to_feather=False, fraction=1):
    '''
    if missing_rate is less than 1 percent
    if there are still missing values after imputation process, do not force yourself drop it down baby!!
    '''
    import pandas as pd

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    #df = df.sort_values(by=['Accommodation ID', 'Occupant ID in HH'])
    #df = df.head(1000)
    #df = df.sample(frac=0.01, replace=True, random_state=1)

    #1ST PART
    # complete missing members of household
    from Census_Functions import occ_in_HH_completion
    df = occ_in_HH_completion(df)

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

    if to_feather == True:
        df.reset_index(drop=True, inplace=True) # Reset the index and convert it to a column
        df.to_feather(output_path)
        print('editing function: writing as .ftr is done')

def finalStep_occ_completion_imputation02(input_path, output_path=None, output_path_csv=None, to_csv=False, to_feather=False):
    '''
    if missing_rate is less than 1 percent
    if there are still missing values after imputation process, do not force yourself drop it down baby!!
    '''
    import pandas as pd
    import numpy as np

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    from Census_Functions import cont_to_cat, simplify_ids

    #prerequisites
    df = cont_to_cat(df, "House Area", slices=13, labeling='middle')
    df = df[df["Occupant ID in HH"].notnull()]

    df = df[["Family ID", "Occupant ID in HH", "Accommodation ID", "Administrative Region", "Region", "Number Family Members",
            "Family Typology", "Gender", "Age Classes",  "Employment status",
            "Hours Worked", "Work Hours", "Full_Part time", "Job type",
            "Departure Hour for work/study",
            #"Departure Minute for work/study",
            "Room Count", "Floor Count", "House Area",]]

    df = df.sort_values(by=['Accommodation ID', 'Occupant ID in HH'])

    # some members of an household are duplicated
    df.drop_duplicates(subset=['Accommodation ID', 'Occupant ID in HH'], keep='first', inplace=True)

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

    if to_feather == True:
        df.reset_index(drop=True, inplace=True) # Reset the index and convert it to a column
        df.to_feather(output_path)
        print('finalStep_occ_completion_imputation02: writing as .ftr is done')

def finalStep_occ_completion_prepare_classification(input_path, output_path=None, output_path_csv=None, to_csv=False, to_feather=False):
    '''
    if missing_rate is less than 1 percent
    if there are still missing values after imputation process, do not force yourself drop it down baby!!
    '''
    import pandas as pd
    import numpy as np

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    # Assuming your DataFrame is called 'df' and the column name is 'column_name'
    df_rfc = df.drop_duplicates(subset=['Accommodation ID'], keep='first')
    df_rfc_train = df_rfc.dropna()
    missing_data = df[df['Room Count'].isnull() | df['House Area'].isnull()]

    training = ["Administrative Region", "Region", "Number Family Members", "Family Typology"]
    target = ["Room Count", "Floor Count", "House Area"]

    #classifier
    # Define the features and target for the second dataset
    X = df_rfc_train[training]
    y = df_rfc_train[target]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    rf = RandomForestClassifier()
    multi_output_clf = MultiOutputClassifier(rf)
    multi_output_clf.fit(X, y)

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

    if to_feather == True:
        df.reset_index(drop=True, inplace=True) # Reset the index and convert it to a column
        df.to_feather(output_path)
        print('finalStep_occ_completion_prepare_classification: writing as .ftr is done')

def finalStep_occ_completion_imputation(input_path, output_path=None, output_path_csv=None, to_csv=False, to_feather=False):
    '''
    if missing_rate is less than 1 percent
    if there are still missing values after imputation process, do not force yourself drop it down baby!!
    '''
    import pandas as pd

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    from preProcessing_Func import imputation
    from Census_Functions import cont_to_cat

    OCCs = ["Gender", "Age Classes", "Marital Status", "Education Degree", "Employment status", "Hours Worked", "Work Hours", "Full_Part time", "Job type", "Departure Hour for work/study", "Departure Minute for work/study"]

    # imputation of HHs features
    df = imputation.iterative_imputation_allCols(df=df, cols_to_extract=OCCs)
    df = cont_to_cat(df, "House Area", slices=13, labeling='middle')

    # CENSUS MAIN COMPLETION FOR WP3: ALIGNMENT #######################
    from Census_Functions import cluster_kmodes, cluster_kmodes_elbow
    df = df[df["Occupant ID in HH"].notnull()]

    # Add the word 'TUS' at the end of each column header
    suffix = '_CENSUS'
    df.columns = [col + suffix for col in df.columns]

    # imputation of OCCs features
    #df = cluster_kmodes(df, selected_columns=["Occupant ID in HH", "Family Typology", "Region", "Number Family Members"],
    #                    cols_to_imput=OCCs)

    #df = df.dropna()
    #cluster_kmodes_elbow(df, selected_columns=["Region_CENSUS", "Number Family Members_CENSUS","Family Typology_CENSUS","Occupant ID in HH_CENSUS",
    #                                    "Gender_CENSUS", "Age Classes_CENSUS", "Education Degree_CENSUS","Employment status_CENSUS", "Job type_CENSUS",])

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

    if to_feather == True:
        df.reset_index(drop=True, inplace=True) # Reset the index and convert it to a column
        df.to_feather(output_path)
        print('editing function: writing as .ftr is done')

def census_enlarged(input_path1, input_path2, output_path_csv=None, to_csv=False):
    '''
    to merge CENSUS MAIN & TUS subset
    '''
    import pandas as pd

    if input_path1.endswith(".csv"):
        census = pd.read_csv(input_path1)
    elif input_path1.endswith(".ftr"):
        census = pd.read_feather(input_path1)

    census = census.dropna() #some of OCCs columns are not fully imputed
    census = census.drop('cluster_CENSUS', axis=1)
    #print(census.head(5))

    if input_path2.endswith(".csv"):
        tus = pd.read_csv(input_path2)
    elif input_path1.endswith(".ftr"):
        tus = pd.read_feather(input_path2)

    from Census_Functions import census_enlarged
    df = census_enlarged(census=census, census_cols_train=["Region_CENSUS",
                                        "Number Family Members_CENSUS",
                                        "Family Typology_CENSUS",
                                        "Occupant ID in HH_CENSUS",
                                        "Gender_CENSUS",
                                        "Age Classes_CENSUS",
                                        "Education Degree_CENSUS",
                                        "Employment status_CENSUS",
                                        "Job type_CENSUS",],
                         tus=tus, tus_cols_predict=["Region_TUS",
                                        "Number Family Members_TUS",
                                        "Family Typology_TUS",
                                        "Occupant ID in HH_TUS",
                                        "Gender_TUS",
                                        "Age Classes_TUS",
                                        "Education Degree_TUS",
                                        "Employment status_TUS",
                                        "Job type_TUS"])


    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)
        print('census_enlarged: writing as .csv is done')

##########################################
#ANALYSIS
def read_df(input_path):
    import pandas as pd

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    return df

def detail_analysis(input_path, columns_to_impute,output_path=None, output_path_csv=None, to_csv=False, to_feather=False):
    import pandas as pd

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    # Define the columns to impute
    target_cols = columns_to_impute

    # Separate the complete data and the missing data
    complete_data = df.dropna(subset=target_cols).drop(target_cols, axis=1)
    print("complete_data.shape:", complete_data.shape)
    missing_data = df[df[target_cols].isnull().any(axis=1)].drop(target_cols, axis=1)
    print("missing_data.shape:", missing_data.shape)
    return complete_data, missing_data

def histogram_from_pd_df_by_column(df, column):
  """
  This function creates a histogram of the specified column in the Pandas DataFrame.

  Args:
    df: The Pandas DataFrame.
    column: The name of the column to create a histogram of.

  Returns:
    The histogram of the specified column.
  """

  # Check if the column exists in the DataFrame.
  if column not in df.columns:
    raise ValueError("The column {} does not exist in the DataFrame.".format(column))

  # Create the histogram.
  histogram = df[column].hist()

  import matplotlib.pyplot as plt
  plt.show()
  # Return the histogram.
  return histogram

##########################################
#OCCUPANT ACTIVITY

if __name__ == '__main__':
    from preProcessing_Func import analysis_func as dfppaf
    from preProcessing_Func import imputation as dfppimp
    from preProcessing_Func import NN_classification as dfppNN

    #INPUTS
    census_occ = r'dataset_CENSUS_occupant/CensusOcc_final.csv'
    census_housing =  r'dataset_CENSUS_housing/Census_Housing_final.csv'
    #MERGE
    census_main_merged =  r'dataset_CENSUS_main/Census_main_merged.csv'

    #_______________________________________________________________________________________________________________________________________
    # CENSUS MAIN: PIPELINE
    merge(df1_path=census_occ, df2_path=census_housing, refCol='Residential_ID', out_csv=census_main_merged)

    # MERGE PREPROCESS
    #census_main_prepocess =  r'dataset_CENSUS_main/Census_main_merged_preprocess.csv'
    # IMPUTATION BY KNN
    #census_main_impute =  r'dataset_CENSUS_main/Census_main_imputed.csv'
    #census_main_final = "dataset_CENSUS_main/Census_main_final.csv"

    # OCCUPANT SCHEDULE #######################
    #merge(df1_path=census_occ_path, df2_path=census_housing_path, refCol='Accommodation ID', to_csv=True, out_csv=census_main_merged)
    #merge_preprocessing(input_path=census_main_merged, output_path_csv=census_main_prepocess, to_csv=True)
    #impute_KNN_evaluate(input_path=census_main_prepocess, output_path_csv=census_main_impute, to_csv=True)
    #impute_RF_evaluate(input_path=census_main_prepocess, output_path_csv=census_main_impute, to_csv=True)

    #dfppimp.randomforestclassification(input_path=census_data_final, columns_to_impute= ['Room Count', 'House Area'], errorPrint=True, to_csv=True, out_csv=census_data_imputed)
    #dfppimp.iterative_imputation(input_path=census_data_merged, columns_to_impute= ['Room Count', 'House Area'], errorPrint=True, to_csv=True, out_csv=census_data_imputed)
    #finalStep(input_path=census_data_merged, output_path_csv=census_data_final, to_csv=True,)

    #CENSUS MAIN COMPLETION FOR WP2 ##########################
    #finalStep_occ_completion(input_path=census_data_final, output_path_csv=census_data_final_clss, to_csv=True, fraction=1)
    #finalStep_occ_completion_imputation(input_path=census_data_final_clss, output_path_csv=census_data_final_clss_imputed, to_csv=True)

    #070623 - 080623
    from Census_Functions import imputation_auto
    #finalStep_occ_completion(input_path=census_data_final, output_path_csv=census_data_final_clss, to_csv=True, fraction=1)
    #finalStep_occ_completion_imputation02(input_path=census_data_final_clss, output_path_csv=census_data_final_clss_imputed02, to_csv=True)
    #finalStep_occ_completion_prepare_classification(input_path=census_data_final_clss_imputed02, output_path_csv=census_data_final_clss_prepare_classification, to_csv=True)

    # CENSUS MAIN COMPLETION FOR WP3: ALIGNMENT #######################
    #finalStep_occ_completion(input_path=census_data_final, output_path_csv=census_wp3_alignment, to_csv=True, fraction=1)
    #finalStep_occ_completion_imputation(input_path=census_wp3_alignment, output_path_csv=census_wp3_alignment_imputed, to_csv=False)

    # CENSUS ENLARGED COMPLETION FOR WP3: ALIGNMENT
    #census_enlarged(input_path1=census_wp3_alignment_imputed, input_path2=TUS_subset, output_path_csv=census_enlarged_wp3, to_csv=True)

    from preProcessing_Func import analysis_func as dfppaf
    input = census_main_merged

    visual = False
    non_visual = True
    #cols_drop = ["Residential_ID", "Family_ID",]
    cols_drop = ["Residential_ID", "Household_ID",]
    #cols_drop = ["Residential_ID",]
    from preProcessing_Func import analysis_func as dfppaf
    #dfppaf.analysis(input_path=input, describe=non_visual)
    dfppaf.analysis(input_path=input, data_len=non_visual)
    #dfppaf.analysis(input_path=input, data_types=non_visual)
    dfppaf.analysis(input_path=input, missingness=non_visual)
    dfppaf.analysis(input_path=input, columns=non_visual)
    dfppaf.analysis(input_path=input, unique=non_visual,uniqueIDcolstoDrop=cols_drop)
    dfppaf.analysis(input_path=input, count_unique_values=non_visual, uniqueIDcolstoDrop=cols_drop)

    print(" ") # visual representation
    dfppaf.analysis(input_path=input, missingness_visual_oriented=visual, missingness_visual_oriented_title="Census Main")
    dfppaf.analysis(input_path=input, fraction=1, unique_visual=visual, uniqueIDcolstoDrop=cols_drop)
    #dfppaf.analysis(input_path=input, missingness_rowbased=visual)
    dfppaf.analysis(input_path=input, multiple_hist=visual, dropID_multiple_hist=cols_drop)

    print(" ") # tests
    #dfppaf.test_mcar(input_path=census_main_prepocess) # census_main dataset is not MACR, it is either MAR, MNAR
    #dfppaf.compute_cramers_v(input_path=census_main_prepocess) # categorical varialbe correlation analysis

    # OCCUPANT SCHEDULE IMPUTATION #######################
    #dfppaf.analysis(input_path=census_data_final, visualize_distribution=True, visualize_distribution_column=['House Area'])
    #multi-label classifications --------------------------------
    #dfppNN.NN_classify_pytorch_earlystop(input_path=census_data_final, columns_to_impute= ['Room Count', 'House Area'], errorPrint=True, to_csv=True, out_csv=census_data_imputed)
    #dfppNN.NN_classify_pytorch_earlystop_ncross(input_path=census_data_final, columns_to_impute= ['Room Count', 'House Area'], errorPrint=True, to_csv=True, out_csv=census_data_imputed)
    #dfppNN.NN_classify_softmax_pytorch(input_path=census_data_main, columns_to_impute= ['RoomCount', 'HouseArea'], errorPrint=True, to_csv=True, out_csv=census_data_main)

    # multi-class classifications -------------------------------
    #dfppNN.NN_classify(input_path=census_data_final, columns_to_impute= ['House Area'], errorPrint=True, to_csv=True, out_csv=census_data_imputed)
    #dfppNN.NN_regression(input_path=census_data_final, columns_to_impute= ['House Area'], errorPrint=True, to_csv=True, out_csv=census_data_imputed)
    #dfppNN.RFC_classify(input_path=census_data_final, columns_to_impute= ['House Area'], errorPrint=False,to_csv=True, out_csv=census_data_imputed,
    #                    single_training=True, gridSearch_training=False, trial_mode=True, fraction=0.25)

    #dfppaf.analysis(input_path=census_data_final, unique=True, uniqueIDcolstoDrop=['Family ID','Accommodation ID',])
    #dfppaf.analysis(input_path=census_data_final, missingness_visual=True, missingness=True)
    #dfppaf.analysis(input_path=census_data_final, data_len=True, columns=False)
    #dfppaf.analysis(input_path=census_data_final, describe=True)
    #dfppaf.analysis(input_path=census_data_main, missingness_rowbased=True)
    #dfppaf.analysis(input_path=census_data_final, fraction=1, unique_visual=True, uniqueIDcolstoDrop=['Family ID','Accommodation ID',])
    #dfppaf.analysis(input_path=census_data_final, fraction=1, multiple_hist=True, dropID_multiple_hist=['Family ID','Accommodation ID',])

    # OCCUPANT ACTIVITY #######################
    #census_house_path_activity = r'dataset_CENSUS_housing/Census_Housing.ftr'
    #census_occ_path_activity = r'dataset_CENSUS_occupant/Census_Occupant05_final_activity.csv'
    #census_occ_path_activity = r'dataset_CENSUS_occupant/Census_Occupant03_imput_manual.ftr'
    #census_data_merged_activity =  r'dataset_CENSUS_main/Census_merged_housing_occupant.csv'
    #census_data_final_activity = r'dataset_CENSUS_main/Census_final_occ_activity.csv'

    #merge(df1_path=census_occ_path_activity, df2_path=census_house_path_activity, refCol='Accommodation ID', to_csv=True, out_csv=census_data_merged_activity, fraction=1)
    #finalStep_activity(input_path=census_data_merged_activity, output_path_csv=census_data_final_activity, to_csv=True, )

    #DETAIL ANALYSIS: DATA DISTRIBUTIONS #######################
    #complete_data = detail_analysis(input_path=census_data_final, columns_to_impute= [ 'House Area'])[0]
    #df = read_df(census_data_final)
    #histogram_from_pd_df_by_column(df,'House Area')
    #imbalance_test(df, 'House Area')
    #for i in df.columns:
    #    imbalance_test(df, i)
    #    compute_class_imbalance(df, i)

    #multipleHistogram(df=complete_data, dropID=['Family ID','Accommodation ID',])

    #missing_data = detail_analysis(input_path=census_data_final, columns_to_impute= ['Room Count', 'House Area'])[1]
    #multipleHistogram(df=missing_data, dropID=['Family ID','Accommodation ID',])