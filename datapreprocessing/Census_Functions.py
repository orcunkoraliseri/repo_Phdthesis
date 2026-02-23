def cont_to_cat(df, columnname, slices, labeling='middle'):
    import numpy as np
    import pandas as pd

    # Determine the minimum and maximum values of your column
    min_val = df[columnname].min()
    max_val = df[columnname].max()

    # Create a list of 11 values from min_val to max_val, which will give you 10 equally spaced bins
    bins = np.linspace(min_val, max_val, slices)

    # The labels are the mid-point of each bin
    if labeling == 'middle':
        labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        labels = list(map(int, labels))
    elif labeling == 'max':
        labels = [bins[i + 1] for i in range(len(bins) - 1)]
        labels = list(map(int, labels))

    # Use pd.cut to bin the data and assign labels
    df[columnname] = pd.cut(df[columnname], bins=bins, labels=labels, include_lowest=True)
    return df

def replace(df, columnname, previous, new):
    # Replace multiple values in a column
    df[columnname].replace({previous:new}, inplace=True)
    return df

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def imbalance_test_imbalance_ratio_visualize(df):
    coefficients = []
    names = []
    colors = []

    for column in df.columns:
        # Check if the column exists in the DataFrame.
        if column not in df.columns:
            raise ValueError("The column {} does not exist in the DataFrame.".format(column))

        class_counts = df[column].value_counts(normalize=True)
        imbalance_ratio = class_counts.max() / class_counts.min()
        coefficients.append(imbalance_ratio)
        names.append(column)

        if imbalance_ratio <= 1:
            colors.append('blue')
        elif imbalance_ratio <= 10:
            colors.append('orange')
        else:
            colors.append('red')

    # Visualization
    plt.figure(figsize=(20, 12))
    plt.bar(names, coefficients, color=colors)
    plt.xlabel('Column')
    plt.ylabel('imbalance_ratio')
    plt.title('imbalance_ratio for Each Column')
    plt.xticks(rotation=45)

    for i in range(len(coefficients)):
        if coefficients[i] <= 1:
            plt.text(i, coefficients[i] + 0.1, 'balanced', ha='center')
        else:
            plt.text(i, coefficients[i] + 0.1, 'imbalanced', ha='center')

    plt.show()

def imbalance_test_chi2_visualize(df):
    coefficients = []
    names = []
    colors = []
    for column in df.columns:
        # Check if the column exists in the DataFrame.
        if column not in df.columns:
            raise ValueError("The column {} does not exist in the DataFrame.".format(column))

        # Get the number of rows in each class.
        counts = df[column].value_counts()

        # Perform the chi-square test.
        chi2, chi2_pvalue = stats.chisquare(counts)

        coefficients.append(chi2)
        names.append(column)

    # Visualization
    plt.figure(figsize=(20, 12))
    plt.bar(names, coefficients)
    plt.xlabel('Column')
    plt.ylabel('chi2')
    plt.title('chi2 for Each Column')
    plt.xticks(rotation=45)

    plt.show()


def imbalance_test(df, column):
  import scipy.stats as stats
  import pandas as pd
  '''
  Imbalance test 
   statistic / P-value /	Interpretation
   Large       Small	    Reject the null hypothesis that the classes are balanced. The distribution of the classes is significantly imbalanced.
   Small       Large	    Fail to reject the null hypothesis that the classes are balanced. The distribution of the classes is not significantly imbalanced.
   Large	   Large	    The test is inconclusive. The results are not significant enough to reject the null hypothesis.
   Small	   Small	    The test is inconclusive. The results are not significant enough to fail to reject the null hypothesis.

    Chi-square test: The chi-square statistic measures the discrepancy between the observed and expected frequencies. 
        A larger chi-square statistic indicates a larger discrepancy and suggests a more significant imbalance. 
        The p-value associated with the chi-square test represents the probability of observing such an extreme imbalance by chance. 
        A smaller p-value indicates a more significant imbalance. Typically, if the p-value is below a pre-defined 
        significance level (e.g., 0.05), we reject the null hypothesis and conclude that the distribution of classes is significantly imbalanced.
    
    Fisher's exact test: Fisher's exact test is applicable when sample sizes are small or when the chi-square test assumptions are violated. 
        The p-value associated with Fisher's exact test represents the probability of observing the observed distribution or a more 
        imbalanced distribution under the null hypothesis of independence. Similarly, a smaller p-value suggests a more significant imbalance. 
        The interpretation of the p-value for Fisher's exact test is the same as for the chi-square test.
        
    the imbalance ratio and Gini coefficient: you can assess the degree of class imbalance in the dataset. Higher imbalance ratios and 
        higher Gini coefficients indicate a more severe class imbalance, while lower values suggest a more balanced distribution.
            If the imbalance ratio is 1, then the dataset is perfectly balanced. Every class has the same number of instances.
            If the imbalance ratio is greater than 1, then there is some degree of imbalance. The higher the ratio, the more severe the imbalance. 
                For example, if the imbalance ratio is 2, then the most common class has twice as many instances as the least common class. 
        If the imbalance ratio is 10, the most common class has ten times as many instances, and so on.
            In extreme cases, the imbalance ratio can be very high. For instance, in some fraud detection datasets, 
            the imbalance ratio could be 1000 or more because fraudulent transactions are rare compared to non-fraudulent ones.
    '''

  # Check if the column exists in the DataFrame.
  if column not in df.columns:
    raise ValueError("The column {} does not exist in the DataFrame.".format(column))

  # Get the number of rows in each class.
  counts = df[column].value_counts()

  # Compute the imbalance test statistic.
  imbalance_test_statistic = counts.min() / counts.max()

  # Compute the p-value.
  p_value = stats.chi2_contingency([counts]).pvalue

  # Perform the chi-square test.
  chi2, chi2_pvalue = stats.chisquare(counts)

  # Perform Fisher's exact test if there are two levels in the column.
  fisher_pvalue = None
  if len(counts) == 2:
      _, fisher_pvalue = stats.fisher_exact(pd.crosstab(df[column], df[column]))

  class_counts = df[column].value_counts(normalize=True)
  gini_coefficient = 1 - 2 * (class_counts * (1 - class_counts)).sum()

  # Print the results.
  print("Column name:", column)
  # A Gini coefficient of zero expresses perfect equality, where all values are the same
  print(f'Gini Coefficient for {column}:', gini_coefficient)
  print("Chi-square statistic:", chi2) #if it is large, the data is imbalance
  print("Chi-square p-value:", chi2_pvalue) # if it is small or zero, the data is imbalance
  if fisher_pvalue is not None:
      print("Fisher's exact test p-value:", fisher_pvalue)
  else:
      print("Fisher's exact test cannot be performed as there are more than two levels in the column.")

  print("Imbalance test statistic:", imbalance_test_statistic)
  print("Imbalance_test P-value:", p_value)
  print(' ')

  # Return the imbalance test statistic and p-value.
  return imbalance_test_statistic, p_value,gini_coefficient, chi2, chi2_pvalue

def imbalance_solver(df, target_columns):
    import pandas as pd
    from imblearn.over_sampling import SMOTE

    # Create an empty DataFrame to store the oversampled data
    df_oversampled = pd.DataFrame()

    for column in target_columns:
        # Separate features and target variable
        X = df.drop(column, axis=1)
        y = df[column]

        # Print class distribution before oversampling
        print("Class distribution before oversampling for column:", column)
        print(y.value_counts())

        # Apply SMOTE oversampling
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Create a new DataFrame with the oversampled data
        oversampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        oversampled_df[column] = y_resampled

        # Print class distribution after oversampling
        print("Class distribution after oversampling for column:", column)
        print(oversampled_df[column].value_counts())

        # Concatenate the oversampled data with the original DataFrame
        df_oversampled = pd.concat([df_oversampled, oversampled_df], ignore_index=True)

    return df_oversampled

def changeNames(df):
    df.rename(columns={'People Count': "Number Family Members"}, inplace=True)
    df.rename(columns={'Age in Classes': 'Age Classes'}, inplace=True)
    df.rename(columns={'Course Type 24month': 'Middle School'}, inplace=True)
    df.rename(columns={'Residence GeoDistribution': 'Region'}, inplace=True)
    df.rename(columns={'Family Type': 'Family Typology'}, inplace=True)
    df.rename(columns={'Employment Status': 'Employment status'}, inplace=True)
    df.rename(columns={'Professional Position': 'Job type'}, inplace=True)
    df.rename(columns={'RoomCount': 'Room Count'}, inplace=True)
    df.rename(columns={'HouseArea': 'House Area'}, inplace=True)
    df.rename(columns={'Highest Education': 'Education Degree'}, inplace=True)
    df.rename(columns={'Full/Part_time': 'Full_Part time'}, inplace=True)
    df.rename(columns={'FloorCount': 'Floor Count'}, inplace=True)
    return df

def change_values_ES(value): # for "Employment Status"
    if value == 0:
        return 0
    elif value == 1:
        return 1
    elif value == 2:
        return 2
    elif value == 3:
        return 2
    elif value == 4:
        return 3
    elif value == 5:
        return 4
    elif value == 6:
        return 5
    elif value == 7:
        return 6
    elif value == 7:
        return 6
    return value

def change_values_jt(value): # for "Job type"
    if value == 6:
        return 1
    elif value == 7:
        return 2
    elif value == 1:
        return 3
    elif value == 2:
        return 4
    elif value == 3:
        return 5
    elif value == 5:
        return 6
    elif value == 4:
        return 7
    return value

def change_values_ac(value): # for "Age Classes"
    if value == 5:
        return 4
    elif value == 6:
        return 5
    elif value == 7:
        return 5
    elif value == 8:
        return 6
    elif value == 9:
        return 6
    elif value == 10:
        return 7
    elif value == 11:
        return 7
    elif value == 12:
        return 8
    elif value == 13:
        return 8
    elif value == 14:
        return 9
    return value

def change_values_ft(value): # for "Family Typology"
    if 2 <= value <= 3:
        return 2
    elif value == 5 or value == 9:
        return 3
    elif value == 6 or value == 10:
        return 4
    elif value == 7 or value == 11:
        return 5
    elif value == 8 or value == 12:
        return 6
    elif value == 13:
        return 7
    return value

def change_values_ed(value): # for "Education Degree"
    if value == 4:
        return 3
    elif value == 5:
        return 3
    elif value == 6:
        return 4
    elif value == 7:
        return 4
    return value

def re_arrangeColsOCCschedule(df):
    # RE-arrange column values OCCUPANCY SCHEDULES, WORK PACKAGE 3-A for 1st step of thesis
    # Change column values based on conditions
    df['Employment status'] = df['Employment status'].apply(change_values_ES)
    df['Job type'] = df['Job type'].apply(change_values_jt)
    df['Age Classes'] = df['Age Classes'].apply(change_values_ac)
    df['Family Typology'] = df['Family Typology'].apply(change_values_ft)
    df['Education Degree'] = df['Education Degree'].apply(change_values_ed)

    #print("AC:", df['Age Classes'].unique())
    #print( "jt:", df['Job type'].unique())
    #print("ES:", df['Employment status'].unique())
    #print("ft:", df['Family Typology'].unique())
    #print("ed:", df['Education Degree'].unique())
    #print("rc:", df['Room Count'].unique())
    #print("nfm:", df['Number Family Members'].unique())
    return df

def smoothenCols(df):
    df['Departure Minute for work/study'] = (df['Departure Minute for work/study'] // 10).round() * 10
    return df


def imputation_auto(input_path,
                    to_feather=False, output_path=False,
                    to_csv=False, output_path_csv=None,
                    fraction=1,
                    linear_imputation=False, columns_to_impute_linear=None,
                    tuning_imput=False, trialMode_imput=False, pipe_imput=False,
                    kde_imput=False, kde_cols=None,
                    find_target_variable=False):
    '''
    7TH STEP:CLASSIFICATION FOR IMPUTATION
    '''
    from preProcessing_Func import imputation as dfppIMP
    import pandas as pd

    #INPUT

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)


    #SAMPLING
    #df = df.sample(frac=fraction, replace=True, random_state=1)
    #print('length of dataframe is:', len(dfSamp))

    if find_target_variable==True:
        target_variable = dfppIMP.find_most_important_target_variable(df)
        print("Most important target variable:", target_variable)

    if linear_imputation == True:
        df = dfppIMP.linear_imp(df, columns_to_impute_linear, errorPrint=True)
        df = df.astype(str).astype('float64')

    if kde_imput == True:
        for i in kde_cols:
            df = dfppIMP.kde_impute_categorical(df,columnToSelect=i,
                                                bandwidth=1, fraction=fraction,
                                                kde_visual=False)

    if tuning_imput == True:
        #accuracy: 0.919 for rf
        dfppIMP.iterImpute_classify(df, missingness=True,
                                    colToOutput="location",
                                    #colToDrop=['act_OCC_PC','act_OCC'],
                                    tuning=True,
                                    fraction=fraction,
                                    )

    if trialMode_imput == True:
        import warnings
        warnings.filterwarnings("ignore", message="The least populated class in y has only")

        dfppIMP.iterImpute_classify(df, missingness=False,
                                    colToOutput='Ownership',
                                    #colToDrop=['act_OCC','act_OCC_PC'],
                                    pipe=False,
                                    trialMode=True,
                                    fraction=fraction,
                                    kfold_repeat=2, kfold_splits=25
                                    )

    if pipe_imput== True:
        import warnings
        warnings.filterwarnings("ignore", message="The least populated class in y has only")
        df = dfppIMP.iterImpute_classify(df,
                                        missingness=False,
                                        colToOutput='Ownership',
                                        #colToDrop=['act_OCC_PC','act_OCC'],
                                        pipe=True,
                                        fraction=fraction,
                                        kfold_repeat=3, kfold_splits=2,
                                        model_to='default',
                                        )

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path_csv, index=None)

    if to_feather == True:
        df.reset_index(drop=True, inplace=True) # Reset the index and convert it to a column
        df.to_feather(output_path)
        print('imputation auto: writing as .ftr is done')

###########################################
def occ_in_HH_completion(df):
    import pandas as pd
    import numpy as np
    from itertools import product

    df = pd.DataFrame(df)

    # Create a mapping of family IDs to their complete range of occupant IDs
    fam_occupant_map = df.groupby(['Accommodation ID', 'Number Family Members'])\
        .apply(lambda g: list(set(range(1, int(g['Number Family Members'].values[0]) + 1)) - set(g['Occupant ID in HH'].values)))\
        .to_frame('missing_ids')

    # Explode this mapping so that we get a row for each missing occupant ID
    fam_occupant_map = fam_occupant_map.explode('missing_ids')

    # Join this back with the original data
    completed_df = df.merge(fam_occupant_map.reset_index(), how='outer', on=['Accommodation ID', 'Number Family Members'])

    # Fill the missing 'Occupant ID in HH' values
    mask = completed_df['Occupant ID in HH'].isna()
    completed_df.loc[mask, 'Occupant ID in HH'] = completed_df.loc[mask, 'missing_ids']

    # Drop the 'missing_ids' column
    completed_df = completed_df.drop(columns='missing_ids')

    ######################
    #fill "Occupant ID in HH"
    #if Accommodation ID is unique and Occupant ID in HH is empty row, then give 1 to Occupant ID in HH
    ######################
    completed_df = check_unique_values_fill_col(completed_df, "Accommodation ID", "Occupant ID in HH", value_to_imput=1)
    completed_df = check_unique_values_fill_col(completed_df, "Accommodation ID", "Number Family Members", value_to_imput=1)
    completed_df = check_unique_values_fill_col(completed_df, "Accommodation ID", "Family Typology", value_to_imput=1)

    #######################
    # fill emtpy Family ID rows
    completed_df = fill_column_c(completed_df, "Accommodation ID","Occupant ID in HH","Family ID")

    ########################
    # fill NaN with most common rows for "Administrative Region"
    #completed_df = impute_most_common(completed_df, columns_to_impute="Administrative Region")
    #completed_df = fill_col_region_col(completed_df)

    sorted_df = completed_df.sort_values(by=['Accommodation ID', 'Occupant ID in HH'])

    df.drop_duplicates(keep='first', inplace=True)

    return sorted_df


def check_unique_values_fill_col(df, column_a, column_b, value_to_imput=None):
    # Check if column_a exists in the DataFrame.
    if column_a not in df.columns:
        raise ValueError("The column {} does not exist in the DataFrame.".format(column_a))

    # Check if column_b exists in the DataFrame.
    if column_b not in df.columns:
        raise ValueError("The column {} does not exist in the DataFrame.".format(column_b))

    # Identify rows where column_a has a unique value and column_b is empty
    mask = df[column_a].duplicated(keep=False) | df[column_b].notna()

    # Fill these rows with the specified value
    df.loc[~mask, column_b] = value_to_imput # vectorized option

    return df

def fill_column_c(df, column_a, column_b, column_c):
    import pandas as pd
    import numpy as np

    # Check if column_a, column_b, and column_c exist in the DataFrame.
    if column_a not in df.columns or column_b not in df.columns or column_c not in df.columns:
        raise ValueError("One or more columns do not exist in the DataFrame.")

    # Get the maximum value in column_c (if it contains numerical values).
    max_value = df[column_c].max() if np.issubdtype(df[column_c].dtype, np.number) else 0

    # Identify rows where column_a and column_b are not null, and column_c is null
    mask = df[column_a].notna() & df[column_b].notna() & df[column_c].isna()

    # For rows satisfying the mask, check if the value in column_a is unique
    unique_mask = df.loc[mask, column_a].map(df[column_a].value_counts() == 1)

    # Calculate number of rows to be filled
    num_rows = sum(mask & unique_mask)

    # If there are rows to be filled, create an increment_values array and fill these rows
    if num_rows > 0:
        increment_values = np.arange(1, num_rows + 1)
        df.loc[mask & unique_mask, column_c] = max_value + increment_values

    return df



def impute_most_common(df,columns_to_impute):
  from sklearn.impute import SimpleImputer

  # Create the imputer object
  imputer = SimpleImputer(strategy="most_frequent")

  X = df[columns_to_impute].values.reshape(-1, 1)

  # Perform iterative imputation on the selected columns
  df[columns_to_impute] = imputer.fit_transform(X)

  return df

def fill_col_region_col(df): # fil Region according to Administrative Region column
    import numpy as np

    # fill Region column
    df['Region'] = np.where(df['Administrative Region'].isin([1, 3]), 1,
                              np.where(df['Administrative Region'].isin([4, 5, 6, 7, 8]), 2,
                                       np.where(df['Administrative Region'].isin([9, 10, 11, 12]), 3,
                                                np.where(df['Administrative Region'].isin([13, 14, 15, 16, 17, 18]), 4,
                                                         np.where(df['Administrative Region'].isin([19, 20]), 5, np.nan)
                                                         )
                                                )
                                       )
                              )

    return df

import pandas as pd

def simplify_ids(df, id_column):
    # Get unique values in the 'id' column
    unique_ids = df[id_column].unique()

    # Create a new column for simplified ids
    df['simplified_id'] = df[id_column]

    # Assign simplified values starting from 1
    simplified_value = 1
    for unique_id in unique_ids:
        df.loc[df[id_column] == unique_id, 'simplified_id'] = simplified_value
        simplified_value += 1

    df[id_column] = df['simplified_id']

    return df

###########################################

def cluster_kmodes(df, selected_columns, cols_to_imput=None):
  from kmodes.kmodes import KModes
  import pandas as pd
  print("before:", len(df.columns))
  print(df.columns)
  # Create a new DataFrame with only the selected columns
  data = df[selected_columns].copy()

  # Create an instance of the K-Modes algorithm
  km = KModes(n_clusters=180, init='random', n_init=1, verbose=1)

  # Fit the K-Modes model to the data
  clusters = km.fit_predict(data)

  # Add the clustering column to the original DataFrame
  df['cluster'] = clusters

  # Print the updated DataFrame with the clustering column
  print("after:", len(df.columns))

  '''
  # Elbow curve to find optimal K
  cost = []
  #K = range(6, 10)
  K = [100, 200, 300, 400, 500]
  for num_clusters in list(K):
      kmode = KModes(n_clusters=num_clusters, init="random", n_init=1, verbose=1)
      print(" ")
      print("num_clusters:", num_clusters)
      print(" ")
      kmode.fit_predict(data)
      cost.append(kmode.cost_)

  plt.plot(K, cost, 'bx-')
  plt.xlabel('No. of clusters')
  plt.ylabel('Cost')
  plt.title('Elbow Method For Optimal k')
  plt.show()
  '''
  import random
  # Create a mask to identify the empty rows in columns 'A' and 'B'
  mask = df[cols_to_imput[0]].isnull()

  # Get a list of unique clusters from the 'cluster' column
  unique_clusters = df['cluster'].unique()

  # Iterate over the unique clusters and fill empty rows with random values
  for cluster in unique_clusters:
      cluster_rows = df.loc[mask & (df['cluster'] == cluster)].index
      valid_values = df.loc[df['cluster'] == cluster, cols_to_imput].dropna()

      if len(valid_values) > 0:
          random_values = valid_values.sample(n=len(cluster_rows), replace=True)
          df.loc[cluster_rows, cols_to_imput] = random_values.values

  return df

def cluster_kmodes_elbow(df, selected_columns): #to find optimal cluster number
  from kmodes.kmodes import KModes
  import pandas as pd
  print("before:", len(df.columns))
  print(df.columns)
  # Create a new DataFrame with only the selected columns
  data = df[selected_columns].copy()

  # Elbow curve to find optimal K
  cost = []
  #K = range(6, 10)
  K = [
      100, 200, 300, 400, 500,
       600, 700, 800, 900, 1000]
  for num_clusters in list(K):
      kmode = KModes(n_clusters=num_clusters, init="random", n_init=1, verbose=1)
      #kmode = KModes(n_clusters=num_clusters, init="Huang", n_init=1, verbose=1)
      print(" ")
      print("num_clusters:", num_clusters)
      print(" ")
      kmode.fit_predict(data)
      cost.append(kmode.cost_)

  plt.plot(K, cost, 'bx-')
  plt.xlabel('No. of clusters')
  plt.ylabel('Cost')
  plt.title('Elbow Method For Optimal k')
  plt.show()

def census_enlarged(census, census_cols_train, tus, tus_cols_predict):
    #alignment of CENSUS_MAIN and TUS_SUBSET
    from kmodes.kmodes import KModes
    import pandas as pd
    import numpy as np

    # Create a new DataFrame with only the selected columns
    data = census[census_cols_train].copy()

    # 1ST: CLUSTER TRAINING
    # Create an instance of the K-Modes algorithm
    km = KModes(n_clusters=900, init='random', n_init=1, verbose=0)

    # 2ND: CLUSTER FITTING & CLUSTER COLUMN ADDITION FOR TRAINING DATAFRAME
    # Fit the K-Modes model to the data
    clusters = km.fit_predict(data)
    # Add the clustering column to the original DataFrame
    census['cluster'] = clusters

    #print(tus.head(5))
    # 3RD: CLUSTER PREDICTION & CLUSTER COLUMN ADDITION FOR PREDICTING DATAFRAME
    dfs2 = tus[tus_cols_predict]
    predicted_clusters = km.predict(dfs2)
    # Add the clustering column to the original DataFrame
    tus['cluster'] = predicted_clusters

    # Shuffle the rows of both datasets
    census = census.sample(frac=1, random_state=42)
    tus = tus.sample(frac=1, random_state=42)

    # 4TH: MERGING WITH TUS DATASET
    # Add a new column 'merged_index' to the first dataset to store the index of the corresponding row in the second dataset
    census['merged_index'] = np.nan

    # Iterate over each cluster in the first dataset
    for cluster in census['cluster'].unique():
        # Get the rows from the first dataset that belong to the current cluster
        first_cluster_rows = census[census['cluster'] == cluster]

        # Get the rows from the second dataset that belong to the same cluster
        second_cluster_rows = tus[tus['cluster'] == cluster]
        second_cluster_rows = second_cluster_rows.sample(frac=1, replace=True, random_state=42)

        # Check if there are available samples in the second dataset for the current cluster
        if len(second_cluster_rows) > 0:
            # Sample rows from the second dataset that match the number of rows in the first dataset cluster
            sampled_rows = second_cluster_rows.sample(len(first_cluster_rows), replace=True)
            sampled_rows = sampled_rows.sample(frac=1, replace=True, random_state=42)

            # Assign the indices of the second dataset rows to the 'merged_index' column of the first dataset
            census.loc[first_cluster_rows.index, 'merged_index'] = sampled_rows.index

    # Merge the first and second datasets based on the 'merged_index' column
    merged_dataset = census.merge(tus, left_on='merged_index', right_index=True)

    # Drop the 'merged_index' column from the merged dataset if needed
    merged_dataset.drop('merged_index', axis=1, inplace=True)
    '''
    # Print the updated DataFrame with the clustering column
    print('training dataframe')
    # print(census)
    print(' ')
    print('predict dataframe')
    # print(tus)
    print(' ')
    print('MERGED dataframe')
    # print(merged_dataset)
    '''

    return merged_dataset

def incremental_clustering(df, selected_cols):
    from kmodes.kmodes import KModes
    import numpy as np

    dataset = df[selected_cols]
    # Initialize an empty clustering object
    clustering = KModes(n_clusters=3, init='Huang', n_init=1, verbose=1)

    # Iterate over your large dataset in batches
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        # Select the current batch
        batch = dataset[i:i + batch_size]

        # Perform clustering on the current batch
        clustering.fit(batch)

        # Update the clustering with the current batch
        if i == 0:
            # If it's the first batch, assign the clusters directly
            cluster_labels = clustering.labels_
            cluster_centroids = clustering.cluster_centroids_
        else:
            # If it's not the first batch, update the clusters incrementally
            cluster_labels = clustering.predict(batch, categorical_input=True)
            cluster_centroids = clustering.cluster_centroids_
            clustering.n_clusters += len(cluster_centroids)
            clustering.cluster_centroids_ = np.concatenate((clustering.cluster_centroids_, cluster_centroids), axis=0)

    # Final clustering results
    print("Final cluster labels:", cluster_labels)
    print("Final cluster centroids:", cluster_centroids)






