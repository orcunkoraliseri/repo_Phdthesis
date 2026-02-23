import pandas as pd
#from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

import time
from matplotlib.ticker import MaxNLocator
import sklearn
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score

class functions_general():
    pass

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

class analysis_func():
    pass
    def analysis(input_path, fraction=1,
                 dropna=False,
                 headPrint=False,
                 columns=False,data_len=False,
                 heatmap=False, heatmap_dropCols=None,
                 missingness=False,
                 missingness_visual=False,
                 missingness_visual_oriented=False, missingness_visual_oriented_title=None,
                 missingness_rowbased=False,
                 imbalance_test=False, dropID_imbalance_test=None,
                 unique_visual=False, uniqueIDcolstoDrop=None, unique=False,
                 count_unique_values=False, unique_visual_byCols=False,
                 feature_importance=False,
                 bar_chart=False, bar_column_name=None,
                 multiple_hist=False, dropID_multiple_hist=None,
                 data_types=False,
                 describe=False,
                 describe_visual=False, describeIDcolstoDrop=None,
                 describe_selCols=False, selCols=[],
                 sel_nan_df=False,
                 visualize_distribution=False, visualize_distribution_column=None):

        '''
        3RD PART: statistical and visual analysis for dataset_TUS_daily
        '''

        from preProcessing_Func import visuals as dfppvis
        from preProcessing_Func import stats_visuals as dfppsv

        # INPUT
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        df = df.sample(frac=fraction, replace=True, random_state=1)

        # df = df.drop(['level_0', 'index', ], axis=1)
        if columns == True:
            print("Columns of dataset: ", df.columns.tolist())

        if data_len == True:
            row_count = df.shape[0]
            print("Total row count of the dataset:", row_count)
            print("Total column count of the dataset:", len(df.columns))

        if headPrint == True:
            # Set display options
            pd.set_option('display.max_colwidth', None)  # None means unlimited width
            pd.set_option('display.max_columns', None)  # Display all columns
            print(df.head(50))
            #print(df.iloc[:10])

        if sel_nan_df == True:
            # select NaN values in the dataframe
            nan_df = df[df.isna().any(axis=1)]
            df = nan_df

        if dropna == True:
            df = df.dropna()

        if missingness == True:
            for i in range(df.shape[1]):
                n_miss = df.iloc[:, i].isna().sum()
                perc = n_miss / df.shape[0] * 100
                colName = df.columns[i]
                print('{}:'.format(colName), '%d (%.1f%%)' % (n_miss, perc))

        if missingness_visual == True:
            import missingno as msno
            import matplotlib.pyplot as plt
            # Visualize missing values
            msno.bar(df, figsize=(35, 70), fontsize=24)

            # Adjust x-tick and y-tick label font sizes
            # ax.tick_params(axis='both', which='major', labelsize=18)
            # ax.tick_params(axis='both', which='minor', labelsize=15)

            # Show the plot
            plt.show()

        if visualize_distribution==True:
            import matplotlib.pyplot as plt
            y = df[visualize_distribution_column]
            # Assuming y is your target variable
            y.value_counts().plot(kind='bar')
            print(y.value_counts())
            plt.title('Class distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.show()

        if imbalance_test == True:
            from Census_Functions import imbalance_test, imbalance_test_imbalance_ratio_visualize, imbalance_test_chi2_visualize
            for i in df.columns:
                imbalance_test(df, i)

            #imbalance_test_imbalance_ratio_visualize(df)
            #imbalance_test_chi2_visualize(df)

        if missingness_rowbased ==True:
            import missingno as msno
            import matplotlib.pyplot as plt
            msno.matrix(df, figsize=(40, 15), fontsize=30)
            plt.tight_layout()
            plt.show()

        if missingness_visual_oriented == True:
            import matplotlib.pyplot as plt

            # Calculate the percentage of missing values per column
            missing_percent = df.isnull().sum() / len(df) * 100
            # Calculate the completeness ratio per column
            completeness_ratio = 100 - missing_percent

            # Create a vertical bar plot
            # Set figure size based on number of columns
            fig_size = (45, 45) if len(df.columns) <= 10 else (60, 45)
            plt.figure(figsize=fig_size)

            '''
            # Choose a palette for census_housing
            num_colors = 2
            palette = sns.color_palette("tab10", num_colors)
            color_groups = {
                "group_1": [palette[0]] * 1,
                "group_2": [palette[1]] * 2,
            }
            '''

            # Choose a palette for census_occupant
            num_colors = 6
            palette = sns.color_palette("tab10", num_colors)
            color_groups = {
                "group_1": [palette[0]] * 3,
                "group_2": [palette[1]] * 2,
                "group_3": [palette[2]] * 2,
                "group_4": [palette[3]] * 2,
                "group_5": [palette[4]] * 3,
                "group_6": [palette[5]] * 2,
            }

            # Concatenate the groups to create a single color list
            colors = []
            for group in color_groups.values():
                colors.extend(group)

            ax = sns.barplot(x=completeness_ratio.index, y=completeness_ratio,
                             palette=colors,
                             color='blue',
                             alpha=0.5, width=0.6)

            ax.set_ylabel("Percentage of Completeness of columns", fontsize=60)

            # Increase the size of the x-axis tick labels
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=60, ha='center', rotation=60)

            # Set y tick label size
            plt.gca().tick_params(axis='y', labelsize=36)

            # Add completeness ratio values as text on top of the bars
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 0.5,
                        '{:.1f}%'.format(completeness_ratio[i]),
                        ha='center', va='bottom', fontsize=60, rotation=90)

                # Add column names as text in the middle of the bars
                ax.text(p.get_x() + p.get_width() / 2., height / 3,
                        completeness_ratio.index[i],
                        ha='center', va='center', fontsize=60, rotation=90)

                # Add the total number of rows of columns as text in the middle of the bars
                ax.text(p.get_x() + p.get_width() / 2, height / 1.25,  # adjusted position for better visibility
                        f'{df[completeness_ratio.index[i]].count()} rows',
                        # Display the number of non-NA entries per column
                        ha='center', va='center', fontsize=60, rotation=90)

            plt.grid(axis="y", linestyle="--", alpha=1, linewidth=5, color="purple")
            # Remove the xticklabels
            plt.xticks([])

            title_fontsize = 120  # Change this value to set a different font size for the title
            plt.title(missingness_visual_oriented_title, fontsize=title_fontsize, y=-0.05, fontweight='bold')  # Positioned below the graph and bold font

            plt.show()

        if unique == True:
            # Count unique values in each categorical column
            # loop over columns and print unique values
            if uniqueIDcolstoDrop == None:
                pass
            else:
                df = df.drop(uniqueIDcolstoDrop, axis=1)

            for col in df.columns:
                unique_values = sorted(df[col].unique())
                decimal_places = 3
                rounded_values = [round(value, decimal_places) if isinstance(value, float) else value for value in
                                  unique_values]

                print(f"Column {col} unique values: {rounded_values}")

            # Count unique values in each categorical column
            print(df.nunique())

        if count_unique_values == True:
            # Count unique values in each categorical column
            # loop over columns and print unique values
            if uniqueIDcolstoDrop is None:
                pass
            else:
                df = df.drop(uniqueIDcolstoDrop, axis=1)

            counts_dict = {}
            percentages_dict = {}

            for col in df.columns:
                value_counts = df[col].value_counts()
                total_count = len(df[col])
                counts_dict[col] = value_counts.to_dict()
                percentages_dict[col] = (value_counts / total_count * 100).round(4).to_dict()

            
            # Write counts and percentages to CSV
            import csv
            with open('unique_values_summary.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Column', 'Value', 'Count', 'Percentage'])
                for col in counts_dict:
                    for value, count in counts_dict[col].items():
                        percentage = percentages_dict[col].get(value, 0)
                        writer.writerow([col, value, count, percentage])

            print(counts_dict)
            print("Percentages of occurrence:", percentages_dict)

        if unique_visual == True:
            df = df.sample(frac=1, replace=True, random_state=1)
            if uniqueIDcolstoDrop == None:
                pass
            else:
                df = df.drop(uniqueIDcolstoDrop, axis=1)
            import matplotlib.pyplot as plt

            # Count unique values in each categorical column
            unique_counts = df.nunique()

            plt.figure(figsize=(50, 25))

            # Create a bar chart
            plt.bar(unique_counts.index, unique_counts.values)
            plt.xlabel('Categorical Columns', fontsize=36)
            plt.ylabel('Number of Unique Values', fontsize=36)
            plt.title('Unique Values per Categorical Column', fontsize=36)

            plt.xticks(rotation=90, fontsize=24)  # Rotates X-Axis Ticks by 45-degrees

            # Add labels to bars
            for i, v in enumerate(unique_counts.values):
                plt.text(i, v + 0.1, str(v), fontsize=25)

            plt.show()

        if unique_visual_byCols== True:
            import matplotlib.pyplot as plt

            if uniqueIDcolstoDrop == None:
                pass
            else:
                df = df.drop(uniqueIDcolstoDrop, axis=1)

            # Plot histograms for each column
            for col in df.columns:
                unique_values = df[col].nunique()

                # Check if the column is numeric for histogram
                if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                    plt.figure(figsize=(20, 8))
                    plt.hist(df[col], bins=unique_values, edgecolor='k', alpha=0.7)
                    plt.title(f'Histogram of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    # For non-numeric columns, we can use value_counts() to get a count and then plot
                    value_counts = df[col].value_counts()
                    plt.figure(figsize=(10, 4))
                    value_counts.plot(kind='bar')
                    plt.title(f'Bar Chart of {col}')
                    plt.ylabel('Frequency')
                    plt.show()

        if bar_chart == True:
            # INPUT
            annot = []

            # DELETE: values if less than 1% of all values
            # Define the ratio threshold for column 'B'
            threshold = 0.01
            # Calculate the frequency of each value in column 'B'
            value_counts = df[bar_column_name].value_counts(normalize=True)
            # Get the values in column 'B' that are below the threshold
            values_below_threshold = value_counts[value_counts < threshold].index
            # Filter the DataFrame to remove rows with values below the threshold
            dfbar = df[~df[bar_column_name].isin(values_below_threshold)]

            # PRINT: the filtered DataFrame
            dfppvis.singleBarChart(dfbar, column_name=bar_column_name, x_axis=50)

        if multiple_hist == True:
            df = df.dropna()
            dfppvis.multipleHistogram(df, dropID_multiple_hist)

        if heatmap == True:
            # working columns: catpri & catcon
            dfppsv.heatMap_correlation(df, corrMethod="pearson", printColumns=False,
                                       colToDrop=heatmap_dropCols,
                                       )
            '''
            not a strong correlation between working columns thus,
            an andvanced algorithm should be applied for imputation
            '''

        if feature_importance == True:
            # pca analysis for feature importance
            dfppsv.pca_analysis(df, dropNAN=True,
                                dropIrrelevant=['actCat_OCC_PC_edited']
                                )
            ''' 
            'company' column does not have strong impact, thus, it is should be eliminated 
            from classification. however, company column have missing rows so it should be included
            into classification for imputation.
            '''

        if data_types == True:
            pd.set_option('display.max_columns', 500)
            # print(df.dtypes)
            for col in df.columns:
                print( col,': ', df[col].dtypes)

        if describe:
            pd.set_option('display.max_columns', None)
            print(df.describe(include='all'))
            description = df.describe(include='all')
            description.to_csv('description.csv')

        if describe_visual == True:
            if describeIDcolstoDrop == None:
                pass
            else:
                df = df.drop(describeIDcolstoDrop, axis=1)
            import matplotlib.pyplot as plt
            df_describe = df.describe()

            plt.bar(df_describe.columns, df_describe.values.tolist())

            # Set the linewidth of each bar to 1.0
            for bar in plt.gca().patches:
                bar.set_linewidth(1.0)

            plt.show()

        if describe_selCols == True:
            for col in selCols:
                print(df[col].unique())

    def stat_analysis_descriptive_distribution(input_path, exclude_cols=[]):
        """
        Analyze and visualize all categorical columns from a DataFrame, excluding specified columns.

        Returns:
        - Visualizations for all categorical columns
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings

        # INPUT
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Exclude specified columns
        df = df.drop(columns=exclude_cols)

        categorical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(categorical_cols)

        # Determine subplot layout
        if num_cols <= 4:
            layout = (2, 2)
        elif num_cols <= 9:
            layout = (3, 3)
        elif num_cols <= 16:
            layout = (4, 4)
        elif num_cols <= 25:
            layout = (5, 5)
        else:
            warnings.warn("The number of columns exceeds 25. The plotting layout might not be optimal.")

        plt.figure(figsize=(25, 25))

        # Initialize an empty dictionary to store distribution data
        distribution_data = {}

        for idx, col in enumerate(categorical_cols, 1):
            plt.subplot(layout[0], layout[1], idx)

            # Descriptive statistics
            mode = df[col].mode().iloc[0]
            counts = df[col].value_counts(dropna=False)
            freqs = counts / counts.sum()

            # Bar plot for counts
            sns.barplot(x=counts.index, y=counts.values, order=counts.index, label=f"Mode: {mode}")
            plt.title(f'Distribution of {col}', fontsize=24)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend()

            # Annotate with statistics without mode
            for i, v in enumerate(counts):
                plt.text(i, plt.ylim()[1] / 2, f"Count: {v} || Freq: {freqs.iloc[i]:.2f}",
                         ha='center', va='center', rotation=90, fontsize=24)

            # Store the distribution in the dictionary
            distribution_data[col] = counts.to_dict()

        # Set display options to show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # Display descriptive statistics as tabular data
        print(" ")
        print(" ")
        print("descriptive statistics")
        desc_stats = analysis_func.custom_describe_categorical(df)  # Assuming you have this function defined elsewhere
        print(desc_stats)

        # After storing the data in the distribution_data dictionary
        print(" ")
        print(" ")
        print("distributions")
        for column, dist in distribution_data.items():
            print(f"{column}: {dist}")
            print("-" * 50)

        plt.tight_layout()
        plt.show()
    def stat_analysis_correlation(input_path, exclude_cols=[]):
        """Plot correlation matrix for all categorical variables in the dataframe using Cramér's V."""

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        # INPUT
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Exclude specified columns
        df = df.drop(columns=exclude_cols)

        columns = df.columns
        corr_matrix = pd.DataFrame(index=columns, columns=columns)
        for col1 in columns:
            for col2 in columns:
                if col1 == col2:
                    corr_matrix.loc[col1, col2] = 1.0
                else:
                    value = analysis_func.cramers_v_compute(df[col1], df[col2])
                    corr_matrix.loc[col1, col2] = float(
                        value) if value is not None else 0.0  # Convert to float and handle None values

        # Convert the entire matrix to float and handle NaN values
        corr_matrix = corr_matrix.astype(float).fillna(0.0)

        # Print the correlation matrix without truncation
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            print(corr_matrix)

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
        plt.title("Categorical Correlation using Cramér's V")
        plt.show()
    def stat_analysis_subRelation_chiSquare(input_path, exclude_cols=[]):
        import pandas as pd
        from scipy.stats import chi2_contingency
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # INPUT
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Exclude specified columns
        df = df.drop(columns=exclude_cols)

        # Calculate number of comparisons
        heatmap_count = sum(1 for i, _ in enumerate(df.columns) for j, _ in enumerate(df.columns) if i < j)
        side_length_heatmap = int(np.ceil(np.sqrt(heatmap_count)))  # Calculate side length for square layout
        fig, axes = plt.subplots(side_length_heatmap, side_length_heatmap, figsize=(25, 25))
        axes = axes.ravel()  # Flatten axes for easy indexing

        idx = 0
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:  # To avoid duplicate and self-comparisons
                    crosstab_result = pd.crosstab(df[col1], df[col2])
                    chi2, p, _, _ = chi2_contingency(crosstab_result)

                    # Visualize the cross-tabulation result with heatmap
                    sns.heatmap(crosstab_result, annot=True, cmap='viridis', fmt='g', ax=axes[idx])
                    axes[idx].set_title(f'Heatmap {col1} and {col2}\np-value: {p:.4f}')

                    # Print the cross-tabulation result and Chi-Squared test results in tabular form
                    print(f"\nCross-tabulation between {col1} and {col2}:\n")
                    print(crosstab_result)
                    print(f"\nChi-Squared p-value between {col1} and {col2}: {p:.4f}")
                    print("-" * 50)

                    idx += 1

        # Turn off any remaining unused subplots
        for j in range(idx, side_length_heatmap * side_length_heatmap):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    def stat_analysis_MCAR_missingness(input_path, exclude_cols=[]):
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Exclude specified columns
        df = df.drop(columns=exclude_cols)

        results = []
        for column_to_check in df.columns:

            missing = df[df[column_to_check].isnull()]
            not_missing = df[~df[column_to_check].isnull()]

            if df[column_to_check].dtype in ['int64', 'float64']:
                # Compare means and variances for numerical columns
                mean_comparison = missing[column_to_check].mean() == not_missing[column_to_check].mean()
                var_comparison = missing[column_to_check].var() == not_missing[column_to_check].var()

                # data is MCAR if both mean and variance are equal
                mcar = mean_comparison and var_comparison
                results.append([column_to_check, "Numerical", mcar])

            else:
                # Compare proportions for categorical columns
                comparison = missing[column_to_check].value_counts(normalize=True) == not_missing[
                    column_to_check].value_counts(normalize=True)
                prop_comparison = all(comparison.fillna(False))

                # data is MCAR if proportions are equal
                results.append([column_to_check, "Categorical", prop_comparison])

        # create DataFrame from the results and print it
        results_df = pd.DataFrame(results, columns=['Column', 'Type', 'MCAR'])
        print(" ")
        print(" ")
        print("missingness analysis, MCAR")
        print(results_df) #
    def cramers_v_compute(x, y):
        """Calculate Cramér's V statistic for categorial-categorial association."""

        import pandas as pd
        import numpy as np
        from scipy.stats import chi2_contingency


        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    def custom_describe_categorical(df):
        """
        Provide custom descriptive statistics for categorical columns.

        Parameters:
        - df: DataFrame with categorical columns

        Returns:
        - DataFrame with custom descriptive statistics
        """
        desc = pd.DataFrame(index=['count', 'unique', 'top', 'freq', 'mode'])

        for col in df.columns:
            desc[col] = [df[col].count(),
                         df[col].nunique(),
                         df[col].value_counts().idxmax(),
                         df[col].value_counts().max(),
                         df[col].mode().iloc[0]]

        return desc

class stats(): #statistical functions
    pass

    def remOutCat(df, q1, q3): #outlier Analysis for categorical variables
        Q1 = df.quantile(q1)
        Q3 = df.quantile(q3)
        IQR = Q3 - Q1
        return(df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)])

    def interpolateNanValues(df, nameToEdit, replace=np.nan, integer=False):
        if integer == False:
            sfilled=df[nameToEdit].interpolate(method ='linear', limit_direction ='forward') # fill Nan with Interpolation
        else:
            sfilled=round(df[nameToEdit].interpolate(method ='linear', limit_direction ='forward')) # fill Nan with Interpolation
        df[nameToEdit] = sfilled
        return df

    def bruteForceImput(df, nameToEdit, namedfAccMain=None, nameddfAcc=None, val2comp=None, inputsCom=None, val2imp=None, imputSelection=None, rename=False):
        '''
        :param nameToEdit:  column to edit
        :param namedfAccMain: main reference column
        :param val2comp: value to compare
        :param val2imp: value to imputate
        :param rename: rename the edited column
        :return:
        '''
        if imputSelection ==1:
            mask1 = (df[namedfAccMain] == val2comp) & (df[nameToEdit].isnull())
            df.loc[mask1, nameToEdit] = val2imp
        elif imputSelection == 2:
            mask1 = (df[namedfAccMain] != val2comp) & (df[namedfAccMain].notnull()) &(df[nameToEdit].isnull())
            df.loc[mask1, nameToEdit] = val2imp
        elif imputSelection == 3:
            mask1 = (df[namedfAccMain].notnull()) & (df[nameToEdit].isnull())
            df.loc[mask1, nameToEdit] = val2imp
        elif imputSelection == 4:
            mask1 = ((df[namedfAccMain].notnull()) | (df[nameddfAcc].notnull())) & (df[nameToEdit].isnull())
            df.loc[mask1, nameToEdit] = val2imp
        elif imputSelection == 5:
            mask1 = (df[namedfAccMain].isin(inputsCom)) &(df[nameToEdit].isnull())
            df.loc[mask1, nameToEdit] = val2imp
        elif imputSelection == 6:
            mask1 = df[nameToEdit].isnull()
            df.loc[mask1, nameToEdit] = val2imp
        elif imputSelection == 7:
            mask1 = (df[namedfAccMain].isin(inputsCom)) &(df[nameToEdit].isnull() & df[nameddfAcc] == val2comp)
            df.loc[mask1, nameToEdit] = val2imp
        else:
            pass

        if rename is not None:
            df.rename(columns = {nameToEdit:rename}, inplace = True)
        return df

    def impute_marital_status(df, age_col='Age Classes', marital_col='Marital Status', impute_value=1):
        mask = df[marital_col].isnull() & df[age_col].isin([1, 2])
        df.loc[mask, marital_col] = impute_value
        return df

    def interpolateAccOtherCol(df, nameToEdit, namedfAccMain, integer= False,namedfAcc=None, rename=False):# interpolate Nan values according to other columns
        '''
        :param nameToEdit: column to edit
        :param namedfAcc:  reference column
        :param namedfAccMain: main reference column
        :param integer: integer interpolation
        :return: dataframe
        '''

        if namedfAcc == None:
            mask1 = (df[namedfAccMain].notnull()) & (df[nameToEdit].isnull())
            df[nameToEdit].loc[mask1] = 0
            mask2 = (df[namedfAccMain].isnull())
            #df[nameToEdit].loc[mask2] = 99
            dfcopy = df[nameToEdit].loc[mask2].copy()
            #print(len(dfcopy))
        else:
            # give 0 for non-workers
            mask1 = (df[namedfAccMain].notnull()) & (df[namedfAcc].isnull())
            df[nameToEdit].loc[mask1] = 0
            # give Nan for empty rows of the specified columns
            mask2 = (df[namedfAccMain].notnull()) & (df[namedfAcc].notnull()) & (df[nameToEdit].isnull())
            df[nameToEdit].loc[mask2] = np.nan
            # filter only mask 2 nan values
            mask3 = (df[namedfAccMain].notnull())
            dfcopy = df[nameToEdit].loc[mask3].copy()

        #interpolate mask2 filtered Nan Values
        if integer == False:
            sfilled= dfcopy.interpolate(method ='linear', limit_direction ='forward') # fill Nan with Interpolation
        else:
            sfilled = round(dfcopy.interpolate(method ='linear', limit_direction ='forward')) # fill Nan with Interpolation

        df[nameToEdit].loc[dfcopy.index] = sfilled

        if rename is not None:
            df.rename(columns = {nameToEdit:rename}, inplace = True)

        #print(len(df[nameToEdit]))
        return df

    def interpolateOutCat(s, q1, q3, replace=np.nan, integer=False, missingRaw=True):
        '''
        :param q1: percentage of first quartile
        :param q3: percentage of third quartile
        :param replace: replace outlier with NaN
        :param missingRaw: if there are missing rows for the column from the Raw data
        :return:
        '''
        #remove outliers and apply linear interpolations for categorical variables
        #INFO#
        #https://colab.research.google.com/drive/1TSz9aDCcEzg_12B3Td4mgfFwB1IYdJjd#scrollTo=CIIzxZmlyEeA

        #1st step
        if missingRaw == True: # replace blank rows with NaN
            s = s.replace(r'^\s*$', np.nan, regex=True)
        else:
            pass

        #2nd step: find outliers
        Q1, Q3 = np.percentile(s, [q1 ,q3])
        IQR = Q3-Q1

        #3rd step: replace outlier with Nan
        withNan = s.where((s > (Q1 - 1.5 * IQR)) & (s < (Q3 + 1.5 * IQR)), replace)
        if integer == False:
            sfilled=withNan.interpolate(method ='linear', limit_direction ='forward') # fill Nan with Interpolation
        else:
            sfilled=round(withNan.interpolate(method ='linear', limit_direction ='forward')) # fill Nan with Interpolation
        return {'filled': sfilled, 'Nan': withNan}

    def interpolateWrongInput(s, inputs, replace=np.nan, integer=False, bruteForceReplacement=False):
        '''
        :param wrngInpt: wrong input
        :param q3: percentage of third quartile
        :param replace: replace outlier with NaN
        :return:
        '''
        #remove outliers and apply linear interpolations for categorical variables
        #INFO#
        #https://colab.research.google.com/drive/1TSz9aDCcEzg_12B3Td4mgfFwB1IYdJjd#scrollTo=CIIzxZmlyEeA

        withNan = s.where(s.isin(inputs), replace) # replace outlier with Nan

        if integer == False:
            sfilled=withNan.interpolate(method ='linear', limit_direction ='forward') # fill Nan with Interpolation
        else:
            sfilled=round(withNan.interpolate(method ='linear', limit_direction ='forward')) # fill Nan with Interpolation
        return sfilled

    def bruteForceReplacement(s,  replace= None, nameToEdit=None, val2comp=None,
                              inputs=None, imputSelection=None,
                              dictionary=None):
        if imputSelection == 0:
            s = s.where(s.isin(inputs), replace)
        elif imputSelection == 1:
            #mask1 = (s[nameToEdit] == val2comp)
            #s[nameToEdit].loc[mask1] = replace
            s = s.where(s[nameToEdit] == val2comp, replace)
        elif imputSelection ==2:
            s = s.replace(dictionary, regex=True)
        else:
            pass
        return s

    def convertBinary(s, equal, replace):
        #INFO#
        #https://colab.research.google.com/drive/1TSz9aDCcEzg_12B3Td4mgfFwB1IYdJjd#scrollTo=CIIzxZmlyEeA

        s.loc[s > equal] = replace
        return s

    def mergeKitchenColumns(df,newColumn, col1, col2, limit, replace1, replace2):
        '''
        :param s1: first binary data
        :param bin2: second binary data
        :param replace: replace outlier with NaN
        :return:
        '''
        #remove wrong inputs from binary data and apply linear interpolations for categorical variables
        #INFO#
        #https://colab.research.google.com/drive/1TSz9aDCcEzg_12B3Td4mgfFwB1IYdJjd#scrollTo=CIIzxZmlyEeA

        #merge two columns
        df[newColumn] = df[col1] + df[col2]
        dfChanged = df[newColumn].where(df[newColumn] == limit, replace1)
        dfChanged = dfChanged.where(dfChanged < limit, replace2)
        return dfChanged

    def get_chi2_p_value(crosstab):
        '''
        the p-value of the chi-squared test for each pair of columns in the dataset_TUS_daily.
        Then it computes the average p-value for each column and
        selects the one with the lowest average p-value as the most important target variable.
        '''
        import pandas as pd
        import numpy as np
        from scipy.stats import chi2_contingency
        chi2, p, _, _ = chi2_contingency(crosstab)
        return p
class stats_visuals():
    def heatMap_correlation(df, corrMethod, colToDrop=None, printColumns=False):
        import seaborn as sns
        import matplotlib.pyplot as plt

        if printColumns == True:
            print(df.columns)

        f, ax = plt.subplots(figsize=(25, 15))
        df = df.drop(colToDrop, axis=1)
        corr = df.corr(method=corrMethod)
        # creating mask
        #mask = np.triu(np.ones_like(df.corr()))
        sns.heatmap(corr,
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    vmin=-1.0, vmax=1.0,
                    square=True, ax=ax,
                    #mask=mask,
                    annot=True, annot_kws={'size': 6})

        miniFontSize = 10
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=miniFontSize, rotation=90)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=miniFontSize)
        plt.title('Correlation Matrix', fontsize=16)

        plt.show()
    

    def pca_analysis(df, dropNAN=False, dropIrrelevant=None,threshold=0.95, print_less_effective=False):
        '''
        example:
        pca_analysis(df2013, dropNAN=True, 
                    dropIrrelevant=['actCat_OCC_PC', 'act_OCC_PC'])
        '''
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt

        # Drop any columns that are not relevant to the analysis
        df = df.drop(dropIrrelevant, axis=1)

        # Drop any columns that are not relevant to the analysis
        if dropNAN == True:
            df = df.dropna()
        else:
            pass

        # Normalize the data using StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Instantiate a PCA object
        pca = PCA(n_components=df.shape[1])

        # Fit the PCA model to the data
        pca.fit(df_scaled)

        # Transform the data to the new coordinate system
        df_pca = pca.transform(df_scaled)

        # to find ideal number of components
        prop_var = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_

        #PCA CHART
        PC_numbers = np.arange(pca.n_components_) + 1
        plt.figure(figsize=(60, 20))
        plt.xticks(fontsize=32)
        plt.plot(PC_numbers, eigenvalues, 'ro-')
        plt.title('Figure 1: Scree Plot', fontsize=32)
        plt.ylabel('Proportion of Variance', fontsize=32)
        plt.show()

        # Create a new DataFrame with the transformed data
        df_pca = pd.DataFrame(df_pca, columns=df.columns)

        # Print the explained variance ratio for each principal component
        #print(pca.explained_variance_ratio_)

        if print_less_effective == True:
            # Find the less effective columns
            cum_explained_var = np.cumsum(pca.explained_variance_ratio_)
            n_less_effective = np.argmax(cum_explained_var >= threshold)
            less_effective_columns = df.columns[n_less_effective + 1:]

            print(f"\nLess effective columns: {list(less_effective_columns)}")

        # BAR CHART
        # Plot the explained variance ratio for each principal component
        plt.figure(figsize=(60, 20))
        plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
        plt.xticks(range(len(pca.explained_variance_ratio_)), df.columns, rotation=90, fontsize=16)
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.show()
class visuals():
    # INTEGER TICK LABELS
    from matplotlib.ticker import MaxNLocator

    def histMultiple(df, fontsize): #fast solution
        #https://stackoverflow.com/questions/46188580/pandas-dataframe-hist-change-title-size-on-subplot
        import matplotlib.pyplot as plt
        params = {'axes.titlesize':'0', #remove the subtitle
              'xtick.labelsize':fontsize,
              'ytick.labelsize':fontsize,
              'legend.fontsize': fontsize,
              'legend.handlelength': 5,
                 "font.family": "Times New Roman" }
        matplotlib.rcParams.update(params)

        #https://www.analyticsvidhya.com/blog/2021/07/how-to-perform-data-visualization-with-pandas/
        length = len(df.columns.values)
        #axes = df.hist(layout=(length, 1), figsize=(length*25,25), legend=True, sharey=True)
        axes = df.hist(figsize=(length*5,5), legend=True, sharey=True)
        for ax in axes.ravel():
            #https://stackoverflow.com/questions/66267260/giving-x-and-y-labels-titles-and-legends-to-individual-histograms-in-python
            ax.set_ylabel(f'count', fontdict={'fontsize':fontsize*1})
            ax.xaxis.set_major_locator(visuals.MaxNLocator(integer=True)) #https://stackoverflow.com/questions/30914462/how-to-force-integer-tick-labels

        plt.show()

    def histMultiple2(df, figSize_x,figSize_y):
        import matplotlib.pyplot as plt
        #https://stackoverflow.com/questions/55396416/plotting-multiple-overlapped-histogram-with-pandas
        fig, ax = plt.subplots(figsize=(figSize_x,figSize_y),
                               #dpi=50,
                               )

        #https://stackoverflow.com/questions/46188580/pandas-dataframe-hist-change-title-size-on-subplot
        params = {'axes.titlesize':'16',
              'xtick.labelsize':'16',
              'ytick.labelsize':'16'}
        plt.rcParams.update(params)

        c = 1
        for i in df.columns:
          plt.subplot(7, 7, c) #c is subplot iterator
          #plt.title('{}'.format(i))
          #plt.xlabel(i)

          #https://stackoverflow.com/questions/60081599/dataframe-hist-with-different-bin-size
          df[i].hist(bins=len(df[i].value_counts()), #bins size is equal to NoOfCategorical Variables
                     alpha=0.5,
                     grid=True
                     )
          c = c + 1
        #fig.tight_layout()
        plt.show()

    def scatter_Subplot(df, nROWS=250):
        '''
        :param nROWS: selected rows, default 250
        :return:
        '''
        import matplotlib.pyplot as plt
        #visualisation of SCATTER plot by indices
        #https://stackoverflow.com/questions/41912064/how-to-plot-scatter-subplots-of-columns-from-pandas
        #https://stackoverflow.com/questions/60409913/how-to-use-pandas-df-plot-scatter-to-make-a-figure-with-subplots
        fig, axes = plt.subplots(len(df.columns.values),1, figsize=(20, 80))
        for i, column in enumerate(df.columns): #[:-1] ignores the index column for my random sample
            df.iloc[:nROWS].reset_index().plot(kind="scatter", x="index", y=column, ax=axes[i], grid=True)
            plt.yticks(df[column].dropna().unique())
            #plt.yticks(df.column.dropna().unique())
        #plt.tight_layout()
        plt.show()

    def scatter_SinglePlot(df, nROWS=1000):
        '''
        :param nROWS: selected rows, default 1000
        :return:
        '''
        import matplotlib.pyplot as plt
        markers= ['o', "v", "1", "s", "*", "+", "^", "<", ">", "d"]
        colors = ["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"]
        fig, axes = plt.subplots(1,1, figsize=(50, 50))

        for i, column in enumerate(df.columns): #[:-1] ignores the index column for my random sample
            df.iloc[:nROWS].reset_index().plot(kind="scatter", x="index", y=column, ax=axes,
                                              marker=markers[i], c=colors[i], legend=True)
        #plt.tight_layout()
        plt.show()

    def singleBarChart(df, column_name, x_axis=6, y_axis=12): #works on local PyCharm
        '''
        pd.df['columnName'], categorical variable
        :return:
        '''
        import matplotlib.pyplot as plt
        #print(df)
        #drop NaN, if exist
        df = df[column_name]
        #print(df)
        df = df.dropna()
        #print(df[column_name].dtype)

        # define unique values on column
        unique_vals = sorted(df.unique().tolist())
        print(unique_vals)
        #binwidth = (unique_vals[1] - unique_vals[0])

        # get bar values
        u, inv = np.unique(df, return_inverse=True)
        counts = np.bincount(inv)

        new_xticks = np.arange(len(unique_vals))

        # define plot
        fig, ax = plt.subplots(figsize=(x_axis, y_axis))
        #graph = ax.bar(u, counts,width=int(binwidth / 2))
        graph = ax.bar(new_xticks, counts )

        #plt.xticks(np.arange(df.min() - binwidth, df.max() + binwidth, binwidth))
        #plt.xticks(unique_vals, unique_vals)
        ax.set_ylim([0, max(counts.tolist()) * 1.25])
        ax.set_xticks(new_xticks)
        ax.set_xticklabels(unique_vals)

        tick_size = 36
        for tick in ax.get_xticklabels():
            tick.set_fontsize(tick_size)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(tick_size)

        #listAct = ["Out/Education", "Chores", "Work-related", "Cooking", "Dishwashing", "Cleaning", "Shopping",
        #    "Caretaking", "In Socializing", "Relaxing", "out/Walking", "WatchingTV", "Out/Healthcare",
        #        "Out/Work", "Errands", "Out/Socializing", "Out/In-Hobbies"]

        #listAct = ["Piemonte-Valle", "Lombardia", "Trentino-Alto", "Veneto",
        #           "Friuli-Venezia", "Liguria", "Emilia-Romagna", "Toscana", "Umbria",
        #           "Marche", "Lazio", "Abruzzo", "Molise", "Campania", "Puglia", "Basilicata",
        #           "Calabria", "Sicilia", "Sardegna"]

        listAct = ["Northwest", "Northeast", "Center", "South", "Island"]

        # insert annotation
        for i,p in enumerate(graph):
            perc = round((p.get_height() / sum(counts.tolist())) * 100, 2)
            height = p.get_height() * 1.03

            plt.text(x=p.get_x() + p.get_width() / 2, y=height + .10,
                     s="{}%".format(perc),
                     ha='center', fontsize=tick_size/3*2)
            plt.text(x=p.get_x() + p.get_width() / 2, y=height + 5000,
                     s="{}".format(listAct[i]),
                     ha='center', fontsize=tick_size/3*2)

        # Calculate the center positions of each bar
        plt.show()

    def multipleHistogram(df, dropID=None):

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import math

        # DROP ID columns
        if dropID is not None:
            df = df.drop(columns=dropID, axis=1)

        num_columns = len(df.columns)

        # Find the next perfect square number greater than or equal to num_columns
        next_square = math.ceil(math.sqrt(num_columns)) ** 2

        # Calculate subplot grid dimensions
        y_axis_count = x_axis_count = int(math.sqrt(next_square))

        figure_size = 45

        params = {
            'xtick.labelsize': figure_size / 2,
            'ytick.labelsize': figure_size / 2}
        plt.rcParams.update(params)

        # Create a figure and subplots
        fig, axes = plt.subplots(y_axis_count, x_axis_count, figsize=(figure_size, figure_size))
        axes = axes.flatten()

        for i, col in enumerate(df.columns):
            unique_values = np.sort(df[col].unique())

            # Assign each unique value an even position on x-axis
            mapping = {v: idx for idx, v in enumerate(unique_values)}
            mapped_values = df[col].map(mapping)

            # Bins and xticks
            bin_edges = np.arange(-0.5, len(unique_values) + 0.5, 1)
            xticks = np.arange(0, len(unique_values), 1)

            # Plotting
            if i == num_columns - 1:  # check if the plot is the last one
                sns.histplot(mapped_values, ax=axes[i], bins=bin_edges, kde=False, color='green', edgecolor="white",
                             linewidth=2.5)
            else:
                sns.histplot(mapped_values, ax=axes[i], bins=bin_edges, kde=False, edgecolor="white", linewidth=2.5)

            axes[i].set_xticks(xticks)
            axes[i].set_xlabel(None)
            axes[i].set_ylabel(None)
            axes[i].legend([col], fontsize=figure_size/3)
            axes[i].set_xticklabels(unique_values, rotation=45, ha='center')  # set alignment to 'center'

        # delete empty subplots
        empty_subplots = next_square - num_columns
        for i in range(empty_subplots):
            axes[-(i + 1)].set_visible(False)

        fig.tight_layout()
        plt.show()
class preprocessing_df():
    def merge_categories(df, column, categories_to_merge, category_to_merge_under):
        '''
        merge categories for dimension reduction,
        for ex.,
        df = merge_categories(df, 'my_column', [5, 6, 7], 5)
        '''
        df[column] = df[column].replace(categories_to_merge, category_to_merge_under)
        return df

    def filter_dataframe(df, column, max_value):
        '''
        hard filter, manual
        for ex.,
        df = filter_dataframe(df, 'rooms', 6) # Filter out rows where 'rooms' is greater than 6
        '''

        df = df[df[column] <= max_value]
        return df
class census_housing():
    pass

    def filterRes(df):
        #df = df[df['ResidentialType'] ==1]
        #df = df.drop(['ResidentialType'], axis=1)
        df = df[df['Room Count'].notnull()]

        # filter outliers, manually
        df = df[df['Room Count'] < 16]
        df = df[df['Room Count'] != 0]
        df = df[df['Room Count'] != 1]

        #df = df[df['FloorCount']<7] #max number of family member is 7 thus, a room per person
        #df = df[df['FloorCount']==1]
        return df

    def identifier(df):
        df = df.drop(['Homeowner'], axis=1)
        return df

    def kitchen(df):

        from preProcessing_Func import dfTUS_data as dftus
        #1ST STEP
        selCols = ['Kitchen small_existence', 'Kitchen corner_existence','separate Kitchen','No Kitchen']
        df = dftus.MultipleEncoding(df, listForCols=selCols, namingCol="kitchen")

        '''
        #kitchen columns
        #1ST STEP: merge CUCINI & ANGCOT
        df['cucini'] = stats.mergeKitchenColumns(df, 'cucini', 'Kitchen small, existence', 'Kitchen corner, existence', 4, 0, 1)
        df = df.drop(['Kitchen small, existence'], axis=1)
        df = df.drop(['Kitchen corner, existence'], axis=1)

        #2ND STEP: all transform CICSTRARIC column to binary
        df['separate Kitchen'] = df['separate Kitchen'].replace(2, 1) # replace all multiple kitchen to kitchen exist info
        df['No Kitchen'] = stats.interpolateWrongInput(df['No Kitchen'], [1,2])

        #3RD STEP:
        df['kitchen'] = np.nan
        #4TH STEP:
        df['kitchen'] = np.where((df['separate Kitchen'] == 1) & (df['No Kitchen'] == 2) & (df['cucini']==1), 1, df['kitchen']) # cucina exist, thus no room division for kitchen activities

        df['kitchen'] = np.where((df['separate Kitchen'] == 0) & (df['No Kitchen'] == 1) & (df['cucini']==1), 2, df['kitchen']) # cucini exist, thus no room division for kitchen activities
        df['kitchen'] = np.where((df['separate Kitchen'] == 0) & (df['No Kitchen'] == 2) & (df['cucini']==1), 2, df['kitchen']) # cucini exist, thus no room division for kitchen activities
        df['kitchen'] = np.where((df['separate Kitchen'] == 0) & (df['No Kitchen'] == 1) & (df['cucini']==0), 2, df['kitchen']) # cucini exist, thus no room division for kitchen activities

        df['kitchen'] = np.where((df['separate Kitchen'] == 1) & (df['No Kitchen'] == 2) & (df['cucini']==0), 1, df['kitchen']) # cucina exist, thus no room division for kitchen activities

        df['kitchen'] = np.where((df['separate Kitchen'] == 0) & (df['No Kitchen'] == 2) & (df['cucini']==0), 0, df['kitchen']) # cucina exist, thus no room division for kitchen activities

        #4TH STEP:
        df = df.drop(['separate Kitchen'], axis=1)
        df = df.drop(['No Kitchen'], axis=1)
        df = df.drop(['cucini'], axis=1)

        df.rename(columns = {'kitchen':'kitchenType'}, inplace = True)
        '''

        return df

    def acqua(df): #filter() the houses that there is no water inside home, based on 17th variable, then drop the column
        df['No Indoor Water'] = stats.interpolateWrongInput(df['No Indoor Water'], [1,2])
        df = df[df['No Indoor Water'] ==2]
        #df = df.drop(['No Indoor Water'], axis=1)
        return df

    def internetCon(df): #internet connection
        from preProcessing_Func import dfTUS_data as dftus
        selCols = ['Traditional_ISDN Line', 'ADSL', 'Other Broadband', 'Internet Key/Mobile'] # selected columns
        #1ST STEP
        df = dftus.MultipleEncoding(df, listForCols=selCols, namingCol="webConnection")
        return df

    def mobilePhone(df):
        from preProcessing_Func import dfTUS_data as dftus
        selCols = ['Fixed Telephone Line', 'Mobile Phone','Family Mobile Phones'] # selected columns
        df = dftus.MultipleEncoding(df, listForCols=selCols, namingCol="phone")
        return df

    def auto(df):
        from preProcessing_Func import stats as dfppss
        df['Automobiles'] = df['Automobiles'].replace({3: 0})

        #private parking, existence
        dfppss.bruteForceImput(df, nameToEdit='Private Parking', val2imp=0, imputSelection=6, rename='Private Parking')  # https://docs.google.com/document/d/1hN8400ZsH2eXd716fE-K8HTGi_5JuxbB0EjxRuM25u0/edit
        stats.convertBinary(df['Private Parking'], 0, 1)
        return df

    def DHW(df): #Domestic Hot Water
        from preProcessing_Func import dfTUS_data as dftus
        selCols = ['Hot Water Availability','Heating-Hot Water System','Methane Heating','Electric Heating', 'Solar Heating', 'Other Heating','Shower/Bathtub Count', 'Toilet Count']
        df = dftus.MultipleEncoding(df, listForCols=selCols, namingCol="DHW_energySource")
        return df

    def heatingSystemSource(df):
        from preProcessing_Func import dfTUS_data as dftus
        #1ST STEP
        selCols = ['Methane Home Heating', 'Diesel Home Heating', 'LPG Home Heating','Oil Home Heating' ,'Solid Fuel', 'Electric Home Heating', 'Other Home Heating']
        df = dftus.MultipleEncoding(df, listForCols=selCols, namingCol="heatSysSource")

        '''
        # merge all columns related with Fluid Fossil Fuel
        selCols = ['Methane Home Heating', 'Diesel Home Heating', 'LPG Home Heating','Oil Home Heating']
        df[selCols] = df[selCols].apply(np.int64)
        df[selCols] = df[selCols].replace({1: 1, 2: 0})
        # convert them binary info, fff exist(1) or not(0)
        df['fossilFuel'] = df[selCols].sum(axis=1).astype(int)
        df['fossilFuel'] =  stats.convertBinary(df['fossilFuel'],1,1)

        # 2ND STEP
        # remove alternative fuel column by spreading the information to other columns
        df['Other Home Heating'] = stats.interpolateWrongInput(df['Other Home Heating'], [1,2])
        sel2Cols = ['fossilFuel','Solid Fuel', 'Electric Home Heating', 'Other Home Heating']
        df[sel2Cols] = df[sel2Cols].apply(np.int64)
        df[sel2Cols] = df[sel2Cols].replace({1: 1, 2: 0})
        df['heatSysSource'] = df[sel2Cols].astype(str).sum(axis=1).astype(int)
        #df['heatSysSource'] = stats.bruteForceReplacement(df['heatSysSource'], [0,1,10, 100, 1000], replace=1)
        df['heatSysSource'] = df['heatSysSource'].replace({1000:1, 100: 2, 10: 3, 1:4})
        df['heatSysSource'] = stats.interpolateWrongInput(df['heatSysSource'], [0,1,2,3], integer=True)
        #print(df['heatingSystem'].describe())

        selCols = ['Methane Home Heating', 'Diesel Home Heating', 'LPG Home Heating','Oil Home Heating', 'fossilFuel' ,'Solid Fuel', 'Electric Home Heating', 'Other Home Heating']
        for i in selCols:
            df = df.drop(i, axis=1)
        '''
        return df

    def heatingSystem(df):
        from preProcessing_Func import dfTUS_data as dftus

        selCols = ['Fixed Appliances (Whole)','Fixed Appliances (Partial)','Centralized System', 'Independent System']
        #1ST STEP
        df = dftus.MultipleEncoding(df, listForCols=selCols, namingCol="heatingSystem")

        '''
        #1st step: merge independent room-based heating system columns
        selCols = ['Fixed Appliances (Whole)','Fixed Appliances (Partial)']
        df['roomHeating']= df['Fixed Appliances (Whole)'] + df['Fixed Appliances (Partial)']
        df['roomHeating'] = stats.interpolateWrongInput(df['roomHeating'], [3,4], integer=True)
        df['roomHeating'] = df['roomHeating'].replace({4: 0})

        #2nd step: clean and merge central and independent residential-based heating system columns
        selCols2 = ['Centralized System', 'Independent System']
        mask = (df['Centralized System'] == 1) & (df['Independent System'] == 1)
        df.loc[mask, 'Independent System'] = 2
        df['Centralized System']=df['Centralized System'].replace({1:1, 2:0})
        df['Independent System']=df['Independent System'].replace({1:2, 2:0})
        df['heatingSystem01'] = df['Centralized System'] + df['Independent System']

        #3rd step: merge all heating systems
        mask = (df['heatingSystem01'] == 1) | (df['heatingSystem01'] == 2)
        df.loc[mask, 'roomHeating'] = 0
        df['heatingSystem'] = df['heatingSystem01'] + df['roomHeating']

        selCols = ['Fixed Appliances (Whole)','Fixed Appliances (Partial)','Centralized System', 'Independent System', 'heatingSystem01', 'roomHeating']
        for i in selCols:
            df = df.drop(i, axis=1)
            
        '''
        return df

    def coolingSystem(df):
        df['Air Conditioning'] = stats.interpolateWrongInput(df['Air Conditioning'], [1,2], integer=True)
        df['Air Conditioning'] = df['Air Conditioning'].replace({1: 1, 2: 0})
        return df

    def profUse(df):
        from preProcessing_Func import stats as dfppss
        #df = df.drop(['Professional Rooms'], axis=1)
        dfppss.bruteForceImput(df, nameToEdit='Professional Rooms', val2imp=0, imputSelection=6, rename='Professional Rooms')  # https://docs.google.com/document/d/1hN8400ZsH2eXd716fE-K8HTGi_5JuxbB0EjxRuM25u0/edit
        return df
class dfPrePIndividu(): # dataframe PreProcess for Individual Sheets
    pass

    def dropColumns(df,listToDROP):
        selCols = listToDROP # selected columns
        for i in selCols:
            df = df.drop(i, axis=1)
        return df

    def editCol(df, name, visual=False, editedDescribe=False, rawDescribe=False, rename=None):
        from matplotlib.ticker import MaxNLocator

        if rawDescribe == True:
            print(df[name].describe(include = 'all'))

        # REPLACE empty rows with NaN
        df[name] = pd.to_numeric(df[name], errors='coerce', downcast='float')

        #CONVERT DTYPE
        #dfEdit = df[name].astype(str).astype('float64')
        dfEdit = df[name].astype('float64')

        if editedDescribe == True:
            print(dfEdit.describe(include = 'all'))

        if visual == True:
            min = int(dfEdit.min())
            max = int(dfEdit.max())
            axes = dfEdit.hist(legend=True, rwidth=0.75,bins=np.arange(min-0.5, max+1))
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.set_xticks(ticks=np.arange(min, max+1)) # the default tick labels will be these same numbers
            plt.show()
        else:
            pass

        #INSERT edited column
        df[name] = dfEdit

        if rename is not None:
            df.rename(columns = {name:rename}, inplace = True)
        return df

    def KMeans(df,name,nrows=None, rawDescribe= False, editDescribe=False, visual=False, showCluster=False, tocsv=False): #find optimal cluster numbers
        import time
        # duration of a simulation
        start = time.time()

        #O STEP: Load the data
        df[name] = df[name].head(nrows) # if nrows==None, all df will be executed

        #1ST STEP: IF there are missing values
        #REPLACE empty rows with NaN
        df[name].replace('', 0, inplace=True)
        df[name].replace(' ', 0, inplace=True)
        df[name].replace('  ', 0, inplace=True)
        df[name].replace('   ', 0, inplace=True)

        #CONVERT DTYPE
        dfEdit = df[name].astype(str).astype('int64')
        dfEdit[dfEdit== 0] = np.nan

        #2ND STEP: Create a copy of the column with the missing values replaced by -1
        X = dfEdit.values
        X = X.reshape(-1,1)
        X_copy = X.copy()
        X_copy[dfEdit.isna()] = -1

        if rawDescribe == True: # describe before KMeans
            print(dfEdit.describe(include = 'all'))

        #3RD STEP: KMEANS CLUSTERING
        from sklearn.cluster import KMeans
        from sklearn.model_selection import cross_val_score

        # Choose a range of possible values for the number of clusters
        n_clusters_range = range(2, 11) #numbers pre-defined

        # Create a list to store the scores for each value of n_clusters
        scores = []

        # Iterate over the range of values for n_clusters
        for n_clusters in n_clusters_range:
            # Create a KMeans model with the current value of n_clusters
            kmeans = KMeans(n_clusters=n_clusters)

            # Use cross-validation to evaluate the model with the current value of n_clusters
            score = cross_val_score(kmeans, X_copy, cv=5)

            # Append the mean of the scores to the scores list
            scores.append(score.mean())

        # Find the value of n_clusters that gives the highest score
        best_n_clusters = n_clusters_range[scores.index(max(scores))]

        # Train the model with the optimal value of n_clusters
        kmeans = KMeans(n_clusters=best_n_clusters)
        kmeans.fit(X_copy)

        # Extract the cluster centroids
        centroids = kmeans.cluster_centers_

        # Predict the cluster labels for the copied column
        cluster_labels = kmeans.predict(X_copy)

        # Select the corresponding cluster centroids
        centroids_selected = centroids[cluster_labels]

        # Replace the values in the original column with the cluster labels
        df['cluster' + name] = centroids_selected
        df['cluster' + name] = df['cluster' + name].round(0)
        df['cluster' + name][df['cluster' + name]== -1] = np.nan

        #df['cluster' + name].loc[df['column'] == 'column_value', 'column'] = 'new_column_value'

        #print(df['cluster' + name].describe(include = 'all'))

        if tocsv == True:
            df['cluster' + name].to_csv('kmeans.csv')

        if visual == True:
            df['cluster' + name].hist(legend=True)
            plt.show()

        if editDescribe == True: # describe before KMeans
            print(df['cluster' + name].describe(include = 'all'))

        if showCluster == True:
            for label in np.unique(cluster_labels):
                # Get the indices of the points in the current cluster
                cluster_indices = np.where(cluster_labels == label)[0]

                # Get the points in the current cluster
                cluster = X[cluster_indices]

                # Get the unique values in the cluster
                unique_values = np.unique(cluster)

                # Print the cluster
                print(f'Cluster {label}:')
                print(unique_values)
        else:
            pass

        # 4TH STEP: REPLACE raw column with Edited column
        df[name]= df['cluster' + name]
        #simulation execution period
        end = time.time()

        print('')
        print('duration:', end - start)

        return df

    def BIRCH(df,name,nrows=None, rawDescribe= False, editDescribe=False, visual=False, showCluster=False, tocsv=False, duration=False, rename=None): #find optimal cluster numbers
        # duration of a simulation
        start = time.time()

        #O STEP: Load the data
        df[name] = df[name].head(nrows) # if nrows==None, all df will be executed

        #1ST STEP: IF there are missing values
        #REPLACE empty rows with NaN
        df[name].replace('', 0, inplace=True)
        df[name].replace(' ', 0, inplace=True)
        df[name].replace('  ', 0, inplace=True)
        df[name].replace('   ', 0, inplace=True)

        #CONVERT DTYPE
        dfEdit = df[name].astype(str).astype('int64')
        dfEdit[dfEdit== 0] = np.nan

        #2ND STEP: Create a copy of the column with the missing values replaced by -1
        X = dfEdit.values
        X = X.reshape(-1,1)
        X_copy = X.copy()
        X_copy[dfEdit.isna()] = -1

        if rawDescribe == True: # describe before KMeans
            print(dfEdit.describe(include = 'all'))

        #3RD STEP: KMEANS CLUSTERING
        from sklearn.cluster import Birch

        # Train the model with the optimal value of n_clusters
        clustering = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)
        clustering.fit(X_copy)

        # Extract the cluster centroids
        centers = clustering.subcluster_centers_

        # Predict the cluster labels for the copied column
        cluster_labels = clustering.labels_

        # Select the corresponding cluster centroids
        centers_selected = centers[cluster_labels]

        # Replace the values in the original column with the cluster labels
        df[name] = centers_selected
        df[name] = df[name].round(0)
        df[name][df[name]== -1] = np.nan
        #print(df['cluster' + name].describe(include = 'all'))

        if tocsv == True:
            df[name].to_csv('birch.csv')

        if visual == True:
            df[name].hist(legend=True)
            plt.show()

        if editDescribe == True: # describe before KMeans
            print(df[name].describe(include = 'all'))

        if showCluster == True:
            for label in np.unique(cluster_labels):
                # Get the indices of the points in the current cluster
                cluster_indices = np.where(cluster_labels == label)[0]

                # Get the points in the current cluster
                cluster = X[cluster_indices]

                # Get the unique values in the cluster
                unique_values = np.unique(cluster)

                # Print the cluster
                print(f'Cluster {label}:')
                print(unique_values)
        else:
            pass

        # 4TH STEP: REPLACE raw column with Edited column
        df[name]= df[name]

        if rename is not None:
            df.rename(columns = {name:rename}, inplace = True)

        if duration == True:#simulation execution period
            end = time.time()
            print('')
            print('duration:', end - start)

        return df

    def location(df):
        selCols = ['REGIONE', 'RIP_GEO']
        df.dropna(subset=['REGIONE'], inplace=True)
        #renameCols = ['regionHH', 'directionHH']
        #for i in range(len(selCols)):
        #    df.rename(columns = {selCols[i]:renameCols[i]}, inplace = True)
        #print(df[renameCols].describe(include = 'all'))
        return df

    def filter_occ(df):  # filter df based on accuracy reasons
        colsToDrop = ["ID_IND_CONV", "TIPCONV", "MOPERC", "STATO_NAS", "CITT_ITA_ACQ", "MESE_TRASFERIMENTO",
                      "ANNO_TRASFERIMENTO", "STATO_EST_ULT_RESID", "STATO_CIV_PREC", "HA_DIP"]
        df = df.drop(colsToDrop, axis=1)

        df = df[df['ID_ALL'].notnull()]

        # filter according to 'returnFromOut', when the occupants go to job or study from another location
        # df = df[df['workLocation'] != 5]
        # df = df[df['accomOrNot'] == 1]
        # df = df[df['returnFromOut'] == 1]

        #df = df[df['FATTE_ORE'] != 10]

        return df

    def ETA_CLASSI(df, visual=False):
        #df['ETA_CLASSI'] = df['ETA_CLASSI'].replace({13: 12, 14: 12})

        if visual == True:
            min = 1
            max = 12
            axes = df['ETA_CLASSI'].hist(legend=True, rwidth=0.75,bins=np.arange(min-0.5, max+1))
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.set_xticks(ticks=np.arange(min, max+1)) # the default tick labels will be these same numbers
            plt.show()
        else:
            pass

        df.rename(columns = {'ETA_CLASSI':'AgeInClasses'}, inplace = True)
        return df

    def household(df):
        selCols = ['NROCOMPO', 'TIPOLOGIA_FAM', 'REL_PAR', 'SESSO', 'ETA_CLASSI', 'STATO_CIV']
        renameCols = ['HHsize', 'HHCombinations', 'FamRelation', 'gender', 'AgeInClasses','MaritalStatus']
        for i in range(len(selCols)):
            df.rename(columns = {selCols[i]:renameCols[i]}, inplace = True)

        renameCols = ['HHsize', 'HHCombinations', 'FamRelation', 'gender', 'AgeInClasses','MaritalStatus']
        #print(df[renameCols].describe(include = 'all'))
        return df

    def unemployed_stat(df):
        #selCols = ["CERCA_LAV", "DISPO_LAV", "SVOLTO_LAV", "reasonInactive", "ASSENZA_LAVORO"]
        #df = dfTUS_data.MultipleEncoding(df, listForCols=selCols, namingCol="unemployed_status")
        df = df.drop(["CERCA_LAV", "DISPO_LAV", "SVOLTO_LAV", "ASSENZA_LAVORO"], axis=1)
        return df

    def TIPOLOGIA_FAM(df, visual=False, editedDescribe=False):

        #df['TIPOLOGIA_FAM'] = df['TIPOLOGIA_FAM'][df['TIPOLOGIA_FAM'] > 0]
        #f['TIPOLOGIA_FAM'] = df['TIPOLOGIA_FAM'].replace({3:2, 6:4, 7:4, 8:4, 9:5, 11:10, 12:10, 13:10})
        #df['TIPOLOGIA_FAM'] = df['TIPOLOGIA_FAM'].replace({4:3, 5:4, 10:5})

        if visual == True:
            min = 1
            max = 5

            axes = df['TIPOLOGIA_FAM'].hist(legend=True, rwidth=0.75,bins=np.arange(min- 0.5, max+1))
            axes.xaxis.set_major_locator(MaxNLocator(integer=True))
            axes.set_xticks(ticks=np.arange(min, max+1)) # the default tick labels will be these same numbers
            plt.show()
        else:
            pass

        if editedDescribe == True:
            print(df['TIPOLOGIA_FAM'].describe(include = 'all'))

        df.rename(columns = {'TIPOLOGIA_FAM':'HHCombinations'}, inplace = True)
        return df

    def workStats(df): #WORK-STATISTICS
        # reasonInactive columns: {0: employed, 1: retired, 2:student, 3: house-person, 4: other}
        df = stats.bruteForceImput(df=df, nameToEdit='CONDIZ_INATTIVI', namedfAccMain='COND_PROF',
                                  inputsCom=[1,2], val2imp=0, rename='reasonInactive', imputSelection=5)

        # reasonInactive columns: {0: employed, 1: retired, 2:student, 3: house-person, 4: other, 5: looking for job}
        df = stats.bruteForceImput(df=df, nameToEdit='reasonInactive', namedfAccMain='COND_PROF',
                                val2comp=3, val2imp=5, rename='reasonInactive', imputSelection=1)

        # permFixed columns: {0: unemployed, 1: Indefinitely, 2:Temporary}
        df = stats.bruteForceImput(df=df, nameToEdit='LAV_A_TEMPO', namedfAccMain='COND_PROF',
                                  inputsCom=[3,4,5,6,7], val2imp=0, rename='permFixed', imputSelection=5)

        df = stats.bruteForceImput(df=df, nameToEdit='permFixed', val2imp=0, rename='permFixed', imputSelection=6)

        df = stats.bruteForceImput(df=df, nameToEdit='POS_PROF', val2imp=0, rename='POS_PROF', imputSelection=6)

        df = stats.bruteForceImput(df=df, nameToEdit='FATTE_ORE', val2imp=2, rename='FATTE_ORE', imputSelection=6)

        df = stats.bruteForceImput(df=df, nameToEdit='ORE_LAV_SETT_PREC', namedfAccMain='COND_PROF',
                                       val2imp=0, rename='workingHours', imputSelection=3)

        df = stats.bruteForceImput(df=df, nameToEdit='TEMPO_PIENO', namedfAccMain='COND_PROF',
                                       val2imp=0, rename='Full_Part_time', imputSelection=3)

        df = stats.bruteForceImput(df=df, nameToEdit='SETT_ATECO', namedfAccMain='COND_PROF',
                                       val2imp=0, rename='econSector', imputSelection=3)

        df = stats.bruteForceImput(df=df, nameToEdit='TIPO_LAV', namedfAccMain='COND_PROF',
                                       val2imp=0, rename='jobType', imputSelection=3)

        #df.rename(columns = {'COND_PROF':'Pro_NonPro'}, inplace = True)

        df = stats.bruteForceImput(df=df, nameToEdit='COND_PROF', val2imp=7, rename='COND_PROF', imputSelection=6)

        workSelCols = [ 'Full_Part_time', 'econSector', 'jobType', 'workingHours',
                       'POS_PROF', 'permFixed', 'reasonInactive']
        for i in workSelCols:
            df = stats.bruteForceImput(df=df, nameToEdit=i, namedfAccMain='eduStatAll',
                                           inputsCom=[1,2,3], val2imp=0, rename=i, imputSelection=5)
            df = stats.bruteForceImput(df=df, nameToEdit=i, namedfAccMain='eduStatChild',
                                           inputsCom=[1,2,3,4], val2imp=0, rename=i, imputSelection=5)


        #print(df[workSelCols].describe(include = 'all'))
        return df

    def parent_status(df):
        selCols = ["LUONAS_MADRE", "STATO_NAS_MADRE", "LUONAS_PADRE", "STATO_NAS_PADRE"]
        df = dfTUS_data.MultipleEncoding(df, listForCols=selCols, namingCol="parent_status")

        df = df.drop("parent_status", axis=1)
        return df

    def workSchStats(df): #WORK-SCHEDULE-STATISTICS
        selCols = ['ALL_STUD_LAV', 'RIENTRA',
                   'LUOGO_STUD_LAV', 'REG_STUD_LAV', 'STATO_STUD_LAV',
                   'ORA_VA_STUD_LAV', 'MIN_VA_STUD_LAV', 'TEMPO_VA_STUD_LAV',
                   'MEZZO_TRAS_STUD_LAV']

        #print(df[selCols].describe(include = 'all'))

        # trainStatAdult columns: {0: baby (0-3 month), 1,2,3,4,5,6}
        df = stats.bruteForceImput(df=df, nameToEdit='STUD_LAV_OGNI_G', namedfAccMain='AgeInClasses',
                                       val2imp=0, rename='STUD_LAV_OGNI_G', imputSelection=3)

        renameCols = ['accomOrNot', 'returnFromOut', 'workLocation', 'regionStudyWork', 'workCountry', 'exitHours',
                      'exitMinutes', 'travelTime', 'typeOftransport']

        for i, col in enumerate(selCols):
            #  columns: {0: does not work or study}
            df = stats.bruteForceImput(df=df, nameToEdit=col, namedfAccMain='STUD_LAV_OGNI_G',
                                      inputsCom=[3,4,5,6], val2imp=0, rename=renameCols[i], imputSelection=5)

        sel2Cols = ['accomOrNot', 'returnFromOut', 'workLocation', 'regionStudyWork',
                    'workCountry', 'exitHours', 'exitMinutes', 'travelTime', 'typeOftransport']

        for i, col in enumerate(sel2Cols):
            #  columns: {0: does not work or study}
            df = stats.bruteForceImput(df=df, nameToEdit=col, namedfAccMain='STUD_LAV_OGNI_G',
                                           val2comp=0, val2imp=0, rename=sel2Cols[i], imputSelection=1)

        #print(df[sel2Cols].describe(include = 'all'))
        return df

    def eduStats(df): #EDUCATION-STATISTICS
        #{0: child (under age of 6), 1:Illiterate or elementary school, 2:Middle School diploma, 3:Qualifying school diploma (2-3 year school course),
        # 4:High school diploma, 5:Non-university tertiary diploma, 6:University degree, 7:Degree Diploma}
        df = stats.bruteForceImput(df=df, nameToEdit='TIT_STUD', namedfAccMain='FREQ_SCUOLA_FIGLI',
                                       val2imp=0, rename='eduStatAll', imputSelection=3)

        # no higher education because he/she is child
        df = stats.bruteForceImput(df=df, nameToEdit='TIT_STUD_EST', namedfAccMain='eduStatAll',
                                       inputsCom=[0,1], val2imp=0, rename='highEduExist', imputSelection=5)

        # trainAdult6m columns: {0: other Courses, 1: 6months course, 2:no-6months course}
        df = stats.bruteForceImput(df=df, nameToEdit='CORSO_STAT6', namedfAccMain='FREQ_SCUOLA_FIGLI', nameddfAcc='CORSO_STAT24',
                                       val2imp=0, rename='trainAdult6m', imputSelection=4)

        # trainAdult24m columns: {0: other Courses, 1: 24months course, 2:no-24months course}
        df = stats.bruteForceImput(df=df, nameToEdit='CORSO_STAT24', namedfAccMain='FREQ_SCUOLA_FIGLI', nameddfAcc='trainAdult6m',
                                       val2imp=0, rename='trainAdult24m', imputSelection=4)

        # uniORmore columns: {0: noHigher Education, 1: 1st level masters, 2:second level masters, 3: graduate School, 4:Ph.D}
        df = stats.bruteForceImput(df=df, nameToEdit='POST_LAUREA', namedfAccMain='eduStatAll',
                                       val2imp=0, rename='uniORmore', imputSelection=3)

        # eduStatChild columns: {0: Not-A-Child, 1:Nursery, micro-asylum, baby-parking, etc. (3-36 months), 2:primary school, 3:First grade, 4:neither kindergarten, nor childhood school, nor first grade}
        df = stats.bruteForceImput(df=df, nameToEdit='FREQ_SCUOLA_FIGLI', namedfAccMain='eduStatAll',
                                       val2imp=0, rename='eduStatChild', imputSelection=3)

        # eduAdultEnroll columns: {0: eduChild, 1:education enrolled, 2: no-education enrolled}
        df = stats.bruteForceImput(df=df, nameToEdit='ISCR_SCUOLA', namedfAccMain='eduStatChild',
                                       val2imp=0, rename='eduAdultEnroll', imputSelection=3)

        # trainStatAdult columns: {0: eduChild, 1:training course, 2: no-tarining course}
        df = stats.bruteForceImput(df=df, nameToEdit='FREQ_CORSO_FORM', namedfAccMain='eduStatChild',
                                       val2imp=0, rename='trainStatAdult', imputSelection=3)

        # trainAdult6m columns: {0: other Courses, 1: 6months course, 2:no-6months course, 3: child (TIT_STUD:1)}
        df = stats.bruteForceImput(df=df, nameToEdit='trainAdult6m', namedfAccMain='eduStatAll',
                                       val2imp=3, rename='trainAdult6m', imputSelection=3)

        # trainAdult6m columns: {0: other Courses, 1: 6months course, 2:no-6months course, 3: child (TIT_STUD:1,2)}
        df = stats.bruteForceImput(df=df, nameToEdit='trainAdult24m', namedfAccMain='eduStatAll',
                                       val2imp=3, rename='trainAdult24m', imputSelection=3)

        adultEduCols = ['REGIONE', 'eduStatChild', 'eduStatAll', 'uniORmore', 'TIT_STUD_EST',
                    'eduAdultEnroll', 'trainStatAdult','trainAdult6m','trainAdult24m']
        #print(df[adultEduCols].describe(include = 'all'))
        return df

    def habitualRes(df):
        selCols = ['DIM_5ANNO_PREC', 'PROV_DIM_5ANNO_PREC', 'STATO_DIM_5ANNO_PREC', 'IND_MODELLO']
        #renameCols = ['HHsize', 'HHCombinations', 'FamRelation', 'gender', 'AgeInClasses','MaritalStatus']
        #for i in range(len(selCols)):
        #    df.rename(columns = {selCols[i]:renameCols[i]}, inplace = True)

        #STATO_DIM_5ANNO_PREC columns -> 0 : italy
        df = stats.bruteForceImput(df=df, nameToEdit='STATO_DIM_5ANNO_PREC', namedfAccMain='PROV_DIM_5ANNO_PREC',
                              inputsCom=[1,2,3,4], val2imp=0, rename='STATO_DIM_5ANNO_PREC', imputSelection=5)

        sel2Cols = ['DIM_5ANNO_PREC', 'PROV_DIM_5ANNO_PREC', 'STATO_DIM_5ANNO_PREC']
        for i, col in enumerate(sel2Cols):
            #  columns: {0: does not work or study}
            df = stats.bruteForceImput(df=df, nameToEdit=col, namedfAccMain='STUD_LAV_OGNI_G',
                                           val2comp=0, val2imp=0, rename=sel2Cols[i], imputSelection=1)

        #print(df[selCols].describe(include = 'all'))
        return df

    def unrelatedExtraCols(df):

        df = stats.bruteForceImput(df=df, nameToEdit='STATO_CITT', namedfAccMain='LUOGO_NAS',
                                   val2imp=4, rename='STATO_CITT', imputSelection=3)
        df = stats.bruteForceImput(df=df, nameToEdit='CITT_NAS_ACQ', namedfAccMain='CITTADINANZA',
                                   val2imp=2, rename='CITT_NAS_ACQ', imputSelection=3)
        df = stats.bruteForceImput(df=df, nameToEdit='DIM_ANNO_PREC', namedfAccMain='PROV_DIM_ANNO_PREC',
                                   val2imp=4, rename='DIM_ANNO_PREC', imputSelection=3)


        selCols = ['DIM_5ANNO_PREC','PROV_DIM_5ANNO_PREC','STATO_DIM_5ANNO_PREC']
        for i, col in enumerate(selCols):
            #  columns: {0: does not work or study}
            df = stats.bruteForceImput(df=df, nameToEdit=col, namedfAccMain='AgeInClasses',
                                           val2comp=1, val2imp=0, rename=selCols[i], imputSelection=1)


        return df
class dfTUS_data():
    pass

    def oneHotEncode(df, listForCols): # binary categorization
        '''This function is used only for columns with empty cells.
        :param df:
        :param listForCols:
        :return:
        '''
        from sklearn.preprocessing import OneHotEncoder

        # fill missing values with 0
        df[listForCols].fillna(0, inplace=True)

        # instantiate OneHotEncoder object
        ohe = OneHotEncoder(sparse=False)

        # transform the dataframe into one-hot encoded array
        one_hot_encoded = ohe.fit_transform(df[listForCols])
        new_list = [x for x in listForCols for i in range(2)]
        namesCols = []
        for i in range(len(new_list)):
            namesCols.append(str(new_list[i] + "_" + str(i)))
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=namesCols)

        # drop previous columns
        df = df.drop(listForCols, axis=1)

        # merge generated columns with df

        dfoUT = pd.concat([df, one_hot_encoded_df], axis=1)
        return dfoUT

    def MultipleEncoding(df, listForCols, namingCol):
        from sklearn.preprocessing import LabelEncoder

        # Fill missing values with a unique value
        data = df[listForCols].copy()

        # Concatenate columns into one
        data.loc[:, namingCol] = data.apply(lambda x: ''.join(x.astype(str)), axis=1)

        # Encode the merged column using LabelEncoder
        le = LabelEncoder()
        data.loc[:, namingCol] = le.fit_transform(data.loc[:, namingCol])

        # if there are rows that are 'nan' for all columns
        for i in range(len(le.classes_.tolist())):
            if le.classes_.tolist()[i] == (
                    'nan' * len(listForCols)):
                x = i
                data.loc[:, namingCol] = data.loc[:, namingCol].replace(x, np.nan)
            else:
                pass

        # Remove the original columns
        data_merged = data.loc[:,namingCol]

        # drop previous columns
        df = df.drop(listForCols, axis=1)

        # merge generated columns with df
        dfout = pd.concat([df, data_merged], axis=1)

        return dfout
class imputation():
    pass

    def randomforestclassification(input_path, columns_to_impute, errorPrint=False, to_csv=False, out_csv=None):
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Separate data with missing values and data without missing values
        missing_data = df[df['Room Count'].isnull() | df['House Area'].isnull()]
        complete_data = df.dropna(subset=columns_to_impute)

        # Define the target columns
        target_columns = columns_to_impute

        # Define categorical, binary, and continuous columns
        categorical_columns = ["Residence Region",
                               "Region",
                               "Family Typology",
                               #"Full_Part_time",
                               "Number Family Members",
                               #"Age Classes",
                               #"Education Degree",
                               #"Employment status",
                               #"Job type",
                               #"Hours Worked",
                               #"Marital Status",
                               ]
        binary_columns = [ "Gender"]
        continuous_columns = ["Work Hours",
                              #"Departure Hour for work/study",
                              #"Departure Minute for work/study",
                              #"Commute Duration for work/study",
                              ]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(complete_data.drop(target_columns, axis=1),
                                                            complete_data[target_columns], test_size=0.30,
                                                            random_state=42)


        # Define the column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                ('bin', SimpleImputer(strategy='most_frequent'), binary_columns),
                ('cont', StandardScaler(), continuous_columns)
            ])

        # Create the pipeline with the preprocessor and the Random Forest Regressor
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42)))])

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Import the necessary functions
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        #MODEL EVALUATION
        y_pred = pipeline.predict(X_test)
        accuracy_scores = [accuracy_score(y_test[col], y_pred[:, i]) for i, col in enumerate(target_columns)]
        precision_scores = [precision_score(y_test[col], y_pred[:, i], average='weighted', zero_division=False) for i, col in
                            enumerate(target_columns)]
        recall_scores = [recall_score(y_test[col], y_pred[:, i], average='weighted', zero_division=False) for i, col in
                         enumerate(target_columns)]
        f1_scores = [f1_score(y_test[col], y_pred[:, i], average='weighted', zero_division=False) for i, col in enumerate(target_columns)]

        # Calculate the average scores
        mean_accuracy = np.mean(accuracy_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_f1 = np.mean(f1_scores)

        # Print the metrics
        print("Accuracy: {:.4f}".format(mean_accuracy))
        print("Precision: {:.4f}".format(mean_precision))
        print("Recall: {:.4f}".format(mean_recall))
        print("F1 Score: {:.4f}".format(mean_f1))

        #IMPUTATION TO THE DATAFRAME
        # Impute missing values
        imputed_values = pipeline.predict(missing_data.drop(target_columns, axis=1))

        # Assign imputed values to missing_data
        missing_data = missing_data.copy()
        missing_data.loc[:, target_columns] = imputed_values

        # Combine imputed data with complete data
        imputed_data = pd.concat([complete_data, missing_data], axis=0)

        # OUTPUT
        if to_csv == True:
            imputed_data.to_csv(out_csv, index=None)
            print('imputation by classification is completed: writing as .csv is done')

        return imputed_data

    def iterative_imputation(input_path, columns_to_impute, errorPrint=False, to_csv=False, out_csv=None):
        '''
        OrdinalEncoder for categorical variables and evaluates the model using
        regression metrics (Mean Squared Error, Root Mean Squared Error, and R2 Score)

        '''

        import numpy as np
        import pandas as pd
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Separate data with missing values and data without missing values
        missing_data = df[df['Room Count'].isnull() | df['House Area'].isnull()]
        complete_data = df.dropna(subset=columns_to_impute)

        # Define the target columns
        target_columns = columns_to_impute

        # Define categorical, binary, and continuous columns
        categorical_columns = ["Residence Region", "Residence GeoDistribution", "Family Type", "Full_Part_time",
                               "People Count", "Age in Classes",
                               "Highest Education", "Employment Status", "Professional Position", "Hours Worked", ]
        binary_columns = ["Gender"]
        continuous_columns = ["Work Hours at 2001",
                              "Departure Hour for work/study", "Departure Minute for work/study",
                              "Commute Duration for work/study"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(complete_data.drop(target_columns, axis=1),
                                                            complete_data[target_columns], test_size=0.30,
                                                            random_state=42)

        # Define the column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),
                ('bin', SimpleImputer(strategy='most_frequent'), binary_columns),
                ('cont', StandardScaler(), continuous_columns)
            ])

        # Define the column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),
                #('bin', SimpleImputer(strategy='most_frequent'), binary_columns),
                #('cont', StandardScaler(), continuous_columns)
            ])

        # Create the pipeline with the preprocessor and the Iterative Imputer
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('imputer', IterativeImputer(random_state=42,
                                                                estimator=RandomForestRegressor(n_estimators=100,
                                                                                                random_state=42)))])

        # Train the pipeline and impute missing values
        pipeline.fit(X_train, y_train)
        imputed_values = pipeline.transform(missing_data.drop(target_columns, axis=1))

        # Assign imputed values to missing_data
        missing_data = missing_data.copy()
        for idx, col in enumerate(target_columns):
            missing_data[col] = imputed_values[:, idx]

        # Combine imputed data with complete data
        imputed_data = pd.concat([complete_data, missing_data], axis=0)

        if to_csv:
            if out_csv is None:
                out_csv = "imputed_data.csv"
            imputed_data.to_csv(out_csv, index=False)
            print(f"Imputed data saved to {out_csv}")

        # Evaluate the pipeline
        X_test_transformed = pipeline.transform(X_test)

        # Predict the target variable for the test data
        y_pred = X_test_transformed[:, -len(target_columns):]

        # Calculate evaluation metrics for regression
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        if errorPrint:
            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error: {rmse}")
            print(f"R2 Score: {r2}")

    def iterative_imputation_allCols(df, cols_to_extract=None):

        import numpy as np
        import pandas as pd

        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        # Create the imputer
        imputer = IterativeImputer(missing_values=np.nan,initial_strategy ="most_frequent",) #for categorical dataset

        # Get the ID columns
        id_cols = ["Residential_ID","Occupant ID in HH","Family ID"] + cols_to_extract

        # Create a new DataFrame that only contains the ID columns
        id_df = df[id_cols]

        # Impute the missing values in the remaining columns
        imputed_df = imputer.fit_transform(df.drop(columns=id_cols))

        imputed_df = pd.DataFrame(imputed_df, columns=df.drop(columns=id_cols).columns)

        # Combine the imputed DataFrame with the ID DataFrame
        imputed_df = pd.concat([imputed_df, id_df], axis=1)

        # Round values for each column, handling NaN values
        # Specify the replacement value for NaN
        replacement_value = -1
        for column in imputed_df.columns:
            imputed_df[column] = imputed_df[column].fillna(replacement_value).apply(round)
            imputed_df[column] = imputed_df[column].replace(replacement_value, np.nan)

        return imputed_df


    def linear_imp(df, columns_to_impute,  errorPrint=False): #works on local PyCharm
        from sklearn.metrics import mean_absolute_error, accuracy_score

        # Shuffle the DataFrame until there are no NaN values in the first row, otherwise, problem occurs
        while df.isna().iloc[0].any():
            df = df.sample(frac=1)

        # Create a copy of the original DataFrame to store the imputed values
        imputed_df = df.copy()

        for column_to_impute in columns_to_impute:
            # Impute the missing values using linear interpolation
            imputed_df[column_to_impute] = df[column_to_impute].interpolate(method='linear').round().astype(int)

        # Create a new DataFrame that contains only the non-missing values of the A column
        ground_truth_df = df.dropna(subset=columns_to_impute)

        if errorPrint == True:
            # Calculate the accuracy score between the imputed values and the ground truth values for each column

            accuracy_scores = {}
            for column_to_impute in columns_to_impute:
                imputed_vals = imputed_df[column_to_impute].loc[ground_truth_df.index]
                ground_truth_vals = ground_truth_df[column_to_impute].loc[ground_truth_df.index]
                accuracy_scores[column_to_impute] = accuracy_score(imputed_vals, ground_truth_vals)
                print(f"Accuracy score for column {column_to_impute}: {accuracy_scores[column_to_impute]:.2f}")

            # Calculate the MAE between the imputed values and the ground truth values for each column
            mae_scores = {}
            for column_to_impute in columns_to_impute:
                mae_scores[column_to_impute] = mean_absolute_error(imputed_df[column_to_impute].loc[ground_truth_df.index], ground_truth_df[column_to_impute].loc[ground_truth_df.index])
                print(f"MAE for column {column_to_impute}: {mae_scores[column_to_impute]:.2f}")

        return imputed_df

    def KNN_imp(df, columns, errorPrint=False):
        # https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/
        from sklearn.impute import KNNImputer
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        df_imputed = imputer.fit_transform(df[columns])
        dfImputed = pd.DataFrame(df_imputed)

        df[columns] = dfImputed
        return df

    def corrImp(df, coltoEdit, colRef, newColName, errorPrint=False):
        from sklearn.impute import KNNImputer

        # create a KNN imputer object with k=3
        imputer = KNNImputer(n_neighbors=2,
                             weights='distance',
                             metric='nan_euclidean')

        # impute missing values in column1 using values from column2
        column1 = pd.DataFrame(df[coltoEdit].values.reshape(-1, 1))
        column2 = pd.DataFrame(df[colRef].values.reshape(-1, 1))

        df[newColName] = imputer.fit_transform(pd.concat([column1, column2], axis=1))[:, 0]
        df[newColName] = df[newColName].round(decimals=0)

        if errorPrint == True:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            # lower values fo mae is better
            mae = mean_absolute_error(df[coltoEdit].dropna(), df[newColName].sample(n=len(df[coltoEdit].dropna())))
            # lower values fo mse is better
            mse = mean_squared_error(df[coltoEdit].dropna(), df[newColName].iloc[:len(df[coltoEdit].dropna())])
            # lower values fo r2 is better
            r2 = r2_score(df[coltoEdit].dropna(), df[newColName].iloc[:len(df[coltoEdit].dropna())])

            print('mae for linearImput:', mae)
            print('mse for linearImput:', mse)
            print('r2 for linearImput:', r2)

        df = df.drop(coltoEdit, axis=1)
        return df

    def kde_impute_categorical(df, columnToSelect, bandwidth=1, fraction=1, kde_visual=False):
        '''
        :param columnToSelect: to edit
        :param bandwidth: for kde
        :return: df
        '''

        df = df.sample(frac=fraction, replace=True, random_state=1)

        # identify NaN values and create a copy of the dataframe
        nan_df = df.isna()
        imputed_df = df.copy()

        # get the non-NaN values of the column
        values = df[columnToSelect][~nan_df[columnToSelect]].values.reshape(-1, 1)

        from sklearn.neighbors import KernelDensity
        # fit a KDE model to the non-NaN values
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)

        # get the missing values and fill them using the KDE model
        missing = df[columnToSelect][nan_df[columnToSelect]].values.reshape(-1, 1)
        imputed = kde.sample(missing.shape[0])

        # update the imputed dataframe with the imputed values
        # round the imputed values nearest 10
        imputed_df.loc[nan_df[columnToSelect], columnToSelect] = np.round(imputed.reshape(-1), -1)

        df.reset_index(drop=True, inplace=True)
        #print(len(df))
        #imputed_df = imputed_df.reset_index()
        imputed_df.reset_index(drop=True, inplace=True)
        #print(len(imputed_df))

        if kde_visual == True:
            import seaborn as sns
            import matplotlib.pyplot as plt
            # create a subplot with a grid of 3 columns and 1 row
            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))

            g = sns.histplot(x=df[columnToSelect], ax=axs[0],
                             kde=True,
                             #bins=10,
                             stat="probability", color='b',
                             kde_kws={'bw_adjust': bandwidth})
            d = sns.histplot(x=imputed_df[columnToSelect], ax=axs[1],
                             kde=True,
                             #bins=10,
                             stat="probability", color='r',
                             kde_kws={'bw_adjust': bandwidth})
            plt.show()

        df = imputed_df

        return df

    def find_most_important_target_variable(df):
        '''
        the p-value of the chi-squared test for each pair of columns in the dataset_TUS_daily.
        Then it computes the average p-value for each column and
        selects the one with the lowest average p-value as the most important target variable.
        '''

        import pandas as pd
        import numpy as np
        p_values = pd.DataFrame(index=df.columns, columns=df.columns)

        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    crosstab = pd.crosstab(df[col1], df[col2])
                    p_values.loc[col1, col2] = stats.get_chi2_p_value(crosstab)

        average_p_values = p_values.mean()
        most_important_target = average_p_values.idxmin()
        return most_important_target

    def iterImpute_classify(df, colToOutput, model_to='random_forest',
                            kfold_repeat=3, kfold_splits=10,
                            fraction=0.1, pipe=False,
                            missingness=False, colToDrop=False,
                            trialMode=False, singleShot=False,
                            tuning=False,
                            scaler=False):
        
        # all dataframe will be filled simultaneously
        # for categorical outputs

        '''
        :param colToOutput: output column, it should not contain missing values
        :param model_to:
        :param fraction:
        :param pipe:
        :param missingness:
        :param colToDrop:
        :return:
        '''

        from numpy import isnan
        # data normalization with sklearn
        from sklearn.preprocessing import MinMaxScaler

        df_to_impute = df.sample(frac=fraction, replace=True, random_state=1)

        #print(df_to_impute.columns)

        if colToDrop != False:
            df_to_impute = df_to_impute.drop(colToDrop, axis=1)
        else:
            pass

        if missingness == True:
            # summarize the number of rows with missing values for each column
            print('Missingness imput auto iterative imputation:')
            for i in range(df_to_impute.shape[1]):
                n_miss = df_to_impute.iloc[:,i].isna().sum()
                perc = n_miss / df_to_impute.shape[0] * 100
                colName = df_to_impute.columns[i]
                print('{}:'.format(colName), '%d (%.1f%%)' % (n_miss, perc))
        else:
            pass

        # TRANSFORM DATA FOR CLASSIFICATION
        #split into input and output elements
        data = df_to_impute.values
        if scaler == True:
            # fit scaler on your training data
            data = MinMaxScaler().fit(data)
        else:
            pass

        #df_to_impute
        yIndex = df_to_impute.columns.get_loc(colToOutput)
        dfTrain = df_to_impute.drop(columns=colToOutput)
        X = dfTrain.values
        y = df_to_impute.iloc[:, yIndex].values

        #FIND 1 MEMBER CLASSES THEN DELETE IT: it is important for hist_gradBoostingClassifier
        # Find the classes with only one member
        unique_classes, class_counts = np.unique(y, return_counts=True)
        one_member_classes = unique_classes[class_counts < 3]

        # Remove rows with one-member classes
        indices_to_remove = np.isin(y, one_member_classes)
        X = X[~indices_to_remove]
        y = y[~indices_to_remove]

        from sklearn.preprocessing import OrdinalEncoder
        # Ordinal encoding for categorical variables
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded = ordinal_encoder.fit_transform(X)

        # INTRODUCTION OF MODELS
        #dictionary of models
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
            AdaBoostClassifier, GradientBoostingClassifier,HistGradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.svm import SVC

        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor

        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.model_selection import cross_val_score, cross_val_predict        
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.pipeline import Pipeline

        # Define the parameter grid to search
        params_RFC = {
                    'n_estimators': 100,
                    'max_features': 'sqrt',
                    'max_depth': 6,
                    'min_samples_split': 2,
                    'min_samples_leaf': 4,
                    'bootstrap': False,
                    'criterion': 'gini',
                    'n_jobs': -1
                    }

        params_iterImp = {
                          'initial_strategy': 'mean',
                          'max_iter': 200,
                          'imputation_order': 'ascending', #absolute
                          'n_nearest_features': None, #absolute
                          }

        models = {
                  'random_forest': RandomForestClassifier(**params_RFC),
                  'default': RandomForestClassifier(),
                  'knn':KNeighborsClassifier(),
                  'decision_tree':DecisionTreeClassifier(),
                  'extra': ExtraTreesClassifier(),
                  'ada': AdaBoostClassifier(),
                  'hist_gradBoost': HistGradientBoostingClassifier(),
                  'sdgc': SGDClassifier(),
                  'mlp': MLPClassifier(),
                  'svm': SVC(),
                  'gauss_NB': GaussianNB(),
                  }

        if pipe==True:
            import time
            # duration of a simulation
            start = time.time()

            from numpy import mean, std

            model= models[model_to]
            imputer = IterativeImputer(**params_iterImp)
            pipeline = Pipeline(steps=[('i', imputer), ('m', model)])

            # define model evaluation
            '''Either KFold or StratifiedKFold are used by GridSearchCV 
            depending if your model is for regression (KFold) or 
            classification (then StratifiedKFold is used).'''

            cv = RepeatedStratifiedKFold(n_splits=kfold_splits,
                                         n_repeats=kfold_repeat,
                                         random_state=1)
            # evaluate model
            scores = cross_val_score(pipeline, X, y,
                                     scoring='accuracy', cv=cv,
                                     n_jobs=-1, error_score='raise')

            print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

            #simulation execution period
            end = time.time()
            print('')
            print('duration:', end - start)

            df_imputed = df_to_impute.copy()

            # Impute missing values in the encoded dataset_TUS_daily
            X_imputed_encoded = imputer.fit_transform(X_encoded)

            # Inverse transform the imputed values back to their original form
            X_imputed = ordinal_encoder.inverse_transform(X_imputed_encoded)

            # Replace missing values in the original dataset_TUS_daily with the imputed values
            df_imputed[dfTrain.columns] = X_imputed

            df_to_impute = df_imputed
            return df_to_impute
        else:
            pass

        if singleShot == True:
            # print total missing
            print('Missing: %d' % sum(isnan(X).flatten()))
            # define imputer
            imputer = IterativeImputer()
            # fit on the dataset_TUS_daily
            imputer.fit(X)
            # transform the dataset_TUS_daily
            X_trans = imputer.transform(X)
            # print total missing
            print('Missing: %d' % sum(isnan(X_trans).flatten()))
        else:
            pass

        if trialMode==True:
            # print total missing
            print('Missing: %d' % sum(isnan(X).flatten()))
            from numpy import mean, std
            for k in models:
                model = models[k]
                imputer = IterativeImputer()
                pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
                # define model evaluation
                cv = RepeatedStratifiedKFold(n_splits=kfold_splits,
                                             n_repeats=kfold_repeat,
                                             random_state=1)
                # evaluate model
                scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
                print('{}:'.format(k),'Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

            print('Missing: %d' % sum(isnan(X).flatten()))

        if tuning==True:
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.model_selection import GridSearchCV
            from scipy.stats import randint

            # first drop all NaN values
            df_to_impute = df_to_impute.dropna()

            # split into input and output elements
            data = df_to_impute.values
            yIndex = df_to_impute.columns.get_loc(colToOutput)
            dfTrain = df_to_impute.drop(columns=colToOutput)
            X = dfTrain.values
            y = df_to_impute.iloc[:, yIndex].values

            # Define the pipeline
            pipeline = Pipeline([
                ('imputer', IterativeImputer(random_state=0)),
                ('classifier', RandomForestClassifier(random_state=0)),
            ])

            # Define the parameter grid to search for RandomForestClassifier()
            param_dist = {
                          'classifier__n_estimators': [200],
                          #'classifier__max_features': ['sqrt', 'log2'],
                          'classifier__max_features': ['sqrt'],
                          #'classifier__max_depth': [None, 5, 20],
                          'classifier__max_depth': [20],
                          'classifier__min_samples_split': [2],
                          'classifier__min_samples_leaf': [4],
                          'classifier__bootstrap': [False],
                          #'classifier__criterion': ['gini', 'entropy', 'log_loss'],
                          'classifier__criterion': ['gini'],
                          'imputer__initial_strategy': ['mean', 'median', 'most_frequent', 'constant'],
                          #'imputer__initial_strategy': ['mean'],
                          'imputer__max_iter': [100],
                          #'imputer__imputation_order': ['ascending'],
                          'imputer__imputation_order': ['ascending', 'descending', 'roman', 'arabic', 'random'],
                          #'imputer__n_nearest_features': [None,10],
                          'imputer__n_nearest_features': [None],
                          }

            # Define the search strategy
            grid_search = GridSearchCV(pipeline, param_grid=param_dist, scoring='accuracy', n_jobs=-1)

            # Fit the GridSearchCV object to the data
            grid_search.fit(X, y)

            # Print the best hyperparameters and the corresponding accuracy score
            print("Model: ", model_to)
            print("Best hyperparameters: ", grid_search.best_params_)
            print("Best accuracy score: ", grid_search.best_score_)

    def endswith(self, param):
        pass
class NN_classification():

    def NN_classify(input_path, columns_to_impute, errorPrint=False, to_csv=False, out_csv=None):
        '''
        Preprocessing:
            a. One-hot encode categorical features.
            b. Normalize continuous features using standard scaling.
            c. Use an appropriate ordinal encoding for the two ordinal columns (HouseArea and RoomCount).
            Split the data:
            Separate the dataset into two subsets: one with complete data (27k instances) and the other with missing values in the ordinal columns (40k instances).
            Train a neural network:
            a. Use the 27k instances with complete data to train the neural network.
            b. Design a network architecture that accepts the preprocessed input features and predicts the two ordinal columns (HouseArea and RoomCount).
            c. Use a suitable loss function, such as mean squared error, for ordinal columns.
            d. Train the network with an appropriate optimizer, such as Adam, and tune hyperparameters (e.g., learning rate, batch size, number of layers, and number of neurons per layer).
            Impute missing values:
            a. Use the trained neural network to predict missing values in the ordinal columns for the 40k instances.
            b. Replace missing values with the predicted values.
            Postprocessing:
            Decode the ordinal columns back to their original format, if necessary.
            Evaluate performance:
            Assess the imputation performance using appropriate metrics, such as root mean squared error (RMSE) or mean absolute error (MAE), and compare with your previous imputation methods.
        '''

        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.model_selection import train_test_split

        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        import numpy as np

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Select the columns to be imputed
        target_columns = columns_to_impute
        # Select your feature columns
        X = df.drop(columns=target_columns)

        # Define the column names for each type of transformation
        onehot_cols = ['Administrative Region', 'Region', 'Number Family Members',
                       'Family Typology', 'Gender', 'Marital Status', 'Employment status', 'Full_Part_time', 'Job type']
        ordinal_cols = ['Age Classes', 'Education Degree']
        scaled_cols = ['Work Hours']
        circular_cols = ['Departure Hour for work/study', 'Departure Minute for work/study']

        # Initialize the encoders/scalers
        onehot_encoder = OneHotEncoder(sparse_output=False)
        ordinal_encoder = OrdinalEncoder()
        scaler = StandardScaler()
        minmax_scaler = MinMaxScaler((-1, 1))  # for circular variables

        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', onehot_encoder, onehot_cols),
                ('ordinal', ordinal_encoder, ordinal_cols),
                ('scale', scaler, scaled_cols),
                ('circular', minmax_scaler, circular_cols)],
            remainder='passthrough')

        # Assume X is your input features dataframe
        X_transformed = preprocessor.fit_transform(X)

        # For circular features, apply the sin and cos transformations
        X_transformed[:, -2] = np.sin(X_transformed[:, -2] * 2 * np.pi)
        X_transformed[:, -1] = np.cos(X_transformed[:, -1] * 2 * np.pi)

        # Encode target columns
        ohe2 = LabelEncoder()
        target_encoded = ohe2.fit_transform(df[target_columns].dropna())

        # Create a mask for rows with missing values in ordinal columns
        mask = df[target_columns[0]].isna()

        # Split data into complete and incomplete subsets
        X_complete = X_transformed[~mask]
        X_incomplete = X_transformed[mask]
        y_complete = target_encoded

        #Train the neural network
        input_size = X_complete.shape[1]
        hidden_size = 128
        # number of unique values in your target column
        num_classes = len(df['House Area'].unique())

        #model parameters
        #model = MulticlassClassifier(input_size, hidden_size, num_classes)
        #model = ImputeNN(input_size, hidden_size, num_classes)
        model = TwoLayerModel(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()  # Cross entropy loss for multi-class classification
        learning_rate=0.01
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               #betas=(0.9, 0.999), eps=1e-08,
                               )
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate) #best
        #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #best

        num_epochs = 25
        batch_size = 32

        # Some additional variables for early stopping
        n_epochs_stop = 5
        epochs_no_improve = 0
        min_val_loss = np.inf
        early_stop = False

        #initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        model.apply(weights_init)

        X_train, X_val, y_train, y_val = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        from torch.utils.data import TensorDataset, DataLoader

        # Convert your data into TensorDataset, which is a dataset provided by PyTorch that wraps tensor pairs (inputs and targets)
        train_dataset = TensorDataset(X_train, y_train.long())
        val_dataset = TensorDataset(X_val, y_val.long())

        # Create a DataLoader for each of your datasets. This will handle batching and shuffling for you.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)  # No need to shuffle validation data

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_dataloader:  # DataLoader will automatically batch and shuffle
                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

            # Validation
            val_loss = 0
            with torch.no_grad():  # No need to track gradients in validation
                for batch_x, batch_y in val_dataloader:
                    val_outputs = model(batch_x)
                    val_loss += criterion(val_outputs, batch_y).item()

            val_loss /= len(val_dataloader)  # Calculate average validation loss
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}")

            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                # Save the model
                torch.save(model, 'best_model.pt')
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
                    # Load the best state dict
                    model = torch.load('best_model.pt')
                    break
            if early_stop:
                print("Stopped")
                break

        # After training:
        model.eval()  # Set the model to evaluation mode

        # Get the model's prediction
        y_val_pred = model(X_val)

        # Get the class with the highest probability for each example
        _, y_val_pred_class = torch.max(y_val_pred, 1)

        # Detach from the computational graph and convert to numpy arrays
        y_val_pred_class = y_val_pred_class.detach().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_val.ravel(), y_val_pred_class)
        print(f"Accuracy: {accuracy}")

    def NN_regression(input_path, columns_to_impute, errorPrint=False, to_csv=False, out_csv=None):

        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.model_selection import train_test_split

        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        import numpy as np

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Select the columns to be imputed
        target_columns = columns_to_impute
        # Select your feature columns
        X = df.drop(columns=target_columns)

        # Define the column names for each type of transformation
        #onehot_cols = ['Administrative Region', 'Region', 'Number Family Members', 'Family Typology', 'Gender',
        # 'Marital Status', 'Employment status', 'Full_Part_time', 'Job type']
        #ordinal_cols = ['Age Classes', 'Education Degree']

        onehot_cols = ['Administrative Region', 'Region', 'Number Family Members',
                       'Family Typology', 'Gender', 'Marital Status', 'Employment status', 'Full_Part_time', 'Job type', 'Age Classes', 'Education Degree']
        scaled_cols = ['Work Hours']
        circular_cols = ['Departure Hour for work/study', 'Departure Minute for work/study']

        # Initialize the encoders/scalers
        onehot_encoder = OneHotEncoder(sparse_output=False)
        ordinal_encoder = OrdinalEncoder()
        scaler = StandardScaler()
        minmax_scaler = MinMaxScaler((-1, 1))  # for circular variables

        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', onehot_encoder, onehot_cols),
                #('ordinal', ordinal_encoder, ordinal_cols),
                ('scale', scaler, scaled_cols),
                ('circular', minmax_scaler, circular_cols)],
            remainder='passthrough')

        # Assume X is your input features dataframe
        X_transformed = preprocessor.fit_transform(X)

        # For circular features, apply the sin and cos transformations
        X_transformed[:, -2] = np.sin(X_transformed[:, -2] * 2 * np.pi)
        X_transformed[:, -1] = np.cos(X_transformed[:, -1] * 2 * np.pi)

        # Encode target columns
        sscale_target = StandardScaler()
        target_scaled = sscale_target.fit_transform(df[target_columns].dropna().values.reshape(-1, 1))

        # Create a mask for rows with missing values in ordinal columns
        mask = df[target_columns[0]].isna()

        # Split data into complete and incomplete subsets
        X_complete = X_transformed[~mask]
        X_incomplete = X_transformed[mask]
        y_complete = target_scaled

        #Train the neural network
        input_size = X_complete.shape[1]
        hidden_size = 12

        #model parameters
        model = RegressionModel(input_size, hidden_size)
        criterion = nn.MSELoss()
        learning_rate=0.001
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate,)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate) #best
        #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #best

        num_epochs = 25
        batch_size = 128

        # Some additional variables for early stopping
        n_epochs_stop = 5
        epochs_no_improve = 0
        min_val_loss = np.inf
        early_stop = False

        #initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        model.apply(weights_init)

        print(X_complete.shape)
        print(y_complete.shape)
        X_train, X_val, y_train, y_val = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        from torch.utils.data import TensorDataset, DataLoader

        # Convert your data into TensorDataset, which is a dataset provided by PyTorch that wraps tensor pairs (inputs and targets)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # Create a DataLoader for each dataset
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_dataloader:
                optimizer.zero_grad()

                outputs = model(batch_x).view(-1, 1)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

            # Validation
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_dataloader:
                    val_outputs = model(batch_x).view(-1, 1)
                    val_loss += criterion(val_outputs, batch_y).item()

            val_loss /= len(val_dataloader)  # Calculate average validation loss
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}")

            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                # Save the model
                torch.save(model, 'best_model.pt')
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
                    # Load the best state dict
                    model = torch.load('best_model.pt')
                    break
            if early_stop:
                print("Stopped")
                break

        # After training:
        model.eval()

        # Get the model's prediction
        y_val_pred = model(X_val)

        # Convert to numpy arrays
        y_val = y_val.detach().numpy()
        y_val_pred = y_val_pred.detach().numpy()

        from sklearn.metrics import mean_squared_error

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        print(f"RMSE: {rmse}")

    def NN_classify_pytorch_earlystop(input_path, columns_to_impute, errorPrint=False, to_csv=False, out_csv=None):
        '''
        Preprocessing:
            a. One-hot encode categorical features.
            b. Normalize continuous features using standard scaling.
            c. Use an appropriate ordinal encoding for the two ordinal columns (HouseArea and RoomCount).
            Split the data:
            Separate the dataset into two subsets: one with complete data (27k instances) and the other with missing values in the ordinal columns (40k instances).
            Train a neural network:
            a. Use the 27k instances with complete data to train the neural network.
            b. Design a network architecture that accepts the preprocessed input features and predicts the two ordinal columns (HouseArea and RoomCount).
            c. Use a suitable loss function, such as mean squared error, for ordinal columns.
            d. Train the network with an appropriate optimizer, such as Adam, and tune hyperparameters (e.g., learning rate, batch size, number of layers, and number of neurons per layer).
            Impute missing values:
            a. Use the trained neural network to predict missing values in the ordinal columns for the 40k instances.
            b. Replace missing values with the predicted values.
            Postprocessing:
            Decode the ordinal columns back to their original format, if necessary.
            Evaluate performance:
            Assess the imputation performance using appropriate metrics, such as root mean squared error (RMSE) or mean absolute error (MAE), and compare with your previous imputation methods.
        '''

        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.model_selection import train_test_split

        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
        from sklearn.compose import ColumnTransformer
        import numpy as np

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Select the columns to be imputed
        target_columns = columns_to_impute
        # Select your feature columns
        X = df.drop(columns=target_columns)


        # Define the column names for each type of transformation
        onehot_cols = ['Occupant ID in HH', 'Administrative Region', 'Region', 'Number Family Members',
                       'Family Typology', 'Gender', 'Marital Status', 'Employment status', 'Full_Part_time', 'Job type']
        ordinal_cols = ['Age Classes', 'Education Degree']
        scaled_cols = ['Work Hours']
        circular_cols = ['Departure Hour for work/study', 'Departure Minute for work/study']

        # Initialize the encoders/scalers
        onehot_encoder = OneHotEncoder(sparse=False)
        ordinal_encoder = OrdinalEncoder()
        scaler = StandardScaler()
        minmax_scaler = MinMaxScaler((-1, 1))  # for circular variables

        # Combine all transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', onehot_encoder, onehot_cols),
                ('ordinal', ordinal_encoder, ordinal_cols),
                ('scale', scaler, scaled_cols),
                ('circular', minmax_scaler, circular_cols)],
            remainder='passthrough')

        # Assume X is your input features dataframe
        X_transformed = preprocessor.fit_transform(X)

        # For circular features, apply the sin and cos transformations
        X_transformed[:, -2] = np.sin(X_transformed[:, -2] * 2 * np.pi)
        X_transformed[:, -1] = np.cos(X_transformed[:, -1] * 2 * np.pi)

        # Encode target columns
        ohe2 = OneHotEncoder(sparse_output=False)
        target_encoded = ohe2.fit_transform(df[target_columns].dropna())

        # Create a mask for rows with missing values in ordinal columns
        mask = df[target_columns[0]].isna()

        # Split data into complete and incomplete subsets
        X_complete = X_transformed[~mask]
        X_incomplete = X_transformed[mask]
        y_complete = target_encoded

        #Train the neural network
        input_size = X_complete.shape[1]
        hidden_size = 128
        output_size = target_encoded.shape[1]

        #model parameters
        model = ImputeNN(input_size, hidden_size, output_size)
        #model = ImputeCNN(input_size, hidden_size, output_size) # slow
        #model = MultiLabelClassifier(input_size, hidden_size, output_size)
        #model = TwoLayerModel(input_size, hidden_size, output_size)
        #model = DropoutBatchnormModel(input_size, hidden_size, output_size)

        criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy loss for multi-label classification

        learning_rate=0.01
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,)
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate) #best
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #best

        num_epochs = 5
        batch_size = 128

        # Some additional variables for early stopping
        n_epochs_stop = 10
        epochs_no_improve = 0
        min_val_loss = np.inf
        early_stop = False

        #initialization
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        model.apply(weights_init)

        X_train, X_val, y_train, y_val = train_test_split(X_complete, y_complete, test_size=0.3, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        from torch.utils.data import TensorDataset, DataLoader

        # Convert your data into TensorDataset, which is a dataset provided by PyTorch that wraps tensor pairs (inputs and targets)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # Create a DataLoader for each of your datasets. This will handle batching and shuffling for you.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)  # No need to shuffle validation data

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_dataloader:  # DataLoader will automatically batch and shuffle
                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

            # Validation
            val_loss = 0
            with torch.no_grad():  # No need to track gradients in validation
                for batch_x, batch_y in val_dataloader:
                    val_outputs = model(batch_x)
                    val_loss += criterion(val_outputs, batch_y).item()

            val_loss /= len(val_dataloader)  # Calculate average validation loss
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}")

            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                # Save the model
                torch.save(model, 'best_model.pt')
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
                    # Load the best state dict
                    model = torch.load('best_model.pt')
                    break
            if early_stop:
                print("Stopped")
                break

        # After training:
        model.eval()  # Set the model to evaluation mode

        # Get the model's prediction
        y_val_pred = model(X_val)
        y_val_pred = torch.sigmoid(y_val_pred)
        y_val_pred = y_val_pred.detach().numpy()

        # Convert the probabilities to binary predictions
        y_val_pred_binary = np.where(y_val_pred > 0.5, 1, 0)

        # Calculate metrics
        accuracy = accuracy_score(y_val.numpy(), y_val_pred_binary)
        precision = precision_score(y_val.numpy(), y_val_pred_binary, average='micro', zero_division=False)
        recall = recall_score(y_val.numpy(), y_val_pred_binary, average='micro', zero_division=False)
        f1 = f1_score(y_val.numpy(), y_val_pred_binary, average='micro', zero_division=False)

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

        from sklearn.metrics import roc_auc_score,average_precision_score
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_val.numpy(), y_val_pred)
        except ValueError:
            roc_auc = None
            print("ROC AUC score cannot be calculated because y_true contains only one class.")

        average_precision = average_precision_score(y_val.numpy(), y_val_pred)
        print(f"ROC AUC: {roc_auc}, Average Precision: {average_precision}")

    def NN_classify_pytorch_earlystop_ncross(input_path, columns_to_impute, errorPrint=False, to_csv=False, out_csv=None):
        '''
        Preprocessing:
            a. One-hot encode categorical features.
            b. Normalize continuous features using standard scaling.
            c. Use an appropriate ordinal encoding for the two ordinal columns (HouseArea and RoomCount).
            Split the data:
            Separate the dataset into two subsets: one with complete data (27k instances) and the other with missing values in the ordinal columns (40k instances).
            Train a neural network:
            a. Use the 27k instances with complete data to train the neural network.
            b. Design a network architecture that accepts the preprocessed input features and predicts the two ordinal columns (HouseArea and RoomCount).
            c. Use a suitable loss function, such as mean squared error, for ordinal columns.
            d. Train the network with an appropriate optimizer, such as Adam, and tune hyperparameters (e.g., learning rate, batch size, number of layers, and number of neurons per layer).
            Impute missing values:
            a. Use the trained neural network to predict missing values in the ordinal columns for the 40k instances.
            b. Replace missing values with the predicted values.
            Postprocessing:
            Decode the ordinal columns back to their original format, if necessary.
            Evaluate performance:
            Assess the imputation performance using appropriate metrics, such as root mean squared error (RMSE) or mean absolute error (MAE), and compare with your previous imputation methods.
        '''


        '''Learning Rate: The learning rate might be too high or too low. A learning rate of 0.005 is a good start, but you may want to experiment with different values.
           Model Complexity: Your model might be too simple to capture the complexity of the data. You can try increasing the hidden size or adding more layers to the model. Be aware that increasing model complexity might lead to overfitting. You should monitor both training loss and validation loss. If your training loss continues to go down but your validation loss starts to increase, it's a sign of overfitting.
            Dropout: You might want to experiment with different dropout values. Dropout can help with overfitting, but too much dropout can also hurt the model's performance.
            Batch Size: You could experiment with different batch sizes. A smaller batch size will make the model update weights more frequently, while a larger batch size will make the gradient descent direction more accurate.
            Data Preprocessing: Check the distribution of your target values. If the classes are imbalanced, consider using techniques like oversampling the minority class or undersampling the majority class.
            Model Architecture: If improving the current model doesn't help, consider using a different model architecture. Depending on the nature of your data and problem, other types of models (like tree-based models or SVMs) might work better.
        '''
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
        from sklearn.model_selection import train_test_split

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        # Select the columns to be imputed
        target_columns = columns_to_impute

        # Preprocess the data
        categorical_columns = ["Residence Region", "Region", "Family Typology", "Full_Part_time",
                               "Number Family Members", "Age Classes", "Education Degree", "Employment status",
                               "Job type", "Hours Worked"]
        binary_columns = ["Gender"]
        continuous_columns = ["Work Hours", "Departure Hour for work/study",
                              "Departure Minute for work/study", ]

        # One-hot encoding
        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(df[categorical_columns])
        one_hot_encoded = ohe.transform(df[categorical_columns])

        # Scaling continuous features
        scaler = StandardScaler()
        scaler.fit(df[continuous_columns])
        scaled = scaler.transform(df[continuous_columns])

        # Encode target columns
        #ohe2 = OneHotEncoder(sparse_output=False)
        #target_encoded = ohe2.fit_transform(df[target_columns].dropna())
        mlb = MultiLabelBinarizer()
        target_encoded = mlb.fit_transform(df[target_columns].dropna().values.tolist())

        # Combine all preprocessed features
        X = np.hstack((one_hot_encoded, df[binary_columns].values, scaled))
        #X = np.hstack((one_hot_encoded,))
        #X = np.hstack((one_hot_encoded, scaled))

        # Create a mask for rows with missing values in ordinal columns
        mask = df[target_columns[0]].isna()

        # Split data into complete and incomplete subsets
        X_complete = X[~mask]
        X_incomplete = X[mask]

        # print("X_complete:", X_complete.shape)
        # print("X_incomplete:",X_incomplete.shape)
        y_complete = target_encoded
        # print("y_complete:", y_complete.shape)

        # Define neural network architecture

        # Train the neural network
        input_size = X_complete.shape[1]
        hidden_size = 40
        output_size = target_encoded.shape[1]

        # model parameters
        model = ImputeNN(input_size, hidden_size, output_size)
        #model = ImputeCNN(input_size, hidden_size, output_size)

        criterion = nn.CrossEntropyLoss() #suitable for softmax, multi-label classification
        optimizer = optim.Adam(model.parameters(), lr=0.01,)

        # initialization for nn
        #def weights_init(m):
        #    if isinstance(m, nn.Linear):
        #        nn.init.zeros_(m.weight)
        #        nn.init.zeros_(m.bias)

        #def weights_init(m):
        #    if isinstance(m, nn.Linear):
        #        nn.init.xavier_uniform_(m.weight)
        #        if m.bias is not None:
        #            nn.init.zeros_(m.bias)

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # initialization for cnn
        #def weights_init(m):
        #    if isinstance(m, (nn.Conv1d, nn.Linear)):
        #        nn.init.kaiming_normal_(m.weight)
        #        if m.bias is not None:
        #            nn.init.zeros_(m.bias)

        model.apply(weights_init)

        X_train, X_val, y_train, y_val = train_test_split(X_complete, y_complete,
                                                          test_size=0.3, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        num_epochs = 50
        batch_size = 128
        patience = 5  # Define how many epochs you want to wait before stopping training when validation loss is not decreasing
        best_val_loss = np.inf
        epochs_no_improve = 0

        # training loop
        for epoch in range(num_epochs):
            permutation = torch.randperm(X_train.size()[0])

            for i in range(0, X_train.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                outputs = model(batch_x)
                batch_y_class_indices = batch_y.argmax(dim=1)
                loss = criterion(outputs, batch_y_class_indices)

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()

            val_permutation = torch.randperm(X_val.size()[0])

            # Validation
            val_losses = []
            val_accuracy = []
            val_precision = []
            val_recall = []
            val_f1_score = []
            for i in range(0, X_val.size()[0], batch_size):
                indices = val_permutation[i:i + batch_size]
                batch_x_val, batch_y_val = X_val[indices], y_val[indices]
                val_outputs = model(batch_x_val)
                batch_y_val_class_indices = batch_y_val.argmax(dim=1)
                val_loss = criterion(val_outputs, batch_y_val_class_indices)
                val_losses.append(val_loss.item())

                # Convert model output to discrete predictions
                _, predictions = torch.max(val_outputs, 1)
                predictions = predictions.cpu().numpy()
                true_labels = batch_y_val_class_indices.cpu().numpy()

                # Calculate metrics
                val_accuracy.append(accuracy_score(true_labels, predictions))
                val_precision.append(precision_score(true_labels, predictions, average='micro'))
                val_recall.append(recall_score(true_labels, predictions, average='micro'))
                val_f1_score.append(f1_score(true_labels, predictions, average='micro'))

            avg_val_loss = np.mean(val_losses)
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {avg_val_loss}")

            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracy)
            avg_val_precision = np.mean(val_precision)
            avg_val_recall = np.mean(val_recall)
            avg_val_f1_score = np.mean(val_f1_score)

            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {avg_val_loss}")
            print(
                f"Val Accuracy: {avg_val_accuracy}, Val Precision: {avg_val_precision}, "
                f"Val Recall: {avg_val_recall}, Val F1-Score: {avg_val_f1_score}")

            # early stopping
            # https://www.educative.io/answers/what-is-early-stopping#
            if avg_val_loss < best_val_loss: # if the validation loss improved
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    break  # Exit loop

    def NN_classify_softmax_pytorch(input_path, columns_to_impute, errorPrint=False, to_csv=False, out_csv=None, visualize_NN=False):
        ''' evaluate two ordinal column as categorical, thus softmax is applied
        Preprocessing:
            a. One-hot encode categorical features.
            b. Normalize continuous features using standard scaling.
            c. Use an appropriate ordinal encoding for the two ordinal columns (HouseArea and RoomCount).
            Split the data:
            Separate the dataset into two subsets: one with complete data (27k instances) and the other with missing values in the ordinal columns (40k instances).
            Train a neural network:
            a. Use the 27k instances with complete data to train the neural network.
            b. Design a network architecture that accepts the preprocessed input features and predicts the two ordinal columns (HouseArea and RoomCount).
            c. Use a suitable loss function, such as mean squared error, for ordinal columns.
            d. Train the network with an appropriate optimizer, such as Adam, and tune hyperparameters (e.g., learning rate, batch size, number of layers, and number of neurons per layer).
            Impute missing values:
            a. Use the trained neural network to predict missing values in the ordinal columns for the 40k instances.
            b. Replace missing values with the predicted values.
            Postprocessing:
            Decode the ordinal columns back to their original format, if necessary.
            Evaluate performance:
            Assess the imputation performance using appropriate metrics, such as root mean squared error (RMSE) or mean absolute error (MAE), and compare with your previous imputation methods.
        '''

        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
        from sklearn.model_selection import train_test_split

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        #print(df.shape)

            # Select the columns to be imputed
        target_cols = columns_to_impute

        # Preprocess the data
        categorical_columns = ["Residence Region", "Residence GeoDistribution", "Family Type", "Full/Part_time",
                            "People Count", "Age in Classes", "Highest Education", "Employment Status",
                            "Professional Position", "Hours Worked"]
        binary_columns = ["Gender"]
        continuous_columns = ["Work Hours at 2001", "Departure Hour for work/study",
                           "Departure Minute for work/study", "Commute Duration for work/study"]

        # One-hot encoding
        ohe = OneHotEncoder(sparse_output=False)
        ohe.fit(df[categorical_columns])
        one_hot_encoded = ohe.transform(df[categorical_columns])

        # Scaling continuous features
        scaler = StandardScaler()
        scaler.fit(df[continuous_columns])
        scaled = scaler.transform(df[continuous_columns])

        # Encode ordinal columns as categorical indices
        ohe2 = OneHotEncoder(sparse_output=False)
        target_cols_encoded = ohe2.fit_transform(df[target_cols].dropna()).astype(np.int64)

        # Get the number of unique categories for each ordinal column
        output_sizes = [len(np.unique(target_cols_encoded[:, 0])), len(np.unique(target_cols_encoded[:, 1]))]
        #print("output_sizes:", output_sizes)

        # Combine all preprocessed features
        X = np.hstack((one_hot_encoded, df[binary_columns].values, scaled))

        # Create a mask for rows with missing values in ordinal columns
        mask = df[target_cols[0]].isna()

        # Split data into complete and incomplete subsets
        X_complete = X[~mask]
        X_incomplete = X[mask]

        #print("X_complete:", X_complete.shape)
        #print("X_incomplete:",X_incomplete.shape)

        y_complete = target_cols_encoded
        #print("y_complete:", y_complete.shape)

        #Define neural network architecture
        class ImputeNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_sizes):
                super(ImputeNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization
                #self.dropout = nn.Dropout(0.1)  # Dropout

                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.bn2 = nn.BatchNorm1d(hidden_size)  # Batch normalization
                #self.dropout = nn.Dropout(0.1)  # Dropout

                self.fc3 = nn.Linear(hidden_size, output_sizes[0])
                self.fc4 = nn.Linear(hidden_size, output_sizes[1])

            def forward(self, x):
                x = self.fc1(x)
                x = self.bn1(x)
                x = torch.relu(x)
                #x = torch.nn.functional.leaky_relu(x)
                #x= torch.nn.functional.sigmoid(x)
                #x = self.dropout(x)

                x = self.fc2(x)
                x = self.bn2(x)
                x = torch.relu(x)
                #x = torch.nn.functional.leaky_relu(x)
                #x= torch.nn.functional.sigmoid(x)
                #x = self.dropout(x)

                x1 = torch.softmax(self.fc3(x), dim=1)
                x2 = torch.softmax(self.fc4(x), dim=1)
                return torch.cat((x1, x2), dim=1)  # concatenates two tensors

        #Train the neural network
        input_size = X_complete.shape[1]
        hidden_size = 64
        output_size = sum(output_sizes)

        model = ImputeNN(input_size, hidden_size, output_sizes)
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(model.parameters(), lr=0.002)

        X_train, X_val, y_train, y_val = train_test_split(X_complete, y_complete, test_size=0.1, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        if visualize_NN ==True:
            from torchviz import make_dot
            dot = make_dot(model(X_train).mean(), params=dict(model.named_parameters()), show_attrs=False, show_saved=True)
            dot.render("architecture")  # This will save a .pdf file with the graph

        num_epochs = 100
        batch_size = 256

        for epoch in range(num_epochs):
            permutation = torch.randperm(X_train.size()[0])

            for i in range(0, X_train.size()[0], batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                outputs = model(batch_x)
                loss1 = criterion(outputs[:, :output_sizes[0]], batch_y[:, 0])
                loss2 = criterion(outputs[:, output_sizes[0]:], batch_y[:, 1])
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

            # Validation 1
            val_outputs = model(X_val)
            val_outputs1 = val_outputs[:, :output_sizes[0]]
            val_outputs2 = val_outputs[:, output_sizes[0]:]

            # Get the predicted categories
            y_val_imputed1 = torch.argmax(val_outputs1, dim=1)
            y_val_imputed2 = torch.argmax(val_outputs2, dim=1)

            val_loss1 = criterion(val_outputs1, y_val[:, 0])
            val_loss2 = criterion(val_outputs2, y_val[:, 1])
            val_loss = val_loss1 + val_loss2
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}")

            # Import the necessary functions
            from sklearn.metrics import precision_recall_fscore_support

        # Convert PyTorch tensor to NumPy array for sklearn metrics
        y_val_np = y_val.numpy()
        y_val_imputed1_np = y_val_imputed1.numpy()
        y_val_imputed2_np = y_val_imputed2.numpy()

        # Calculate precision, recall, and F1-score for RoomCount
        precision1, recall1, f1_score1, _ = precision_recall_fscore_support(y_val_np[:, 0], y_val_imputed1_np,
                                                                            average='weighted', zero_division=False)

        print("Precision1 for RoomCount:", precision1)
        print("Recall1 for RoomCount:", recall1)
        print("F1-Score1 for RoomCount:", f1_score1)

        # Calculate precision, recall, and F1-score for HouseArea
        precision2, recall2, f1_score2, _ = precision_recall_fscore_support(y_val_np[:, 1], y_val_imputed2_np,
                                                                            average='weighted',zero_division=False)

        print("Precision2 for HouseArea:", precision2)
        print("Recall2 for HouseArea:", recall2)
        print("F1-Scor2 for HouseArea:", f1_score2)

    def RFC_classify(input_path, columns_to_impute, errorPrint=False, fraction=1,
                     to_csv=False, out_csv=None,
                     single_training=False,
                     gridSearch_training=False,
                     trial_mode=False,
                     ):
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
        from sklearn.compose import ColumnTransformer

        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        elif input_path.endswith(".ftr"):
            df = pd.read_feather(input_path)

        df = df.sample(frac=fraction, replace=True, random_state=1)

        # Assuming your dataframe is called df and your target column is 'House Area'
        # Encode 'House Area' column as it is categorical
        le = LabelEncoder()
        #df['House Area'] = le.fit_transform(df['House Area'].astype(str))
        df['House Area'] = le.fit_transform(df['House Area'])

        # Define which columns should be encoded vs scaled
        feature_columns = ['Administrative Region', 'Region', 'Number Family Members', 'Family Typology',
                           'Gender', 'Marital Status', 'Employment status', 'Full_Part_time',
                           'Job type', 'Age Classes', 'Education Degree', 'Work Hours',
                           'Departure Hour for work/study', 'Departure Minute for work/study']

        # Define which columns should be encoded vs scaled
        columns_to_encode = ['Administrative Region', 'Region', 'Number Family Members',
                             'Family Typology', 'Gender', 'Marital Status', 'Employment status', 'Full_Part_time',
                             'Job type', 'Age Classes', 'Education Degree']
        columns_to_scale = ['Work Hours']
        columns_to_minmax_scale = ['Departure Hour for work/study', 'Departure Minute for work/study']

        # Instantiate encoder/scaler
        scaler = StandardScaler()
        ohe = OneHotEncoder(drop='first')
        minmax_scaler = MinMaxScaler((-1, 1))

        # Build transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', ohe, columns_to_encode),
                ('scale', scaler, columns_to_scale),
                ('minmax_scale', minmax_scaler, columns_to_minmax_scale)
            ],
            remainder='passthrough'
        )

        # Split the data into features (X) and target (y)
        X = df[feature_columns]
        y = df['House Area']

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing on training data
        X_train = preprocessor.fit_transform(X_train)

        # Preprocessing on testing data
        X_val = preprocessor.transform(X_val)

        # Create a Random Forest Classifier model
        # Define the parameter grid to search
        params_RFC = {
                    'n_estimators': 300,
                    'max_features': 'sqrt',
                    'max_depth': None,
                    'min_samples_split': 1,
                    'min_samples_leaf': 1,
                    'bootstrap': True,
                    'criterion': 'gini',
                    'n_jobs': -1
                    }

        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
            AdaBoostClassifier, GradientBoostingClassifier,HistGradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        from sklearn.svm import SVC

        if single_training == True:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            # model = RandomForestClassifier(**params_RFC)
            #model =  KNeighborsClassifier() #0.59
            #model = DecisionTreeClassifier() #0.06
            #model = ExtraTreesClassifier() #0.60
            #model = AdaBoostClassifier() #0.60
            # model = HistGradientBoostingClassifier() #0.60
            #model = SGDClassifier()  # 0.60
            #model = MLPClassifier() #0.59
            #model = GaussianNB() #0.59

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions on the validation set
            y_pred = model.predict(X_val)

            if errorPrint:
                # Print the accuracy of the model
                print("Accuracy score:", accuracy_score(y_val, y_pred))

        if trial_mode == True:
            models = {
                'RandomForest': RandomForestClassifier(),
                'ExtraTrees': ExtraTreesClassifier(),
                'HistGradientBoosting': HistGradientBoostingClassifier(),
                'SGD': SGDClassifier(),
                'MLP': MLPClassifier(),
                'SVM': SVC(),
            }

            # Define the number of folds for cross-validation
            num_folds = 3
            import scipy.sparse as sp
            from sklearn.model_selection import cross_val_score
            for name, model in models.items():
                try:  # Add a try-except block to handle potential errors
                    # If the data is sparse and the model can't handle sparse data, convert it to dense
                    if sp.issparse(X_train) and name in ['HistGradientBoosting', 'GaussianNB']:
                        X_train_dense = X_train.toarray()
                        scores = cross_val_score(model, X_train_dense, y_train, cv=num_folds)
                    else:
                        scores = cross_val_score(model, X_train, y_train, cv=num_folds)

                    # Print the average accuracy across all folds
                    print(f"Classifier: {name}, Cross-validation Accuracy: {np.mean(scores)}")
                except Exception as e:
                    print(f"Error with classifier {name}: {e}")

        if gridSearch_training == True:
            from sklearn.model_selection import GridSearchCV

            # Define the parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }

            # Create a GridSearchCV object
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

            # Perform Grid Search by fitting the model with the data
            grid_search.fit(X_train, y_train)

            if errorPrint:
                # Print the best parameters and the best score
                print("Best Parameters: ", grid_search.best_params_)
                print("Best Score: ", grid_search.best_score_)

        # Print the confusion matrix
        #print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

        # Print the classification report for each class
        #print("Classification Report:\n", classification_report(y_val, y_pred))

        #from sklearn.model_selection import cross_val_score

        # Perform cross-validation
        #cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        #print("Cross-validation scores: ", cv_scores)
        #print("Average cross-validation score: ", np.mean(cv_scores))
class imbalance():
    pass

    def balance_columns(df, columns_to_balance, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = []

        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import LabelEncoder

        # Remove the columns to exclude from the dataset
        df_no_excluded = df.drop(exclude_columns, axis=1)

        balanced_df = df_no_excluded.copy()

        for column in columns_to_balance:
            # Extract the imbalanced column and its corresponding data
            X = df_no_excluded.drop(column, axis=1)
            y = df_no_excluded[column]

            # Encode the target column if it's not numerical
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)

            # Calculate the number of neighbors to use with SMOTE
            min_class_count = min(pd.Series(y_encoded).value_counts())
            k_neighbors = max(1, min(min_class_count - 1, 5))

            # Apply SMOTE with the adjusted number of neighbors
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

            # Decode the target column back to its original form
            y_resampled = encoder.inverse_transform(y_resampled)

            # Combine resampled features and target column
            balanced_data = pd.concat(
                [pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=column)], axis=1)

            # Replace the imbalanced column in the balanced dataset with the resampled data
            balanced_df[column] = balanced_data[column]

        # Add the excluded columns back into the dataset
        balanced_df[exclude_columns] = df[exclude_columns]

        return balanced_df


