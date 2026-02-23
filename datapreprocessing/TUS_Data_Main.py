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
def merge(df1_path, df2_path, refCol,to_csv=False,out_csv=None, columns=False):
    import pandas as pd

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    df1 = convert_dtyes(df1)
    df1.drop_duplicates(inplace=True)
    df2 = convert_dtyes(df2)
    df2.drop_duplicates(inplace=True)

    # Merge DataFrames using the 'id' column
    df = pd.merge(df1, df2, on=refCol, how='left')

    '''housing and occupant datasets do not completely match each other due to missingness, thus, NaN rows are deleted
    '''
    df = df.dropna()

    #some columns are repeated, it can be detected using 'Accommodation ID' columns
    df.drop_duplicates(inplace=True)

    if columns ==True:
        # Assuming 'df' is your DataFrame
        print("Column count:", len(df.columns))
        # Alternatively, you can use the shape attribute
        row_count = df.shape[0]
        print("Row count:", row_count)
        #for col_name in df.columns:
        #    print(col_name)

    #OUTPUT
    if to_csv==True:
        df.to_csv(out_csv, index=None)

    #print(merged_df.head(500))
if __name__ == '__main__':
    from preProcessing_Func import analysis_func as dfppaf

    # INPUT PATHS: TUS individual
    tus_subset_final = r'dataset_TUS_individual/TUS_indiv_03_final.csv'
    tus_subset_final_data5 = r'dataset_TUS_individual/TUS_indiv_03_final_data5.csv'
    tus_subset_final_moreCol = r'dataset_TUS_individual/TUS_indiv_03_final_moreCol.csv'

    # INPUT PATHS: TUS daily
    tus_daily_final = r'dataset_TUS_daily/TUS_daily_final.csv'
    tus_daily_final_simplified = r'dataset_TUS_daily/TUS_daily_final_simplified.csv'
    tus_daily_final_DATA3 = r'dataset_TUS_daily/tus_daily_final_DATA3.csv'
    tus_daily_final_DATA4 = r'dataset_TUS_daily/tus_daily_final_DATA4.csv'

    # OUTPUT PATHS
    tus_main = r'dataset_TUS_main/TUS_main.csv'
    tus_main_simplified = r'dataset_TUS_main/TUS_main_simplified.csv'
    tus_main_DATA3 = r'dataset_TUS_main/TUS_main_DATA3.csv'
    tus_main_DATA4 = r'dataset_TUS_main/TUS_main_DATA4.csv'
    tus_main_DATA5 = r'dataset_TUS_main/TUS_main_DATA5.csv'
    tus_main_RAWDATA_25 = r'dataset_TUS_main/tus_main_RAWDATA_25.csv'
    tus_main_RAWDATA_24 = r'dataset_TUS_main/tus_main_RAWDATA_24.csv'
    tus_main_RAWDATA_23 = r'dataset_TUS_main/tus_main_RAWDATA_23.csv'
    tus_main_RAWDATA_22 = r'dataset_TUS_main/tus_main_RAWDATA_22.csv'
    tus_main_RAWDATA_31 = r'dataset_TUS_main/tus_main_RAWDATA_31.csv'

    # RAW_DATA
    #merge(df1_path=tus_subset_final, df2_path=tus_daily_final, refCol=["Household_ID", "Occupant_ID_in_HH"], to_csv=True, out_csv=tus_main)
    #merge(df1_path=tus_subset_final_moreCol, df2_path=tus_daily_final, refCol=["Household_ID", "Occupant_ID_in_HH"], to_csv=True, out_csv=tus_main_RAWDATA_22)
    #merge(df1_path=tus_subset_final_moreCol, df2_path=tus_daily_final, refCol=["Household_ID", "Occupant_ID_in_HH"], to_csv=True, out_csv=tus_main_RAWDATA_31)

    #VISUALISATION & ANALYSIS
    input = tus_main_RAWDATA_31
    non_visual = True
    visual = False
    ID_DROP = ["Household_ID",]

    from preProcessing_Func import analysis_func as dfppaf
    #dfppaf.analysis(input_path=input, describe=non_visual)
    dfppaf.analysis(input_path=input, data_len=non_visual)
    #dfppaf.analysis(input_path=input, data_types=non_visual)
    dfppaf.analysis(input_path=input, columns=non_visual)
    dfppaf.analysis(input_path=input, missingness=non_visual)
    dfppaf.analysis(input_path=input, unique=non_visual, uniqueIDcolstoDrop=ID_DROP)
    print(" ")
    dfppaf.analysis(input_path=input, missingness_visual_oriented=visual)
    dfppaf.analysis(input_path=input, fraction=1, unique_visual=visual, uniqueIDcolstoDrop=ID_DROP)
    dfppaf.analysis(input_path=input, multiple_hist=False, dropID_multiple_hist=ID_DROP)
    dfppaf.analysis(input_path=input, missingness_rowbased=visual)
    dfppaf.analysis(input_path=input, multiple_hist=visual, dropID_multiple_hist=ID_DROP)


