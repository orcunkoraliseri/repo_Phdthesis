import pandas as pd
from preProcessing_Func import census_housing as dfppCH
from preProcessing_Func import dfPrePIndividu as dfppInd
# ignore future warnings
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
def selection_byCol(input_path, output_path_csv=None):
    df = pd.read_csv(input_path)

    row_count = df.shape[0]
    print("Row count before:", row_count)

    df = df[df['Res_ID'] == 6486202]
    df.drop_duplicates(inplace=True)
    print(df)

    #OUTPUT
    df.to_csv(output_path_csv, index=None)
##################################################################################
def reading(input_path, output_path_csv, to_csv=False):
    '''
    read file & convert from .txt to .csv
    https://stackoverflow.com/questions/13250046/how-to-keep-leading-zeros-in-a-column-when-reading-csv-with-pandas
    '''

    df = pd.read_csv(input_path,  sep='\t', )
    #print(df.head(5))
    #print(df.columns)

    if to_csv==True:
        df.to_csv(output_path_csv, index=None)
        print('reading function: writing as .csv is done')
def editing(input_path, output_path_csv=False):
    #INPUT
    df = pd.read_csv(input_path)

    new_names = ['Residential_ID', 'Home Ownership', 'Homeowner', 'ResidentialType', 'Room Count', 'Professional Rooms',
               'separate Kitchen', 'Kitchen small_existence', 'Kitchen corner_existence', 'No Kitchen', 'numFloors',
               'House Area', 'Aqueduct Water', 'Well Water', 'Alternative Water', 'Non-potable Water',
               'No Indoor Water', 'Shower/Bathtub Count', 'Toilet Count', 'Hot Water Availability',
               'Heating-Hot Water System', 'Methane Heating', 'Electric Heating', 'Solar Heating', 'Other Heating',
               'Centralized System', 'Independent System', 'Fixed Appliances (Whole)', 'Fixed Appliances (Partial)',
               'Methane Home Heating', 'Diesel Home Heating', 'LPG Home Heating', 'Solid Fuel', 'Electric Home Heating',
               'Oil Home Heating', 'Other Home Heating', 'Renewable Energy', 'Air Conditioning', 'Private Parking',
               'Car Ownership', 'Fixed Telephone Line', 'Mobile Phone Ownership', 'Family Mobile Phones', 'Internet Access',
               'Landline Ownership', 'ADSL', 'Other Broadband', 'Internet Key/Mobile']

    my_dict = dict(zip(df, new_names))
    df = df.rename(columns=my_dict)

    #3RD STEP: from raw to readable dataframe
    for i in df.columns:
        df = dfppInd.editCol(df, i, rename=i)

    #df.drop_duplicates(keep='first', inplace=True)

    df = df[['Residential_ID', 'ResidentialType', 'House Area', 'Room Count', 'numFloors', 'Internet Access',
             'Landline Ownership', 'Mobile Phone Ownership', 'Car Ownership', 'Home Ownership']]
    df.to_csv(output_path_csv, index=None)
def imputation_manual(input_path, output_path_csv=None, ):
    #INPUT
    df = pd.read_csv(input_path) #INPUT

    df = df[['Residential_ID', 'ResidentialType', 'House Area', 'Room Count', 'numFloors', 'Internet Access',
             'Landline Ownership', 'Mobile Phone Ownership', 'Car Ownership', 'Home Ownership']]

    #selecting residential units
    df = dfppCH.filterRes(df)
    #df = dfppCH.identifier(df)
    #df = dfppCH.kitchen(df)
    #df = dfppCH.acqua(df)
    #df = dfppCH.internetCon(df)
    #df = dfppCH.mobilePhone(df)
    #df = dfppCH.auto(df)
    #df = dfppCH.DHW(df)
    #df = dfppCH.heatingSystemSource(df)
    #df = dfppCH.coolingSystem(df)
    #df = dfppCH.heatingSystem(df)
    #df = dfppCH.profUse(df)
    def map_column_values(df, column_name, mapping_dict, new_column_name=None):
        if new_column_name:
            # Map values and create a new column
            df[new_column_name] = df[column_name].map(mapping_dict)
        else:
            # Map values and update the existing column
            df[column_name] = df[column_name].map(mapping_dict)
        return df

    mapping_Car = {1: 1, 2:1, 3:2} # 1: Si, 2: No
    df["Car Ownership"] = df["Car Ownership"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name="Car Ownership",mapping_dict=mapping_Car)

    mapping_roomCount = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15,
                       16: 15, 17: 15, 18: 15, 19: 15, 20: 15, 21: 15, 22: 15, 23: 15, 24: 15, 27: 15, 28: 15, 30: 15,
                       31: 15, 33: 15, 34: 15, 35: 15, 39: 15, 40: 15, 41: 15, 42: 15, 43: 15, 45: 15, 50: 15, 51: 15,
                       60: 15, 61: 15, 63: 15, 71: 15, 74: 15, 83: 15, 92: 15, 93: 15, 94: 15}
    df['Room Count'] = df['Room Count'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Room Count',mapping_dict=mapping_roomCount, new_column_name='Room Count')

    # set display options to print all columns without truncating
    pd.set_option('display.max_columns', None)

    df.to_csv(output_path_csv, index=None)
def feature_importance(input_path, print_less_effective=False): #COLUMNS TO DROP because they are not directly related with daily schedules of members of household

    from preProcessing_Func import stats_visuals as dfppsv
    import pandas as pd

    #INPUT
    df = pd.read_feather(input_path)
    print(df.columns)

    # pca analysis for feature importance
    dfppsv.pca_analysis(df, dropNAN=True, print_less_effective=print_less_effective,
                        dropIrrelevant=['Residential_ID']
                        )
    ''' 
    'company' column does not have strong impact, thus, it is should be eliminated 
    from classification. however, company column have missing rows so it should be included
    into classification for imputation.
    '''
def finalStep(input_path, output_path_csv=None):
    import pandas as pd

    # Read the input CSV file
    df = pd.read_csv(input_path)

    print(df.columns)

    # Import preprocessing functions
    from preProcessing_Func import functions_general as dfppfg

    # Apply preprocessing functions
    df = df.applymap(dfppfg.convert_if_integer)
    df = df.applymap(dfppfg.remove_trailing_zero)

    # Remove duplicate rows based on all columns
    df.drop_duplicates(keep='first', inplace=True)

    # Filter rows where 'House Area' is 0, 'Room Count' is 0, or 'numFloors' is NaN
    df = df[~((df['House Area'] == 0) | (df['Room Count'] == 0) | (df['numFloors'].isna())| (df['numFloors'] == "nan"))]

    # Select relevant columns
    df = df[['Residential_ID', 'ResidentialType', 'House Area', 'Room Count', 'numFloors', 'Internet Access',
             'Landline Ownership', 'Mobile Phone Ownership', 'Car Ownership', 'Home Ownership']]

    df['Internet Access'] = pd.to_numeric(df['Internet Access'], errors='coerce')
    df['Mobile Phone Ownership'] = pd.to_numeric(df['Mobile Phone Ownership'], errors='coerce')
    df['Landline Ownership'] = pd.to_numeric(df['Landline Ownership'], errors='coerce')
    df = df.dropna()

    # Write the processed DataFrame to a CSV file
    if output_path_csv:
        df.to_csv(output_path_csv, index=None)

    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    from Census_Functions import imputation_auto

    """EXTRA
    imputation_manual(input_path=Census_Housing_editing, output_path_csv=Census_Housing_imputMan)
    imputation_auto(input_path=path_ftr , output_path=path_ftr, output_path_csv=path_csv, kde_imput=True, kde_cols=["Floor Count"], to_feather=True, to_csv=True)
    """

    #INPUT PATHS
    txt_input =r'dataset_CENSUS_housing/CensPop2011_1%_Microdati_Anno_2011_alloggi.txt' # census data, housing
    Census_Housing_reading = r'dataset_CENSUS_housing/Census_Housing_reading.csv'
    Census_Housing_editing = r'dataset_CENSUS_housing/Census_Housing_editing.csv'
    Census_Housing_imputMan = r'dataset_CENSUS_housing/Census_Housing_imputMan.csv'
    Census_Housing_final = r'dataset_CENSUS_housing/Census_Housing_final.csv'

    #reading(input_path=txt_input, output_path_csv=Census_Housing_reading, to_csv=True) # 245535 rows, 48 columns
    #editing(input_path=Census_Housing_reading, output_path_csv= Census_Housing_editing,) # 245535 rows, 48 columns
    #imputation_manual(input_path=Census_Housing_editing, output_path_csv= Census_Housing_imputMan,) # 232856, 10 columns
    #finalStep(input_path=Census_Housing_imputMan, output_path_csv=Census_Housing_final) # 244216 rows, 5 columns

    from preProcessing_Func import analysis_func as dfppaf
    input = Census_Housing_final
    non_visual = True
    visual = False

    #dfppaf.analysis(input_path=input, describe=non_visual)
    dfppaf.analysis(input_path=input, data_len=non_visual)
    dfppaf.analysis(input_path=input, data_types=non_visual)
    dfppaf.analysis(input_path=input, missingness=non_visual)
    dfppaf.analysis(input_path=input, unique=non_visual, uniqueIDcolstoDrop=['Residential_ID'])
    print(" ")
    dfppaf.analysis(input_path=input, missingness_visual_oriented=visual, missingness_visual_oriented_title="Census Housing")
    dfppaf.analysis(input_path=input, fraction=1, unique_visual=visual, uniqueIDcolstoDrop=['Residential_ID'])
    dfppaf.analysis(input_path=input, multiple_hist=visual, dropID_multiple_hist=["Residential_ID"])
    print(" ")
    #from UBEMsim_functions import unit_randomization as unit_rand
    #bins = 8
    #unit_rand.plot_histogram(input_path=input, column="House Area", kde=False, xtick_count=10, bins=bins, print_bin_info=True)