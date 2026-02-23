import pandas as pd
from preProcessing_Func import dfPrePIndividu as dfppInd
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
######################################################
def reading(input_path, output_path_csv, to_csv=False):
    '''
    read file & convert from .txt to .csv
    https://stackoverflow.com/questions/13250046/how-to-keep-leading-zeros-in-a-column-when-reading-csv-with-pandas
    '''

    df = pd.read_csv(input_path,  sep='\t', )
    #print(df.head(5))

    if to_csv==True:
        df.to_csv(output_path_csv, index=None)
        print('reading function: writing as .csv is done')
def editing(input_path, output_path_csv=False, fraction=1):
    #INPUT
    df = pd.read_csv(input_path)

    df = df.sample(frac=fraction, replace=True, random_state=1)

    #df['TEMPO_PIENO'] = pd.to_numeric(df['TEMPO_PIENO'], errors='coerce', downcast='float')

    # only column that consist of strings
    df['IND_MODELLO'] = df['IND_MODELLO'].replace({'C': 0, 'L': 1})

    #3RD STEP: from raw to readable dataframe
    for i in df.columns:
        df = dfppInd.editCol(df, i, rename=i)

    df.to_csv(output_path_csv, index=None)
    print('editing function: writing as .csv is done')
def imputation_manual(input_path, output_path_csv=None, fraction=1, naming=False):
    from preProcessing_Func import dfPrePIndividu as dfppInd
    #INPUT
    df = pd.read_csv(input_path)

    #SAMPLING
    df = df.sample(frac=fraction, replace=True, random_state=1)

    # 3RD STEP: from raw to readable dataframe
    df = dfppInd.ETA_CLASSI(df)
    df = dfppInd.TIPOLOGIA_FAM(df)
    df = dfppInd.location(df)
    df = dfppInd.household(df)
    df = dfppInd.eduStats(df)
    df = dfppInd.workSchStats(df)
    df = dfppInd.workStats(df)
    df = dfppInd.habitualRes(df)
    df = dfppInd.unrelatedExtraCols(df)
    df = dfppInd.unemployed_stat(df)
    df = dfppInd.parent_status(df)
    df = dfppInd.filter_occ(df)

    if naming == True:
        with open(name_path, 'r') as f:
            new_names = [line.strip() for line in f.readlines()]
        my_dict = dict(zip(df.columns, new_names))
        df = df.rename(columns=my_dict)
    else:
        pass

    # Usage for 'Family Typology'
    def map_column_values(df, column_name, mapping_dict, new_column_name=None):
        if new_column_name:
            # Map values and create a new column
            df[new_column_name] = df[column_name].map(mapping_dict)
        else:
            # Map values and update the existing column
            df[column_name] = df[column_name].map(mapping_dict)
        return df
    mapping_family_typology = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 5, 10: 6, 11: 7, 12: 8, 13: 13}
    df['Family Typology'] = df['Family Typology'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Family Typology',mapping_dict=mapping_family_typology, new_column_name='Family_Typology_Simple')

    mapping_empStat = {1: 1, 2: 2, 3: 6, 4: 5, 5: 4, 6: 3, 7: 7}
    df['Employment status'] = df['Employment status'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Employment status',mapping_dict=mapping_empStat, new_column_name='Employment status')

    mapping_econSector = {1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4,
            17: 4, 18: 4, 19: 4, 20: 4, 21: 4, 0: 0}
    df['Economic Sector, Profession'] = df['Economic Sector, Profession'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Economic Sector, Profession',mapping_dict=mapping_econSector, new_column_name='Economic Sector, Profession')

    mapping_ageClasses = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6, 12: 7, 13: 7, 14: 8}
    df['Age Classes'] = df['Age Classes'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Age Classes',mapping_dict=mapping_ageClasses, new_column_name='Age Classes')

    mapping_maritalStatus = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 6}
    df['Marital Status'] = df['Marital Status'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Marital Status',mapping_dict=mapping_maritalStatus, new_column_name='Marital Status')

    mapping_kinship = {1: 1, 2: 2, 3: 2, 4: 5, 5: 5, 6: 5, 7: 3, 8: 10, 9: 8, 10: 8, 11: 9, 12: 6, 13: 7, 14: 7, 15: 7, 16: 10, 17: 10}
    df['Kinship Relationship'] = df['Kinship Relationship'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Kinship Relationship',mapping_dict=mapping_kinship, new_column_name='Kinship Relationship')

    df.drop_duplicates(keep='first', inplace=True)
    df.to_csv(output_path_csv, index=None)
    print('imputation_manual: writing as .csv is done')
def finalStep(input_path,output_path_csv=None):
    df = pd.read_csv(input_path)
    #print(df.columns)

    df = df[['Family_ID', 'Occupant_ID_in_HH', 'Residential_ID', 'Region', 'Number Family Members', 'Family Typology',
             'Gender', 'Age Classes', 'Marital Status', 'Education Degree', 'Employment status', "Kinship Relationship",
             "Work Activity Type", "Job Type", 'Family_Typology_Simple', "Citizenship", "Economic Sector, Profession",
             "Full_Part_time", "Permanent/fixed"]]

    from preProcessing_Func import functions_general as dfppfg
    df = df.applymap(dfppfg.convert_if_integer)
    df = df.applymap(dfppfg.remove_trailing_zero)

    # Sort the DataFrame based on column 'column_to_sort'
    # df = df.sort_values(by=column_to_sort)

    # some columns are repeated, it can be detected using 'Accommodation ID' columns
    df.drop_duplicates(keep='first', inplace=True)

    def map_column_values(df, column_name, mapping_dict, new_column_name=None):
        if new_column_name:
            # Map values and create a new column
            df[new_column_name] = df[column_name].map(mapping_dict)
        else:
            # Map values and update the existing column
            df[column_name] = df[column_name].map(mapping_dict)
        return df

    mapping_eduDegree = {1: 4, 2: 3, 3: 4, 4: 2, 5: 2, 6: 1, 7: 1, 0: 4}
    df['Education Degree'] = df['Education Degree'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Education Degree',mapping_dict=mapping_eduDegree, new_column_name='Education Degree')

    df = df.sample(frac=1, replace=True, random_state=1)
    def generate_main_income_source(df):
        # Initialize the new column with NaN or default value
        df['main_income_source'] = None

        # Apply the conditions
        # Condition 1: Employment status (COND_PROF) == 1
        df.loc[df['Employment status'] == 1, 'main_income_source'] = 1

        # Condition 2: Job Type (POS_PROF) == 5
        df.loc[df['Job type'] == 5, 'main_income_source'] = 2

        # Condition 3: Employment status (COND_PROF) == 5
        df.loc[df['Employment status'] == 5, 'main_income_source'] = 3

        # Condition 4: Job Type (POS_PROF) == 9 and Age Classes in (1, 2)
        df.loc[(df['Job type'] == 0) & (df['Age Classes'].isin([1, 2])), 'main_income_source'] = 6

        return df
    # Apply the function to the dataframe
    #df = generate_main_income_source(df)

    # OUTPUT
    df.to_csv(output_path_csv, index=None)

    return df
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from preProcessing_Func import analysis_func as dfppaf

    #INPUT PATH
    txt_input =r'dataset_CENSUS_occupant/CensPop2011_1%_Microdati_Anno_2011_individui.txt' # census data, housing
    name_path = r'dataset_CENSUS_occupant/census_occupant_renaming.txt'

    CensusOcc_read = r'dataset_CENSUS_occupant/CensusOcc_read.csv'
    CensusOcc_edit = r'dataset_CENSUS_occupant/CensusOcc_edit.csv'
    CensusOcc_imput = r'dataset_CENSUS_occupant/CensusOcc_imput.csv'
    CensusOcc_final = r'dataset_CENSUS_occupant/CensusOcc_final.csv'

    reading(input_path=txt_input, output_path_csv=CensusOcc_read, to_csv=True)
    editing(input_path=CensusOcc_read, output_path_csv= CensusOcc_edit, fraction=1)
    imputation_manual(input_path=CensusOcc_edit, output_path_csv=CensusOcc_imput, naming=True)
    finalStep(input_path=CensusOcc_imput, output_path_csv=CensusOcc_final,)

    from preProcessing_Func import analysis_func as dfppaf
    input = CensusOcc_final

    visual = True
    non_visual = True
    ID_DROP = ["Residential_ID", "Family_ID"]
    from preProcessing_Func import analysis_func as dfppaf
    dfppaf.analysis(input_path=input, columns=non_visual)
    #dfppaf.analysis(input_path=input, describe=non_visual)
    dfppaf.analysis(input_path=input, data_len=non_visual)
    dfppaf.analysis(input_path=input, data_types=non_visual)
    dfppaf.analysis(input_path=input, missingness=non_visual)
    dfppaf.analysis(input_path=input, unique=non_visual, uniqueIDcolstoDrop=ID_DROP)
    dfppaf.analysis(input_path=input, count_unique_values=non_visual, uniqueIDcolstoDrop=ID_DROP)

    print(" ")
    dfppaf.analysis(input_path=input, missingness_visual_oriented=visual, missingness_visual_oriented_title="Census Occupant")
    dfppaf.analysis(input_path=input, fraction=1, unique_visual=visual, uniqueIDcolstoDrop=["Residential_ID", "Family_ID"])
    dfppaf.analysis(input_path=input, multiple_hist=visual, dropID_multiple_hist=["Residential_ID", "Family_ID"])
