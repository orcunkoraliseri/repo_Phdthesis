import TUS_Functions

def reading(txt_file, csv_file, name_txt, naming=False):
    import pandas as pd
    import numpy as np

    #READING
    # Read the dataset_TUS_daily from .txt file
    df = pd.read_csv(txt_file, sep="\t", dtype=str)  # Assuming tab-separated values in the .txt file

    #CONVERTING DATA TYPES
    # Replace empty strings with '999'
    df = df.applymap(lambda x: '999' if x.strip() == '' else x)

    # Convert columns to float64
    for column in df.columns:
        df[column] = df[column].astype('float64')

    # Replace 999 with NaN
    df = df.replace(999, np.nan)
    df = df.replace(99, np.nan)

    if naming == True:
        #new column names of dataframe
        with open(name_txt, 'r') as f:
            new_names = [line.strip() for line in f.readlines()]
        my_dict = dict(zip(df.columns, new_names))
        df = df.rename(columns=my_dict)
    else:
        pass

    df.drop_duplicates(keep='first', inplace=True)

    df = df[["Household_ID",
             "Occupant_ID_in_HH",
             "Number Family Members",
             "Gender",
             "Marital Status",
             "Education Degree",
             "Nuclear Family, Occupant Profile",
             "Family Typology",
             "Region",
             "Age Classes",
             "Employment status",
             "Main Income Source",
             "School Enrollment",
             "Job Type",
             "Full_Part_time",
             "Home Type",
             "Room Count",
             "Economic Resources",
             "Kinship Relationship",
             "Nuclear Family, Typology",
             "Nuclear Family, Occupant Sequence Number",
             "Citizenship",  # new additions
             "Pet Ownership",
             "Computer Ownership",
             "Internet Access",
             "Landline Ownership",
             "Mobile Phone Ownership",
             "Bicycle Ownership",
             "Moped/Scooter Ownership",
             "Motorcycle Ownership",
             "Car Ownership",
             "Economic Sector, Profession",
             'Home Ownership',
             "Permanent/fixed"
             ]]

    # Save the DataFrame to a .csv file
    df.to_csv(csv_file, index=False)
    print('reading: writing as .csv is done')

def imputation_manual(input_path, output_path_csv=None, missingness=False):
    import pandas as pd
    from preProcessing_Func import functions_general as dfppfg
    from TUS_Functions import re_arrangeColsOCCschedule_moreRepresentation
    df = pd.read_csv(input_path)
    df = df[["Household_ID",
             "Occupant_ID_in_HH",
             "Number Family Members",
             "Gender",
             "Marital Status",
             "Education Degree",
             "Nuclear Family, Occupant Profile",
             "Family Typology",
             "Region",
             "Age Classes",
             "Employment status",
             "Main Income Source",
             "School Enrollment",
             "Job Type",
             "Full_Part_time",
             "Home Type",
             "Room Count",
             "Economic Resources",
             "Kinship Relationship",
             "Nuclear Family, Typology",
             "Nuclear Family, Occupant Sequence Number",
             "Citizenship",  # new additions
             "Pet Ownership",
             "Computer Ownership",
             "Internet Access",
             "Landline Ownership",
             "Mobile Phone Ownership",
             "Bicycle Ownership",
             "Moped/Scooter Ownership",
             "Motorcycle Ownership",
             "Car Ownership",
             "Economic Sector, Profession",
             'Home Ownership',
             "Permanent/fixed"
             ]]

    from TUS_Functions import imput_man_bruteForce
    df = imput_man_bruteForce(df)
    df = df.dropna()
    df = df.applymap(dfppfg.convert_if_integer)
    df = df.applymap(dfppfg.remove_trailing_zero)

    # Re-arrange column values for alignment with census datasets
    df = re_arrangeColsOCCschedule_moreRepresentation(df)
    def map_column_values(df, column_name, mapping_dict, new_column_name=None):
        if new_column_name:
            # Map values and create a new column
            df[new_column_name] = df[column_name].map(mapping_dict)
        else:
            # Map values and update the existing column
            df[column_name] = df[column_name].map(mapping_dict)
        return df
    # Usage for 'Job Type'
    mapping_home_jobType = {1: 6, 2:7, 3:7, 4:1, 5:2, 6:3, 7:5, 8:4, 0:0}
    df['Job Type'] = df['Job Type'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Job Type',mapping_dict=mapping_home_jobType, new_column_name='Job Type')

    mapping_empStat = {1: 1, 2: 2, 4: 3, 5: 4, 7: 5, 8: 6, 0: 7}
    df['Employment status'] = df['Employment status'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Employment status',mapping_dict=mapping_empStat, new_column_name='Employment status')

    mapping_eduDegree = {1: 1, 2: 2, 3: 3, 4: 4, 5: 4}
    df["Education Degree"] = df["Education Degree"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df, column_name="Education Degree", mapping_dict=mapping_eduDegree, new_column_name="Education Degree")

    # Usage for 'Family Typology', updated
    mapping_family_typology = {1: 1, 2: 3, 3: 1, 4: 1, 5: 1, 6: 5, 7: 2, 8: 6, 9: 2, 10: 8, 11: 8, 14: 8, 15: 3, 16: 7,
                               19: 7, 20: 5, 21: 2, 22: 6, 23: 2, 24: 8, 25: 8, 28: 8, 29: 7, 30: 7, 33: 7, 34: 13,
                               35: 13, 36: 13, 37: 13, 38: 13, 39: 13, 40: 13}

    df['Family Typology'] = df['Family Typology'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Family Typology',mapping_dict=mapping_family_typology,new_column_name='Family_Typology_Simple')

    mapping_ageClasses = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8}
    df['Age Classes'] = df['Age Classes'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Age Classes',mapping_dict=mapping_ageClasses, new_column_name='Age Classes')

    # Usage for 'Home Ownership'
    mapping_home_ownership = {1: 2, 2: 1, 3: 1, 4: 3, 5: 3}
    df['Home Ownership'] = df['Home Ownership'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Home Ownership',mapping_dict=mapping_home_ownership)

    # Usage for "Full_Part_time"
    mapping_Permanentfixed = {1: 2, 2:1, 0:0}
    df["Permanent/fixed"] = df["Permanent/fixed"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name="Permanent/fixed",mapping_dict=mapping_Permanentfixed)

    # Usage for "Full_Part_time"
    mapping_Internet = {3: 1, 4:2}
    df['Internet Access'] = df['Internet Access'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df, column_name='Internet Access',mapping_dict=mapping_Internet)

    mapping_Car = {3: 2, 4:1} # 1: Si, 2: No
    df["Car Ownership"] = df["Car Ownership"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name="Car Ownership",mapping_dict=mapping_Car)

    # Usage for "Full_Part_time"
    mapping_Mobile = {3: 2, 4:1}
    df["Mobile Phone Ownership"] = df["Mobile Phone Ownership"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name="Mobile Phone Ownership",mapping_dict=mapping_Mobile)

    # Usage for "Full_Part_time"
    mapping_LandLine = {1: 2, 2:1}
    df["Landline Ownership"] = df["Landline Ownership"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name="Landline Ownership",mapping_dict=mapping_LandLine)

    # Usage for  "Computer Ownership"
    mapping_Comp = {1: 2, 2:1}
    df["Computer Ownership"] = df["Computer Ownership"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name="Computer Ownership",mapping_dict=mapping_Comp)

    # Usage for  "Computer Ownership"
    mapping_Citizen = {1: 1, 3:2}
    df["Citizenship"] = df["Citizenship"].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name="Citizenship",mapping_dict=mapping_Citizen, new_column_name='Citizenship')

    mapping_kinship = {1: 1, 2: 2, 4: 3, 5: 3, 6: 5, 8: 6, 10: 7, 12: 8, 14: 9, 16: 10}
    df['Kinship Relationship'] = df['Kinship Relationship'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Kinship Relationship',mapping_dict=mapping_kinship, new_column_name='Kinship Relationship')

    mapping_ecoSector = {1: 1, 2: 2, 4: 3, 5: 4, 0: 0}
    df['Economic Sector, Profession'] = df['Economic Sector, Profession'].astype(int)  # Ensure correct data type
    df = map_column_values(df=df,column_name='Economic Sector, Profession',
                           mapping_dict=mapping_ecoSector, new_column_name='Economic Sector, Profession')

    # set display options to print all columns without truncating
    pd.set_option('display.max_columns', None)

    if missingness == True:
        print('Missingness imput manual before:')
        from TUS_Functions import missing
        missing(df)
        print(" ")

    #OUTPUT
    df.to_csv(output_path_csv, index=False)
    print('imput_manual: writing as .csv is done')

def finalStep_moreRep_moreColumns(input_path, output_path_csv=None):
    import pandas as pd
    # Load and preprocess the dataset
    df = pd.read_csv(input_path)

    # Update column order
    new_column_order = [
        "Household_ID", "Occupant_ID_in_HH", "Region", "Number Family Members", "Family Typology", "Family_Typology_Simple",
        "Age Classes", "Employment status", "Gender", "Education Degree", "Marital Status", "Main Income Source",
        "School Enrollment", "Home Type", "Economic Resources", "Kinship Relationship", "Nuclear Family, Occupant Profile",
        "Nuclear Family, Typology", "Nuclear Family, Occupant Sequence Number", "Citizenship", "Pet Ownership",
        "Computer Ownership", "Internet Access", "Landline Ownership", "Mobile Phone Ownership", "Bicycle Ownership",
        "Moped/Scooter Ownership", "Motorcycle Ownership", "Car Ownership","Job Type","Economic Sector, Profession",
        'Home Ownership', "Room Count","Full_Part_time", "Permanent/fixed"]
    df = df.reindex(columns=new_column_order)

    # Output to CSV
    if output_path_csv:
        df.to_csv(output_path_csv, index=False)
        print('final step: writing as .csv is done')

    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from preProcessing_Func import analysis_func as dfppaf

    # INPUT PATHS
    txt_input = r'dataset_TUS_individual/uso_tempo_Microdati_Anno_2013_Individui.txt'
    name_path=r'dataset_TUS_individual/TUS-Data - TUS_individual_naming.txt'

    # OUTPUT PATHS
    tus_subset_reading = r'dataset_TUS_individual/TUS_indiv_01_reading.csv'
    tus_subset_imputMan = r'dataset_TUS_individual/TUS_indiv_02_imputMan.csv'
    tus_subset_final = r'dataset_TUS_individual/TUS_indiv_03_final.csv'
    tus_subset_final_moreCol = r'dataset_TUS_individual/TUS_indiv_03_final_moreCol.csv'

    # FUNCTIONS
    #reading(txt_file=txt_input, csv_file=tus_subset_reading, name_txt=name_path, naming=True) #original name headers
    #imputation_manual(input_path=tus_subset_reading, output_path_csv=tus_subset_imputMan, missingness=False ,)
    #finalStep_moreRep_moreColumns(input_path=tus_subset_imputMan, output_path_csv=tus_subset_final_moreCol)

    #VISUALISATION & ANALYSIS
    input = tus_subset_final_moreCol
    non_visual = True
    visual = True
    ID_DROP = ["Household_ID"]

    from preProcessing_Func import analysis_func as dfppaf
    #dfppaf.analysis(input_path=input, describe=non_visual)
    dfppaf.analysis(input_path=input, data_len=non_visual)
    dfppaf.analysis(input_path=input, count_unique_values=non_visual, uniqueIDcolstoDrop=ID_DROP)
    dfppaf.analysis(input_path=input, data_types=non_visual)
    #dfppaf.analysis(input_path=input, columns=non_visual)
    dfppaf.analysis(input_path=input, missingness=non_visual)
    dfppaf.analysis(input_path=input, unique=non_visual, uniqueIDcolstoDrop=ID_DROP)
    print(" ")
    #dfppaf.analysis(input_path=input, unique_visual_byCols=visual, uniqueIDcolstoDrop=ID_DROP)
    #dfppaf.analysis(input_path=input, missingness_visual_oriented=visual)
    #dfppaf.analysis(input_path=input, fraction=1, unique_visual=visual, uniqueIDcolstoDrop=ID_DROP)
    #dfppaf.analysis(input_path=input, missingness_rowbased=visual)
    dfppaf.analysis(input_path=input, multiple_hist=visual, dropID_multiple_hist=ID_DROP)
    #dfppaf.analysis(input_path=input, missingness_visual_oriented=visual, missingness_visual_oriented_title="tusOccupant")
    #dfppaf.analysis(input_path=input, fraction=1, unique_visual=visual, uniqueIDcolstoDrop=ID_DROP)