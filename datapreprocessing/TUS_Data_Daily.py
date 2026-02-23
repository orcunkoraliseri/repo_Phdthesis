'''
for vscode installation
python3 -m pip install <package-name>
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preProcessing_Func import dfPrePIndividu as dfppInd
from preProcessing_Func import imputation as dfppIMP
from preProcessing_Func import stats as dfppss
import seaborn as sns
# ignore future warnings
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3f}'.format # for ID columns
pd.set_option('display.max_columns', 500)

def reading(input_path, output_path_csv=None, naming=False):
    '''
    1ST PART: TUS_DATA.py: read file & convert from .txt to .csv
    read_csv, however preserve 00245 as str(00245)
    https://stackoverflow.com/questions/13250046/how-to-keep-leading-zeros-in-a-column-when-reading-csv-with-pandas
    '''

    import pandas as pd
    from preProcessing_Func import stats as dfppss

    #INPUT
    read_file = pd.read_csv(input_path, sep='\t', converters={'catpri': str})

    #Task: convert some strings to reasonable values before converting to integer
    # resource: https://docs.google.com/spreadsheets/d/1IOjNvJaW7wXtOOXiuE3al79tQbnJeybT5AlRgLkeuPE/edit#gid=368165643
    initialReplacement_dict = {'011': '11', '012': '11', '031': '31',  '021': '21', '039': '39'}
    read_file = brute_force_replacement(read_file, 'catpri', initialReplacement_dict)
    #converting the values to the simplified order for the issue of 59%
    #https://docs.google.com/document/d/1SttfZUfyhpbdjW0FjrMm3ufH8cmZCjWZK2Www-SoI0A/edit?tab=t.0
    #https://docs.google.com/spreadsheets/d/1TJu15ql9LKaxcjazERkyUqI-VugOnZNUXsppd_FfnS0/edit?gid=0#gid=0
    """
    # DATA 2:
    #       https://docs.google.com/spreadsheets/d/1TJu15ql9LKaxcjazERkyUqI-VugOnZNUXsppd_FfnS0/edit?gid=0#gid=0
    #       https://docs.google.com/presentation/d/1x6yKKuXxuJlE6sJiiKYl7Y7YMebyF8GgamuUE3_Gxkc/edit#slide=id.g316b3caf76f_0_19
    dictToReplace_simplified_order = \
                     {'411': '429', '421': '429', '422': '429', '423': '429', '424': '429', '425': '429', '426': '429', '427': '429',
                      '428': '429', '432': '431', '439': '431', '512': '511', '513': '511', '514': '511', '515': '511', '516': '511',
                      '517': '511', '519': '511', '521': '529', '522': '529', '523': '529', '524': '529', '525': '529',  '531': '529',
                      '611': '619', '612': '619', '613': '619', '614': '619', '615': '619', '616': '619', '617': '619','631': '621',
                      '712': '711', '713': '711', '714': '711', '719': '711', '722': '721', '723': '721','732': '731', '733': '731',
                      '734': '731', '735': '731', '736': '731', '739': '731', '812': '811', '813': '811', '814': '811', '819': '811',
                      '822': '821', '832': '831', '910': '900', '921': '900', '922': '900', '931': '900', '936': '900', '938': '900',
                      '939': '900', '941': '900', '942': '900', '943': '900', '951': '900', '960': '900', '971': '900', '972': '900',
                      '981': '900', '982': '900', '983': '900', '989': '900', '997': '995'}
    
    # DATA 3:
    #       https://docs.google.com/spreadsheets/d/1TJu15ql9LKaxcjazERkyUqI-VugOnZNUXsppd_FfnS0/edit?gid=0#gid=0
    #       https://docs.google.com/presentation/d/1x6yKKuXxuJlE6sJiiKYl7Y7YMebyF8GgamuUE3_Gxkc/edit#slide=id.g316b3caf76f_0_25

    dictToReplace = {'90': '111', '112': '111', '113': '111', '121': '111', '122': '111', '123': '111', '132': '111', '139': '111',
                     '219': '211', '221': '211', '222': '211', '223': '211', '224': '211', '229': '211', '341': '349', '342': '349',
                     '343': '349', '344': '349', '352': '351', '353': '351', '354': '351', '359': '351', '362': '361', '363': '361',
                     '364': '361', '365': '361', '369': '361', '411': '429', '421': '429', '422': '429', '423': '429', '424': '429',
                     '425': '429', '426': '429', '427': '429', '428': '429', '432': '431', '439': '431', '512': '511', '513': '511',
                     '514': '511', '515': '511', '516': '511', '517': '511', '519': '511', '521': '529', '522': '529', '523': '529',
                     '524': '529', '525': '529', '531': '529', '611': '619', '612': '619', '613': '619', '614': '619', '615': '619',
                     '616': '619', '617': '619', '631': '621', '712': '711', '713': '711', '714': '711', '719': '711', '722': '721',
                     '723': '721', '729': '721', '732': '731', '733': '731', '734': '731', '735': '731', '736': '731', '739': '731',
                     '812': '811', '813': '811', '814': '811', '819': '811', '822': '821', '832': '831', '910': '900', '921': '900',
                     '922': '900', '931': '900', '936': '900', '938': '900', '939': '900', '941': '900', '942': '900', '943': '900',
                     '951': '900', '960': '900', '971': '900', '972': '900', '981': '900', '982': '900', '983': '900', '989': '900',
                     '995': '995', '997': '995'}
    """
    # DATA 4:
    #       https://docs.google.com/spreadsheets/d/1TJu15ql9LKaxcjazERkyUqI-VugOnZNUXsppd_FfnS0/edit?gid=0#gid=0
    #       https://docs.google.com/presentation/d/1x6yKKuXxuJlE6sJiiKYl7Y7YMebyF8GgamuUE3_Gxkc/edit#slide=id.g316b3caf76f_0_41
    reduction_dict = {
        '531': '90',
        '111': '113', '112': '113', '121': '113', '122': '113', '123': '113', '132': '113',
        '139': '113',
        '211': '219', '212': '219', '221': '219', '222': '219', '223': '219', '224': '219', '229': '219',
        '311': '319', '312': '319',
        '321': '329', '322': '329', '323': '329', '324': '329',
        '331': '339', '332': '339', '333': '339',
        '341': '349', '342': '349', '343': '349', '344': '349', '351': '349', '352': '349', '353': '349', '354': '349', '359': '349',
        '361': '369', '362': '369', '363': '369', '364': '369', '365': '369',
        '371': '399', '372': '399', '381': '399', '382': '399', '383': '399', '384': '399', '389': '399', '391': '399', '392': '399', '393': '399', '394': '399',
        '411': '429', '421': '429', '422': '429', '423': '429', '424': '429', '425': '429', '426': '429', '427': '429', '428': '429',
        '431': '439', '432': '439',
        '511': '519', '512': '519', '513': '519', '514': '519',  '515': '519', '516': '519', '517': '519',
        '521': '529', '522': '529', '523': '529', '524': '529', '525': '529',
        '611': '631', '612': '631', '613': '631', '614': '631', '615': '631', '616': '631', '617': '631', '619': '631', '621': '631',
        '711': '739', '712': '739', '713': '739', '714': '739', '719': '739',
        '721': '729', '722': '729', '723': '729',
        '731': '739', '732': '739', '733': '739', '734': '739', '735': '739', '736': '739',
        '811': '819', '812': '819', '813': '819', '814': '819',
        '832': '821', '822': '821', '831': '821',
        '910': '900', '921': '900', '922': '900', '931': '900',
        '936': '900', '938': '900', '939': '900', '941': '900', '942': '900', '943': '900', '951': '900',
        '960': '900', '971': '900', '972': '900', '981': '900', '982': '900', '983': '900', '989': '900',
        '995': '997',
    }
    #read_file = brute_force_replacement(read_file, 'catpri', reduction_dict)

    if naming == True:
        #new column names of dataframe
        with open(name_path, 'r') as f:
            new_names = [line.strip() for line in f.readlines()]
        my_dict = dict(zip(read_file.columns, new_names))
        read_file = read_file.rename(columns=my_dict)
    else:
        pass

    read_file = read_file.drop(['survey_code', 'survey_year', 'Occupant_side_Activity', "Occupant_feeling",
                                "Occupant_Activity_PC", "Occupant_side_Activity_PC",
                                "daily_order_Activity",
                                'minStart_Activity', 'minEnd_Activity',
                                #'months_season',
                                ], axis=1)

    #3RD STEP: from raw to readable dataframe
    for i in read_file.columns:
        read_file = dfppInd.editCol(read_file, i, rename=i)

    read_file.drop_duplicates(keep='first', inplace=True)

    # OUTPUT
    read_file.to_csv(output_path_csv, index=None)
    print('reading function: writing as .csv is done')

import pandas as pd

def brute_force_replacement(df, column_name, replacement_dict):
    """
    This function replaces values in the specified column of a DataFrame according to a given dictionary
    and converts the column values to integers if possible.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The column where replacements will be applied.
    replacement_dict (dict): A dictionary specifying replacements {old_value: new_value}.

    Returns:
    pd.DataFrame: The DataFrame with replaced values and integer conversion.
    """
    # Make sure the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Replace the values using the provided dictionary
    df[column_name] = df[column_name].astype(str).replace(replacement_dict)

    # Convert the values to integers if possible
    try:
        df[column_name] = df[column_name].astype(int)
    except ValueError:
        raise ValueError(f"Column '{column_name}' cannot be converted to integers. Please check for non-integer values.")

    return df

def imputation_manual(input_path, output_path_csv=None):
    '''4TH STEP: MANUAL IMPUTATION'''
    df = pd.read_csv(input_path)

    from TUS_Daily_Functions import feature_extraction, imput_man_bruteForce
    df = feature_extraction(df)
    df = imput_man_bruteForce(df)
    #OUTPUT
    df.to_csv(output_path_csv, index=False)
    print('imputation manual: writing as .csv is done')

def finalStep(input_path,output_path_csv=None,):
    import pandas as pd
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".ftr"):
        df = pd.read_feather(input_path)

    df = df.astype(int)

    new_order = ['Household_ID', 'Occupant_ID_in_HH',
                 'months_season', 'week_or_weekend',
                 'hourStart_Activity', 'hourEnd_Activity',
                 'Occupant_Activity',
                 'location',
                 'withALONE',
                 'withMOTHER', 'withFATHER', 'withSPOUSE', 'withCHILD', 'withBROTHER',
                 'withOTHERFAMILYMEMBER', 'withOTHERPERSON',
                 'witness'
                 ]

    df = df[new_order]

    from preProcessing_Func import functions_general as dfppfg
    df = df.applymap(dfppfg.remove_trailing_zero)

    #OUTPUT
    df.to_csv(output_path_csv, index=None)
    print('final step: writing as .csv is done')

#------------------------------------------------------------------------------------------------------------------
def select_samples_householdID(tus_daily_augmented_path, tus_daily_augmented_sample_path, size=1, specific_household_id=None):
    import pandas as pd
    import numpy as np

    # Read the datasets
    df = pd.read_csv(tus_daily_augmented_path)

    if specific_household_id:
        # Ensure the specific Household ID exists in the data
        if specific_household_id in df['Household_ID'].unique():
            # Filter the DataFrame to only rows with the specific 'Household_ID'
            sample_df = df[df['Household_ID'] == specific_household_id]
            print(f"Sample dataset for specific Household_ID {specific_household_id} saved to {tus_daily_augmented_sample_path}")
        else:
            print(f"Household_ID {specific_household_id} not found in the dataset.")
            return
    else:
        # Select x random 'Household_ID'
        random_ids = np.random.choice(df['Household_ID'].unique(), size=size, replace=False)

        # Filter the DataFrame to only rows with the selected 'Household_ID'
        sample_df = df[df['Household_ID'].isin(random_ids)]

        print(f"Sample dataset for {size} random Household_IDs saved to {tus_daily_augmented_sample_path}")

    # Save the sample_df to a new CSV
    sample_df.to_csv(tus_daily_augmented_sample_path, index=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from preProcessing_Func import analysis_func as dfppaf

    #INPUT PATHS
    txt_input = r'dataset_TUS_daily/uso_tempo_Microdati_Anno_2013_DiarioGiornaliero.txt'
    name_path = r'dataset_TUS_daily/TUS_daily_naming.txt'

    # OUTPUT PATHS
    tus_daily_reading = r'dataset_TUS_daily/TUS_daily_reading.csv'
    tus_daily_imputMan = r'dataset_TUS_daily/TUS_daily_imput_manual.csv'
    tus_daily_final = r'dataset_TUS_daily/TUS_daily_final.csv'
    tus_daily_final_simplified = r'dataset_TUS_daily/TUS_daily_final_simplified.csv'
    tus_daily_final_DATA3 = r'dataset_TUS_daily/tus_daily_final_DATA3.csv'
    tus_daily_final_DATA4 = r'dataset_TUS_daily/tus_daily_final_DATA4.csv'

    # OUTPUT PATHS - TRIAL
    #tus_daily_final_1HHID = r'dataset_TUS_daily/TUS_daily_final_1HHID.csv'

    #reading(input_path=txt_input, naming=True, output_path_csv=tus_daily_reading)     #read .txt and convert to .csv
    #imputation_manual(input_path=tus_daily_reading, output_path_csv=tus_daily_imputMan)
    #finalStep(input_path=tus_daily_imputMan, output_path_csv=tus_daily_final)

    #___DATA SELECTION___________________________________________________________________________________________________________________________
    #select_samples_householdID(tus_daily_final, tus_daily_final_1HHID, specific_household_id=8) # select number of household

    input = tus_daily_final

    non_visual = True
    visual = True
    ID_DROP = ["Household_ID", 'Occupant_ID_in_HH']
    from preProcessing_Func import analysis_func as dfppaf
    #dfppaf.analysis(input_path=input, describe=non_visual)
    dfppaf.analysis(input_path=input, data_len=non_visual)
    #dfppaf.analysis(input_path=input, data_types=non_visual)
    dfppaf.analysis(input_path=input, missingness=non_visual)
    dfppaf.analysis(input_path=input, columns=non_visual)
    dfppaf.analysis(input_path=input, unique=non_visual, uniqueIDcolstoDrop=ID_DROP)

    print(" ")
    #dfppaf.analysis(input_path=input, missingness_visual_oriented=visual)
    #dfppaf.analysis(input_path=input, fraction=1, unique_visual=visual, uniqueIDcolstoDrop=['Household_ID'])
    dfppaf.analysis(input_path=input, multiple_hist=visual, dropID_multiple_hist=["Household_ID"])




