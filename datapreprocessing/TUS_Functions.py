
def columns_to_drop(df): # COLUMNS TO DROP because they are not directly related with daily schedules of members of household
    dropList = [
                #"Marital Status",
                "Workplace Distance",
                 "Nuclear Family, Typology",
                 #"Position in the Job",
                 #"Formal Registration, Profession",
                 #"Economic Sector, Profession",
                  "survey_year", "Survey Quarter",
                "Kinship Relationship", "Previous Marital Status", "Daily Diary Attendance",
                 #"Employee in Resident, how many",
                #"Economic Resources",
                #"Employee in Resident, existence",
     "Weekly Diary Attendance","Individual Universe Ratio", "Daily Diary Ratio", "Citizenship", "Father Birth State", "Mother Birth State",
     "Support Availability, emergence", "Mother Support, emergence", "Father Support, emergence",
     "Child Support, emergence", "Sibling Support, emergence", "Grandparent Support, emergence",
     "Grandchild Support, emergence", "Relative Support, emergence", "Friend Support, emergence",
     "Neighbor Support, emergence", "See Friends Frequency", "Non_cohabiting Children", "Job Retention",
     "Time Out of Work", "No Work Reason, last week", "Additional Jobs","Days in Week, additional job", "Hours in Week, additional job",
     "Variable Work Time", "Term Job?", "Schedule Flexibility, Work", "Work time with family A",
     "Work time with family B", "Work time with family C", "Worked in Past?", "Dont Work: Not Interested",
     "Dont Work: No Need", "Dont Work: Finish Studies", "Dont Work: Personal Reasons", "Dont Work: Childcare",
     "Dont Work: Family Disagrees", "Dont Work: Family Reasons", "Dont Work: No Job Found",
     "Dont Work: Uninteresting Job", "Dont Work: Other Reasons", "Last Job Position", "Had Employees?",
     "Employed Position", "Economic Sector, Profession, Last Work", "Stopped Working Year", "Why Stopped Working?",
     "Happy Stopping Work?", "Undated Resignation?", "Job Loss?", "Pregnancy/Job Loss, Timing",
     "Position was in Profession", "Had Employees?", "Had Employees?.1", "Job Lost: Pregnancy", "Economic Sector", "Job Search?",
     "Job Search Activities", "Job search, public center", "Job search, private interviews", "Job search, public tests",
     "Job search, competition application", "Job search, newspaper vacancies", "Job search, ads response",
     "Job search, resumes sent", "Job search, friends/relatives", "Job search, internet", "Job search, temp agency",
     "Job search, self_employment resources", "Job search, permits/licenses/funding", "Job search, other methods",
     "Preferred work hours", "Main job search reason", "Services absent/expensive", "Job start availability",
     "Mothers work when respondent14", "Fathers work when respondent14",
     "Diary completion day", "With parents at work", "Satisfied: self_time", "Satisfied: partner_time",
     "Satisfied: children_time", "Satisfied: parents_time", "Satisfied: work_time", "Satisfied: social_time",
     "Satisfied: leisure_time", "Satisfied: rest_time", "Satisfied: Couple Life", "Satisfied: Work_Family",
     "Desire more social time", "Time trouble feelings", "Partner Schedule Conflict", "Children School Conflict",
     "Office Hours Conflict", "Leisure Place Hours Conflict", "Shop Hours Conflict", "Economic Satisfaction",
     "Free Time Satisfaction", "Free Time Quality", "Work Satisfaction", "Couple Life Satisfaction",
     "Work_Family Balance", "Overall Life Satisfaction", "Trust People?", "Traditional Roles", "Equal Housework",
     "Sick Child Care", "Fathers Childcare Skill", "Tidy Home Importance", "Activity Organization Talks",
     "More Time Requested", "Time Requested by Partner", "Housework Contributor", "Housework Division Satisfaction",
     "Income Contributor", "Childcare Contributor", "Childcare Division Satisfaction", "Family Farming",
     "Raising Animals", "Farm Products Sales", "Pet Ownership", "No Work Diary Week", "Diary Compilation Situation",
     "Diary Compilation Method", "Second Home", "Building House",
     "b_Working Hours, Reason", "b_Child Care, Family Need", "b_Elderly Care, Family Need",
    "b_House Care, Family Need", "b_Partner Time, Family Need", "b_Family Time, Family Need",
    "b_Other, Family Need", "a_Working Hours, Reason", "a_Child Care, Family Need", "a_Elderly Care, Family Need",
    "a_House Care, Family Need", "a_Partner Time, Family Need", "a_Family Time, Family Need",
    "a_Other, Family Need", "Outside Work Hours, Reason A", "Outside Work Hours, Reason B", "Outside Work Hours, Reason C",
    "Major Home Work", "unknown hour, action", "Worked Last Week", "Job category",
    "Domestic collaborator: Italian", "Domestic collaborator: foreigner", "Baby_sitter: Italian", "Baby_sitter: foreigner", "Elderly assistant: Italian", "Elderly assistant: foreigner",
    "Home Ownership",
    'Time with friends', 'Time with schoolmates', 'Time with cousins', 'Time with siblings',
    'Time with mom', 'Time with dad', 'Time with grandparents', 'Time with others', 'Time with nobody',
    "Outside Work: PC", "Outside Work: Internet", "Outside Work: Phone", "Outside Work: Materials",
    "Outside Work: Meetings", "Outside Work: Accounting", "Outside Work: Reading",
    "Men Housework Skill",
    'Telecommuting Interest', 'Remote Work, Time Need', 'Remote Work, Work Need',
    'Remote Work, Concentration', 'Remote Work, Work/Family Balance',
    'Remote Work, Travel Time Reduction', 'Remote Work, Other',
    'Choose Part_Time?', 'Time Period, Reason', 'If Part Time, Reason', 'If Childcare, Reason', 'if Elderly Care, Reason',
    'if Domestic Work, Reason', 'if Family Time, Reason', 'if Other, Reason','Part_time Service, Reason',
    "Spend time playing", "Spend time outdoors", "Spend time school", "Spend time homework",
    "Spend time sports", "Spend time extra_curricular", "Spend time other", "Nuclear Family, Occupant Sequence Number"]

    df = df.drop(dropList, axis=1)
    return df

def merging_manual(df):  # feature reduction
    from preProcessing_Func import dfTUS_data as dftus

    ####################################################################################################################
    # TRANSPORT:
    '''
    columns_transport = ['Bicycle Ownership', 'Bicycle Count', 'Moped/Scooter Ownership',
                         'Moped/Scooter Count', 'Motorcycle Ownership', 'Motorcycle Count',
                         'Car Ownership', 'Car Count']
    # company columns: merge & labelEncoding
    df = dftus.MultipleEncoding(df, columns_transport, namingCol="transport")
    '''
    df = df.drop(['Bicycle Ownership', 'Bicycle Count', 'Moped/Scooter Ownership',
                         'Moped/Scooter Count', 'Motorcycle Ownership', 'Motorcycle Count',
                         'Car Ownership', 'Car Count'], axis=1)
    ####################################################################################################################
    #TECHNOLOGY:
    # PC_exist: "Computer Ownership"
    # TV_exist: "Satellite Dish Ownership"
    '''
    #tech = ["Computer Ownership", "Computer Count", "Internet Access",
    #        "Internet Access Number", "Satellite Dish Ownership", "Satellite Dish Count",
    #        "Landline Ownership", "Landline Count", "Mobile Ownership", "Mobile Count"]
    # company columns: merge & labelEncoding
    #df = dftus.MultipleEncoding(df, tech, namingCol="tech_existence")
    '''
    df = df.drop(["Computer Ownership", "Computer Count", "Internet Access",
                  "Internet Access Number", "Satellite Dish Ownership", "Satellite Dish Count",
                  "Landline Ownership", "Landline Count", "Mobile Ownership", "Mobile Count"], axis=1)
    ####################################################################################################################

    #OVERTIME WORK COLUMNS
    df['Fixed Work Time'] = df['Fixed Work Time'].replace({1: 2, 2: 1})
    df['OvertimeWork'] = df['Overtime Work'].combine_first(df['Fixed Work Time'])
    df = df.drop(['Overtime Work', 'Fixed Work Time'], axis=1)

    return df

def update_column_values_by_condition(df, condition_column, condition_values, target_column, new_value):
    """
    Update values in the target column based on the condition,
    but only if the target column's rows are empty (NaN).

    Parameters:
    df (pd.DataFrame): The DataFrame to update.
    condition_column (str): The column to check the condition.
    condition_values (list): The values to match in the condition column.
    target_column (str): The column to update.
    new_value: The value to set in the target column.

    Returns:
    pd.DataFrame: The updated DataFrame.
    """
    # Apply the condition only where target_column is NaN
    mask = df[condition_column].isin(condition_values) & df[target_column].isna()
    df.loc[mask, target_column] = new_value
    return df
def fill_columns_by_age_classes(df, age_classes_column, age_classes_values, target_columns, fill_value):

    # Create a mask for rows where the age classes column matches the specified values
    mask = df[age_classes_column].isin(age_classes_values)

    # Update target columns only if they are NaN and the mask condition is true
    for column in target_columns:
        # Check if the fill_value matches the column's data type, and convert if necessary
        if df[column].dtype != type(fill_value):
            df[column] = df[column].astype(object)  # Convert to object to allow mixed types
        df.loc[mask & df[column].isna(), column] = fill_value

    return df
def convert_columns_to_int(df, target_columns):
    for column in target_columns:
        df[column] = df[column].fillna(0).astype(int)
    return df
def update_education_degree(df):
        """
        Updates the 'Education Degree' column in the DataFrame based on the condition:
        If 'Age Classes' is in [1, 2, 3, 4], set 'Education Degree' to 4.

        Parameters:
            df (pd.DataFrame): The DataFrame containing 'Age Classes' and 'Education Degree' columns.

        Returns:
            pd.DataFrame: The updated DataFrame.
        """
        condition = df['Age Classes'].isin([1, 2, 3, 4])
        df.loc[condition, 'Education Degree'] = 4
        return df

def imput_man_bruteForce(df): # to fullfill NaN values with zeros and other integers
    from preProcessing_Func import stats as dfppss

    # Task: Fill the children's job-related information with zeros using "AGE CLASSES"
    child_by_ageClassesList = ["Main Income Source", "Job Type", "Full_Part_time", "Employment status", "Full_Part_time",
                               "Economic Sector, Profession", "Permanent/fixed", ]
    # Convert columns to int where applicable, handling NaN
    df = convert_columns_to_int(df, [col for col in child_by_ageClassesList if df[col].dtype != 'object'])
    df = fill_columns_by_age_classes(df, "Age Classes", [1, 2, 3, 4], child_by_ageClassesList, 0)

    df = update_education_degree(df) # for children

    df = dfppss.impute_marital_status(df, "Age Classes", "Marital Status", impute_value=1)

    update_column_values_by_condition(df, "Employment status", [0,2,4,5,7,8], "Job Type", 0)
    update_column_values_by_condition(df, "Employment status", [0, 2, 4, 5, 7, 8], "Full_Part_time", 0)
    update_column_values_by_condition(df, "Employment status", [0, 2, 4, 5, 7, 8], "Permanent/fixed", 0)
    update_column_values_by_condition(df, "Employment status", [0,2,4,5,7,8], "Economic Sector, Profession", 0)
    # Task: Fill the children's education-related information with 11 (no title) using "AGE CLASSES"
    # resource for education degree: https://docs.google.com/spreadsheets/d/1IOjNvJaW7wXtOOXiuE3al79tQbnJeybT5AlRgLkeuPE/edit#gid=1866035734
    little_child_by_ageClassesList= ['Education Degree']
    for i in little_child_by_ageClassesList:
        dfppss.bruteForceImput(df, i, "Age Classes", inputsCom=[1,2], val2imp=11, imputSelection=5, rename=i)

    """
    # Currently, we do not use these columns - 27.05.2024
    # from self_employed
    self_employed = ["Full_Part time", "Job type" ]
    for i in self_employed:
        dfppss.bruteForceImput(df, i, "Main Income Source", inputsCom=[3,4,6], val2imp=0, imputSelection=5, rename=i)

    # Currently, we do not use these columns - 27.05.2024
    # from MARITAL
    maritalList = ["Marital Status"]
    for i in maritalList:
        dfppss.bruteForceImput(df, i, "Age Classes", inputsCom=[1,2], val2imp=1, imputSelection=5, rename=i) #https://docs.google.com/document/d/1hN8400ZsH2eXd716fE_K8HTGi_5JuxbB0EjxRuM25u0/edit
    """

    return df

def feature_extraction(df):
    # Define the function and apply it to the DataFrame (same code as before)
    '''
    tus_data_subset,
    - If age classes is 5, education degree is 1, then occupant 19-24
    - If age classes is 5, main income source is 1,2,  then occupant 19-24
    - If age classes is 5, course distance is not 0, then occupant 19-24
    - If age classes is 5, Nuclear Family, Occupant Profile is 0,1,2, then,  then occupant 19-24
    '''

    def assign_age_category(row):
        if row['Age Classes'] == 5:
            # Condition 1: If age classes is 5 and education degree is 1, then new age classes is 5B
            if row['Education Degree'] == 1:
                return 12

            # Condition 2: If age classes is 5 and main income source is 1 or 2, then new age classes is 5B
            if row['Main Income Source'] in [1, 2]:
                return 12

            # Condition 3: If age classes is 5 and course distance is not 0, then new age classes is 5A
            #if row['Course Distance'] == 0:
            #    return 12

            # Condition 4: If age classes is 5 and main income source is 1 or 2, then new age classes is 5B
            #if row['Nuclear Family, Occupant Profile'] in [0, 1, 2]:
            #    return 12

            # If none of the conditions hold, return '5A'
            return 5
        else:
            return row['Age Classes']

    # Apply the function to each row in the dataframe
    df['Age Classes'] = df.apply(assign_age_category, axis=1)
    return df

def missing(df):
    for i in range(df.shape[1]):
        n_miss = df.iloc[:, i].isna().sum()
        perc = n_miss / df.shape[0] * 100
        colName = df.columns[i]
        print('{}:'.format(colName), '%d (%.1f%%)' % (n_miss, perc))

def feature_importance(input_path,to_csv=False, output_path=False, print_less_effective=False, columns=False): #COLUMNS TO DROP because they are not directly related with daily schedules of members of household

    from preProcessing_Func import stats_visuals as dfppsv
    import pandas as pd

    #INPUT
    df = pd.read_csv(input_path)
    print(df.columns)

    # pca analysis for feature importance
    dfppsv.pca_analysis(df, dropNAN=True, print_less_effective=print_less_effective,
                        dropIrrelevant=["Survey Quarter"]
                        )
    ''' 
    'company' column does not have strong impact, thus, it is should be eliminated 
    from classification. however, company column have missing rows so it should be included
    into classification for imputation.
    '''

    #after feature importance analysis
    # threshold= 0.95
    less_effective_columns = ['Partner Schedule Conflict', 'Children School Conflict', 'Office Hours Conflict',
                              'Leisure Place Hours Conflict', 'Shop Hours Conflict', 'Economic Satisfaction',
                              'Free Time Satisfaction', 'Free Time Quality', 'Work Satisfaction', 'Couple Life Satisfaction',
                              'Work-Family Balance', 'Overall Life Satisfaction', 'Trust People?', 'Traditional Roles',
                              'Equal Housework', 'Sick Child Care', 'Men Housework Skill', 'Fathers Childcare Skill',
                              'Tidy Home Importance', 'Activity Organization Talks', 'More Time Requested',
                              'Time Requested by Partner', 'Housework Contributor', 'Housework Division Satisfaction',
                              'Income Contributor', 'Childcare Contributor', 'Childcare Division Satisfaction',
                              'Diary Compilation Situation', 'Diary Compilation Method', 'Home Type', 'Home Occupancy',
                              'Room Count', 'Second Home', 'Building House', 'Major Home Work', 'Family Farming',
                              'Raising Animals', 'Farm Products Sales', 'Pet Ownership', 'Domestic Worker Hours',
                              'Baby-Sitter Hours', 'Elderly/Disabled Hours', 'Economic Resources', 'domestic_assisstant',
                              'baby_sitter', 'num_domestic_assistant', 'Elderly_disabled_assistant', 'transport', 'tech',
                              'accompany', 'outside_work', 'outside_work_type', 'outside_work_reason',
                              'worktimeWithFamily', 'b_workHoursReason', 'a_workHoursReason', 'fixed_overtime', 'remoteWork',
                              'partTime_period_day', 'fullparttime_choice', 'support', 'spend_time_activity_children']

    df = df.drop(less_effective_columns, axis=1)

    if columns ==True:
        # Assuming 'df' is your DataFrame
        print(len(df.columns))
        #for col_name in df.columns:
        #    print(col_name)

    # OUTPUT
    if to_csv == True:
        df.to_csv(output_path, index=None)

    return df


def imputation_auto(input_path, output_path=False, to_csv=False, fraction=1, columns=False, missingness=False,
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
    df = pd.read_csv(input_path)
    #print(df.columns)

    #SAMPLING
    dfSamp = df.sample(frac=fraction, replace=True, random_state=1)
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
                                    colToOutput='Job Retention',
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
                                        colToOutput="Job Retention",
                                        #colToDrop=['act_OCC_PC','act_OCC'],
                                        pipe=True,
                                        fraction=fraction,
                                        kfold_repeat=3, kfold_splits=2,
                                        model_to='random_forest',
                                        )

    if columns ==True:
        # Assuming 'df' is your DataFrame
        #print(len(df.columns))
        for col_name in df.columns:
            print(col_name)

    #OUTPUT
    if to_csv==True:
        df.to_csv(output_path, index=None)

def filter_df(df): # filter df based on accuracy reasons
    #df = df[df['Questions answered directly'].notna()] # only directly answered questions are included
    #df = df.drop('Questions answered directly', axis=1)

    df = df[df['Home Type'] == 2]  # only residential units are included
    df = df.drop('Home Type', axis=1)
    #df = df[df["Room Count"] < 6]

    #df = df[df['Questions answered directly'].isin([1,2,5])]
    #df = df[df['Number Family Members'] != 7]
    #df = df.loc[~((df['Age Classes'] == 5) & (df['Main Income Source'] == 3))]
    return df

def change_values_Age_Classes(value): # for "Age Classes"
    if value == "1":
        return 1
    elif value == "2":
        return 1
    elif value == "3":
        return 1
    elif value == "4":
        return 1
    elif value == "5":
        return 2
    elif value == "6":
        return 2
    elif value == "7":
        return 3
    elif value == "8":
        return 4
    elif value == "9":
        return 5
    elif value == "10":
        return 6
    elif value == "11":
        return 6
    return value

def change_values_FT(value): # for "Family, Typology"
    if value in ['1','2','3', '4', '5']:
        return 2
    elif value in ['6', '7', '20', '21', '34', '35', '36', '37', '38', '39', '40']:
        return 5
    elif value in ['8', '9', '22', '23']:
        return 3
    elif value in ['10', '11', '14','15', '16', '19',  '24', '25', '28','29', '30', '33',]:
        return 4
    return value

def change_values_ES(value): # for "Employment Status"
    if value == "7":
        return 3
    elif value == "5":
        return 4
    elif value == "1":
        return 1
    elif value == "2":
        return 2
    elif value == "0": #Age classes: 1
        return 6
        #return 7
    elif value == "4":
        return 5
    elif value == "8":
        return 6
    return value

def change_values_ES_moreRepresentation (value): # for "Employment Status"
    if value == "7":
        return 3
    elif value == "5":
        return 4
    elif value == "1":
        return 1
    elif value == "2":
        return 2
    elif value == "0": #Age classes: 1
        return 7
    elif value == "4":
        return 5
    elif value == "8":
        return 6
    return value

def change_values_jt(value): # for "Job Type"
    if value == "0":
        return 0
    elif value == "1":
        return 1
    elif value == "2":
        return 2
    elif value == "3":
        return 2
    elif value == "4":
        return 3
    elif value == "5":
        return 4
    elif value == "6":
        return 5
    elif value == "7":
        return 6
    elif value == "8":
        return 7
    return value

def change_values_ed(value): # for "Education Degree"
    if value == "11":
        return 1
    elif value == "10":
        return 2
    elif value == "9":
        return 3
    elif value == "7":
        return 4
    elif value == "1":
        return 5
    return value

def change_values_nfm(value): # for "Number Family Members"
    if value == "7":
        return 6
    return value

def re_arrangeColsOCCschedule(df):
    # Task: RE-arrange column values
    # resource: https://docs.google.com/spreadsheets/d/1IOjNvJaW7wXtOOXiuE3al79tQbnJeybT5AlRgLkeuPE/edit#gid=1866035734
    # Change column values based on conditions
    df['Age Classes'] = df['Age Classes'].apply(change_values_Age_Classes)
    df['Family Typology'] = df['Family Typology'].apply(change_values_FT)
    df['Employment status'] = df['Employment status'].apply(change_values_ES)
    df['Education Degree'] = df['Education Degree'].apply(change_values_ed)
    df['Number Family Members'] = df['Number Family Members'].apply(change_values_nfm)

    #print("AC:", df['Age Classes'].unique())
    #print( "FT:", df['Family Typology'].unique())
    #print("ES:", df['Employment status'].unique())
    #print("jt:", df['Job type'].unique())
    #print("ed:", df['Education Degree'].unique())
    return df

def re_arrangeColsOCCschedule_moreRepresentation(df):
    # Task: RE-arrange column values
    # resource: https://docs.google.com/spreadsheets/d/1IOjNvJaW7wXtOOXiuE3al79tQbnJeybT5AlRgLkeuPE/edit#gid=1866035734
    # Change column values based on conditions
    df['Education Degree'] = df['Education Degree'].apply(change_values_ed)

    #print("AC:", df['Age Classes'].unique())
    #print( "FT:", df['Family Typology'].unique())
    #print("ES:", df['Employment status'].unique())
    #print("jt:", df['Job type'].unique())
    #print("ed:", df['Education Degree'].unique())
    return df

def make_binary_min_max(df, columns):
    binary_df = df.copy()

    for column in columns:
        binary_df[column] = binary_df[column].apply(lambda x: 1 if x == 0 else 0 if x != 0 else x)

    return binary_df

def merging_fordataimbalance(df):

    #MERGING AND CONVERTING TO BINARY: data imbalance problem
    #convert binary to all columns
    columns_to_convert = ['Domestic assistant: none', 'Baby_sitter: none', 'Elderly assistant: none',]
    df = make_binary_min_max(df, columns_to_convert)
    # Merge the binary columns into a single one
    df['assistant_existence'] = df[columns_to_convert].any(axis=1).astype(int)
    # Drop the original columns
    df = df.drop(columns=columns_to_convert, axis=1)

    # MERGING AND CONVERTING HOURS COLUMNS: data imbalance problem
    # Columns to sum
    assisstant_hours_per_week = ['Domestic Worker Hours per week', 'Baby_Sitter Hours per week', 'Elderly Hours per week']
    # Calculate the sum of the specified columns
    df['assisstant_hours_per_week'] = df[assisstant_hours_per_week].sum(axis=1)
    # Drop the original columns
    df = df.drop(columns=assisstant_hours_per_week, axis=1)

    #hours
    hours = ['Hours, school', 'Hours, kindergarten', 'Hours, work']
    df['hours_wsk'] = df[hours].sum(axis=1)
    df = df.drop(columns=hours, axis=1)

    #minutes
    minutes = ['Minutes, work', 'Minutes, school']
    df['minutes_wsk'] = df[minutes].sum(axis=1)
    df = df.drop(columns=minutes, axis=1)

    return  df

class TUS_daily:
    pass
    import pandas as pd

    def sample_daily_schedules(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
        """
        Sample a fraction of daily schedules based on Household_ID and Occupant_ID_in_HH.

        Parameters:
        - df: DataFrame containing the data.
        - frac: Fraction of daily schedules to sample. Default is 0.1 (10%).

        Returns:
        - Sampled DataFrame.
        """
        import pandas as pd

        # Reset the index of the DataFrame to ensure continuous indexing
        df = df.reset_index(drop=True)

        # Group by 'Household_ID' and 'Occupant_ID_in_HH' and sample a fraction of these groups
        # Check for the existence of the column and get the correct name
        if 'Occupant_ID_in_HH' in df.columns:
            occupant_id_column = 'Occupant_ID_in_HH'
        elif 'Occupant ID in HH' in df.columns:
            occupant_id_column = 'Occupant ID in HH'
        else:
            raise ValueError("Neither 'Occupant ID_in_HH' nor 'Occupant ID in HH' found in the dataframe")

        # Use the identified column name in the rest of the code
        unique_groups = df[['Household_ID', occupant_id_column]].drop_duplicates()

        sampled_groups = unique_groups.sample(frac=frac)

        # Merge the sampled groups with the original dataframe to get the rows that belong to these groups
        sampled_df = pd.merge(sampled_groups, df, on=['Household_ID', occupant_id_column], how='left')

        return sampled_df