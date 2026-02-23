def feature_extraction(df):
    # Define the function and apply it to the DataFrame (same code as before)
    '''
    tus_data_subset,
    - location 0: outside
    - location 1: in residential
    '''

    def assign_loc_category(row):
        if row['location'] > 11:
            return 0
        elif row['location'] == 11:
            return 1
        else:
            return row['location']

    # Apply the function to each row in the dataframe
    df['location'] = df.apply(assign_loc_category, axis=1)

    return df

def imput_man_bruteForce(df): # to fullfill NaN values with zeros and other integers
    from preProcessing_Func import stats as dfppss

    accompany = ["withALONE",
                 "withMOTHER", "withFATHER", "withSPOUSE", "withCHILD", "withBROTHER",
                 "withOTHERFAMILYMEMBER", "withOTHERPERSON", "witness"
                 ]
    for i in accompany:
        dfppss.bruteForceImput(df, nameToEdit=i, val2imp=0, imputSelection=6, rename=i)
    return df

'''
    #mainAct_OCC = actCat_OCC  /  actCat_OCC_PC = mainAct_OCC_PC

    #sideAct_OCC = act_OCC  /  act_OCC_PC = sideAct_OCC_PC 
    #action for occupant with(out) PC
    df = dfppss.bruteForceImput(df, nameToEdit="actCat_OCC_PC",
                                namedfAccMain="actCat_OCC",
                                val2imp=0,
                                imputSelection=5,
                                rename="actCat_OCC_PC_edited",
                                inputsCom=[0,3, 4, 5, 6, 7, 9, 10, 20, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 53, 61,
                                           62, 63, 83, 90, 11, 12, 21, 31, 39, 311, 312, 321, 324, 331, 332,
                                           333, 339, 341, 342, 343, 344, 349, 351, 352, 353, 354, 361, 362, 363,
                                           364, 365, 371, 381, 391, 392, 393, 421, 422, 423, 424, 426, 427, 428,
                                           432, 512, 513, 514, 516, 517, 521, 522, 523, 524, 525, 529, 531, 611,
                                           612, 613, 614, 615, 616, 617, 619, 621, 631, 711, 712, 713, 714, 719,
                                           735, 736, 739, 814, 831, 900, 910, 921, 922, 931, 936, 938, 939, 941,
                                           942, 943, 951, 960, 971, 972, 981, 982, 983, 989, 821,111,813,322,732,112,383])
    #action for occupant with PC
    df = dfppss.bruteForceImput(df, nameToEdit="actCat_OCC_PC_edited",
                                namedfAccMain="actCat_OCC",
                                val2imp=1,
                                imputSelection=5,
                                rename="actCat_OCC_PC_edited",
                                inputsCom=[8, 72, 372, 721, 722, 723, 729, 731, 733, 734, 822, 832])

    df = dfppss.bruteForceImput(df, nameToEdit="feeling",
                                val2imp=0,
                                imputSelection=6,
                                rename="feeling")
                                
    df = df[df['actCat_OCC_PC_edited'].notna()]
'''

