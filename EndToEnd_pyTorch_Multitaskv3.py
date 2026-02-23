
if __name__ == '__main__':
    import pandas as pd
    import v3_mainRNN
    import v3_mainTrans
    import v3_mainN_HITS

    # INPUT PATHS
    tus_mainEqualPad50HHID = r'dataset_TUS_equalized/tus_mainEqualPad50HHID.csv'
    tus_mainEqualPad100HHID = r'dataset_TUS_equalized/tus_mainEqualPad100HHID.csv'
    tus_mainEqualPad107HHID = r'dataset_TUS_equalized/tus_mainEqualPad107HHID.csv'
    tus_mainEqualPad1000HHID = r'dataset_TUS_equalized/tus_mainEqualPad1000HHID.csv' #all the unique activities included in the sample
    tus_mainEqualPad = r'dataset_TUS_equalized/tus_mainEqualPad.csv'

    csv_input = tus_mainEqualPad107HHID
    df = pd.read_csv(csv_input)

    df = df[['Household_ID',
             'months_season', 'week_or_weekend',
             'Occupant_ID_in_HH',
             'Number Family Members',
             'Family Typology',
             'Employment status','Education Degree',
             'Gender',
             'hourStart_Activity', 'hourEnd_Activity',
             'Occupant_Activity', 'location', 'withNOBODY']]

    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.max_rows', None)  # Display all rows

    # LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM LSTM

    # test data shapes
    #print_test(X_train, y_activity_train, y_location_train, y_withNOB_train, X_valid, y_activity_valid, y_location_valid, y_withNOB_valid, X_test, y_activity_test, y_location_test, y_withNOB_test,)

    # LSTM: IMPACT ANALYSIS---------------------------------------------------------------------------------------------
    #impact_analysis(X_train, y_activity_train, y_location_train, y_withNOB_train, target_column='Occupant_Activity')

    # RNNs: BASEMODEL---------------------------------------------------------------------------------------------------
    #v3_mainRNN.BaseTrain(df,rnnType="LSTM",n_epoch=200,b_size=48, l_rate=0.001, n_h_layer=1,h_unit = [100],embedSize = 250,
    #                     activation_func_act = "relu", activation_func_bi="tanh", drLoc = 0.1, drNOB = 0.1, drRNN = 0.1, drEmbed = 0.1)

    # LSTM:TUNING NO CROSS-VALIDATION ---------------------------------------------------------------------------
    #v3_mainRNN.main_tuningLSTM(df=df, csv_filepath=csv_input, epochs=2, n_trials=2) # defualt epochs= 20-30, default n_trials= 50-100, default n_split=3-5
    #v3_mainRNN.trainEvaluate_afterTuning(df=df, csv_filepath=csv_input, rnn_type="LSTM", epochBase=10)

    # LSTM:TUNING CROSS-VALIDATION ---------------------------------------------------------------------------
    v3_mainRNN.main_tuningLSTM_kFold(csv_filepath=csv_input, epochs=3, n_trials=5, num_split=2, df=df) # defualt epochs= 20-30, default n_trials= 50-100, default n_split=3-5
    v3_mainRNN.trainEvaluate_afterTuning(df=df, csv_filepath=csv_input, rnn_type="LSTM", epochBase=5)

    # TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER TRANSFORMER
    # TRANSFORMER: BASEMODEL--------------------------------------------------------------------------------------------
    #v3_mainTFT.modelBaseTransformerTraining(df=df, epochNum=1000, batch_size=48, learning_rate=0.001)
    #v3_mainTFT.modelBaseTransformerTrainingNoEmbed(df=df, epochNum=10, batch_size=96, learning_rate=0.001, nhead=4, numHiddenLayers=2, dropout_loc = 0.1, dropout_withNOB = 0.1, dropout_TFTs = 0.1)

    # TRANSFORMER:TUNING NO CROSS-VALIDATION ---------------------------------------------------------------------------
    #main_tuningTFT(csv_filepath=csv_input, epochs=3, n_trials=3)
    #trainEvaluate_afterTuning_TFT(csv_filepath=csv_input, epochBase=10)

    # TRANSFORMER:TUNING ---------------------------------------------------------------------------
    #v3_mainTrans.main_tuningTransformer_kFold(csv_filepath=csv_input,  epochs=5, n_trials=5, num_split=2, df=df)
    #v3_mainTrans.trainEvaluate_afterTuning_Transformer(df=df, csv_filepath=csv_input, epochBase=2)

    # N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS N-HITS
    # N-HITS: BASEMODEL-------------------------------------------------------------------------------------------------
    #v3_mainN_HITS.BaseTrainNoEmbed(df=df, n_epoch=100, b_size=48, l_rate=0.01, h_unit=64, n_block=4, n_layer=4)
    #v3_mainN_HITS.BaseTrain(df=df, h_unit=128, n_epoch = 300, b_size = 24, l_rate = 0.005, n_block = 5, n_layer = 5,
    #                        embed_size = 100, dr_embed = 0.00, drop_act = 0.0, drop_loc = 0.25, drop_NOB = 0.25, w_act=1, w_loc=1, w_NOB=1,)

    # N-HITS: TUNING ---------------------------------------------------------------------------------------------------
    #v3_mainN_HITS.main_tuning_N_HITS_kFold(csv_filepath=csv_input,  epochs=2, n_trials=2, num_split=2, df=df)
    #v3_mainN_HITS.trainEvaluate_afterTuning_N_HITS(df=df, csv_filepath=csv_input, epochBase=5)

    # MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA MAMBA

    #VISUALIZATION: GENERAL_____________________________________________________________________________________________
    from preProcessing_Func import analysis_func as dfppaf
    input_Data= csv_input
    ID_DROP = ["Household_ID", 'Occupant_ID_in_HH']

    binary=False
    dfppaf.analysis(input_path=input_Data, missingness=binary)
    dfppaf.analysis(input_path=input_Data, columns=binary)
    dfppaf.analysis(input_path=input_Data, data_len=binary)
    dfppaf.analysis(input_path=input_Data, unique=binary, uniqueIDcolstoDrop=ID_DROP)
    #dfppaf.analysis(input_path=input_Data, headPrint=binary)
    #dfppaf.analysis(input_path=input_Data, describe=binary)
    #dfppaf.analysis(input_path=input_Data, unique_visual_byCols=binary, uniqueIDcolstoDrop=ID_DROP)