# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
# set pandas options to show all columns
pd.set_option('display.max_columns', None)

def sampling(input, output):
    # Use a breakpoint in the code line below to debug your script.
    df = pd.read_csv(input)
    #df = df.dropna()
    df = df.astype(float).round(2)
    print(df.columns)
    #path = df.sample(frac=0.25).to_csv(output)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #csv_path_in = r'dataset_TUS_daily/Giornaliero.csv'
    csv_path_in = r'dataset_TUS_daily/Giornaliero_imput_auto_afterkde.csv'

    csv_path_out = r'dataset_TUS_daily/Giornaliero_imput_auto_afterkde_sample.csv'
    sampling(csv_path_in, csv_path_out)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
