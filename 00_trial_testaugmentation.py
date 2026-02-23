import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast

def filter_by_household(input_csv, household_id, output_filename=None):
    """
    Reads the CSV file at `input_csv`, filters rows where the 'Household_ID'
    column matches `household_id`, and writes the result to a new CSV in the same folder.

    If `output_filename` is not provided, the output will be named
    'filtered_<household_id>.csv'.
    """
    # Determine directory of input file
    input_dir = os.path.dirname(os.path.abspath(input_csv))

    # Set default output filename if not provided
    if output_filename is None:
        output_filename = f"filtered_{household_id}.csv"

    # Full path for output in same folder
    output_csv = os.path.join(input_dir, output_filename)

    # Read the input CSV
    df = pd.read_csv(input_csv)
    print("Available columns:", df.columns.tolist())

    # Determine which column to use for filtering
    possible_columns = ['Household_ID', 'profam']
    filter_column = None
    for col in possible_columns:
        if col in df.columns:
            filter_column = col
            break

    if filter_column is None:
        print("None of the expected columns found. Please check column names.")
        return

    print(f"Filtering using column: {filter_column}")

    # Filter rows by the identified column
    filtered_df = df[df[filter_column] == household_id]

    # Save filtered rows to a new CSV in same folder
    filtered_df.to_csv(output_csv, index=False)
    print(filtered_df.head(5))

    print(f"Filtered data saved to: {output_csv}")

#VISUALIZATION----------------------------------------------------------------------------------------------------------
# — before any plotting —
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman']
# if you need math text in Times as well:
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm']      = 'Times New Roman'
#-----------------------------------------------------------------------------------------------------------------------
def load_data(csv_path, demo_csv_path, header_dict=None):
    """
    Load main data, demo data, labels, and category mappings.
    Returns:
      df, demo_df, highlight_combos,
      season_map, week_map, labels_df, category_mappings
    """
    # main classification data
    df = pd.read_csv(csv_path)

    # demo data
    demo_df = pd.read_csv(demo_csv_path)
    if header_dict:
        demo_df.rename(columns=header_dict, inplace=True)

    # parse category mappings from fixed CSV path
    cat_csv = r"dataset_CENTUS/00CategoriesFORDemographics.csv"
    cat_df = pd.read_csv(cat_csv)
    category_mappings = {}
    for entry in cat_df['Category']:
        col, mapping_str = entry.split('=', 1)
        col = col.strip()
        category_mappings[col] = ast.literal_eval(mapping_str.strip())

    # mappings for season/week names
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
    week_map   = {1: 'Weekdays', 2: 'Saturday', 3: 'Sunday'}

    # highlight combos based on demo
    highlight_combos = set(zip(demo_df['months_season'], demo_df['week_or_weekend']))

    # activity labels from fixed labels CSV
    base_dir = os.path.dirname(csv_path)
    labels_df = pd.read_csv(os.path.join(base_dir, '00Labels-Names-OccAct.csv'))

    # ensure location is categorical
    df['location'] = pd.Categorical(df['location'], categories=[0,1])

    return df, demo_df, highlight_combos, season_map, week_map, labels_df, category_mappings
def plot_single_subplot(ax, subset, season_map, week_map, combo, highlight_combos):
    """Plot co-presence and activity annotations on one Axes."""
    month, week = combo
    # highlight styling
    if combo in highlight_combos:
        for spine in ax.spines.values():
            spine.set_edgecolor('gold'); spine.set_linewidth(2)
        ax.set_facecolor((1,1,0,0.05))
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor('dimgray'); spine.set_linewidth(1)
        ax.set_facecolor('white')

    # sort by hour
    subset = subset.sort_values('hourEnd_Activity')
    hours  = subset['hourEnd_Activity'].values
    codes  = subset['location'].cat.codes * 0.5
    colors = subset['withNOBODY'].map({1:'coral',0:'dodgerblue'})
    #colors = subset['withALONE'].map({1: 'coral', 0: 'dodgerblue'})
    acts   = subset['Occupant_Activity'].values

    # line segments
    for i in range(len(hours)-1):
        ax.plot(hours[i:i+2], codes.values[i:i+2], color=colors.values[i], linewidth=3)
    # annotate activities
    for x,y,act,code in zip(hours, codes, acts, subset['location'].cat.codes):
        y_off, va = (y-0.03,'top') if code==0 else (y+0.02,'bottom')
        ax.text(x, y_off, str(act), ha='center', va=va,
                fontsize=9, fontweight='bold', fontstyle='italic',
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

    ax.set_title(f"Season: {season_map[month]}, {week_map[week]}", fontsize=12, loc='left')
    ax.set_xticks(sorted(hours))
    ax.set_yticks([0,0.5])
    ax.set_yticklabels(['outside','inside'], fontsize=10, rotation=90, va='center')
    ax.set_ylim(-0.15,0.65)
    ax.grid(axis='x', linestyle='--', linewidth=0.25, color='gray')
def add_legends(fig, df, demo_df, labels_df, category_mappings):
    """Add legends: co-presence, augmentation, activity, demographics."""
    # separate legends for co-presence and augmentation
    # co-presence legend
    patches_cp = [
        mpatches.Patch(color='dodgerblue', label='Alone'),
        mpatches.Patch(color='coral',      label='Accompanied')
    ]
    leg_cp = fig.legend(handles=patches_cp,
                        loc='upper left',
                        bbox_to_anchor=(0.025,0.99),
                        ncol=1,
                        title='Co-presence')
    fig.add_artist(leg_cp)

    # augmentation legend
    patches_aug = [
        mpatches.Patch(color='gold',  label='Existing TUS schedule'),
        mpatches.Patch(color='black', label='Augmented schedule(s)')
    ]
    leg_aug = fig.legend(handles=patches_aug,
                         loc='upper left',
                         bbox_to_anchor=(0.095,0.99),
                         ncol=1,
                         title='Augmentation')
    fig.add_artist(leg_aug)

    # occupant activity legend
    acts    = sorted(df['Occupant_Activity'].unique())
    act_map = labels_df.set_index('Category Label')['Category Name'].to_dict()
    patches_act = [
        mpatches.Patch(facecolor='white', edgecolor='black',
                       label=f"{act}: {act_map.get(act,'')}" )
        for act in acts
    ]
    leg_act = fig.legend(handles=patches_act,
                         loc='upper center',
                         bbox_to_anchor=(0.595,0.99),
                         ncol=9,
                         title='Occupant Activity')
    fig.add_artist(leg_act)

    # demographics per occupant
    ignore = {'hourStart_Activity', 'hourEnd_Activity',
              'months_season', 'week_or_weekend',
              'withALONE', 'withMOTHER', 'withFATHER', 'withSPOUSE', 'withCHILD',
              'withBROTHER', 'withOTHERFAMILYMEMBER', 'withOTHERPERSON', 'witness',
              'Occupant_Activity', "location", "withNOBODY"}
    demo_cols = [c for c in demo_df.columns if c not in ignore]
    patches_demo = []
    for col in demo_cols:
        mapping = category_mappings.get(col, {})
        for v in sorted(demo_df[col].dropna().unique()):
            label = mapping.get(v, str(v))
            patches_demo.append(
                mpatches.Patch(facecolor='white', edgecolor='black',
                               label=f"{col}: {label}")
            )
    fig.legend(handles=patches_demo,
               loc='lower center',
               bbox_to_anchor=(0.5,0),
               ncol=10,
               title='Demographics')
def visualize_predictions_by_combo_st9(csv_path,demo_csv_path,header_dict=None,occupant_ids=None):
    # load data
    df, demo_df, highlight_combos, season_map, week_map, labels_df, category_mappings = \
        load_data(csv_path, demo_csv_path, header_dict)

    # determine occupants
    all_occ = sorted(df['Occupant_ID_in_HH'].unique())
    if occupant_ids is None:
        occupant_ids = all_occ
    else:
        occupant_ids = [occ for occ in occupant_ids if occ in all_occ]

    # season/week combos
    combos = [(m, w)
              for m in sorted(df['months_season'].unique())
              for w in sorted(df['week_or_weekend'].unique())]

    for occ in occupant_ids:
        # subset for this occupant
        df_occ = df[df['Occupant_ID_in_HH'] == occ]
        demo_df_occ = demo_df[demo_df['Occupant_ID'] == occ]
        # compute occupant-specific highlight combos
        highlight_combos_occ = set(zip(demo_df_occ['months_season'], demo_df_occ['week_or_weekend']))

        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(24,11), sharex=True, sharey=True)
        fig.subplots_adjust(top=0.90, bottom=0.12)
        axes = axes.flatten()

        for ax, combo in zip(axes, combos):
            subset = df_occ[(df_occ['months_season']==combo[0]) & (df_occ['week_or_weekend']==combo[1])]
            plot_single_subplot(ax, subset, season_map, week_map, combo, highlight_combos_occ)

        #fig.suptitle(f"Occupant ID: {occ}", fontsize=16)
        add_legends(fig, df_occ, demo_df_occ, labels_df, category_mappings)
        plt.tight_layout(rect=[0,0.08,1,0.92])
        plt.show()

#-----------------------------------------------------------------------------------------------------------------------
def select_first_rows(input_csv, output_csv):
    """
    Reads `input_csv`, selects the first 100 rows, and writes them to `output_csv`.
    """
    df = pd.read_csv(input_csv, nrows=1000)
    df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    data_raw22_v2 = '/Users/orcunkoraliseri/PycharmProjects/thesis_OCCmodel/dataset_TUS_equalized/tus_main_EqPadHHID_RAWDATA_22_v2.csv'
    data_raw2 = '/Users/orcunkoraliseri/PycharmProjects/thesis_OCCmodel/dataset_raw/Giornaliero.csv'
    data_filtered22_v2 = '/Users/orcunkoraliseri/PycharmProjects/thesis_OCCmodel/dataset_raw/filtered_data.csv'
    data_filtered_raw = '/Users/orcunkoraliseri/PycharmProjects/thesis_OCCmodel/dataset_raw/filtered_data_raw.csv'
    data_22_v3 = "/Users/orcunkoraliseri/Desktop/ThesisFirstJournal/AI:DL/train_st7_LSTM_RAWDATA22_v3/tus_main_EqPadHHID_RAWDATA_22_v3_model_predictions_LSTM.csv"
    data_31_v4 = '/Users/orcunkoraliseri/PycharmProjects/thesis_OCCmodel/dataset_raw/tus_main_EqPadHHID_RAWDATA_31.csv'
    data_31_v4_predictions= "/Users/orcunkoraliseri/Desktop/tus_main_EqPadHHID_RAWDATA_31_v4_model_predictions_LSTM.csv"
    header_dict = {
        "Family_Typology_Simple": "SimpleFamilyType",
        "Employment status": "Employment",
        "Job Type": "JobType",
        "Economic Sector, Profession": "JobSector",
        "Family Typology": "FamilyType",
        "Full_Part_time": "Full/PartTime",
        "Permanent/fixed": "EmploymentType",
        "Education Degree": "Education",
        "Age Classes": "AgeClass",
        "Kinship Relationship": "Affiliation",
        "Mobile Phone Ownership": "PhoneOwnership",
        "Nuclear Family, Typology": "NuclearFamilyType",
        "Nuclear Family, Occupant Profile": "OccupantProfileFamily",
        "Nuclear Family, Occupant Sequence Number": "OccupantOrder",
        "Occupant_ID_in_HH": "Occupant_ID",
        "Internet Access": "InternetAccess",
        "Number Family Members": "#FamilyMembers",
        "Marital Status": "MaritalStatus",
        "Room Count": "RoomCount",
        "Car Ownership": "CarOwnership",
        "Home Ownership": "HomeOwnership",
        "House Area": "HouseArea",
    }

    # Example usage
    #filter_by_household(data_raw2, 7, data_filtered_raw)
    #filter_by_household(data_raw22_v2, household_id=15, output_filename=data_filtered22_v2)
    #visualize_predictions_by_combo_st9(csv_path=data_filtered22_v2,demo_csv_path=data_filtered22_v2, header_dict=header_dict,)

    select_first_rows(data_31_v4_predictions, r'dataset_raw/tus_main_EqPadHHID_RAWDATA_31_v4__predictions_sample.csv')
