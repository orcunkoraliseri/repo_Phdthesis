import uuid
import pathlib
import pandas as pd
def assemble_households(csv_file_path, target_year, output_dir, start_id=100):
    """
    Reads forecasted CSV, reconstructs households, saves the LINKED CSV.
    Uses simple sequential IDs (100, 101, 102...) for households.
    """
    print(f"\n--- Assembling Households for {target_year} ---")
    print(f"   Loading data from: {csv_file_path}")

    # 1. LOAD DATA
    df_population = pd.read_csv(csv_file_path)

    # Generate PIDs (Personal IDs) - We keep these as UUIDs or Random strings
    # to distinguish people within the house.
    df_population['PID'] = [str(uuid.uuid4())[:8] for _ in range(len(df_population))]

    # Ensure Types
    df_population['HHSIZE'] = pd.to_numeric(df_population['HHSIZE'], errors='coerce').fillna(1).astype(int)
    df_population['CF_RP'] = df_population['CF_RP'].astype(str).str.replace('.0', '', regex=False)

    final_households = []

    # --- INITIALIZE COUNTER ---
    current_hh_id = start_id

    # --- PHASE 1: SINGLES (HHSIZE = 1) ---
    singles_mask = df_population['HHSIZE'] == 1
    df_singles = df_population[singles_mask].copy()

    if not df_singles.empty:
        # Assign a range of IDs to singles all at once
        num_singles = len(df_singles)
        df_singles['SIM_HH_ID'] = range(current_hh_id, current_hh_id + num_singles)
        current_hh_id += num_singles  # Increment counter
        final_households.append(df_singles)

    print(f"   Processed {len(df_singles)} Single-Person Households.")

    # Remove singles from the pool
    df_remaining = df_population[~singles_mask].copy()

    # --- PHASE 2: FAMILIES (Heads = 1) ---
    df_family_heads = df_remaining[df_remaining['CF_RP'] == '1'].copy()
    df_members_2 = df_remaining[df_remaining['CF_RP'] == '2'].copy()
    df_members_3 = df_remaining[df_remaining['CF_RP'] == '3'].copy()

    pool_family_mem = df_members_2.sample(frac=1.0).to_dict('records')
    pool_non_family = df_members_3.sample(frac=1.0).to_dict('records')

    print(f"   Assembling {len(df_family_heads)} Family Households (Heads)...")

    family_batch = []
    for _, head_series in df_family_heads.iterrows():
        head = head_series.to_dict()

        # Assign Simple ID
        house_id = current_hh_id
        current_hh_id += 1

        head['SIM_HH_ID'] = house_id
        family_batch.append(head)

        slots_needed = head['HHSIZE'] - 1

        for _ in range(slots_needed):
            if pool_family_mem:
                member = pool_family_mem.pop()
            elif pool_non_family:
                member = pool_non_family.pop()
            else:
                # Clone fallback
                if not df_members_2.empty:
                    member = df_members_2.sample(1).to_dict('records')[0]
                else:
                    member = df_remaining.sample(1).to_dict('records')[0]
                member['PID'] = str(uuid.uuid4())[:8]

            member['SIM_HH_ID'] = house_id
            family_batch.append(member)

    if family_batch:
        final_households.append(pd.DataFrame(family_batch))

    # --- PHASE 3: ROOMMATES (Leftover CF_RP 3s) ---
    leftover_roommates = pd.DataFrame(pool_non_family)

    if not leftover_roommates.empty:
        print(f"   Assembling {len(leftover_roommates)} Roommate/Non-Family Agents...")

        for size in sorted(leftover_roommates['HHSIZE'].unique()):
            if size == 1: continue

            mates_of_size = leftover_roommates[leftover_roommates['HHSIZE'] == size]
            mate_list = mates_of_size.to_dict('records')

            roommate_batch = []
            while mate_list:
                head = mate_list.pop()

                # Assign Simple ID
                house_id = current_hh_id
                current_hh_id += 1

                head['SIM_HH_ID'] = house_id
                roommate_batch.append(head)

                slots_needed = size - 1
                for _ in range(slots_needed):
                    if mate_list:
                        member = mate_list.pop()
                    else:
                        member = mates_of_size.sample(1).to_dict('records')[0]
                        member['PID'] = str(uuid.uuid4())[:8]

                    member['SIM_HH_ID'] = house_id
                    roommate_batch.append(member)

            if roommate_batch:
                final_households.append(pd.DataFrame(roommate_batch))

    # --- Combine All ---
    if final_households:
        df_assembled = pd.concat(final_households, ignore_index=True)
    else:
        df_assembled = pd.DataFrame()

    print(f"--- Assembly Complete. Last ID used: {current_hh_id - 1} ---")

    # Final Validation
    if not df_assembled.empty:
        size_counts = df_assembled.groupby('SIM_HH_ID')['PID'].count()
        target_sizes = df_assembled.groupby('SIM_HH_ID')['HHSIZE'].first()
        mismatches = size_counts != target_sizes
        if mismatches.any():
            print(f"⚠️ WARNING: {mismatches.sum()} households have mismatched sizes!")
        else:
            print("✅ VALIDATION SUCCESS: All households have correct member counts.")

    # --- SAVE TO CSV ---
    save_filename = f"{target_year}_LINKED.csv"
    save_path = pathlib.Path(output_dir) / save_filename
    df_assembled.to_csv(save_path, index=False)
    print(f"✅ Saved linked {target_year} data to: {save_path}")

    return df_assembled

if __name__ == '__main__':
    BASE_DIR = pathlib.Path("/Users/orcunkoraliseri/Desktop/Postdoc/eSim/Occupancy")

    DATA_DIR = BASE_DIR / "DataSources_CENSUS"
    OUTPUT_DIR = BASE_DIR / "Outputs_CENSUS"
    OUTPUT_DIR_ALIGNED = BASE_DIR / "Outputs_Aligned"
    OUTPUT_DIR_ALIGNED.mkdir(parents=True, exist_ok=True)

    # --- Edited Files ---
    cen06_filtered2 = OUTPUT_DIR / "cen06_filtered2.csv"
    cen11_filtered2 = OUTPUT_DIR / "cen11_filtered2.csv"
    cen16_filtered2 = OUTPUT_DIR / "cen16_filtered2.csv"
    cen21_filtered2 = OUTPUT_DIR / "cen21_filtered2.csv"

    #ASSEMBLE HOUSEHOLD ------------------------------------------------------------------------------------------------
    #df_linked_2006 = assemble_households(cen06_filtered2, target_year=2006, output_dir=OUTPUT_DIR)
    df_linked_2016 = assemble_households(cen16_filtered2, target_year=2016, output_dir=OUTPUT_DIR)

    #PROFILE MATCHER ---------------------------------------------------------------------------------------------------



