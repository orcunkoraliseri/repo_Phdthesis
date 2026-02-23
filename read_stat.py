import pandas as pd
# Show all columns in output
pd.set_option('display.max_columns', None)
df_iter = pd.read_stata("~/Desktop/MTUS_hef2.dta", iterator=True)
canada_rows = []
chunk_size = 50000  # adjust based on your memory

while len(canada_rows) < 100:
    try:
        chunk = df_iter.get_chunk(chunk_size)
        canada_chunk = chunk[chunk['country'] == "Canada"]
        canada_rows.append(canada_chunk)
    except StopIteration:
        break

# Combine and keep only the first 100 rows
if canada_rows:
    result = pd.concat(canada_rows).head(100)
    print(result)
else:
    print("No rows found for Canada.")