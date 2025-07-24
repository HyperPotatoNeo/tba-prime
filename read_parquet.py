import pyarrow.parquet as pq
import pandas as pd

pd.set_option('display.max_rows', None)

table = pq.read_table("rollouts/step_0/2aa84d37-2882-4d7e-a669-4a09038254e8.parquet")
df = table.to_pandas()  # Convert to a pandas DataFrame if needed
for col in df.columns:
    print(f'{col} | {df[col][0]}')
    print()