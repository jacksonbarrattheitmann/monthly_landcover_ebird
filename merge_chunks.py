import pandas as pd
from pathlib import Path

output_dir = Path("output")
chunks = sorted(output_dir.glob("hotspot_monthly_cover_chunk_*.csv"))
df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)
df.to_csv(output_dir / "hotspot_monthly_cover_final.csv", index=False)
print(f"Merged {len(chunks)} chunks → {len(df)} rows")