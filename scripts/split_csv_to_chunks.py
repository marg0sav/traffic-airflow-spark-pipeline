from pathlib import Path

import pandas as pd


INPUT_FILE = Path("traffic_measurements_history_5_months_virtual_sensors.csv")
OUTPUT_DIR = Path("chunks")

CHUNK_ROWS = 500_000


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    for old_file in OUTPUT_DIR.glob("traffic_part_*.csv"):
        old_file.unlink()

    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_ROWS)

    for index, chunk in enumerate(reader, start=1):
        output_file = OUTPUT_DIR / f"traffic_part_{index:03d}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Saved {output_file} rows={len(chunk)}")

    print("Finished splitting")


if __name__ == "__main__":
    main()