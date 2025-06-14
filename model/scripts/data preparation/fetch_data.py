# scripts/fetch_data.py
import os
import pandas as pd

# Define paths
RAW_DIR = "/data/external_sources"
os.makedirs(RAW_DIR, exist_ok=True)

# Dataset metadata
DATASETS = {
    "phonepe_pulse.csv": {
        "url": None,  # Manual download from https://www.phonepe.com/pulse/
        "expected_columns": ["District", "upi_txn_volume"]
    },
    "census_district.csv": {
        "url": None,  # Manual download from https://censusindia.gov.in/
        "expected_columns": ["District name", "Female_Literate"]
    },
    "census_consolidated.csv": {
        "url": None,  # Manual download from https://data.gov.in/
        "expected_columns": ["DISTRICT NAME", "TOTAL POULATION"]
    },
    "sector_district.csv": {
        "url": None,  # Manual download from https://data.gov.in/
        "expected_columns": [
            "Dist Code",
            "Per Capita Current Prices (1000 in Rs)"
        ]
    },
    "mgnrega_district.csv": {
        "url": None,  # Manual download from https://nrega.nic.in/
        "expected_columns": ["district_name", "Average days of employment provided per Household"]
    },
    "mission_antyodaya.csv": {
        "url": None,  # Manual download from https://missionantyodaya.nic.in/
        "expected_columns": ["district_name", "is_village_connected_to_all_weather_road"]
    },
    "income_expenditure.csv": {
        "url": None,  # Manual download from https://mospi.gov.in/
        "expected_columns": [
            "Mthly_HH_Income",
            "Mthly_HH_Expense",
            "No_of_Fly_Members",
            "Emi_or_Rent_Amt",
            "Annual_HH_Income",
            "Highest_Qualified_Member",
            "No_of_Earning_Members"
        ]
    }
}

def verify_file(filepath, expected_columns):
    """Verify if file exists and has required columns."""
    if not os.path.exists(filepath):
        print(f"❌ {filepath} not found. Please download and place in {RAW_DIR}")
        return False
    df = pd.read_csv(filepath, nrows=5)
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️ {filepath} missing columns: {missing_cols}")
        return False
    print(f"✅ {filepath} verified")
    return True

def main():
    print("📥 Verifying datasets in", RAW_DIR)
    all_verified = True
    for filename, meta in DATASETS.items():
        filepath = os.path.join(RAW_DIR, filename)
        if meta["url"]:
            print(f"📡 Auto-download for {filename} not supported yet.")
        if not verify_file(filepath, meta["expected_columns"]):
            all_verified = False
            print(f"ℹ️ Please download {filename} from {meta['url'] or 'specified source'}")

    if all_verified:
        print("🎉 All datasets verified successfully!")
    else:
        print("❗ Some datasets need to be downloaded or fixed.")

if __name__ == "__main__":
    main()
