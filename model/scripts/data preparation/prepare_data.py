import os
import pandas as pd
import numpy as np

# Directories
RAW_DIR = "D:\Python\Credit_UnderWriting_Model\data\processed"
os.makedirs(RAW_DIR, exist_ok=True)

# Load actual districts and states from census data
df_census = pd.read_csv(os.path.join(RAW_DIR, "D:\Python\Credit_UnderWriting_Model\data\external_sources\Districts.csv"))
# Standardize district and state columns
if 'District'  in df_census.columns:
    df_census.rename(columns={'District': 'district'}, inplace=True)
elif 'district' not in df_census.columns:
    raise KeyError("Expected a 'District name' or 'district' column in census data")

state_column = next((c for c in ['State name', 'State', 'state'] if c in df_census.columns), None)
if not state_column:
    raise KeyError("Expected a 'State name', 'State', or 'state' column in census data")
df_census.rename(columns={state_column: 'state'}, inplace=True)

# Unique district-state list
df_districts = df_census[['district', 'state']].drop_duplicates().reset_index(drop=True)

# Feature lists
sectors = ["Agriculture", "Manufacturing", "Services", "Construction", "IT and Security", "Food", "LifeStyle", "Medical"]
area_types = ["Rural", "Urban"]

# Generate district-level features


# income = gdp*500 + litrarty * 0.95 + 40 + upi tnx /2 + noise( paramter )
def generate_district_features(df):
    np.random.seed(2025)
    df_feat = df.copy()
    n = len(df_feat)
    df_feat['work_sector'] = np.random.choice(sectors, size=n)
    df_feat['area_type'] = np.random.choice(area_types, size=n, p=[0.6, 0.4])
    df_feat['distance_from_city'] = np.round(np.random.uniform(5, 200, size=n), 1)
    df_feat['gdp_per_capita'] = np.round(np.random.normal(loc=80, scale=20, size=n), 2)
    df_feat['literacy_rate_male'] = np.round(np.random.uniform(70, 100, size=n), 2)
    df_feat['literacy_rate_female'] = np.round(np.random.uniform(60, 95, size=n), 2)
    df_feat['avg_upi_txn_volume'] = np.round(np.random.normal(loc=50, scale=15, size=n), 2)
    # Clip values
    df_feat['gdp_per_capita'] = df_feat['gdp_per_capita'].clip(lower=5)
    df_feat['avg_upi_txn_volume'] = df_feat['avg_upi_txn_volume'].clip(lower=0)
    return df_feat

# Save district features
print("🛠️ Generating synthetic district features...")
df_district_features = generate_district_features(df_districts)
df_district_features.to_csv(os.path.join(RAW_DIR, "synthetic_district_features.csv"), index=False)
print(f"✅ {len(df_district_features)} districts saved.")

# Generate individual records

def generate_person_records(df_district, num_individuals=1000):
    np.random.seed(2025)
    districts = df_district['district'].values
    sel = np.random.choice(districts, size=num_individuals)
    df_person = pd.DataFrame({'district': sel})
    df_person = df_person.merge(df_district, on='district', how='left')
    num_individuals = len(df_person)
    df_person['gender'] = np.random.choice(['Male', 'Female'], size=num_individuals)

    # Compute predicted_income
    df_person['predicted_income'] = (
            df_person['gdp_per_capita'] * 1000 * 0.3 +  # Slightly reduced weight
            ((df_person['literacy_rate_male'] + df_person[
                'literacy_rate_female']) / 2) * 10 +  # Increased literacy impact but less than before
            np.sqrt(df_person['avg_upi_txn_volume']) * 1500 +  # Use sqrt to reduce impact of very high txn volumes
            np.random.normal(0, 15000, size=num_individuals)  # Reduced noise
    ).round(2)

    df_person['predicted_income'] = df_person['predicted_income'].clip(lower=5000, upper=200000)

    return df_person

print("🛠️ Generating synthetic person data for stage 1...")
_df_person = generate_person_records(df_district_features, num_individuals=1000)
_df_person.to_csv(os.path.join(RAW_DIR, "synthetic_person_data.csv"), index=False)
print(f"✅ {_df_person.shape[0]} records saved.")

