
import pandas as pd
import numpy as np

# STEP 1: Load uncleaned CSV file
df = pd.read_csv("smart_energy_consumption_uncleaned (1).csv")

print("Original Dataset Shape:", df.shape)

# STEP 2: Remove duplicate records
df.drop_duplicates(subset=["record_id"], keep="first", inplace=True)

# STEP 3: Remove extra spaces from text columns
text_cols = [
    "building_id", "building_name",
    "building_type", "zone", "weather"
]

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# STEP 4: Correct inconsistent text values
for col in ["building_type", "weather", "zone"]:
    df[col] = df[col].str.lower().str.strip()

    df[col] = df[col].replace({
        "education": "Education",
        "office": "Office",
        "laboratory": "Laboratory",
        "residential": "Residential",
        "recreation": "Recreation",
        "food service": "Food Service",
        "clear": "Clear",
        "cloudy": "Cloudy",
        "rainy": "Rainy",
        "humid": "Humid",
        "north": "North",
        "south": "South",
        "east": "East",
        "west": "West"
    })

# STEP 5: Convert timestamp into datetime
df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

# STEP 6: Convert numerical columns into numeric format
numeric_cols = [
    "occupancy_pct",
    "temperature_c",
    "humidity_pct",
    "energy_kwh",
    "peak_demand_kw",
    "solar_generation_kwh",
    "tariff_inr_per_kwh",
    "carbon_emission_kg"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col].astype("string")
        .str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce"
    )

# STEP 7: Handle invalid values
valid_ranges = {
    "occupancy_pct": (0, 100),
    "temperature_c": (-10, 60),
    "humidity_pct": (0, 100),
    "energy_kwh": (0, 500),
    "peak_demand_kw": (0, 500),
    "solar_generation_kwh": (0, 100),
    "tariff_inr_per_kwh": (0, 30),
    "carbon_emission_kg": (0, 500)
}

for col, (low, high) in valid_ranges.items():
    df.loc[~df[col].between(low, high), col] = np.nan

# STEP 8: Handle missing numerical values
for col in numeric_cols:
    df[col] = df[col].fillna(
        df.groupby("building_type")[col].transform("median")
    )
    df[col] = df[col].fillna(df[col].median())

# STEP 9: Handle missing categorical values
categorical_cols = [
    "building_id", "building_name",
    "building_type", "zone", "weather"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# STEP 10: Remove rows with missing essential information
df.dropna(subset=["record_id", "timestamp"], inplace=True)

# STEP 11: Recalculate carbon emissions
df["carbon_emission_kg"] = (
    df["energy_kwh"] * 0.75
).round(2)

# STEP 12: Sort the dataset
df.sort_values(
    by=["building_id", "timestamp"],
    inplace=True
)

df.reset_index(drop=True, inplace=True)

# STEP 13: Save cleaned CSV file
df.to_csv("smart_energy_consumption_cleaned.csv", index=False)

# STEP 14: Final verification
print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")
print("Final Dataset Shape:", df.shape)
print("Duplicate Records:", df["record_id"].duplicated().sum())
print("Missing Values:", df.isnull().sum().sum())

print("\nCleaned Dataset Preview:")
print(df.head())

print("\nCleaned CSV file saved successfully!")
