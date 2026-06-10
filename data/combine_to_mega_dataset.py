import pandas as pd
import numpy as np
import os
# print(os.path.dirname(os.path.abspath(__file__)))

EUROGATE_CSV = "data/eurogate_container_history_weather.csv"
INDONESIA_CSV = "data/clean_and_viz_weather.csv"
OUTPUT_CSV = "data/mega_container_dwell_dataset.csv"

euro = pd.read_csv(EUROGATE_CSV)
indo = pd.read_csv(INDONESIA_CSV)

# start with working on the indonesia dataset
indo_std = indo.copy()
indo_std = indo_std.rename(columns={
    "case_id": "container_id",
    "CTR_SIZE": "container_size",
    "GROSS": "gross_weight",
    "YARD_SLOT": "yard_slot",
    "YARD_BLOCK_enc": "yard_block_enc",
})

indo_std["port"] = "Tanjung Perak"
# indo_std["source_dataset"] = "clean_and_viz_weather"

# CTR_TYPE_RFR and reefer are the same signal; keep a single reefer column
if "CTR_TYPE_RFR" in indo_std.columns:
    indo_std["reefer"] = indo_std["CTR_TYPE_RFR"]
else:
    indo_std["reefer"] = np.nan
indo_std["released"] = np.nan
indo_std["arrival_time"] = indo_std["weather_match_time"]
indo_std["departure_time"] = np.nan
indo_std["arrival_voyage_eta"] = np.nan
indo_std["vessel_eta_delay_hours"] = np.nan
indo_std["arrival_type"] = np.nan
indo_std["departure_type"] = np.nan
indo_std["origin_transport_code"] = np.nan
indo_std["arrival_pol_code"] = np.nan
indo_std["line_code"] = np.nan

# now work on the eurogate dataset, standardize it
euro_std = euro.copy()
euro_std = euro_std.rename(columns={
    "containerId": "container_id",
    "gross": "gross_weight",
    "arrivalTime": "arrival_time",
    "departureTime": "departure_time",
    "arrivalVoyageEta": "arrival_voyage_eta",
    "lineCode": "line_code",
    "arrivalType": "arrival_type",
    "departureType": "departure_type",
    "originOfTransportCode": "origin_transport_code",
    "arrivalPolCode": "arrival_pol_code",
})

euro_std["port"] = "Eurogate Hamburg"
# euro_std["source_dataset"] = "eurogate_container_history_weather"

# teu -> approximate container size
euro_std["container_size"] = np.where(
    euro_std["teu"] == 2, 40,
    np.where(euro_std["teu"] == 1, 20, np.nan)
)

euro_std["yard_slot"] = np.nan
euro_std["yard_block_enc"] = np.nan

# eurogate doesnt have all of the indonesia event/process columns so just nan
euro_std["n_events"] = euro_std.get("n_snapshots", np.nan)
euro_std["n_unique_activities"] = np.nan
euro_std["n_unique_roles"] = np.nan
euro_std["hours_atb_to_discharge"] = np.nan
euro_std["flag_quarantine"] = np.nan
euro_std["flag_customs"] = np.nan
euro_std["flag_damage"] = np.nan

# Eurogate container type approximations
euro_std["CTR_TYPE_DRY"] = np.nan
euro_std["CTR_TYPE_FLT"] = np.nan
euro_std["CTR_TYPE_O/T"] = np.nan
euro_std["CTR_TYPE_OVD"] = np.nan
euro_std["CTR_TYPE_TNK"] = np.where(euro_std["typeCode"].astype(str).str.upper().eq("TK"), 1, 0)

# Eurogate has no Indonesian customs document columns
for col in [
    "JOB_DEL_DOCTYPE_BC23",
    "JOB_DEL_DOCTYPE_BCF26",
    "JOB_DEL_DOCTYPE_LAIN2",
    "JOB_DEL_DOCTYPE_LELANG",
    "JOB_DEL_DOCTYPE_NNMITA",
    "JOB_DEL_DOCTYPE_PLP",
    "JOB_DEL_DOCTYPE_SPPB",
    "JOB_DEL_DOCTYPE_TMBLU",
]:
    euro_std[col] = np.nan

# if container IDs match between ports then one of them should change
# just change the eurogate container ID to have a prefix
overlap_ids = set(indo_std["container_id"].astype(str)).intersection(
    set(euro_std["container_id"].astype(str))
)
euro_std["container_id"] = euro_std["container_id"].astype(str).apply(
    lambda x: f"EUROGATE_{x}" if x in overlap_ids else x
)
indo_std["container_id"] = indo_std["container_id"].astype(str)
print(f"Overlapping container IDs renamed in Eurogate: {len(overlap_ids)}")

# mega data cols
final_cols = [
    "container_id",
    "port",
    # "source_dataset",

    # target!!
    "dwell_hours",

    "container_size",
    "gross_weight",
    "reefer",
    "arrival_time",
    "departure_time",
    "arrival_voyage_eta",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",

    # indonesian 
    "n_events",
    "n_unique_activities",
    "n_unique_roles",
    "hours_atb_to_discharge",
    "flag_quarantine",
    "flag_customs",
    "flag_damage",
    "yard_slot",
    "yard_block_enc",
    "CTR_TYPE_DRY",
    "CTR_TYPE_FLT",
    "CTR_TYPE_O/T",
    "CTR_TYPE_OVD",
    "CTR_TYPE_TNK",
    "JOB_DEL_DOCTYPE_BC23",
    "JOB_DEL_DOCTYPE_BCF26",
    "JOB_DEL_DOCTYPE_LAIN2",
    "JOB_DEL_DOCTYPE_LELANG",
    "JOB_DEL_DOCTYPE_NNMITA",
    "JOB_DEL_DOCTYPE_PLP",
    "JOB_DEL_DOCTYPE_SPPB",
    "JOB_DEL_DOCTYPE_TMBLU",

    # euro
    "first_seen",
    "last_seen",
    "n_snapshots",
    "sizetypeIsoCode",
    "typeCode",
    "line_code",
    "arrival_type",
    "departure_type",
    "origin_transport_code",
    "arrival_pol_code",
    "released",
    "n_unique_locations_x",
    "n_unique_locations_y",
    "n_unique_locations_z",
    "observed_days",
    "vessel_eta_delay_hours",
    "n_observed_location_changes",
    "n_unique_locations",
]

# add missing columns to each dataset
for col in final_cols:
    if col not in indo_std.columns:
        indo_std[col] = np.nan
    if col not in euro_std.columns:
        euro_std[col] = np.nan

indo_std = indo_std[final_cols]
euro_std = euro_std[final_cols]

# Combine
mega = pd.concat([indo_std, euro_std], ignore_index=True)

# Clean booleans -> encode to 1/0
bool_cols = [
    "reefer",
    "CTR_TYPE_DRY",
    "CTR_TYPE_FLT",
    "CTR_TYPE_O/T",
    "CTR_TYPE_OVD",
    "CTR_TYPE_TNK",
]

for col in bool_cols:
    if col in mega.columns:
        mega[col] = mega[col].replace({
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            "Y": 1,
            "N": 0
        })

# make sure dwell hours is numeric (it should be anyways but this is just a double check)
mega["dwell_hours"] = pd.to_numeric(mega["dwell_hours"], errors="coerce")

rows_before = len(mega)
mega = mega[mega["dwell_hours"].notna()].copy()
print(f"Dropped {rows_before - len(mega):,} rows with missing dwell_hours")

mega.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Shape:", mega.shape)

print("\nRows by port:")
print(mega["port"].value_counts(dropna=False))

print("\nDwell hours summary by port:")
print(mega.groupby("port")["dwell_hours"].describe())

print("\nTop missing columns:")
print(mega.isna().mean().sort_values(ascending=False).head(25))

mega.head()