"""Feature engineering for the L&T Finance Farmer Income Prediction Challenge."""
import re
import numpy as np
import pandas as pd

TARGET = "Target_Variable/Total Income"
ID = "FarmerID"

WATER_TYPES = ["water", "river", "reservoir", "riverbank", "wetland"]

# Low-cardinality categoricals -> native LightGBM categorical
LOW_CARD_CATS = [
    "State", "REGION", "SEX", "MARITAL_STATUS", "Address type", "Ownership",
    "K022-Village category based on Agri parameters (Good, Average, Poor)",
    "K022-Village category based on socio-economic parameters (Good, Average, Poor)",
    "R022-Village category based on Agri parameters (Good, Average, Poor)",
    " Village category based on socio-economic parameters (Good, Average, Poor)",
    "Kharif Seasons  Type of soil in 2022",
    "Rabi Seasons Type of soil in 2022",
    "Kharif Seasons  Agro Ecological Sub Zone in 2022",
    "Rabi Seasons Agro Ecological Sub Zone in 2022",
]

# High-cardinality -> out-of-fold target encoding (plus a label-encoded copy)
HIGH_CARD_CATS = ["DISTRICT", "CITY", "VILLAGE", "Zipcode", "K022-Nearest Mandi Name"]

TEMP_COLS = [c for c in [
    "K022-Ambient temperature (min & max)", "R022-Ambient temperature (min & max)",
    "K021-Ambient temperature (min & max)", "R021-Ambient temperature (min & max)",
    "R020-Ambient temperature (min & max)",
]]

RAIN_COLS = [
    "K022-Seasonal Average Rainfall (mm)", "R022-Seasonal Average Rainfall (mm)",
    "K021-Seasonal Average Rainfall (mm)", "R021-Seasonal Average Rainfall (mm)",
    "R020-Seasonal Average Rainfall (mm)",
]

# State-level economic-prosperity tier (1=lowest, 5=highest), sourced from
# RBI "Handbook of Statistics on Indian Economy", Table 10 (Per Capita Net
# State Domestic Product, constant prices, base 1999-2000, Aug 2025 release)
# 2021-22 figures for 13 of the 17 states present in this dataset:
#   Bihar 27,674 / Jharkhand 57,172 / Madhya Pradesh 61,011 / Assam 63,657 /
#   Chhattisgarh 78,350 / Rajasthan 79,490 / Odisha 85,516 / Andhra Pradesh
#   118,349 / Punjab 118,307 / Maharashtra 141,651 / Karnataka 165,517 /
#   Haryana 166,873 / Gujarat 170,519.
# Telangana, Uttar Pradesh, West Bengal and Chandigarh are not in that same
# table vintage; they are placed by well-corroborated general standing
# (multiple independent sources agree UP is bottom-tier, Telangana and
# Chandigarh are top-tier, West Bengal is mid-tier) rather than an
# interpolated number, to avoid mixing incompatible price bases.
STATE_INCOME_TIER = {
    "BIHAR": 1, "UTTAR PRADESH": 1, "JHARKHAND": 1,
    "MADHYA PRADESH": 2, "ASSAM": 2, "WEST BENGAL": 2, "CHATTISGARH": 2,
    "RAJASTHAN": 3, "ODISHA": 3, "ANDHRA PRADESH": 3, "PUNJAB": 3,
    "MAHARASHTRA": 4, "KARNATAKA": 4,
    "HARYANA": 5, "GUJARAT": 5, "TELANGANA": 5, "CHANDIGARH": 5,
}

# State capital (or de facto commercial capital) coordinates, for a
# distance-to-capital market-access proxy. Public knowledge, no external
# fetch needed.
STATE_CAPITALS = {
    "MADHYA PRADESH": (23.2599, 77.4126),   # Bhopal
    "MAHARASHTRA": (19.0760, 72.8777),      # Mumbai
    "TELANGANA": (17.3850, 78.4867),        # Hyderabad
    "KARNATAKA": (12.9716, 77.5946),        # Bengaluru
    "ANDHRA PRADESH": (16.5062, 80.6480),   # Vijayawada/Amaravati area
    "UTTAR PRADESH": (26.8467, 80.9462),    # Lucknow
    "BIHAR": (25.5941, 85.1376),            # Patna
    "HARYANA": (30.7333, 76.7794),          # Chandigarh (shared)
    "ODISHA": (20.2961, 85.8245),           # Bhubaneswar
    "CHATTISGARH": (21.2514, 81.6296),      # Raipur
    "WEST BENGAL": (22.5726, 88.3639),      # Kolkata
    "GUJARAT": (23.2156, 72.6369),          # Gandhinagar
    "PUNJAB": (30.7333, 76.7794),           # Chandigarh (shared)
    "JHARKHAND": (23.3441, 85.3096),        # Ranchi
    "ASSAM": (26.1445, 91.7362),            # Guwahati/Dispur
    "CHANDIGARH": (30.7333, 76.7794),
    "RAJASTHAN": (26.9124, 75.7873),        # Jaipur
}

# A handful of major national metros, for a nearest-large-market proxy
# independent of state boundaries.
MAJOR_METROS = [
    (28.7041, 77.1025),   # Delhi
    (19.0760, 72.8777),   # Mumbai
    (22.5726, 88.3639),   # Kolkata
    (13.0827, 80.2707),   # Chennai
    (12.9716, 77.5946),   # Bengaluru
    (17.3850, 78.4867),   # Hyderabad
    (23.0225, 72.5714),   # Ahmedabad
    (18.5204, 73.8567),   # Pune
]


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _split_temp(s: pd.Series):
    """'23.34 /30.33' -> (23.34, 30.33)."""
    ext = s.astype(str).str.extract(r"(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)")
    return pd.to_numeric(ext[0], errors="coerce"), pd.to_numeric(ext[1], errors="coerce")


def _split_latlon(s: pd.Series):
    ext = s.astype(str).str.extract(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)")
    lat = pd.to_numeric(ext[0], errors="coerce")
    lon = pd.to_numeric(ext[1], errors="coerce")
    # guard against swapped / out-of-India coordinates
    bad = ~(lat.between(6, 38) & lon.between(67, 98))
    return lat.mask(bad), lon.mask(bad)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # ---- raw numeric passthrough -------------------------------------------
    skip = {TARGET, ID, "Location"} | set(LOW_CARD_CATS) | set(HIGH_CARD_CATS) | set(TEMP_COLS)
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if s.dtype.kind in "ifb":
            X[c] = s.astype(float)

    # ---- temperature parsing ------------------------------------------------
    for c in TEMP_COLS:
        if c not in df.columns:
            continue
        tag = c.split("-")[0]
        lo, hi = _split_temp(df[c])
        X[f"{tag}_tmin"], X[f"{tag}_tmax"] = lo, hi
        X[f"{tag}_trange"] = hi - lo
        X[f"{tag}_tmean"] = (hi + lo) / 2

    # ---- geolocation --------------------------------------------------------
    if "Location" in df.columns:
        lat, lon = _split_latlon(df["Location"])
        X["lat"], X["lon"] = lat, lon
        X["latlon_sum"] = lat + lon
        X["latlon_diff"] = lat - lon
        X["has_location"] = df["Location"].notna().astype(int)

        # distance to own state's capital (market-access / remoteness proxy)
        state_upper = df["State"].astype(str).str.upper().str.strip()
        cap_lat = state_upper.map(lambda s: STATE_CAPITALS.get(s, (np.nan, np.nan))[0])
        cap_lon = state_upper.map(lambda s: STATE_CAPITALS.get(s, (np.nan, np.nan))[1])
        X["dist_to_state_capital_km"] = _haversine_km(lat, lon, cap_lat, cap_lon)

        # distance to nearest major national metro, independent of state
        metro_dists = np.vstack([_haversine_km(lat, lon, mlat, mlon) for mlat, mlon in MAJOR_METROS])
        X["dist_to_nearest_metro_km"] = np.nanmin(metro_dists, axis=0)
        X["log_dist_to_metro"] = np.log1p(X["dist_to_nearest_metro_km"])
        X["log_dist_to_capital"] = np.log1p(X["dist_to_state_capital_km"])

    # ---- water bodies -> multi-hot -----------------------------------------
    for col in [c for c in df.columns if "Type of water bodies" in c]:
        year = re.search(r"(20\d\d)", col).group(1)
        season = "K" if col.strip().startswith("Kharif") else "R"
        txt = df[col].astype(str).str.lower()
        for w in WATER_TYPES:
            X[f"{season}{year}_wb_{w}"] = txt.str.contains(w, regex=False).astype(int)
        X[f"{season}{year}_wb_n"] = sum(X[f"{season}{year}_wb_{w}"] for w in WATER_TYPES)

    # ---- soil / agro-zone for non-2022 years (as codes, cheap) --------------
    for col in df.columns:
        if ("Type of soil" in col or "Agro Ecological Sub Zone" in col) and "2022" not in col:
            X[f"code_{col.strip()[:40]}"] = df[col].astype("category").cat.codes.astype(float)

    # ---- domain-derived features -------------------------------------------
    land = pd.to_numeric(df["Total_Land_For_Agriculture"], errors="coerce")
    nonagri = pd.to_numeric(df["Non_Agriculture_Income"], errors="coerce").fillna(0)
    X["log_land"] = np.log1p(land)
    X["log_nonagri"] = np.log1p(nonagri)
    X["has_nonagri"] = (nonagri > 0).astype(int)

    bureau_amt = pd.to_numeric(df["Avg_Disbursement_Amount_Bureau"], errors="coerce")
    nloans = pd.to_numeric(df["No_of_Active_Loan_In_Bureau"], errors="coerce")
    X["log_bureau_amt"] = np.log1p(bureau_amt)
    X["has_bureau"] = bureau_amt.notna().astype(int)
    X["bureau_total_exposure"] = np.log1p(bureau_amt.fillna(0) * nloans.fillna(0))
    X["bureau_amt_per_land"] = bureau_amt / land.replace(0, np.nan)
    X["nonagri_per_land"] = nonagri / land.replace(0, np.nan)

    # village-level land pressure
    net_agri = pd.to_numeric(df["K022-Net Agri area (in Ha)-"], errors="coerce")
    geo_area = pd.to_numeric(df["K022-Total Geographical Area (in Hectares)-"], errors="coerce")
    lhi = pd.to_numeric(df[" Land Holding Index source (Total Agri Area/ no of people)"], errors="coerce")
    X["land_vs_village_avg"] = land / lhi.replace(0, np.nan)
    X["village_pop_proxy"] = net_agri / lhi.replace(0, np.nan)
    X["agri_area_share"] = net_agri / geo_area.replace(0, np.nan)
    X["log_net_agri"] = np.log1p(net_agri)
    X["log_geo_area"] = np.log1p(geo_area)

    # rainfall aggregates and volatility
    rain = df[[c for c in RAIN_COLS if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    X["rain_mean"] = rain.mean(axis=1)
    X["rain_std"] = rain.std(axis=1)
    X["rain_cv"] = X["rain_std"] / X["rain_mean"].replace(0, np.nan)
    if {"K022-Seasonal Average Rainfall (mm)", "K021-Seasonal Average Rainfall (mm)"} <= set(df.columns):
        k22 = pd.to_numeric(df["K022-Seasonal Average Rainfall (mm)"], errors="coerce")
        k21 = pd.to_numeric(df["K021-Seasonal Average Rainfall (mm)"], errors="coerce")
        X["kharif_rain_trend"] = k22 - k21
        X["kharif_rain_ratio"] = k22 / k21.replace(0, np.nan)

    # agricultural score trajectories
    for season in ["Kharif", "Rabi"]:
        sc = [c for c in df.columns if c.startswith(season) and "Agricultural Score" in c]
        if len(sc) >= 2:
            blk = df[sc].apply(pd.to_numeric, errors="coerce")
            X[f"{season}_score_mean"] = blk.mean(axis=1)
            X[f"{season}_score_std"] = blk.std(axis=1)
            c22 = [c for c in sc if "2022" in c]
            c20 = [c for c in sc if "2020" in c]
            if c22 and c20:
                X[f"{season}_score_trend"] = blk[c22[0]] - blk[c20[0]]
        irr = [c for c in df.columns if c.startswith(season) and "Irrigated area" in c]
        if irr:
            blk = df[irr].apply(pd.to_numeric, errors="coerce")
            X[f"{season}_irrig_mean"] = blk.mean(axis=1)
            X[f"{season}_irrig_std"] = blk.std(axis=1)

    # groundwater
    gw_t = [c for c in df.columns if "groundwater thickness" in c]
    gw_r = [c for c in df.columns if "groundwater replenishment" in c]
    X["gw_thickness_mean"] = df[gw_t].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    X["gw_replenish_mean"] = df[gw_r].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    X["gw_balance"] = X["gw_replenish_mean"] / X["gw_thickness_mean"].replace(0, np.nan)

    # prosperity composite from census-style living indicators
    prosp = ["Perc_of_house_with_6plus_room",
             "perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms",
             "perc_of_pop_living_in_hh_electricity",
             "perc_of_Wall_material_with_Burnt_brick",
             "Households_with_improved_Sanitation_Facility"]
    prosp = [c for c in prosp if c in df.columns]
    X["prosperity_mean"] = df[prosp].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if "perc_Households_do_not_have_KCC_With_The_Credit_Limit_Of_50k" in df.columns:
        X["kcc_penetration"] = 100 - pd.to_numeric(
            df["perc_Households_do_not_have_KCC_With_The_Credit_Limit_Of_50k"], errors="coerce")

    # market access
    mandi = pd.to_numeric(df["K022-Proximity to nearest mandi (Km)"], errors="coerce")
    rail = pd.to_numeric(df["K022-Proximity to nearest railway (Km)"], errors="coerce")
    road = pd.to_numeric(df[" Road density (Km/ SqKm)"], errors="coerce")
    X["market_access"] = road / (1 + mandi)
    X["mandi_rail_sum"] = mandi + rail
    X["log_mandi_km"] = np.log1p(mandi)

    # ---- external data: state-level income tier (RBI NSDP, see above) ------
    tier = df["State"].astype(str).str.upper().str.strip().map(STATE_INCOME_TIER)
    X["state_income_tier"] = tier.astype(float)

    # ---- interactions ---------------------------------------------------
    X["land_x_prosperity"] = X["log_land"] * X["prosperity_mean"]
    X["land_x_income_tier"] = X["log_land"] * X["state_income_tier"]
    X["nonagri_x_income_tier"] = X["log_nonagri"] * X["state_income_tier"]
    if "Kharif_score_mean" in X.columns:
        X["kharif_score_x_land"] = X["Kharif_score_mean"] * X["log_land"]
    if "Rabi_score_mean" in X.columns:
        X["rabi_score_x_land"] = X["Rabi_score_mean"] * X["log_land"]
    X["bureau_to_nonagri"] = X["log_bureau_amt"] / (X["log_nonagri"] + 1)
    X["market_access_x_land"] = X["market_access"] * X["log_land"]
    X["dev_composite"] = X[["prosperity_mean", "kcc_penetration"]].mean(axis=1) \
        if "kcc_penetration" in X.columns else X["prosperity_mean"]

    # ---- low-cardinality categoricals --------------------------------------
    for c in LOW_CARD_CATS:
        if c in df.columns:
            X[f"cat_{c.strip()[:45]}"] = df[c].astype(str)

    return X


def add_target_encoding(X_tr, X_te, df_tr, df_te, y_enc, folds, smoothing=20.0):
    """Out-of-fold mean/median target encoding for high-cardinality keys.

    y_enc should already be on the modelling scale (e.g. log income) so the
    encoding is not dominated by the right tail.
    """
    prior = y_enc.mean()
    for c in HIGH_CARD_CATS:
        key_tr = df_tr[c].astype(str).values
        key_te = df_te[c].astype(str).values
        oof = np.full(len(df_tr), np.nan)
        cnt_oof = np.full(len(df_tr), np.nan)
        for tr_idx, va_idx in folds:
            s = pd.Series(y_enc.values[tr_idx]).groupby(pd.Series(key_tr[tr_idx])).agg(["mean", "count"])
            enc = (s["mean"] * s["count"] + prior * smoothing) / (s["count"] + smoothing)
            oof[va_idx] = pd.Series(key_tr[va_idx]).map(enc).values
            cnt_oof[va_idx] = pd.Series(key_tr[va_idx]).map(s["count"]).values
        s = pd.Series(y_enc.values).groupby(pd.Series(key_tr)).agg(["mean", "count"])
        enc_full = (s["mean"] * s["count"] + prior * smoothing) / (s["count"] + smoothing)
        name = f"te_{c.strip()[:30]}"
        X_tr[name] = np.where(np.isnan(oof), prior, oof)
        X_te[name] = pd.Series(key_te).map(enc_full).fillna(prior).values
        X_tr[name + "_cnt"] = np.nan_to_num(cnt_oof)
        X_te[name + "_cnt"] = pd.Series(key_te).map(s["count"]).fillna(0).values
    return X_tr, X_te
