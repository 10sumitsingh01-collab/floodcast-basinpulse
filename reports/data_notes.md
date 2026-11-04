# Data Notes & Ingestion Audit (Phase 1)

## Data Sources & Provenance
1. **Source A (USGS Daily Values NWIS)**: Daily mean discharge (`00060_00003`). Negative, ice-affected, or unmeasured days converted strictly to NaN. Qualifiers retained in `data/processed/discharge_qualifiers.parquet`.
2. **Source B (USGS Site RDB Expanded)**: Official site coordinates, state, drainage area (`drain_area_va`), and gauge altitude (`alt_va`). Gauge altitude is stored separately in `data/processed/station_static_extra.parquet` to prevent inadvertent leakage into `mean_elevation_ft`.
3. **Source C (Open-Meteo Archive API)**: Historical weather fetched in 5-year cached chunks under UTC. Uses `soil_moisture_0_to_7cm_mean` mapped to `soil_moisture`. Backoff logic handled HTTP 429 rate limits automatically.
4. **Source D (USGS StreamStats NSS Services)**: NSS service queried for regression region definitions.

## Station Candidate Review & Rejections
- **Candidate 11447650 (Sacramento River at Freeport, CA)**: Flagged for human review. USGS NWIS discontinued publishing daily discharge (`00060_00003`) at this site on 2015-09-30 (moved to tidal stage-only). As a result, it has 5,200 valid discharge days in train (2000-2015) and 0 valid discharge days during the 2022-2023 holdout period. Per instructions ("keep the 5 mandatory ones" and "test table must contain only rows whose label exists in the holdout labels file"), Sacramento is retained in train and station metadata, but produces 0 rows in `test_features.parquet` and `holdout_labels.parquet`.
- **All other 14 stations**: 100.0% coverage over 2000-2023 (8,766 / 8,766 days) and 100.0% holdout coverage (730 / 730 days).
- **Unverified candidate IDs**: Excluded. Only gauges confirmed via Source B with `site_tp_cd == 'ST'` were ingested.
