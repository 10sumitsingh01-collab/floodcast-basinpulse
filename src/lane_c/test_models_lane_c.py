"""Unit tests for BASINSHIFT Lane C (src/models.py) — Steps 1 through 9."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DATE_COL,
    EXPERIMENT_LOG_COLUMNS,
    FLOOD_THRESHOLDS,
    HORIZONS,
    MAX_N_JOBS,
    QUANTILES,
    RANDOM_STATE,
    STATIC_DERIVED_FEATURE_COLS,
    STATIC_FEATURE_COLS,
    STATIC_RAW_FEATURE_COLS,
    STATION_COL,
    TARGET_COL,
)
from src.data import load_train
from src.features import FEATURE_COLS, build_features
from src.models import (
    _compare_ablation_mode,
    _verify_fold_integrity,
    _verify_no_unseen_station_in_bundle,
    attach_static_attributes,
    calibrate_exceedance_models,
    compute_flood_thresholds,
    compute_horizon_metrics,
    evaluate_basinshift,
    fit_climatology,
    fit_exceedance_models,
    fit_models,
    fit_persistence,
    fit_quantile_models,
    get_feature_groups,
    inverse_transform_target,
    log_basinshift_experiment,
    make_weights,
    predict,
    predict_climatology,
    predict_exceedance_probs,
    predict_multi_horizon,
    predict_persistence,
    predict_quantiles,
    prepare_exceedance_labels,
    prepare_horizon_targets,
    resolve_station_thresholds,
    resolve_target_metadata,
    run_and_log_experiment,
    run_static_ablation,
    transform_target,
    validate_temporal,
    validate_unseen_stations,
)


def test_resolve_target_metadata_data_v2():
    train = load_train().head(50)
    meta = resolve_target_metadata(train)
    assert meta["target_col"] == "discharge_cfs"
    assert meta["target_units"] == "cfs"


def test_log1p_expm1_float32_roundtrip():
    train = load_train().head(500)
    y = train[TARGET_COL].to_numpy(dtype=np.float32)

    z = transform_target(y)
    assert z.dtype == np.float32
    assert np.isfinite(z).all()
    assert (z >= 0.0).all()

    y_recovered = inverse_transform_target(z)
    assert y_recovered.dtype == np.float32
    np.testing.assert_allclose(y_recovered, y, rtol=1e-5, atol=1e-3)


def test_prepare_horizon_targets_full_train():
    train = load_train()
    targets_df = prepare_horizon_targets(train, horizons=HORIZONS)

    # Index, row count, and metadata contract
    assert len(targets_df) == len(train)
    pd.testing.assert_index_equal(targets_df.index, train.index)
    assert targets_df.attrs["target_col"] == "discharge_cfs"
    assert targets_df.attrs["target_units"] == "cfs"
    assert targets_df.attrs["horizons"] == (1, 3, 7)

    # Dtype checks (must be float32)
    assert targets_df["y_origin"].dtype == np.float32
    assert targets_df["z_origin"].dtype == np.float32

    n_stations = train[STATION_COL].nunique()
    for h in HORIZONS:
        assert targets_df[f"y_h{h}"].dtype == np.float32
        assert targets_df[f"z_h{h}"].dtype == np.float32
        assert targets_df[f"valid_h{h}"].dtype == bool

        # Exact target dates t + h days
        expected_dates = pd.to_datetime(train[DATE_COL]).dt.normalize() + pd.Timedelta(days=h)
        assert (targets_df[f"target_date_h{h}"].values == expected_dates.values).all()

        # At the end of the 2000-2021 record, each of the 14 complete stations has
        # exactly h trailing rows where t + h falls in 2022 (outside train).
        missing_count = int((~targets_df[f"valid_h{h}"]).sum())
        assert missing_count >= n_stations * h

        # Verify no area division occurred: valid y_h{h} values match raw discharge_cfs scale
        valid_mask = targets_df[f"valid_h{h}"].values
        valid_y = targets_df.loc[valid_mask, f"y_h{h}"].to_numpy()
        valid_z = targets_df.loc[valid_mask, f"z_h{h}"].to_numpy()
        np.testing.assert_allclose(valid_z, np.log1p(valid_y), rtol=1e-6, atol=1e-6)


def test_prepare_horizon_targets_calendar_gap_alignment():
    """Verify calendar-date shift (t + h days) rather than row-index shift when dates have gaps."""
    train = load_train()
    # Select real Potomac River station slice in May 2015
    potomac = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2015-05-01"))
        & (train[DATE_COL] <= pd.Timestamp("2015-05-15"))
    ].copy()

    # Record true values from complete series
    val_may04 = np.float32(
        potomac.loc[potomac[DATE_COL] == pd.Timestamp("2015-05-04"), TARGET_COL].iloc[0]
    )
    val_may08 = np.float32(
        potomac.loc[potomac[DATE_COL] == pd.Timestamp("2015-05-08"), TARGET_COL].iloc[0]
    )

    # Drop 2015-05-02 and 2015-05-03 from the slice to create a 2-day calendar gap
    gapped = potomac[
        ~potomac[DATE_COL].isin([pd.Timestamp("2015-05-02"), pd.Timestamp("2015-05-03")])
    ].copy()

    res = prepare_horizon_targets(gapped, horizons=(1, 3, 7))
    row_may01 = res.loc[gapped[DATE_COL] == pd.Timestamp("2015-05-01")].iloc[0]

    # h=1 (2015-05-02) is missing from `gapped`, so y_h1 MUST be NaN (not 2015-05-04!)
    assert np.isnan(row_may01["y_h1"])
    assert not row_may01["valid_h1"]

    # h=3 (2015-05-04) is present and must match exact calendar date 2015-05-04
    assert row_may01["valid_h3"]
    assert np.isclose(row_may01["y_h3"], val_may04)

    # h=7 (2015-05-08) is present and must match exact calendar date 2015-05-08
    assert row_may01["valid_h7"]
    assert np.isclose(row_may01["y_h7"], val_may08)


def test_prepare_horizon_targets_max_target_date_leakage_guard():
    """Verify max_target_date strictly masks any target date t + h > max_target_date."""
    train = load_train()
    sub = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2020-06-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-06-20"))
    ].copy()

    cutoff = pd.Timestamp("2020-06-15")
    res = prepare_horizon_targets(sub, horizons=(1, 3, 7), max_target_date=cutoff)

    for _, row in res.iterrows():
        origin_dt = row[DATE_COL]
        if origin_dt > cutoff:
            assert np.isnan(row["y_origin"])
        for h in (1, 3, 7):
            target_dt = row[f"target_date_h{h}"]
            if target_dt > cutoff:
                assert np.isnan(row[f"y_h{h}"]), f"Target leaked past cutoff for h={h} at {origin_dt}"
                assert not row[f"valid_h{h}"]
            else:
                assert row[f"valid_h{h}"]
                assert np.isfinite(row[f"y_h{h}"])


def test_prepare_horizon_targets_preserves_scrambled_index():
    """Verify scrambled row order and arbitrary index are preserved while matching exact dates."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "07010000"]))
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-03-01"))
    ].copy()

    ordered_res = prepare_horizon_targets(sub, horizons=(1, 3, 7))
    scrambled_sub = sub.sample(frac=1.0, random_state=42)
    scrambled_res = prepare_horizon_targets(scrambled_sub, horizons=(1, 3, 7))

    pd.testing.assert_index_equal(scrambled_res.index, scrambled_sub.index)
    pd.testing.assert_frame_equal(scrambled_res.reindex(sub.index), ordered_res)


def test_make_weights():
    train = load_train()
    weights = make_weights(train, q=0.90, boost=5.0)

    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float32
    assert len(weights) == len(train)
    assert (weights >= 1.0).all()

    boosted_count = int((weights == 5.0).sum())
    assert boosted_count > 0
    ratio = boosted_count / len(train)
    assert 0.05 <= ratio <= 0.15


def test_persistence_exact_origin_flow_propagation():
    """Verify persistence baseline predicts y_hat(t+h) = y(t) in float32 across horizons."""
    train = load_train()
    trn_fold = train[train[DATE_COL] <= pd.Timestamp("2020-12-31")]
    val_fold = train[train[DATE_COL] >= pd.Timestamp("2021-01-01")].sample(frac=1.0, random_state=42)

    p_model = fit_persistence(trn_fold)
    assert p_model["model_type"] == "persistence"
    assert p_model["target_col"] == "discharge_cfs"
    assert p_model["target_units"] == "cfs"
    assert isinstance(p_model["global_median"], np.float32)

    preds_df = predict_persistence(p_model, val_fold, horizons=HORIZONS)
    pd.testing.assert_index_equal(preds_df.index, val_fold.index)
    assert preds_df.attrs["target_col"] == "discharge_cfs"
    assert preds_df.attrs["target_units"] == "cfs"
    assert preds_df.attrs["horizons"] == (1, 3, 7)

    expected_y = val_fold[TARGET_COL].to_numpy(dtype=np.float32)
    expected_z = transform_target(expected_y)

    for h in HORIZONS:
        col_y = f"persist_h{h}"
        col_z = f"persist_z_h{h}"
        assert preds_df[col_y].dtype == np.float32
        assert preds_df[col_z].dtype == np.float32
        assert np.isfinite(preds_df[col_y]).all()
        assert (preds_df[col_y] >= 0.0).all()
        np.testing.assert_allclose(preds_df[col_y].to_numpy(), expected_y, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(preds_df[col_z].to_numpy(), expected_z, rtol=1e-6, atol=1e-6)


def test_persistence_causal_ffill_and_station_fallback():
    """Verify persistence uses strictly causal ffill (<= t) or train station median when y_t is NaN."""
    train = load_train()
    trn_fold = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] <= pd.Timestamp("2019-12-31"))
    ].copy()
    val_slice = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2020-05-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-05-05"))
    ].copy()

    p_model = fit_persistence(trn_fold)
    expected_trn_median = p_model["station_medians"]["01646500"]

    # Case 1: First row (2020-05-01) and third row (2020-05-03) have NaN origin flow
    val_missing = val_slice.copy()
    y_may02 = np.float32(
        val_missing.loc[val_missing[DATE_COL] == pd.Timestamp("2020-05-02"), TARGET_COL].iloc[0]
    )
    val_missing.loc[
        val_missing[DATE_COL].isin([pd.Timestamp("2020-05-01"), pd.Timestamp("2020-05-03")]),
        TARGET_COL,
    ] = np.nan

    preds_1 = predict_persistence(p_model, val_missing, horizons=(1, 3, 7))
    # 2020-05-01 has no prior observation in val_missing -> must fall back to train station median (NOT bfill from May 02!)
    pred_may01 = preds_1.loc[val_missing[DATE_COL] == pd.Timestamp("2020-05-01"), "persist_h1"].iloc[0]
    assert np.isclose(pred_may01, expected_trn_median)

    # 2020-05-03 has causal observation on 2020-05-02 -> must forward-fill y_may02 (NOT May 04!)
    pred_may03 = preds_1.loc[val_missing[DATE_COL] == pd.Timestamp("2020-05-03"), "persist_h1"].iloc[0]
    assert np.isclose(pred_may03, y_may02)

    # Case 2: Target column completely omitted from df -> falls back to train station median
    val_no_target = val_slice.drop(columns=[TARGET_COL])
    preds_2 = predict_persistence(p_model, val_no_target, horizons=(1, 3, 7))
    for h in (1, 3, 7):
        np.testing.assert_allclose(
            preds_2[f"persist_h{h}"].to_numpy(),
            np.full(len(val_no_target), expected_trn_median, dtype=np.float32),
        )


def test_persistence_unseen_station_loso_fallback():
    """Verify persistence behavior on an unseen station (LOSO) with and without origin flow."""
    train = load_train()
    # Fit excluding Potomac (01646500)
    trn_other = train[
        (train[STATION_COL] != "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2020-01-01"))
    ].copy()
    unseen_val = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2021-06-01"))
        & (train[DATE_COL] <= pd.Timestamp("2021-06-10"))
    ].copy()

    p_model = fit_persistence(trn_other)
    assert "01646500" not in p_model["station_medians"]

    # When causal origin flow y_t is present on the unseen station, persistence uses y_t
    preds_with_origin = predict_persistence(p_model, unseen_val, horizons=(1, 3, 7))
    np.testing.assert_allclose(
        preds_with_origin["persist_h1"].to_numpy(),
        unseen_val[TARGET_COL].to_numpy(dtype=np.float32),
    )

    # When origin flow is unavailable on the unseen station, persistence falls back to global_median
    unseen_no_origin = unseen_val.drop(columns=[TARGET_COL])
    preds_fallback = predict_persistence(p_model, unseen_no_origin, horizons=(1, 3, 7))
    np.testing.assert_allclose(
        preds_fallback["persist_h1"].to_numpy(),
        np.full(len(unseen_no_origin), p_model["global_median"], dtype=np.float32),
    )


def test_climatology_known_stations_and_target_doy_alignment():
    """Verify DOY climatology uses training-only observations evaluated at target DOY(t + h)."""
    train = load_train()
    trn_fold = train[train[DATE_COL] <= pd.Timestamp("2019-12-31")].copy()
    val_slice = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2020-05-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-05-15"))
    ].copy()

    clim = fit_climatology(trn_fold)
    assert clim["model_type"] == "climatology"
    assert clim["target_col"] == "discharge_cfs"
    assert clim["target_units"] == "cfs"

    # Predict with target present vs target corrupted vs target dropped -> must be identical
    preds_a = predict_climatology(clim, val_slice, horizons=HORIZONS)
    val_corrupted = val_slice.copy()
    val_corrupted[TARGET_COL] = 999999.0
    preds_b = predict_climatology(clim, val_corrupted, horizons=HORIZONS)
    preds_c = predict_climatology(clim, val_slice.drop(columns=[TARGET_COL]), horizons=HORIZONS)

    pd.testing.assert_frame_equal(preds_a, preds_b)
    pd.testing.assert_frame_equal(preds_a, preds_c)
    pd.testing.assert_index_equal(preds_a.index, val_slice.index)

    # Verify exact DOY(t + h) alignment for origin date 2020-05-01 on Potomac (01646500)
    potomac_trn = trn_fold[trn_fold[STATION_COL] == "01646500"]
    potomac_trn_doy = potomac_trn[DATE_COL].dt.day_of_year
    row_may01 = preds_a.loc[val_slice[DATE_COL] == pd.Timestamp("2020-05-01")].iloc[0]

    for h in HORIZONS:
        target_dt = pd.Timestamp("2020-05-01") + pd.Timedelta(days=h)
        target_doy = target_dt.day_of_year
        expected_q50 = np.float32(
            np.quantile(potomac_trn.loc[potomac_trn_doy == target_doy, TARGET_COL].to_numpy(dtype=np.float32), 0.50)
        )
        assert np.isclose(row_may01[f"clim_h{h}"], expected_q50, rtol=1e-5)
        assert np.isclose(row_may01[f"clim_q50_h{h}"], expected_q50, rtol=1e-5)

        # Dtype and quantile non-crossing checks
        for col in [f"clim_h{h}", f"clim_z_h{h}", f"clim_q10_h{h}", f"clim_q50_h{h}", f"clim_q90_h{h}"]:
            assert preds_a[col].dtype == np.float32
            assert np.isfinite(preds_a[col]).all()
            assert (preds_a[col] >= 0.0).all()

        assert (preds_a[f"clim_q10_h{h}"] <= preds_a[f"clim_q50_h{h}"]).all()
        assert (preds_a[f"clim_q50_h{h}"] <= preds_a[f"clim_q90_h{h}"]).all()


def test_climatology_unseen_station_global_doy_fallback():
    """Verify unseen stations fall back strictly to training-only global DOY climatology."""
    train = load_train()
    # Fit on 2018-2020 excluding Potomac (01646500)
    trn_other = train[
        (train[STATION_COL] != "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2018-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-12-31"))
    ].copy()
    unseen_val = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2021-03-01"))
        & (train[DATE_COL] <= pd.Timestamp("2021-03-20"))
    ].drop(columns=[TARGET_COL])

    clim = fit_climatology(trn_other)
    assert "01646500" not in clim["station_summary"]
    assert "01646500" not in clim["station_doy_table"].index.get_level_values(0)

    preds = predict_climatology(clim, unseen_val, horizons=HORIZONS)
    for h in HORIZONS:
        target_doys = (pd.to_datetime(unseen_val[DATE_COL]) + pd.Timedelta(days=h)).dt.day_of_year.to_numpy()
        expected_global_q50 = clim["global_doy_table"].reindex(target_doys)["q50"].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(preds[f"clim_q50_h{h}"].to_numpy(), expected_global_q50, rtol=1e-6)
        assert (preds[f"clim_q10_h{h}"] <= preds[f"clim_q50_h{h}"]).all()
        assert (preds[f"clim_q50_h{h}"] <= preds[f"clim_q90_h{h}"]).all()


def test_climatology_leap_day_handling():
    """Verify safe leap-year handling (DOY=366 and Feb 29) even when trained on a non-leap year."""
    train = load_train()
    # Train ONLY on non-leap year 2019 (max DOY in training is 365; DOY 366 is never observed)
    trn_non_leap = train[
        (train[STATION_COL].isin(["01646500", "01092000"]))
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-12-31"))
    ].copy()

    clim = fit_climatology(trn_non_leap)
    # Verify DOY 366 exists and is finite in both station_doy_table and global_doy_table
    assert 366 in clim["global_doy_table"].index
    assert np.isfinite(clim["global_doy_table"].loc[366].to_numpy()).all()
    assert np.isfinite(clim["station_doy_table"].loc[("01646500", 366)].to_numpy()).all()

    # Predict on leap year 2020 dates including Feb 29 and Dec 24..30 (where t+h lands on Dec 31 = DOY 366)
    leap_val = train[
        (train[STATION_COL].isin(["01646500", "07010000"]))  # 01646500 seen, 07010000 unseen
        & (
            train[DATE_COL].isin(
                [
                    pd.Timestamp("2020-02-28"),
                    pd.Timestamp("2020-02-29"),
                    pd.Timestamp("2020-12-24"),  # t + 7 = 2020-12-31 (DOY 366)
                    pd.Timestamp("2020-12-28"),  # t + 3 = 2020-12-31 (DOY 366)
                    pd.Timestamp("2020-12-30"),  # t + 1 = 2020-12-31 (DOY 366)
                ]
            )
        )
    ].drop(columns=[TARGET_COL])

    preds = predict_climatology(clim, leap_val, horizons=(1, 3, 7))
    assert len(preds) == len(leap_val)
    for h in (1, 3, 7):
        assert np.isfinite(preds[f"clim_q10_h{h}"]).all()
        assert np.isfinite(preds[f"clim_q50_h{h}"]).all()
        assert np.isfinite(preds[f"clim_q90_h{h}"]).all()
        assert (preds[f"clim_q10_h{h}"] <= preds[f"clim_q50_h{h}"]).all()
        assert (preds[f"clim_q50_h{h}"] <= preds[f"clim_q90_h{h}"]).all()


def test_climatology_missing_values_and_date_gaps():
    """Verify circular interpolation over missing training DOYs and safe handling of NaNs/NaT dates."""
    train = load_train()
    trn_sub = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2018-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-12-31"))
    ].copy()

    # Drop DOY 150..160 completely from training and set some target values to NaN
    doy_trn = trn_sub[DATE_COL].dt.day_of_year
    trn_gapped = trn_sub[~doy_trn.between(150, 160)].copy()
    trn_gapped.iloc[:10, trn_gapped.columns.get_loc(TARGET_COL)] = np.nan

    clim = fit_climatology(trn_gapped)

    # Query dates in 2020 whose target dates fall inside DOY 150..160, plus a row with NaT date
    val_sub = train[
        (train[STATION_COL].isin(["01646500", "07010000"]))
        & (train[DATE_COL] >= pd.Timestamp("2020-05-25"))
        & (train[DATE_COL] <= pd.Timestamp("2020-06-05"))
    ].drop(columns=[TARGET_COL]).copy()
    val_sub.iloc[0, val_sub.columns.get_loc(DATE_COL)] = pd.NaT

    preds = predict_climatology(clim, val_sub, horizons=(1, 3, 7))
    for h in (1, 3, 7):
        assert np.isfinite(preds[f"clim_h{h}"]).all()
        assert (preds[f"clim_h{h}"] >= 0.0).all()
        assert (preds[f"clim_q10_h{h}"] <= preds[f"clim_q50_h{h}"]).all()
        assert (preds[f"clim_q50_h{h}"] <= preds[f"clim_q90_h{h}"]).all()

    # First row (NaT date on 01646500) must fall back to station_summary['01646500']['q50']
    first_sid = str(val_sub.iloc[0][STATION_COL])
    expected_fb = clim["station_summary"][first_sid]["q50"]
    assert np.isclose(preds.iloc[0]["clim_q50_h1"], expected_fb)


def test_quantile_models_training_and_resource_limits():
    """Verify all 9 LightGBM quantile models train in log1p float32 with n_jobs <= 4 and seed=42."""
    train = load_train()
    trn_sub = train[
        (train[STATION_COL].isin(["01646500", "01092000"]))
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-06-30"))
    ].copy()

    feats = build_features(trn_sub)
    X_trn = feats[FEATURE_COLS]

    # Pass n_jobs=16 to verify it is capped at MAX_N_JOBS (4)
    q_bundle = fit_quantile_models(
        X_trn,
        trn_sub,
        horizons=HORIZONS,
        quantiles=QUANTILES,
        params={"n_estimators": 15, "learning_rate": 0.1, "n_jobs": 16},
    )

    assert q_bundle["model_type"] == "quantile_lgbm"
    assert q_bundle["target_col"] == "discharge_cfs"
    assert q_bundle["target_units"] == "cfs"
    assert q_bundle["n_jobs"] <= MAX_N_JOBS
    assert q_bundle["random_state"] == RANDOM_STATE
    assert len(q_bundle["models"]) == 9
    assert len(q_bundle["models_by_name"]) == 9

    # Trailing horizon rows past 2019-06-30 must be excluded from training counts
    assert q_bundle["train_counts"][7] < q_bundle["train_counts"][3] < q_bundle["train_counts"][1] < len(trn_sub)

    for h in HORIZONS:
        for q, q_name in zip(QUANTILES, ["q10", "q50", "q90"]):
            reg = q_bundle["models"][(h, q_name)]
            assert reg is not None
            assert reg.objective == "quantile"
            assert np.isclose(reg.alpha, q)
            assert reg.random_state == RANDOM_STATE
            assert reg.n_jobs <= MAX_N_JOBS


def test_quantile_models_prediction_ordering_and_non_negativity():
    """Verify Q10 <= Q50 <= Q90 ordering, float32 dtypes, non-negativity, and expm1 consistency."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "02085000", "07010000"]))
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-03-31"))
    ].copy()

    feats = build_features(sub)
    trn_mask = (sub[DATE_COL] <= pd.Timestamp("2019-12-31")).to_numpy()
    val_mask = (sub[DATE_COL] >= pd.Timestamp("2020-02-01")).to_numpy()

    trn_df = sub.loc[trn_mask]
    X_trn = feats.loc[trn_mask, FEATURE_COLS]
    val_df = sub.loc[val_mask].sample(frac=1.0, random_state=42)
    X_val = feats.loc[val_df.index, FEATURE_COLS]

    q_bundle = fit_quantile_models(
        X_trn,
        trn_df,
        horizons=HORIZONS,
        quantiles=QUANTILES,
        params={"n_estimators": 25, "learning_rate": 0.08},
    )

    preds = predict_quantiles(q_bundle, X_val, val_df)
    pd.testing.assert_index_equal(preds.index, val_df.index)
    assert len(preds) == len(val_df)
    assert preds.attrs["target_col"] == "discharge_cfs"
    assert preds.attrs["target_units"] == "cfs"

    for h in HORIZONS:
        q10 = preds[f"q10_h{h}"]
        q50 = preds[f"q50_h{h}"]
        q90 = preds[f"q90_h{h}"]
        z10 = preds[f"z_q10_h{h}"]
        z50 = preds[f"z_q50_h{h}"]
        z90 = preds[f"z_q90_h{h}"]

        for s in (q10, q50, q90, z10, z50, z90):
            assert s.dtype == np.float32
            assert np.isfinite(s).all()
            assert (s >= 0.0).all()

        # Strict quantile ordering in both log1p and raw space
        assert (z10 <= z50).all(), f"z_q10 > z_q50 at horizon h={h}"
        assert (z50 <= z90).all(), f"z_q50 > z_q90 at horizon h={h}"
        assert (q10 <= q50).all(), f"q10 > q50 at horizon h={h}"
        assert (q50 <= q90).all(), f"q50 > q90 at horizon h={h}"

        # Verify expm1 correspondence between z_q50 and q50
        np.testing.assert_allclose(
            q50.to_numpy(),
            inverse_transform_target(z50.to_numpy()),
            rtol=1e-5,
            atol=1e-4,
        )


def test_quantile_models_unseen_stations_and_zero_target_leakage():
    """Verify quantile models predict on unseen stations without target leakage and reject leaked X."""
    train = load_train()
    trn_sub = train[
        (train[STATION_COL].isin(["01646500", "01092000", "01570500"]))
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-12-31"))
    ].copy()
    unseen_sub = train[
        (train[STATION_COL].isin(["02085000", "07010000"]))
        & (train[DATE_COL] >= pd.Timestamp("2020-04-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-05-15"))
    ].copy()

    X_trn = build_features(trn_sub)[FEATURE_COLS]
    X_unseen = build_features(unseen_sub)[FEATURE_COLS]

    # Passing target column inside X_train must raise ValueError
    X_leaked = X_trn.copy()
    X_leaked[TARGET_COL] = trn_sub[TARGET_COL].to_numpy()
    with pytest.raises(ValueError, match="Target column detected"):
        fit_quantile_models(X_leaked, trn_sub, params={"n_estimators": 5})

    q_bundle = fit_quantile_models(
        X_trn,
        trn_sub,
        horizons=HORIZONS,
        quantiles=QUANTILES,
        params={"n_estimators": 20, "learning_rate": 0.08},
    )

    # Predict on unseen stations with target present vs target dropped -> must be identical
    preds_with_target = predict_quantiles(q_bundle, X_unseen, unseen_sub)
    preds_no_target = predict_quantiles(q_bundle, X_unseen, unseen_sub.drop(columns=[TARGET_COL]))
    pd.testing.assert_frame_equal(preds_with_target, preds_no_target)

    for h in HORIZONS:
        assert (preds_no_target[f"q10_h{h}"] >= 0.0).all()
        assert (preds_no_target[f"q10_h{h}"] <= preds_no_target[f"q50_h{h}"]).all()
        assert (preds_no_target[f"q50_h{h}"] <= preds_no_target[f"q90_h{h}"]).all()


def test_flood_thresholds_q95_q99_and_q90_rejection():
    """Verify Q95 (q_high) and Q99 (q_extreme) threshold calculation and explicit Q90 rejection."""
    train = load_train()
    trn_sub = train[
        (train[STATION_COL].isin(["01646500", "07010000", "02085000"]))
        & (train[DATE_COL] >= pd.Timestamp("2015-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-12-31"))
    ].copy()

    # Explicitly reject Q90 (0.90) as q_high
    with pytest.raises(ValueError, match="never use Q90"):
        compute_flood_thresholds(trn_sub, thresholds={"q_high": 0.90, "q_extreme": 0.99})

    thresh_info = compute_flood_thresholds(trn_sub)
    assert thresh_info["thresholds"] == FLOOD_THRESHOLDS
    assert np.isclose(thresh_info["thresholds"]["q_high"], 0.95)
    assert np.isclose(thresh_info["thresholds"]["q_extreme"], 0.99)

    for sid in ["01646500", "07010000", "02085000"]:
        s_y = trn_sub.loc[trn_sub[STATION_COL] == sid, TARGET_COL].to_numpy(dtype=np.float32)
        exp_q95 = np.float32(np.quantile(s_y, 0.95))
        exp_q99 = np.float32(np.quantile(s_y, 0.99))
        st_t = thresh_info["station_thresholds"][sid]
        assert np.isclose(st_t["q_high"], exp_q95, rtol=1e-5)
        assert np.isclose(st_t["q_extreme"], exp_q99, rtol=1e-5)
        assert 0.0 <= st_t["q_high"] <= st_t["q_extreme"]

    labels_df = prepare_exceedance_labels(trn_sub, thresh_info, horizons=HORIZONS)
    for h in HORIZONS:
        valid = labels_df[f"valid_h{h}"].to_numpy(dtype=bool)
        y_high = labels_df.loc[valid, f"y_q_high_h{h}"].to_numpy()
        y_ext = labels_df.loc[valid, f"y_q_extreme_h{h}"].to_numpy()

        assert y_high.dtype == np.float32
        assert y_ext.dtype == np.float32
        assert set(np.unique(y_high)).issubset({0.0, 1.0})
        assert set(np.unique(y_ext)).issubset({0.0, 1.0})
        # Extreme event is always a subset of high-flow event
        assert (y_ext <= y_high).all()
        # Positive rate on training fold should be ~5% for Q95 and ~1% for Q99
        assert 0.03 <= float(np.mean(y_high)) <= 0.07
        assert 0.005 <= float(np.mean(y_ext)) <= 0.025


def test_exceedance_models_training_imbalance_and_prediction_bounds():
    """Verify 6 classifiers train with class imbalance weights, n_jobs<=4, seed=42, and [0, 1] outputs."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "01092000", "02085000"]))
        & (train[DATE_COL] >= pd.Timestamp("2016-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-06-30"))
    ].copy()

    feats = build_features(sub)
    trn_mask = (sub[DATE_COL] <= pd.Timestamp("2019-12-31")).to_numpy()
    val_mask = (sub[DATE_COL] >= pd.Timestamp("2020-02-01")).to_numpy()

    trn_df = sub.loc[trn_mask]
    X_trn = feats.loc[trn_mask, FEATURE_COLS]
    val_df = sub.loc[val_mask]
    X_val = feats.loc[val_mask, FEATURE_COLS]

    exc_bundle = fit_exceedance_models(
        X_trn,
        trn_df,
        horizons=HORIZONS,
        params={"n_estimators": 20, "learning_rate": 0.08, "n_jobs": 12},
    )

    assert exc_bundle["model_type"] == "exceedance_lgbm"
    assert exc_bundle["n_jobs"] <= MAX_N_JOBS
    assert exc_bundle["random_state"] == RANDOM_STATE
    assert len(exc_bundle["models"]) == 6

    for h in HORIZONS:
        for level_name in ("q_high", "q_extreme"):
            clf = exc_bundle["models"][(h, level_name)]
            assert clf is not None
            assert clf.n_jobs <= MAX_N_JOBS
            assert clf.random_state == RANDOM_STATE
            # Class imbalance weight must be > 1.0 for 5% and 1% rare events
            assert exc_bundle["scale_pos_weights"][(h, level_name)] > 1.0

    probs = predict_exceedance_probs(exc_bundle, X_val, val_df)
    pd.testing.assert_index_equal(probs.index, val_df.index)

    for h in HORIZONS:
        p_high = probs[f"prob_q_high_h{h}"]
        p_ext = probs[f"prob_q_extreme_h{h}"]

        assert p_high.dtype == np.float32
        assert p_ext.dtype == np.float32
        assert np.isfinite(p_high).all()
        assert np.isfinite(p_ext).all()
        assert ((p_high >= 0.0) & (p_high <= 1.0)).all()
        assert ((p_ext >= 0.0) & (p_ext <= 1.0)).all()
        assert (p_ext <= p_high).all(), f"prob_q_extreme > prob_q_high at horizon h={h}"


def test_exceedance_models_single_class_fold_safety():
    """Verify safe constant-probability handling when a fold has only a single target class."""
    train = load_train()
    trn_sub = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-03-31"))
    ].copy()

    X_trn = build_features(trn_sub)[FEATURE_COLS]

    # Set constant discharge so 100% of rows >= q_high and >= q_extreme (single class: all 1s)
    trn_const = trn_sub.copy()
    trn_const[TARGET_COL] = np.float32(500.0)

    exc_bundle = fit_exceedance_models(
        X_trn,
        trn_const,
        horizons=HORIZONS,
        params={"n_estimators": 10},
    )

    for h in HORIZONS:
        for level_name in ("q_high", "q_extreme"):
            assert exc_bundle["models"][(h, level_name)] is None
            assert np.isclose(exc_bundle["constant_probs"][(h, level_name)], 1.0)

    probs = predict_exceedance_probs(exc_bundle, X_trn.head(20), trn_const.head(20))
    for h in HORIZONS:
        assert np.allclose(probs[f"prob_q_high_h{h}"].to_numpy(), 1.0)
        assert np.allclose(probs[f"prob_q_extreme_h{h}"].to_numpy(), 1.0)


def test_exceedance_models_unseen_stations_and_leakage_prevention():
    """Verify unseen-basin threshold policy and zero validation target leakage in classifiers."""
    train = load_train()
    trn_sub = train[
        (train[STATION_COL].isin(["01646500", "01092000", "01570500"]))
        & (train[DATE_COL] >= pd.Timestamp("2017-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-12-31"))
    ].copy()
    unseen_sub = train[
        (train[STATION_COL].isin(["02085000", "07010000"]))
        & (train[DATE_COL] >= pd.Timestamp("2020-04-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-05-15"))
    ].copy()

    X_trn = build_features(trn_sub)[FEATURE_COLS]
    X_unseen = build_features(unseen_sub)[FEATURE_COLS]

    exc_bundle = fit_exceedance_models(
        X_trn,
        trn_sub,
        horizons=HORIZONS,
        params={"n_estimators": 15, "learning_rate": 0.08},
    )

    # Disallowing unseen fallback must raise ValueError rather than using validation targets
    with pytest.raises(ValueError, match="Unseen stations"):
        resolve_station_thresholds(
            exc_bundle["thresh_info"],
            unseen_sub,
            allow_unseen_fallback=False,
        )

    # Allowing training-only unseen fallback must not depend on unseen_sub[TARGET_COL]
    t_with_target = resolve_station_thresholds(
        exc_bundle["thresh_info"],
        unseen_sub,
        allow_unseen_fallback=True,
    )
    t_no_target = resolve_station_thresholds(
        exc_bundle["thresh_info"],
        unseen_sub.drop(columns=[TARGET_COL]),
        allow_unseen_fallback=True,
    )
    pd.testing.assert_frame_equal(t_with_target, t_no_target)
    assert (~t_no_target["is_seen_station"]).all()
    assert (t_no_target["q_high"] <= t_no_target["q_extreme"]).all()

    # Predictions on unseen stations must be identical with or without target column in df
    p_with = predict_exceedance_probs(exc_bundle, X_unseen, unseen_sub)
    p_without = predict_exceedance_probs(exc_bundle, X_unseen, unseen_sub.drop(columns=[TARGET_COL]))
    pd.testing.assert_frame_equal(p_with, p_without)

    for h in HORIZONS:
        assert ((p_without[f"prob_q_high_h{h}"] >= 0.0) & (p_without[f"prob_q_high_h{h}"] <= 1.0)).all()
        assert (p_without[f"prob_q_extreme_h{h}"] <= p_without[f"prob_q_high_h{h}"]).all()


def test_calibration_temporal_split_and_brier_improvement():
    """Verify isotonic calibration uses a disjoint temporal inner split without refitting base models."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "01092000", "01570500", "02085000"]))
        & (train[DATE_COL] >= pd.Timestamp("2015-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2021-06-30"))
    ].copy()

    feats = build_features(sub)
    trn_mask = (sub[DATE_COL] <= pd.Timestamp("2020-12-31")).to_numpy()
    val_mask = (sub[DATE_COL] >= pd.Timestamp("2021-02-01")).to_numpy()

    trn_df = sub.loc[trn_mask]
    X_trn = feats.loc[trn_mask, FEATURE_COLS]
    val_df = sub.loc[val_mask]
    X_val = feats.loc[val_mask, FEATURE_COLS]

    cal_bundle = calibrate_exceedance_models(
        X_trn,
        trn_df,
        horizons=HORIZONS,
        cal_fraction=0.20,
        gap_days=14,
        params={"n_estimators": 25, "learning_rate": 0.08},
    )

    assert cal_bundle["model_type"] == "calibrated_exceedance_lgbm"
    split_info = cal_bundle["calibration_split_info"]
    # Verify inner purge gap strictly separates inner_train and inner_cal
    assert split_info["inner_train_cutoff"] == split_info["cal_start_date"] - pd.Timedelta(days=14)
    assert split_info["n_inner_train_rows"] > 0
    assert split_info["n_inner_cal_rows"] > 0
    # Verify base classifiers were fitted ONLY on inner_train, NOT refitted on all of trn_df
    assert cal_bundle["train_counts"][1] <= split_info["n_inner_train_rows"]

    for h in HORIZONS:
        for lvl in ("q_high", "q_extreme"):
            key = (h, lvl)
            diag = cal_bundle["calibration_diagnostics"][key]
            assert diag["status"] == "calibrated"
            assert cal_bundle["calibrators"][key] is not None
            assert diag["n_cal_positives"] >= 2
            # Isotonic regression on inner_cal reduces or preserves Brier score relative to uncalibrated
            assert diag["brier_calibrated"] <= diag["brier_raw"] + 1e-6

    probs = predict_exceedance_probs(cal_bundle, X_val, val_df, calibrated=True)
    raw_only = predict_exceedance_probs(cal_bundle, X_val, val_df, calibrated=False)

    for h in HORIZONS:
        p_high = probs[f"prob_q_high_h{h}"]
        p_ext = probs[f"prob_q_extreme_h{h}"]
        r_high = probs[f"raw_prob_q_high_h{h}"]
        r_ext = probs[f"raw_prob_q_extreme_h{h}"]

        for s in (p_high, p_ext, r_high, r_ext):
            assert s.dtype == np.float32
            assert np.isfinite(s).all()
            assert ((s >= 0.0) & (s <= 1.0)).all()

        # Monotonicity across severity thresholds for both calibrated and raw outputs
        assert (p_ext <= p_high).all()
        assert (r_ext <= r_high).all()
        np.testing.assert_allclose(r_high.to_numpy(), raw_only[f"prob_q_high_h{h}"].to_numpy(), rtol=1e-6)


def test_calibration_zero_evaluation_leakage():
    """Verify calibration never depends on evaluation targets and respects max_target_date."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "01092000"]))
        & (train[DATE_COL] >= pd.Timestamp("2017-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-06-30"))
    ].copy()

    feats = build_features(sub)
    trn_mask = (sub[DATE_COL] <= pd.Timestamp("2019-12-31")).to_numpy()
    val_mask = (sub[DATE_COL] >= pd.Timestamp("2020-02-01")).to_numpy()

    trn_df = sub.loc[trn_mask]
    X_trn = feats.loc[trn_mask, FEATURE_COLS]
    val_df = sub.loc[val_mask]
    X_val = feats.loc[val_mask, FEATURE_COLS]

    cal_bundle = calibrate_exceedance_models(
        X_trn,
        trn_df,
        horizons=HORIZONS,
        cal_fraction=0.20,
        gap_days=14,
        params={"n_estimators": 15, "learning_rate": 0.1},
    )

    p_normal = predict_exceedance_probs(cal_bundle, X_val, val_df)
    val_corrupted = val_df.copy()
    val_corrupted[TARGET_COL] = 0.0
    p_corrupted = predict_exceedance_probs(cal_bundle, X_val, val_corrupted)
    p_dropped = predict_exceedance_probs(cal_bundle, X_val, val_df.drop(columns=[TARGET_COL]))

    pd.testing.assert_frame_equal(p_normal, p_corrupted)
    pd.testing.assert_frame_equal(p_normal, p_dropped)


def test_calibration_rare_event_and_single_class_cal_split_safety():
    """Verify safe fallback when inner_cal has 0 positives or when training window is too short."""
    train = load_train()
    sub = train[
        (train[STATION_COL] == "01646500")
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2019-12-31"))
    ].copy()
    X_sub = build_features(sub)[FEATURE_COLS]

    # Case 1: Force inner_cal (final 20% of 2019, ~Oct-Dec 2019) to have low flow so 0 positives occur in inner_cal
    sub_no_cal_floods = sub.copy()
    sub_no_cal_floods.loc[
        sub_no_cal_floods[DATE_COL] >= pd.Timestamp("2019-09-15"),
        TARGET_COL,
    ] = np.float32(50.0)

    bundle_no_cal_pos = calibrate_exceedance_models(
        X_sub,
        sub_no_cal_floods,
        horizons=HORIZONS,
        cal_fraction=0.20,
        gap_days=14,
        params={"n_estimators": 10},
    )

    for h in HORIZONS:
        for lvl in ("q_high", "q_extreme"):
            key = (h, lvl)
            assert bundle_no_cal_pos["calibrators"][key] is None
            assert bundle_no_cal_pos["calibration_diagnostics"][key]["status"] == "fallback_single_class_cal"

    preds_1 = predict_exceedance_probs(bundle_no_cal_pos, X_sub.head(30), sub.head(30))
    for h in HORIZONS:
        assert ((preds_1[f"prob_q_high_h{h}"] >= 0.0) & (preds_1[f"prob_q_high_h{h}"] <= 1.0)).all()
        assert (preds_1[f"prob_q_extreme_h{h}"] <= preds_1[f"prob_q_high_h{h}"]).all()

    # Case 2: Very short date window (15 days < gap_days + max_h + 10) -> safe fallback_insufficient_dates
    tiny_sub = sub.head(15).copy()
    X_tiny = X_sub.head(15).copy()
    bundle_tiny = calibrate_exceedance_models(
        X_tiny,
        tiny_sub,
        horizons=HORIZONS,
        cal_fraction=0.20,
        gap_days=14,
        params={"n_estimators": 5},
    )
    for h in HORIZONS:
        for lvl in ("q_high", "q_extreme"):
            assert bundle_tiny["calibration_diagnostics"][(h, lvl)]["status"] == "fallback_insufficient_dates"

    preds_2 = predict_exceedance_probs(bundle_tiny, X_tiny, tiny_sub)
    for h in HORIZONS:
        assert np.isfinite(preds_2[f"prob_q_high_h{h}"]).all()
        assert (preds_2[f"prob_q_extreme_h{h}"] <= preds_2[f"prob_q_high_h{h}"]).all()


def test_fold_integrity_and_leakage_rejection():
    """Verify temporal and unseen-station fold boundary checks reject any leakage."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "01092000"]))
        & (train[DATE_COL] >= pd.Timestamp("2019-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-03-31"))
    ].copy()

    # 1. Temporal leakage: training origin date crosses cutoff or overlaps val_df
    trn_bad_temp = sub[sub[DATE_COL] <= pd.Timestamp("2020-01-05")]
    val_temp = sub[sub[DATE_COL] >= pd.Timestamp("2020-01-01")]
    with pytest.raises(ValueError, match="Temporal fold leakage"):
        _verify_fold_integrity(
            trn_bad_temp,
            val_temp,
            mode="temporal",
            horizons=HORIZONS,
            effective_train_cutoff=pd.Timestamp("2020-01-05"),
        )

    # 2. Temporal horizon leakage: train origin is 2019-12-28, val starts 2020-01-01 (only 4 days < h=7)
    trn_short_gap = sub[sub[DATE_COL] <= pd.Timestamp("2019-12-28")]
    with pytest.raises(ValueError, match="Temporal horizon leakage"):
        _verify_fold_integrity(
            trn_short_gap,
            val_temp,
            mode="temporal",
            horizons=HORIZONS,
            effective_train_cutoff=pd.Timestamp("2019-12-31"),
        )

    # 3. Unseen-station leakage: overlapping station in train_df and val_df
    trn_station_leak = sub[sub[STATION_COL].isin(["01646500", "01092000"])]
    val_station = sub[sub[STATION_COL] == "01092000"]
    with pytest.raises(ValueError, match="Station leakage detected"):
        _verify_fold_integrity(
            trn_station_leak,
            val_station,
            mode="unseen_station",
            horizons=HORIZONS,
        )

    # 4. Bundle-level station leakage check
    feats = build_features(sub)
    bundle = fit_models(
        feats[FEATURE_COLS],
        train_df=sub,
        params={"n_estimators": 5},
        calibrate=False,
    )
    with pytest.raises(ValueError, match="Station leakage"):
        _verify_no_unseen_station_in_bundle(bundle, ["01092000"])


def test_compute_horizon_metrics_calculations_and_no_legacy_composite():
    """Verify multi-horizon metrics (MAE, RMSE, Pinball, Coverage, Q95/Q99 F1, Brier) and absence of legacy composite."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "01092000", "01570500"]))
        & (train[DATE_COL] >= pd.Timestamp("2017-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-06-30"))
    ].copy()
    feats = build_features(sub)

    trn_mask = (sub[DATE_COL] <= pd.Timestamp("2019-11-30")).to_numpy()
    val_mask = (sub[DATE_COL] >= pd.Timestamp("2020-01-01")).to_numpy()

    trn_df = sub.loc[trn_mask]
    X_trn = feats.loc[trn_mask, FEATURE_COLS]
    val_df = sub.loc[val_mask]
    X_val = feats.loc[val_mask, FEATURE_COLS]

    bundle = fit_models(
        X_trn,
        train_df=trn_df,
        params={"n_estimators": 15, "learning_rate": 0.08},
        calibrate=True,
        max_target_date=pd.Timestamp("2019-11-30"),
    )

    preds_df = predict_multi_horizon(bundle, X_val, df=val_df, include_baselines=True, calibrated=True)
    p_1d = predict(bundle, X_val, df=val_df, horizon=1, quantile=0.50)
    assert p_1d.dtype == np.float32
    assert p_1d.shape == (len(val_df),)
    np.testing.assert_allclose(p_1d, preds_df["q50_h1"].to_numpy(dtype=np.float32), rtol=1e-6)

    metrics_df = compute_horizon_metrics(
        val_df,
        preds_df,
        thresh_info=bundle["exceedance"]["thresh_info"],
        horizons=HORIZONS,
    )

    assert list(metrics_df["horizon"]) == [1, 3, 7]
    # Must NOT contain the old FLOODCAST composite score
    assert "weighted_composite" not in metrics_df.columns
    assert "top10_rmse" not in metrics_df.columns

    required_cols = [
        "mae", "rmse", "persistence_mae", "persistence_rmse",
        "climatology_mae", "climatology_rmse",
        "skill_mae_vs_persistence", "skill_mae_vs_climatology",
        "pinball_q10", "pinball_q50", "pinball_q90", "mean_pinball_loss",
        "interval_coverage_80", "interval_width_80",
        "n_pos_q_high", "pos_rate_q_high", "precision_q_high", "recall_q_high", "f1_q_high",
        "brier_q_high", "brier_raw_q_high", "cal_mae_q_high",
        "n_pos_q_extreme", "pos_rate_q_extreme", "precision_q_extreme", "recall_q_extreme", "f1_q_extreme",
        "brier_q_extreme", "brier_raw_q_extreme", "cal_mae_q_extreme",
    ]
    for col in required_cols:
        assert col in metrics_df.columns, f"Missing metric column: {col}"
        assert np.isfinite(metrics_df[col]).all(), f"Non-finite values in metric column: {col}"

    # Verify mathematical identity: pinball_q50 == 0.5 * mae
    np.testing.assert_allclose(
        metrics_df["pinball_q50"].to_numpy(),
        0.5 * metrics_df["mae"].to_numpy(),
        rtol=1e-5,
    )
    assert ((metrics_df["interval_coverage_80"] >= 0.0) & (metrics_df["interval_coverage_80"] <= 1.0)).all()
    assert (metrics_df["interval_width_80"] >= 0.0).all()
    assert ((metrics_df["brier_q_high"] >= 0.0) & (metrics_df["brier_q_high"] <= 1.0)).all()
    assert ((metrics_df["brier_q_extreme"] >= 0.0) & (metrics_df["brier_q_extreme"] <= 1.0)).all()


def test_validate_temporal_fold_boundaries_and_multi_horizon_eval():
    """Verify temporal validation enforces strict target date cutoffs across folds and horizons."""
    train = load_train()
    sub = train[
        (train[STATION_COL].isin(["01646500", "01092000", "01570500"]))
        & (train[DATE_COL] >= pd.Timestamp("2016-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2021-12-31"))
    ].copy()
    feats = build_features(sub)

    temp_res = validate_temporal(
        sub,
        features_df=feats,
        val_years=[2020, 2021],
        gap_days=30,
        horizons=HORIZONS,
        params={"n_estimators": 12, "learning_rate": 0.1},
        calibrate=True,
    )

    assert temp_res["mode"] == "temporal"
    fold_df = temp_res["fold_metrics"]
    sum_df = temp_res["summary_by_horizon"]

    # 2 validation years x 3 horizons = 6 fold-horizon rows
    assert len(fold_df) == 6
    assert set(fold_df["val_year"]) == {2020, 2021}
    assert list(sum_df["horizon"]) == [1, 3, 7]

    # Verify strict temporal fold boundaries: max_train_origin_date + 7 days < min_val_origin_date
    for _, row in fold_df.iterrows():
        max_trn = pd.Timestamp(row["max_train_origin_date"])
        cutoff = pd.Timestamp(row["effective_train_cutoff"])
        min_val = pd.Timestamp(row["min_val_origin_date"])
        assert max_trn <= cutoff
        assert cutoff <= min_val - pd.Timedelta(days=30)
        assert max_trn + pd.Timedelta(days=max(HORIZONS)) < min_val


def test_validate_unseen_stations_loso_and_zero_station_leakage():
    """Verify whole-station (LOSO) validation excludes held-out basins from all model components."""
    train = load_train()
    stations = ["01646500", "01092000", "01570500", "02085000"]
    sub = train[
        (train[STATION_COL].isin(stations))
        & (train[DATE_COL] >= pd.Timestamp("2017-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-12-31"))
    ].copy()
    feats = build_features(sub)

    loso_res = validate_unseen_stations(
        sub,
        features_df=feats,
        val_stations=["01092000", "01570500"],
        horizons=HORIZONS,
        params={"n_estimators": 12, "learning_rate": 0.1},
        calibrate=True,
    )

    assert loso_res["mode"] == "unseen_station"
    fold_df = loso_res["fold_metrics"]
    sum_df = loso_res["summary_by_horizon"]

    # 2 held-out stations x 3 horizons = 6 rows
    assert len(fold_df) == 6
    assert set(fold_df["val_station"]) == {"01092000", "01570500"}
    assert (fold_df["n_train_stations"] == 3).all()
    assert (fold_df["n_val_stations"] == 1).all()
    assert (fold_df["unseen_threshold_policy"] == "area_scaling_prior").all()
    assert list(sum_df["horizon"]) == [1, 3, 7]

    # Also verify unified evaluate_basinshift entry point runs both modes cleanly
    combined = evaluate_basinshift(
        sub,
        features_df=feats,
        val_years=[2020],
        val_stations=["01092000"],
        params={"n_estimators": 10, "learning_rate": 0.1},
        calibrate=True,
    )
    assert "temporal" in combined
    assert "unseen_station" in combined
    assert len(combined["temporal"]["summary_by_horizon"]) == 3
    assert len(combined["unseen_station"]["summary_by_horizon"]) == 3


def test_feature_groups_static_removal_and_partition_integrity():
    """Verify get_feature_groups cleanly separates dynamic_only and static features."""
    train = load_train().head(200)
    feats = build_features(train)

    groups = get_feature_groups(features_df=feats)
    full_cols = groups["full_static_plus_dynamic"]
    dyn_cols = groups["dynamic_only"]
    static_rem = groups["static_removed"]
    static_raw = groups["static_raw_only"]
    static_der = groups["static_derived_only"]

    assert len(full_cols) == 51
    assert len(static_raw) == 6
    assert len(static_der) == 8
    assert len(static_rem) == 14
    assert len(dyn_cols) == 37

    # Disjointness and completeness checks
    assert set(dyn_cols).isdisjoint(set(static_rem))
    assert set(dyn_cols) | set(static_rem) == set(full_cols)
    for col in STATIC_FEATURE_COLS:
        assert col not in dyn_cols
        assert col in static_rem

    # No target or identifier columns in any feature group
    for forbidden in (STATION_COL, DATE_COL, TARGET_COL, "streamflow"):
        assert forbidden not in full_cols
        assert forbidden not in dyn_cols


def test_attach_static_attributes_unseen_station_and_leakage_rejection():
    """Verify static attributes for unseen stations do not depend on targets and reject target leakage."""
    train = load_train()
    unseen_sub = train[
        (train[STATION_COL] == "02085000")
        & (train[DATE_COL] >= pd.Timestamp("2020-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-03-31"))
    ].copy()

    # Extract static catchment attributes table per station (without targets)
    static_table = (
        unseen_sub[[STATION_COL, *STATIC_RAW_FEATURE_COLS]]
        .drop_duplicates(subset=[STATION_COL])
        .copy()
    )
    dynamic_part = unseen_sub.drop(columns=list(STATIC_RAW_FEATURE_COLS))

    reattached_with_target = attach_static_attributes(dynamic_part, static_df=static_table)
    reattached_no_target = attach_static_attributes(
        dynamic_part.drop(columns=[TARGET_COL]),
        static_df=static_table,
    )

    feats_with_target = build_features(reattached_with_target)[FEATURE_COLS]
    feats_no_target = build_features(reattached_no_target)[FEATURE_COLS]
    pd.testing.assert_frame_equal(feats_with_target, feats_no_target)

    # Reject any static_df that attempts to smuggle target values
    leaked_static = static_table.copy()
    leaked_static[TARGET_COL] = 1000.0
    with pytest.raises(ValueError, match="Target column"):
        attach_static_attributes(dynamic_part, static_df=leaked_static)


def test_run_static_ablation_fold_consistency_and_multi_horizon_comparison():
    """Verify static-feature ablation runs on identical temporal and unseen-station folds across horizons."""
    train = load_train()
    stations = ["01646500", "01092000", "01570500", "02085000"]
    sub = train[
        (train[STATION_COL].isin(stations))
        & (train[DATE_COL] >= pd.Timestamp("2017-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-12-31"))
    ].copy()
    feats = build_features(sub)

    ablation = run_static_ablation(
        sub,
        features_df=feats,
        val_years=[2020],
        val_stations=["01092000", "01570500"],
        horizons=HORIZONS,
        params={"n_estimators": 12, "learning_rate": 0.1},
        calibrate=True,
        run_temporal=True,
        run_unseen_station=True,
    )

    assert len(ablation["feature_groups"]["dynamic_only"]) == 37
    assert len(ablation["feature_groups"]["full_static_plus_dynamic"]) == 51

    comp_df = ablation["comparison_by_horizon"]
    # 2 modes (temporal, unseen_station) x 3 horizons (1, 3, 7) = 6 rows
    assert len(comp_df) == 6
    assert set(comp_df["mode"]) == {"temporal", "unseen_station"}
    assert list(comp_df.loc[comp_df["mode"] == "temporal", "horizon"]) == [1, 3, 7]
    assert list(comp_df.loc[comp_df["mode"] == "unseen_station", "horizon"]) == [1, 3, 7]

    expected_diff_cols = [
        "mae_dynamic_only", "mae_full_static_plus_dynamic", "delta_mae", "rel_gain_pct_mae",
        "rmse_dynamic_only", "rmse_full_static_plus_dynamic", "delta_rmse",
        "mean_pinball_loss_dynamic_only", "mean_pinball_loss_full_static_plus_dynamic", "delta_mean_pinball_loss",
        "f1_q_high_dynamic_only", "f1_q_high_full_static_plus_dynamic", "delta_f1_q_high",
        "f1_q_extreme_dynamic_only", "f1_q_extreme_full_static_plus_dynamic", "delta_f1_q_extreme",
        "brier_q_high_dynamic_only", "brier_q_high_full_static_plus_dynamic", "delta_brier_q_high",
        "brier_q_extreme_dynamic_only", "brier_q_extreme_full_static_plus_dynamic", "delta_brier_q_extreme",
        "interval_coverage_80_dynamic_only", "interval_coverage_80_full_static_plus_dynamic", "delta_interval_coverage_80",
    ]
    for col in expected_diff_cols:
        assert col in comp_df.columns, f"Missing ablation comparison column: {col}"
        assert np.isfinite(comp_df[col]).all(), f"Non-finite values in ablation column: {col}"

    # Verify delta_mae == mae_full_static_plus_dynamic - mae_dynamic_only
    np.testing.assert_allclose(
        comp_df["delta_mae"].to_numpy(),
        (comp_df["mae_full_static_plus_dynamic"] - comp_df["mae_dynamic_only"]).to_numpy(),
        rtol=1e-6,
    )

    assert "temporal" in ablation["overall_comparison"]
    assert "unseen_station" in ablation["overall_comparison"]
    assert "mean_delta_mae" in ablation["overall_comparison"]["unseen_station"]

    # Verify fold consistency check raises ValueError if fold boundaries / sample counts differ
    bad_full = {
        "fold_metrics": ablation["variants"]["full_static_plus_dynamic"]["temporal"]["fold_metrics"].copy(),
        "summary_by_horizon": ablation["variants"]["full_static_plus_dynamic"]["temporal"]["summary_by_horizon"],
    }
    bad_full["fold_metrics"]["n_eval"] = bad_full["fold_metrics"]["n_eval"] + 1
    with pytest.raises(ValueError, match="Fold consistency violation"):
        _compare_ablation_mode(
            ablation["variants"]["dynamic_only"]["temporal"],
            bad_full,
            mode="temporal",
        )


def test_log_basinshift_experiment_append_behavior_and_schema(tmp_path):
    """Verify experiment logging saves per-horizon and static-ablation records without overwriting."""
    train = load_train()
    stations = ["01646500", "01092000", "01570500"]
    sub = train[
        (train[STATION_COL].isin(stations))
        & (train[DATE_COL] >= pd.Timestamp("2018-01-01"))
        & (train[DATE_COL] <= pd.Timestamp("2020-12-31"))
    ].copy()
    feats = build_features(sub)

    log_file = tmp_path / "experiments" / "results.csv"
    params = {"n_estimators": 8, "learning_rate": 0.1, "n_jobs": 2}

    # 1. Log temporal validation experiment (3 rows: horizons 1, 3, 7)
    temp_res = validate_temporal(
        sub,
        features_df=feats,
        val_years=[2020],
        gap_days=30,
        horizons=HORIZONS,
        params=params,
        calibrate=True,
    )
    logged_1 = log_basinshift_experiment(
        temp_res,
        description="Temporal validation baseline run",
        model_config=params,
        feature_set="full_static_plus_dynamic",
        n_features=len(FEATURE_COLS),
        validation_strategy="temporal",
        random_seed=RANDOM_STATE,
        experiment_id="exp_temporal_01",
        log_path=log_file,
    )
    assert len(logged_1) == 3
    assert log_file.exists()

    # 2. Log static-feature ablation experiment (6 rows: dynamic_only & full across h=1,3,7 for temporal)
    ablation_res = run_static_ablation(
        sub,
        features_df=feats,
        val_years=[2020],
        horizons=HORIZONS,
        params=params,
        calibrate=True,
        run_temporal=True,
        run_unseen_station=False,
    )
    logged_2 = log_basinshift_experiment(
        ablation_res,
        description="Static feature ablation temporal run",
        model_config=params,
        experiment_id="exp_ablation_02",
        log_path=log_file,
    )
    assert len(logged_2) == 6

    # Verify append behavior: 3 + 6 = 9 rows total on disk, no previous records overwritten
    saved = pd.read_csv(log_file)
    assert len(saved) == 9
    assert list(saved.columns) == list(EXPERIMENT_LOG_COLUMNS)
    assert list(saved["experiment_id"].iloc[:3]) == ["exp_temporal_01"] * 3
    assert list(saved["experiment_id"].iloc[3:]) == ["exp_ablation_02"] * 6
    assert (saved["random_seed"] == RANDOM_STATE).all()
    assert (saved["status"] == "completed").all()
    assert set(saved["horizon"]) == {1, 3, 7}

    # Verify ablation delta fields are populated for the static ablation rows
    ablation_rows = saved.iloc[3:]
    assert set(ablation_rows["feature_set"]) == {"dynamic_only", "full_static_plus_dynamic"}
    assert np.isfinite(ablation_rows["ablation_delta_mae"]).all()
    assert np.isfinite(ablation_rows["ablation_delta_brier_q_high"]).all()


def test_log_basinshift_experiment_legacy_header_preservation(tmp_path):
    """Verify appending to an existing legacy results.csv preserves prior rows without data loss."""
    log_file = tmp_path / "legacy_results.csv"
    legacy_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "commit": "abc1234",
                "description": "legacy_run",
                "val_rmse": 1200.5,
                "val_top10_rmse": 3400.0,
                "val_mae": 500.2,
                "val_r2": 0.75,
            }
        ]
    )
    legacy_df.to_csv(log_file, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "horizon": h,
                "n_eval": 100,
                "mae": 250.0 + h,
                "rmse": 450.0 + h,
                "mean_pinball_loss": 90.0 + h,
                "interval_coverage_80": 0.81,
                "f1_q_high": 0.62,
                "brier_q_high": 0.04,
                "f1_q_extreme": 0.50,
                "brier_q_extreme": 0.01,
            }
            for h in HORIZONS
        ]
    )

    log_basinshift_experiment(
        summary_df,
        description="BASINSHIFT post-legacy run",
        model_config={"n_estimators": 50},
        feature_set="full_static_plus_dynamic",
        n_features=51,
        validation_strategy="temporal",
        experiment_id="exp_post_legacy",
        log_path=log_file,
    )

    combined = pd.read_csv(log_file)
    assert len(combined) == 4
    assert combined.loc[0, "description"] == "legacy_run"
    assert combined.loc[0, "val_rmse"] == pytest.approx(1200.5)
    assert list(combined.loc[1:, "experiment_id"]) == ["exp_post_legacy"] * 3


def test_log_basinshift_experiment_required_fields_and_missing_metrics(tmp_path):
    """Verify validation of required metadata, seed=42, holdout prohibition, and missing metrics."""
    log_file = tmp_path / "validation_checks.csv"
    valid_summary = pd.DataFrame(
        [
            {
                "horizon": 1,
                "n_eval": 50,
                "mae": 100.0,
                "rmse": 200.0,
                "mean_pinball_loss": 40.0,
                "interval_coverage_80": 0.80,
                "f1_q_high": 0.5,
                "brier_q_high": 0.03,
                "f1_q_extreme": 0.4,
                "brier_q_extreme": 0.01,
            }
        ]
    )

    # 1. Wrong random_seed (!= 42) must raise ValueError
    with pytest.raises(ValueError, match="Reproducibility violation"):
        log_basinshift_experiment(
            valid_summary,
            description="Bad seed test",
            model_config={"n_estimators": 10},
            validation_strategy="temporal",
            random_seed=99,
            log_path=log_file,
        )

    # 2. Empty description or model_config must raise ValueError
    with pytest.raises(ValueError, match="description"):
        log_basinshift_experiment(
            valid_summary,
            description="   ",
            model_config={"n_estimators": 10},
            validation_strategy="temporal",
            log_path=log_file,
        )
    with pytest.raises(ValueError, match="model_config"):
        log_basinshift_experiment(
            valid_summary,
            description="Valid description",
            model_config={},
            validation_strategy="temporal",
            log_path=log_file,
        )

    # 3. Attempting to log holdout/test_label/fabricated results must raise ValueError
    for forbidden_strat in ("holdout_labels", "test_label_eval", "fabricated_run", "synthetic_data"):
        with pytest.raises(ValueError, match="Forbidden experiment source"):
            log_basinshift_experiment(
                valid_summary,
                description="Check forbidden source",
                model_config={"n_estimators": 10},
                validation_strategy=forbidden_strat,
                log_path=log_file,
            )

    # 4. Missing required metric columns must raise ValueError unless allow_missing_metrics=True
    incomplete_summary = pd.DataFrame([{"horizon": 1, "n_eval": 50, "mae": 100.0, "rmse": 200.0}])
    with pytest.raises(ValueError, match="missing required metric columns"):
        log_basinshift_experiment(
            incomplete_summary,
            description="Incomplete metrics run",
            model_config={"n_estimators": 10},
            validation_strategy="temporal",
            log_path=log_file,
        )

    logged_partial = log_basinshift_experiment(
        incomplete_summary,
        description="Diagnostic partial metrics run",
        model_config={"n_estimators": 10},
        validation_strategy="temporal",
        allow_missing_metrics=True,
        log_path=log_file,
    )
    assert len(logged_partial) == 1
    assert np.isnan(logged_partial.loc[0, "brier_q_high"])


def test_run_and_log_experiment_failed_experiment_handling(tmp_path):
    """Verify failed experiments are caught and logged clearly with status='failed' and error message."""
    log_file = tmp_path / "failed_experiments.csv"

    def _failing_runner():
        raise RuntimeError("Simulated out-of-memory or fold failure")

    result, logged_df = run_and_log_experiment(
        _failing_runner,
        description="Intentional failure test",
        model_config={"n_estimators": 100, "learning_rate": 0.05},
        feature_set="dynamic_only",
        n_features=37,
        validation_strategy="unseen_station",
        experiment_id="exp_fail_01",
        log_path=log_file,
        raise_on_error=False,
    )

    assert result is None
    assert len(logged_df) == 1
    saved = pd.read_csv(log_file)
    assert len(saved) == 1
    assert saved.loc[0, "experiment_id"] == "exp_fail_01"
    assert saved.loc[0, "status"] == "failed"
    assert "RuntimeError: Simulated out-of-memory or fold failure" in str(saved.loc[0, "error_message"])
    assert saved.loc[0, "random_seed"] == RANDOM_STATE
    assert np.isnan(saved.loc[0, "mae"])

