import pandas as pd
import pytest

from rents.load import load_adverts, path_for, read_input, train_slice, window
from rents.model import combine, fit


@pytest.mark.parametrize("w", ["A", "B", "final"])
def test_training_rows_never_after_cutoff(small_world, w):
    c = small_world
    adv, info = load_adverts(c, "synthetic")
    assert adv["date"].max() <= pd.Timestamp(c["windows"]["hard_cutoff"])
    win = window(c, w)
    tr = train_slice(adv, win["train_end"])
    assert len(tr) and tr["date"].max() <= win["train_end"]
    f = fit(read_input(path_for(c, "synthetic", "input"), c), tr, win["train_end"], c["model"])
    assert f.index.m_hi == win["train_end"].year * 12 + win["train_end"].month - 1


def test_fit_refuses_post_cutoff_rows(small_world):
    c = small_world
    flats = read_input(path_for(c, "synthetic", "input"), c)
    adv, _ = load_adverts(c, "synthetic")
    win = window(c, "A")
    bad = adv.copy()
    bad.loc[bad.index[0], "date"] = win["train_end"] + pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        fit(flats, bad, win["train_end"], c["model"])


@pytest.mark.parametrize("w", ["A", "B"])
def test_predictions_invariant_to_post_cutoff_adverts(small_world, w):
    """Changing every advert after train_end (price x3, duplicated rows) must not change any prediction."""
    c = small_world
    flats = read_input(path_for(c, "synthetic", "input"), c)
    adv, _ = load_adverts(c, "synthetic")
    win = window(c, w)
    p1 = combine(fit(flats, train_slice(adv, win["train_end"]), win["train_end"], c["model"]), "onevote", 5)
    adv2 = adv.copy()
    late = adv2["date"] > win["train_end"]
    assert late.any()
    adv2.loc[late, "price"] *= 3
    adv2 = pd.concat([adv2, adv2[late]], ignore_index=True)
    p2 = combine(fit(flats, train_slice(adv2, win["train_end"]), win["train_end"], c["model"]), "onevote", 5)
    pd.testing.assert_frame_equal(p1, p2)


def test_input_date_columns_are_not_features(small_world):
    """attrs_last_seen / first_ever_listed can be after the cut-off in the real file: predictions must ignore them."""
    c = small_world
    flats = read_input(path_for(c, "synthetic", "input"), c)
    adv, _ = load_adverts(c, "synthetic")
    win = window(c, "final")
    tr = train_slice(adv, win["train_end"])
    p1 = combine(fit(flats, tr, win["train_end"], c["model"]), "onevote", 5)
    f2 = flats.copy()
    f2["attrs_last_seen"] = pd.Timestamp("2026-10-01")
    f2["first_ever_listed"] = pd.Timestamp("2026-09-01")
    f2["ever_advertised_for_lease"] = 1
    p2 = combine(fit(f2, tr, win["train_end"], c["model"]), "onevote", 5)
    pd.testing.assert_frame_equal(p1, p2)


def test_loader_drops_rows_after_task_cutoff(small_world, tmp_path):
    c = dict(small_world)
    pl = pd.read_csv(small_world["paths"]["synthetic"]["past_leases"], dtype=str)
    extra = pl.iloc[:3].copy()
    extra["date"] = "2026-08-15"
    p = tmp_path / "pl.csv"
    pd.concat([pl, extra]).to_csv(p, index=False)
    c["paths"] = dict(small_world["paths"])
    c["paths"]["synthetic"] = dict(small_world["paths"]["synthetic"], past_leases=str(p))
    adv, info = load_adverts(c, "synthetic")
    assert info["n_after_cutoff_dropped"] == 3
    assert adv["date"].max() <= pd.Timestamp("2026-07-31")


def test_public_features_see_no_dated_input(small_world):
    """Public features are built from flats' id / building key / floor / coords only (no adverts, no input dates),
    so they are identical in every window. Reads public-data files locally; no network."""
    import inspect
    from rents import public_features
    assert list(inspect.signature(public_features.flat_features).parameters) == ["flats", "c", "radius_m"]
    c = small_world
    flats = read_input(path_for(c, "synthetic", "input"), c)
    slim = flats[["gnaf_id", "bkey", "floor_level", "latitude", "longitude"]]
    feats = public_features.flat_features(slim, c)
    assert len(feats) == len(flats) and feats["gnaf_id"].is_unique
    assert feats["osm_id"].notna().mean() > 0.9 and (feats["join_dist_m"].dropna() <= 15).all()
