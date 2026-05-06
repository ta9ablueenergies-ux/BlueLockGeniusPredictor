"""Shared feature schema between training and inference."""

FEATURE_SCHEMA_VERSION = "v1"

# Canonical ordering for build_enhanced_features_v81 numeric vector (36 features).
# NOTE: Bookmaker market signals (impl_prob_h etc.) are encoded separately in
# V11_SITUATION_FEATURE_COLUMNS via build_v11_situation_features() — do not
# duplicate them here as it would break V9 backbone weight loading.
FEATURE_COLUMNS_V81 = [
    "exp_h", "exp_a", "exp_total", "exp_diff",
    "pi_h", "pi_a", "pi_diff",
    "mom_h", "mom_a", "mom_diff",
    "sot_h", "sot_a", "sot_diff", "sot_ratio",
    "corn_h", "corn_a", "corn_total",
    "cs_h", "cs_a",
    "ga_h", "ga_a", "ga_diff",
    "imp_h", "imp_d", "imp_a",
    "edge_h", "edge_a",
    "p_h", "p_d", "p_a",
    "p_h_scores", "p_a_scores",
    "p_btts_model", "p_o25_model",
    "n_games_h", "n_games_a",
]


def validate_feature_vector(vector):
    """Raises ValueError when feature vector does not match schema length."""
    expected = len(FEATURE_COLUMNS_V81)
    actual = len(vector)
    if actual != expected:
        raise ValueError(f"Feature schema mismatch: expected {expected}, got {actual}")

