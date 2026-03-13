"""
Empirical example pipeline for RORR and coarsened AIPW.

The proprietary dataset used in the paper cannot be shared. This script includes
fully runnable example code on simulated data, and the same workflow can be run
with user-supplied real data via `run_rorr(data=...)`.

By default, required columns are: `stratum`, `x`, `y`, and `t` (from
`COVARIATES + [OUTCOME, TREATMENT]`). These names can be adapted by editing the
constants below.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from statsmodels.regression.linear_model import OLS

from data import simulate_dataset

## Constants.
SEED = 1024
SAMPLE_SIZE = 100_000
NUM_STRATA = 5
COVARIATES = ["stratum", "x"]  # All features.
CATEGORICAL = ["stratum"]  # Categorical features.
OUTCOME = "y"  # Outcome variable.
TREATMENT = "t"  # Treatment variable (assumed to be continuous).
CHECK_COVARIATE = "x"  # Covariate to check balance on.
BUCKETS = 5
EPS = 1e-6
PSCORE_CLIP = 1e-3


def required_columns():
    return sorted(set([*COVARIATES, OUTCOME, TREATMENT]))


def validate_input_data(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    missing = [col for col in required_columns() if col not in df.columns]
    if missing:
        raise ValueError(
            "Input data is missing required columns: "
            + ", ".join(missing)
            + "."
        )


def require_nonzero_std(value, name):
    if not np.isfinite(value) or abs(float(value)) < EPS:
        raise ValueError(
            f"{name} is near zero or non-finite. "
            "Check that the corresponding variable has non-trivial variation."
        )


def validate_num_bins(num_bins):
    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("`num_bins` must be an integer >= 2.")


def add_intercept(x):
    x = np.asarray(x, dtype=float)
    return np.c_[np.ones(x.shape[0]), x]


def predict_with_model(model, x_df):
    num_iteration = model.best_iteration if getattr(model, "best_iteration", 0) else None
    return model.predict(x_df, num_iteration=num_iteration)


def mock_data(sample_size=SAMPLE_SIZE, num_strata=NUM_STRATA, seed=SEED):
    """
    Placeholder empirical dataset for Section 5 development.

    Uses the simulation DGP so this script is runnable end-to-end until
    the real empirical input data is wired in.
    """
    rng = np.random.default_rng(seed)
    df = simulate_dataset(sample_size, num_strata, seed=seed).copy()
    df["stratum"] = df["x"].astype("category")
    # Keep a continuous covariate to avoid perfect collinearity with one-hot stratum.
    df["x"] = df["x"] + rng.normal(loc=0.0, scale=0.25, size=len(df))
    return df


def stratified_split(df, group_cols, train_frac=0.33, validate_frac=0.33, random_state=SEED):
    train_list, validate_list, test_list = [], [], []

    for _, group in df.groupby(group_cols, observed=True):
        shuffled_group = group.sample(frac=1, random_state=random_state)
        n = len(shuffled_group)
        train_end = int(train_frac * n)
        validate_end = train_end + int(validate_frac * n)

        train_list.append(shuffled_group.iloc[:train_end])
        validate_list.append(shuffled_group.iloc[train_end:validate_end])
        test_list.append(shuffled_group.iloc[validate_end:])

    train = pd.concat(train_list).sample(frac=1, random_state=random_state)
    validate = pd.concat(validate_list).sample(frac=1, random_state=random_state)
    test = pd.concat(test_list).sample(frac=1, random_state=random_state)
    return train, validate, test


def train_lgbm_regressor(train_df, validate_df, label_col, min_data_in_leaf):
    train_data = lgb.Dataset(
        train_df[COVARIATES],
        label=train_df[label_col],
        feature_name=[*COVARIATES],
        categorical_feature=CATEGORICAL,
    )
    validate_data = lgb.Dataset(
        validate_df[COVARIATES],
        label=validate_df[label_col],
        feature_name=[*COVARIATES],
        categorical_feature=CATEGORICAL,
        reference=train_data,
    )

    model = lgb.train(
        params={
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": min_data_in_leaf,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "seed": SEED,
        },
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[validate_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )
    return model


def train_propensity_model(train_df, validate_df, coarse_col, num_bins):
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_classes=num_bins,
        metric="multi_logloss",
        learning_rate=0.05,
        max_depth=-1,
        n_estimators=1500,
        min_data_in_leaf=30,
        max_bin=255,
        num_leaves=127,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=1.0,
        min_gain_to_split=0.0,
        seed=SEED,
        verbosity=-1,
    )
    clf.fit(
        train_df[COVARIATES],
        train_df[coarse_col].astype(int),
        categorical_feature=CATEGORICAL,
        eval_set=[(validate_df[COVARIATES], validate_df[coarse_col].astype(int))],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )
    return clf


def train_aipw_outcome_model(train_df, validate_df, coarse_col):
    features = [*COVARIATES, coarse_col]
    categorical = [*CATEGORICAL, coarse_col]
    train_data = lgb.Dataset(
        train_df[features],
        label=train_df[OUTCOME],
        feature_name=features,
        categorical_feature=categorical,
    )
    validate_data = lgb.Dataset(
        validate_df[features],
        label=validate_df[OUTCOME],
        feature_name=features,
        categorical_feature=categorical,
        reference=train_data,
    )
    return lgb.train(
        params={
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "seed": SEED,
        },
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[validate_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
    )


def compute_nonzero_cutpoints(series, num_bins):
    validate_num_bins(num_bins)
    non_zero_series = series[series > EPS]
    positive_bins = num_bins - 1
    if positive_bins <= 1 or len(non_zero_series) == 0:
        return np.array([], dtype=float)
    quantiles = np.linspace(0, 100, positive_bins + 1)[1:-1]
    return np.percentile(non_zero_series, quantiles)


def coarsener(x, cutpoints):
    if x <= EPS:
        return 0
    return int(np.searchsorted(cutpoints, x, side="right") + 1)


def coarsen_treatment(df, cutpoints):
    out = df.copy()
    out[f"{TREATMENT}_coarse"] = out[TREATMENT].map(lambda x: coarsener(x, cutpoints)).astype(int)
    return out


def compute_aipw_influence_function(level, outcome, treatment, pred_outcome_level, pscores):
    indicator = (treatment.to_numpy() == level).astype(float)
    probs = np.clip(pscores[:, level], PSCORE_CLIP, 1.0)
    aipw = (indicator / probs) * (outcome.to_numpy() - pred_outcome_level) + pred_outcome_level
    return aipw


def compute_ipw_weighted_mean(level, covariate, treatment, pscores):
    indicator = (treatment.to_numpy() == level).astype(float)
    probs = np.clip(pscores[:, level], PSCORE_CLIP, 1.0)
    weights = indicator / probs
    denom = np.sum(weights)
    if denom <= EPS:
        return np.nan
    return np.sum(weights * covariate.to_numpy()) / denom


def run_rorr(
    data: pd.DataFrame | None = None,
    *,
    random_state: int = SEED,
    train_frac: float = 0.33,
    validate_frac: float = 0.33,
):
    if data is None:
        data = mock_data(seed=random_state)
    else:
        validate_input_data(data)
        data = data.copy()
        for col in CATEGORICAL:
            if col in data.columns:
                data[col] = data[col].astype("category")

    train, validate, test = stratified_split(
        data,
        CATEGORICAL,
        train_frac=train_frac,
        validate_frac=validate_frac,
        random_state=random_state,
    )

    outcome_std = train[OUTCOME].std()
    treatment_std = train[TREATMENT].std()
    require_nonzero_std(outcome_std, "outcome_std")
    require_nonzero_std(treatment_std, "treatment_std")

    outcome_model = train_lgbm_regressor(
        train_df=train, validate_df=validate, label_col=OUTCOME, min_data_in_leaf=100
    )
    treatment_model = train_lgbm_regressor(
        train_df=train, validate_df=validate, label_col=TREATMENT, min_data_in_leaf=10
    )

    y_resid = test[OUTCOME] - predict_with_model(outcome_model, test[COVARIATES])
    t_resid = test[TREATMENT] - predict_with_model(treatment_model, test[COVARIATES])
    rorr_result = OLS(
        endog=y_resid / outcome_std, exog=add_intercept(t_resid / treatment_std)
    ).fit()

    print("RORR OLS summary (held-out test set):")
    print(rorr_result.summary())

    return {
        "data": data,
        "train": train,
        "validate": validate,
        "test": test,
        "outcome_model": outcome_model,
        "treatment_model": treatment_model,
        "rorr_result": rorr_result,
        "outcome_std": outcome_std,
        "treatment_std": treatment_std,
    }


def run_coarsened_aipw(state, num_bins=BUCKETS):
    validate_num_bins(num_bins)
    train = state["train"]
    validate = state["validate"]
    test = state["test"]
    data = state["data"]
    outcome_model = state["outcome_model"]
    outcome_std = state["outcome_std"]
    treatment_std = state["treatment_std"]

    cutpoints = compute_nonzero_cutpoints(train[TREATMENT], num_bins=num_bins)
    data = coarsen_treatment(data, cutpoints)
    train = coarsen_treatment(train, cutpoints)
    validate = coarsen_treatment(validate, cutpoints)
    test = coarsen_treatment(test, cutpoints)
    coarse_col = f"{TREATMENT}_coarse"

    propensity_model = train_propensity_model(train, validate, coarse_col, num_bins=num_bins)
    aipw_outcome_model = train_aipw_outcome_model(train, validate, coarse_col)
    ps = propensity_model.predict_proba(test[COVARIATES])

    y_hat_by_level = []
    for level in range(num_bins):
        test_level = test[[*COVARIATES]].copy()
        test_level[coarse_col] = level
        y_hat_level = predict_with_model(aipw_outcome_model, test_level[[*COVARIATES, coarse_col]])
        y_hat_by_level.append(y_hat_level)

    aipw_by_level = [
        compute_aipw_influence_function(
            level=level,
            outcome=test[OUTCOME],
            treatment=test[coarse_col],
            pred_outcome_level=y_hat_by_level[level],
            pscores=ps,
        )
        for level in range(num_bins)
    ]

    ## Check balance on pre-treatment covariate.
    pre_weighting = (
        test.groupby(coarse_col)[CHECK_COVARIATE].mean().reindex(np.arange(num_bins), fill_value=np.nan).to_numpy()
    )
    post_weighting = np.array([
        compute_ipw_weighted_mean(level, test[CHECK_COVARIATE], test[coarse_col], ps)
        for level in range(num_bins)
    ])
    pre_weighting_std = test[CHECK_COVARIATE].std()
    require_nonzero_std(pre_weighting_std, f"{CHECK_COVARIATE}_std")

    pre_weighting_diff = (pre_weighting - pre_weighting[0]) / pre_weighting_std
    post_weighting_diff = (post_weighting - post_weighting[0]) / pre_weighting_std

    bins = np.arange(num_bins)
    plt.figure(figsize=(6, 4))
    plt.axvline(x=0, color="pink", linewidth=1.5, zorder=0)
    plt.axvline(x=0.2, color="grey", linestyle="--", linewidth=1)
    plt.axvline(x=-0.2, color="grey", linestyle="--", linewidth=1)
    for i in bins:
        plt.scatter(pre_weighting_diff[i], i, color="tab:orange", label="Before Weighting" if i == 0 else "")
        plt.scatter(post_weighting_diff[i], i, color="tab:blue", label="After Weighting" if i == 0 else "")
    plt.yticks(bins, labels=[f"{i + 1}" for i in bins], fontsize=12)
    plt.xlabel("Standardized Difference in Mean Pre-Treatment Covariate", fontsize=14)
    plt.ylabel("Treatment Bins", fontsize=14)
    plt.title("Pre-Treatment Covariate Balance", fontsize=16)
    plt.legend(fontsize=11)
    plt.xlim([-0.5, 2.0])
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/emp-balance.png", dpi=300)
    plt.close()

    means = np.array([arr.mean() for arr in aipw_by_level])
    stderrs = np.sqrt(np.array([arr.var(ddof=1) / len(arr) for arr in aipw_by_level]))
    conf_intervals = 1.96 * stderrs

    plt.figure(figsize=(6, 4))
    plt.bar(
        x=np.arange(1, num_bins + 1),
        height=means,
        yerr=conf_intervals,
        capsize=5,
        color="tab:blue",
        alpha=0.5,
    )
    plt.yticks([])
    plt.ylabel("Outcome", fontsize=14)
    plt.xlabel("Treatment Bin", fontsize=14)
    plt.title("Outcome by Treatment Bin (AIPW)", fontsize=16)
    plt.tight_layout()
    plt.savefig("figures/emp-dose-response.png", dpi=300)
    plt.close()

    incremental_means = np.array([means[i + 1] - means[i] for i in range(num_bins - 1)])
    incremental_stderrs = np.array(
        [
            np.sqrt(aipw_by_level[i + 1].var(ddof=1) / len(aipw_by_level[i + 1]) + aipw_by_level[i].var(ddof=1) / len(aipw_by_level[i]))
            for i in range(num_bins - 1)
        ]
    )
    incremental_cis = 1.96 * incremental_stderrs

    plt.figure(figsize=(6, 4))
    plt.bar(
        x=np.arange(1, num_bins),
        height=incremental_means,
        yerr=incremental_cis,
        capsize=5,
        color="tab:blue",
        alpha=0.5,
    )
    plt.yticks([])
    plt.ylabel("Effect on Outcome", fontsize=14)
    plt.xlabel("Treatment Bin Increment", fontsize=14)
    plt.xticks(
        ticks=np.arange(1, num_bins),
        labels=[f"{i} to {i+1}" for i in range(1, num_bins)],
        fontsize=12,
    )
    plt.axhline(y=0, color="pink", linewidth=1.5, zorder=0)
    plt.title("Estimated Effect of Treatment on Outcome (AIPW)", fontsize=16)
    plt.tight_layout()
    plt.savefig("figures/emp-treatment-effects.png", dpi=300)
    plt.close()

    distribution = (
        data[coarse_col].value_counts(normalize=True).reindex(np.arange(num_bins), fill_value=0.0).to_numpy()
    )
    plt.figure(figsize=(6, 4))
    plt.bar(x=np.arange(1, num_bins + 1), height=distribution, alpha=0.5)
    plt.yticks([])
    plt.ylabel("Sample Proportion", fontsize=14)
    plt.xlabel("Treatment Bin", fontsize=14)
    plt.title("Distribution of Treatment", fontsize=16)
    plt.tight_layout()
    plt.savefig("figures/emp-distribution.png", dpi=300)
    plt.close()

    bin_medians = (
        test.groupby(coarse_col)[TREATMENT].median().reindex(np.arange(num_bins), fill_value=np.nan)
    )
    bin_deltas = (bin_medians.iloc[1:].to_numpy() - bin_medians.iloc[:-1].to_numpy()) / treatment_std
    safe_deltas = np.where(np.abs(bin_deltas) < EPS, np.nan, bin_deltas)
    if np.any(np.isnan(safe_deltas)):
        raise ValueError(
            "At least one treatment-bin median delta is near zero. "
            "Cannot compute standardized incremental treatment effects."
        )

    aipw_incremental_arrays = [
        (aipw_by_level[i + 1] - aipw_by_level[i]) / outcome_std / safe_deltas[i]
        for i in range(num_bins - 1)
    ]
    aipw_treatment_effects = np.array([arr.mean() for arr in aipw_incremental_arrays])
    aipw_treatment_effects_var = np.array([arr.var(ddof=1) / len(arr) for arr in aipw_incremental_arrays])

    weights = (
        test[coarse_col].value_counts(normalize=True)
        .reindex(np.arange(num_bins - 1), fill_value=0.0)
        .to_numpy()
    )
    aggregate_if = np.sum([weights[i] * aipw_incremental_arrays[i] for i in range(num_bins - 1)], axis=0)
    aipw_mean_all = float(np.mean(aggregate_if))
    aipw_std_all = float(np.sqrt(np.var(aggregate_if, ddof=1) / len(aggregate_if)))
    aipw_ci = (aipw_mean_all - 1.96 * aipw_std_all, aipw_mean_all + 1.96 * aipw_std_all)

    rorr_coef = np.asarray(state["rorr_result"].params, dtype=float)[1]
    print("\nComparison (held-out test set)")
    print(f"RORR coefficient (standardized): {rorr_coef:.4f}")
    print(
        "AIPW weighted incremental effect "
        f"(standardized): {aipw_mean_all:.4f} "
        f"[95% CI: ({aipw_ci[0]:.4f}, {aipw_ci[1]:.4f})]"
    )

    return {
        "aipw_by_level": aipw_by_level,
        "aipw_incremental_effects": aipw_treatment_effects,
        "aipw_incremental_variance": aipw_treatment_effects_var,
        "aipw_weighted_effect": aipw_mean_all,
        "aipw_weighted_std": aipw_std_all,
        "aipw_weighted_ci": aipw_ci,
        "cutpoints": cutpoints,
        "num_bins": num_bins,
    }


def write_table3_tex(rorr_result, aipw_results, output_path="figures/table-3-section5.tex"):
    rorr_coef = np.asarray(rorr_result.params, dtype=float)[1]
    rorr_se = np.asarray(rorr_result.bse, dtype=float)[1]
    rorr_ci = np.asarray(rorr_result.conf_int(), dtype=float)[1]

    aipw_coef = float(aipw_results["aipw_weighted_effect"])
    aipw_se = float(aipw_results["aipw_weighted_std"])
    aipw_ci = aipw_results["aipw_weighted_ci"]

    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{RORR and AIPW Estimates of the Effect of the Treatment Variable}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & RORR & Std. Err. & 95\% CI \\",
        r"\midrule",
        f"Treatment Variable & {rorr_coef:.4f} & {rorr_se:.3f} & ({rorr_ci[0]:.3f}, {rorr_ci[1]:.3f}) \\\\",
        r"\midrule",
        r" & AIPW & Std. Err. & 95\% CI \\",
        r"\midrule",
        f"Treatment Variable & {aipw_coef:.3f} & {aipw_se:.3f} & ({aipw_ci[0]:.3f}, {aipw_ci[1]:.3f}) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{document}",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote Table 3 LaTeX to {output_path}")


if __name__ == "__main__":
    # To run on real data, pass your DataFrame: run_rorr(data=my_dataframe)
    state = run_rorr()
    aipw_results = run_coarsened_aipw(state)
    write_table3_tex(state["rorr_result"], aipw_results)
