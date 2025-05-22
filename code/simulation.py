from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from data import simulate_dataset, f_t, g_x, h_x
from estimators import compute_acd, estimate_weighted_increments, get_rorr_estimates
from plot import plot_simulation


NUM_STRATA = 5
SEED = 1024
SAMPLE_SIZES = [10_000, 100_000, 1_000_000]


def run_simulation(sample_sizes):
    results_acd = []
    results_rorr = []
    results_aie = []

    true_acd = compute_acd(NUM_STRATA)

    for n in sample_sizes:
        data = simulate_dataset(n, NUM_STRATA, seed=SEED)

        # ACD Estimator
        acd_empirical = data.derivative.mean()
        acd_stderr = data.derivative.std() / np.sqrt(n)
        acd_ci = (acd_empirical - 1.96 * acd_stderr, acd_empirical + 1.96 * acd_stderr)
        results_acd.append([n, acd_empirical, f"({acd_ci[0]:.3f}, {acd_ci[1]:.3f})", true_acd])

        # RORR Estimator
        rorr_empirical, rorr_stderr, rorr_ci = get_rorr_estimates(data, g_x, h_x)
        rorr_target = (data.assign(plm_plim=lambda df: df.weight / (df.t_star + 1))).plm_plim.mean()
        results_rorr.append([n, rorr_empirical, f"({rorr_ci[0]:.3f}, {rorr_ci[1]:.3f})", rorr_target])

        # AIE Estimator
        aie_empirical, aie_stderr, aie_ci = estimate_weighted_increments(data, f_t, g_x, h_x)
        aie_target = data[data.t < data.t.max()].incremental.mean()
        results_aie.append([n, aie_empirical, f"({aie_ci[0]:.3f}, {aie_ci[1]:.3f})", aie_target])

    return results_acd, results_rorr, results_aie


def to_latex(df, estimator_name):
    latex = df.to_latex(index=False, float_format="%.3f")
    print(f"\nLaTeX Table for {estimator_name}:\n")
    print(latex)


if __name__ == "__main__":
    results_acd, results_rorr, results_aie = run_simulation(SAMPLE_SIZES)

    columns_acd = ["Sample Size", "Empirical ACD", "ACD CI", "ACD Target"]
    columns_rorr = ["Sample Size", "Empirical RORR", "RORR CI", "RORR Target"]
    columns_aie = ["Sample Size", "Empirical AIE", "AIE CI", "AIE Target"]

    df_acd = pd.DataFrame(results_acd, columns=columns_acd)
    df_rorr = pd.DataFrame(results_rorr, columns=columns_rorr)
    df_aie = pd.DataFrame(results_aie, columns=columns_aie)

    # NOTE: In the paper, we set the RORR and AIE targets to their empirical estimates with n=1m.
    to_latex(df_acd, "ACD Estimator")
    to_latex(df_rorr, "RORR Estimator")
    to_latex(df_aie, "AIE Estimator")

    plot_data = simulate_dataset(1_000_000, NUM_STRATA, seed=SEED)
    plot_simulation(plot_data).show()
    plt.savefig("figures/figure-1.png", dpi=300)
