import numpy as np
import pandas as pd

from scipy.stats import poisson
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def compute_acd(num_strata):
    strata = np.arange(1, num_strata + 1)
    return np.mean((1 - np.exp(-strata)) / strata)

def get_rorr_estimates(data, g_x, h_x):
    y_hat = data.y - g_x(data.x)
    t_hat = data.t - h_x(data.x)
    fit = OLS(y_hat, add_constant(t_hat)).fit()
    coef = fit.params.loc[0]
    lwr, upr = fit.conf_int().loc[0].values
    se = fit.bse.loc[0]
    return (coef, se, (lwr, upr))

def compute_aipw_matrix(df, f_t, g_x, h_x):
    """
    Computes a wide-form AIPW influence matrix.

    Parameters:
    - df: DataFrame with columns 'y', 't', 'x'
    - f_t: Function of t (can handle array input)
    - g_x: Function of x (can handle array input)
    - h_x: Function of x (returns Poisson mean)

    Returns:
    - AIPW matrix as a DataFrame:
        - index matches df.index
        - columns are influence values for t = 0, 1, ..., max(t)
    """
    y = df['y'].to_numpy()
    t_obs = df['t'].to_numpy()
    x = df['x'].to_numpy()
    idx = df.index

    h_x_vals = h_x(x)
    g_x_vals = g_x(x)

    t_max = np.max(t_obs)
    t_vals = np.arange(t_max + 1)

    # shape (n, len(t_vals))
    m_vals = f_t(t_vals[None, :]) + g_x_vals[:, None]
    pscore = poisson.pmf(t_vals[None, :], mu=h_x_vals[:, None])
    indicator = (t_obs[:, None] == t_vals[None, :]).astype(float)

    # AIPW formula
    influence = indicator / np.clip(pscore, 1e-12, None) * (y[:, None] - m_vals) + m_vals

    # Build DataFrame with t values as column names
    influence_df = pd.DataFrame(influence, index=idx, columns=[f"t_{t}" for t in t_vals])
    return influence_df

def analytical_variance(df, f_t, g_x, h_x, weights):
    aipw_matrix = compute_aipw_matrix(df, f_t, g_x, h_x).values  # shape (n, T+1)
    
    T_max = aipw_matrix.shape[1] - 1

    # Compute the c vector:
    c = np.zeros(T_max + 1)
    c[0] = -weights[0]
    c[1:T_max] = weights[:-1] - weights[1:]
    c[T_max] = weights[T_max-1]

    psi = aipw_matrix @ c
    var_hat = np.var(psi, ddof=1) / len(df)

    return var_hat

def estimate_weighted_increments(df, f_t, g_x, h_x):
    """
    Estimates weighted increments between AIPW estimates for each value of t.

    Steps:
    1. Computes empirical proportions of each value of t (used as weights)
    2. Computes AIPW influence function matrix
    3. Computes difference in mean influence across consecutive t values
    4. Returns weighted differences using empirical t distribution

    Parameters:
    - df: DataFrame with columns 'y', 't', 'x'
    - f_t, g_x, h_x: functions as defined earlier

    Returns:
    - A dictionary with:
        - 'weighted_differences': 1D numpy array of length (max_t)
        - 'empirical_weights': 1D numpy array of length (max_t + 1)
        - 'differences': raw unweighted differences
    """
    # Step 1: Compute empirical weights
    t_vals, counts = np.unique(df['t'], return_counts=True)
    max_t = np.max(t_vals)
    weights = np.zeros(max_t + 1)
    weights[t_vals] = counts / len(df)

    # Step 2: Compute AIPW matrix
    aipw_matrix = compute_aipw_matrix(df, f_t, g_x, h_x)

    # Step 3: Compute consecutive mean differences
    column_names = [f"t_{t}" for t in range(max_t + 1)]
    means = aipw_matrix[column_names].mean(axis=0).to_numpy()
    differences = np.diff(means)  # length = max_t

    # Step 4: Apply weights (set last weight to zero)
    effective_weights = weights[:-1].copy()
    effective_weights[-1] = 0.0  # Set weight for final bin to zero
    weighted_differences = effective_weights * differences
    estimate = weighted_differences.sum()

    variance = analytical_variance(df, f_t, g_x, h_x, effective_weights)
    return (
        estimate,
        np.sqrt(variance),
        (
            estimate - 1.96 * np.sqrt(variance),
            estimate + 1.96 * np.sqrt(variance)
        ),
    )
