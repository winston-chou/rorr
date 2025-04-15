import numpy as np
import pandas as pd


BETA_X = 0.111


def f_t(t_array):
    "The g function in Y = f(t) + g(x) + e"
    return np.log(t_array + 1)

def g_x(x_array, beta_x=BETA_X):
    "The g function in Y = f(t) + g(x) + e"
    return x_array * BETA_X

def h_x(x_array):
    "The h function in T = h(x) + u"
    return x_array

def generate_x(sample_size, num_strata, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(1, num_strata + 1, size=sample_size)

def generate_t(x_array, seed=None):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=h_x(x_array))

def generate_y(t_array, x_array, seed=None):
    rng = np.random.default_rng(seed)
    e = rng.normal(loc=0.0, scale=1.0, size=len(t_array))
    return f_t(t_array) + g_x(x_array) + e

def append_derivative(df):
    """
    Adds 1 / (t + 1), the derivative of log(t + 1), to the input DataFrame

    Parameters:
    - df: DataFrame with at least a 't' column

    Returns:
    - df: Modified DataFrame with `derivative` column
    """
    df = df.copy()
    df['derivative'] = 1 / (df['t'] + 1)
    return df

def append_incremental(df):
    """
    Adds log((t + 2) / (t + 1)), the effect of incrementing t, to the input DataFrame

    Parameters:
    - df: DataFrame with at least a 't' column

    Returns:
    - df: Modified DataFrame with `incremental` column
    """
    df = df.copy()
    df['incremental'] = np.log(df['t'] + 2) - np.log(df['t'] + 1)
    return df

def append_t_star(df):
    """
    Computes t_star based on the rule:
      - If t == x:      t_star = t
      - If t != x:      t_star = (t - x) / log((t + 1) / (x + 1)) - 1

    Parameters:
    - df: DataFrame with columns 't' and 'x'

    Returns:
    - df: Copy of input DataFrame with an added column 't_star'
    """
    df = df.copy()

    t = df['t'].to_numpy()
    x = df['x'].to_numpy()

    # Initialize t_star with the same values as t
    t_star = np.where(
        t == x,
        t,
        (t - x) / np.log((t + 1) / (x + 1)) - 1
    )

    df['t_star'] = t_star
    return df

def append_conditional_variance_weight(df):
    df = df.copy()
    
    t = df['t'].to_numpy()
    x = df['x'].to_numpy()

    df['weight'] = (t - x) ** 2 / np.mean((t - x) **2)
    return df

def simulate_dataset(sample_size, num_strata, seed=None):
    x = generate_x(sample_size, num_strata, seed)
    t = generate_t(x, seed)
    y = generate_y(t, x, seed)
    data = pd.DataFrame({'y': y, 't': t, 'x': x})
    return (
        data
        .pipe(append_derivative)
        .pipe(append_incremental)
        .pipe(append_t_star)
        .pipe(append_conditional_variance_weight)
        .assign(plm_plim=lambda df: df.weight * 1 / (df.t_star + 1))
    )
