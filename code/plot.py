from matplotlib import pyplot as plt
import numpy as np
import pandas as pd   # ensure pandas is available


def plot_simulation(df):
    """
    Three vertically stacked plots:

      1.  f(x)=log(x+1) plus *one* tangent line for each of T and ωT*.
          •  The slope of each tangent is the average derivative E[1/(X+1)].
          •  The tangent point x0 satisfies 1/(x0+1)=E[1/(X+1)].

      2.  Histogram of T with a vertical line at mean(T).

      3.  Histogram of ωT* with a vertical line at mean(ωT*).

    Parameters
    ----------
    df : pd.DataFrame
        Columns required: 't', 'weight', 't_star'.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ---------- data ----------
    T = df["t"].to_numpy()
    T_star = df["t_star"].to_numpy()
    WT_star = (df["weight"] * df["t_star"]).to_numpy()

    mean_T, mean_WT_star = T.mean(), WT_star.mean()

    # ---------- helper: compute slope and tangent point ----------
    def slope_and_anchor(values, weights=None):
        """Return (slope, x0, f(x0)) for the average derivative tangent."""
        if weights is None:
            weights = np.ones(len(values))
        slope = np.mean(1.0 / (values + 1) * weights)
        x0 = 1.0 / slope - 1.0            # solve 1/(x0+1)=slope
        y0 = np.log(x0 + 1.0)
        return slope, x0, y0

    m_T,  x0_T,  y0_T  = slope_and_anchor(T)
    m_WT, x0_WT, y0_WT = slope_and_anchor(T_star, df["weight"].to_numpy())
    print(m_T)
    print(m_WT)
    # ---------- domain ----------
    x = np.linspace(0, 10, 300)
    f_x = np.log(x + 1)

    # ---------- tangent lines ----------
    tangent_T  = y0_T  + m_T  * (x - x0_T)
    tangent_WT = y0_WT + m_WT * (x - x0_WT)

    # ---------- figure ----------
    color_T, color_WT_star = "tab:blue", "tab:red"
    fig, axs = plt.subplots(3, 1, figsize=(4, 8), sharex=True)

    # -- subplot 1: curve + tangents --
    axs[0].plot(x, f_x, color="black", label=r"$\log(x+1)$")
    axs[0].plot(x, tangent_T,  "--", color=color_T,
                label="E[f'(T)]")
    axs[0].plot(x, tangent_WT, "--", color=color_WT_star,
                label="E[ωf'(T*)]")
    axs[0].set_xlim(0, 10)
    axs[0].set_title("f(x)")
    axs[0].set_ylabel("y")
    axs[0].legend()

    # -- subplot 2: histogram of T --
    bins_T = np.arange(0, 12) - 0.5
    axs[1].hist(T, bins=bins_T, color=color_T, alpha=0.5)
    axs[1].axvline(mean_T, color=color_T, linestyle="--", linewidth=2,
                   label=f"Mean = {mean_T:.2f}")
    axs[1].set_xlim(0, 10)
    axs[1].set_title("Observed Treatment Distribution")
    axs[1].set_ylabel("Count")
    axs[1].legend()

    # -- subplot 3: histogram of ωT* --
    bins_WT_star = np.linspace(0, 10, 21)
    axs[2].hist(WT_star, bins=bins_WT_star, color=color_WT_star, alpha=0.5)
    axs[2].axvline(mean_WT_star, color=color_WT_star, linestyle="--", linewidth=2,
                   label=f"Mean = {mean_WT_star:.2f}")
    axs[2].set_xlim(0, 10)
    axs[2].set_title("'Effective' Treatment Distribution")
    axs[2].set_xlabel("Value")
    axs[2].set_ylabel("Count")
    axs[2].legend()

    plt.tight_layout()
    return fig


TALK_FIG_SIZE = (4, 3)


def plot_f():
    fig, axs = plt.subplots(1, 1, figsize=TALK_FIG_SIZE, sharex=True)

    x = np.linspace(0, 10, 300)
    f_x = np.log(x + 1)
    # -- subplot 1: curve + tangents --
    axs.plot(x, f_x, color="black", label=r"$\log(t+1)$")
    axs.set_xlim(0, 10)
    axs.set_ylim(0, 4)
    axs.set_title("Dose-Response Function (f(t))", fontsize=16)
    axs.set_ylabel("f(t)", fontsize=14)
    axs.set_xlabel("t", fontsize=14)
    # axs.legend()

    axs.set_yticks([])
    plt.tight_layout()
    return fig


def plot_dist(df):
    fig, axs = plt.subplots(1, 1, figsize=TALK_FIG_SIZE, sharex=True)

    T = df["t"].to_numpy()
    mean_T = T.mean()
    color_T = "tab:blue"

    # -- subplot 2: histogram of T --
    bins_T = np.arange(0, 12) - 0.5
    axs.hist(T, bins=bins_T, color=color_T, alpha=0.5)
    axs.axvline(mean_T, color=color_T, linestyle="--", linewidth=2, label=f"Mean = {mean_T:.2f}")
    axs.set_xlim(0, 10)
    axs.set_title("Observed Treatment (T)", fontsize=16)
    axs.set_xlabel("t", fontsize=14)
    axs.set_ylabel("Draws", fontsize=14)
    axs.set_yticks([])
    axs.legend()

    plt.tight_layout()
    return fig



def plot_deriv(df):
 # ---------- data ----------
    T = df["t"].to_numpy()
    T_star = df["t_star"].to_numpy()


    # ---------- helper: compute slope and tangent point ----------
    def slope_and_anchor(values, weights=None):
        """Return (slope, x0, f(x0)) for the average derivative tangent."""
        if weights is None:
            weights = np.ones(len(values))
        slope = np.mean(1.0 / (values + 1) * weights)
        x0 = 1.0 / slope - 1.0            # solve 1/(x0+1)=slope
        y0 = np.log(x0 + 1.0)
        return slope, x0, y0

    m_T,  x0_T,  y0_T  = slope_and_anchor(T)
    m_WT, x0_WT, y0_WT = slope_and_anchor(T_star, df["weight"].to_numpy())
    print(m_T)
    print(m_WT)
    # ---------- domain ----------
    x = np.linspace(0, 10, 300)
    f_x = np.log(x + 1)

    # ---------- tangent lines ----------
    tangent_T  = y0_T  + m_T  * (x - x0_T)
    tangent_WT = y0_WT + m_WT * (x - x0_WT)

    color_T, color_WT_star = "tab:blue", "tab:red"
    fig, axs = plt.subplots(1, 1, figsize=TALK_FIG_SIZE, sharex=True)

    # -- subplot 1: curve + tangents --
    axs.plot(x, f_x, color="black", label=r"$\log(t+1)$")
    axs.plot(x, tangent_T,  "--", color=color_T,
                label="E[f'(T)]")
    axs.plot(x, tangent_WT, "--", color=color_WT_star,
                label="E[f'(ωT*)]")
    axs.set_xlim(0, 10)
    axs.set_ylim(0, 4)
    
    axs.set_title("f(t)", fontsize=16)
    axs.set_ylabel("y", fontsize=14)
    axs.set_xlabel("t", fontsize=14)
    axs.legend()
    axs.set_yticks([])
    plt.tight_layout()
    return fig



def plot_tstar_dist(df):
    # ---------- data ----------
    T = df["t"].to_numpy()
    T_star = df["t_star"].to_numpy()
    WT_star = (df["weight"] * df["t_star"]).to_numpy()

    mean_WT_star = WT_star.mean()

    # ---------- helper: compute slope and tangent point ----------
    def slope_and_anchor(values, weights=None):
        """Return (slope, x0, f(x0)) for the average derivative tangent."""
        if weights is None:
            weights = np.ones(len(values))
        slope = np.mean(1.0 / (values + 1) * weights)
        x0 = 1.0 / slope - 1.0            # solve 1/(x0+1)=slope
        y0 = np.log(x0 + 1.0)
        return slope, x0, y0

    m_T,  _,  _  = slope_and_anchor(T)
    m_WT, _, _ = slope_and_anchor(T_star, df["weight"].to_numpy())
    print(m_T)
    print(m_WT)
    
    # ---------- figure ----------
    color_WT_star = "tab:red"
    fig, axs = plt.subplots(1, 1, figsize=TALK_FIG_SIZE, sharex=True)

    # -- subplot 3: histogram of ωT* --
    bins_WT_star = np.linspace(0, 10, 21)
    axs.hist(WT_star, bins=bins_WT_star, color=color_WT_star, alpha=0.5)
    axs.axvline(mean_WT_star, color=color_WT_star, linestyle="--", linewidth=2,
                   label=f"Mean = {mean_WT_star:.2f}")
    axs.set_xlim(0, 10)
    axs.set_yticks([])

    axs.set_title("'Effective' Treatment (ωT*)", fontsize=16)
    axs.set_xlabel("t", fontsize=14)
    axs.set_ylabel("Draws", fontsize=14)
    axs.legend()

    plt.tight_layout()
    return fig
