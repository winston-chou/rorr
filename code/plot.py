from matplotlib import pyplot as plt
import numpy as np


def plot_simulation(df):
    """
    Creates a 3x1 vertically aligned plot with the following subplots:
    
      1. Plot the function f(T)=log(T+1) over the domain [0,10]. Overlay two tangent
         lines to this curve at the x-values given by:
           - mean of df["t"]
           - mean of (df["weight"] * df["t_star"])
         (Each tangent line has slope 1/(x+1) at its corresponding x-value.)
         
      2. Histogram (bar plot) of "t" with a vertical line at its mean.
            
      3. Histogram (bar plot) of "weight * t_star" with a vertical line at its mean.
      
    All subplots share the same x-axis domain [0, 10]. The colors for the tangent lines
    in subplot 1 are coordinated with the vertical mean lines in the corresponding histograms.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns 't', 'weight', and 't_star'.
    
    Returns:
        matplotlib.figure.Figure: The figure object containing the 4 subplots.
    """
    # Compute the means used for tangent lines and histogram vertical lines.
    mean_T = df["t"].mean()
    mean_WT_star = (df["weight"] * df["t_star"]).mean()
    
    # Define the colors used for each quantity for consistency:
    color_T = 'tab:blue'
    color_WT_star = 'tab:red'
    
    # Define the domain for plotting: x from 0 to 10.
    x = np.linspace(0, 10, 300)
    f_x = np.log(x + 1)  # f(x) = log(x+1)
    
    # For each chosen x0, compute the tangent line to f(x).
    # Tangent line at x0: L(x) = log(x0+1) + (1/(x0+1))*(x - x0)
    def tangent_line(x, x0):
        return np.log(x0 + 1) + (1 / (x0 + 1)) * (x - x0)
    
    tangent_T = tangent_line(x, mean_T)
    tangent_WT_star = tangent_line(x, mean_WT_star)
    
    # Create a 3x1 subplot figure (vertical stack).
    fig, axs = plt.subplots(3, 1, figsize=(4, 8), sharex=True)
    
    # --- Subplot 1: Function and tangent lines.
    axs[0].plot(x, f_x, label=r'$\log(T+1)$', color='black')
    axs[0].plot(x, tangent_T, label=f'Tangent at mean(T) = {mean_T:.2f}', 
                color=color_T, linestyle='--')
    axs[0].plot(x, tangent_WT_star, label=f'Tangent at mean(ωT*) = {mean_WT_star:.2f}', 
                color=color_WT_star, linestyle='--')
    axs[0].set_xlim(0, 10)
    axs[0].set_title('f(X) and Tangent Lines')
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('Y')
    axs[0].legend()
    
    # --- Subplot 2: Histogram of T.
    # Use bins for integer data (e.g., bins from -0.5 to 10.5 for T values 0 to 10).
    bins_T = np.arange(0, 12) - 0.5
    axs[1].hist(df["t"], bins=bins_T, color=color_T, alpha=0.5)
    axs[1].axvline(mean_T, color=color_T, linestyle='--', linewidth=2,
                   label=f'Mean = {mean_T:.2f}')
    axs[1].set_xlim(0, 10)
    axs[1].set_title("Observed Treatment Distribution")
    axs[1].set_xlabel('T')
    axs[1].set_ylabel('Count')
    axs[1].legend()

    # --- Subplot 3: Histogram of W*T_star.
    product_WT_star = df["weight"] * df["t_star"]
    bins_WT_star = np.linspace(0, 10, 21)
    axs[2].hist(product_WT_star, bins=bins_WT_star, color=color_WT_star, alpha=0.5)
    mean_product_WT_star = product_WT_star.mean()
    axs[2].axvline(mean_product_WT_star, color=color_WT_star, linestyle='--', linewidth=2,
                   label=f'Mean = {mean_product_WT_star:.2f}')
    axs[2].set_xlim(0, 10)
    axs[2].set_title("'Effective' Treatment Distribution")
    axs[2].set_xlabel('ωT*')
    axs[2].set_ylabel('Count')
    axs[2].legend()
    plt.tight_layout()
    return fig
