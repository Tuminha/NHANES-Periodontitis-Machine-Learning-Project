"""
Periospot Plotting Style Helpers
Author: Francisco Teixeira Barbosa (Cisco)

Purpose: Enforce Periospot brand colors and consistent matplotlib styling
         across all figures generated in this project.

Usage:
    from src.ps_plot import set_style, get_palette, save_figure
    
    set_style()  # Apply once at notebook start
    colors = get_palette()
    
    fig, ax = plt.subplots()
    # ... your plot code ...
    save_figure(fig, "figures/my_plot.png")
"""

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def load_config() -> Dict:
    """
    Load configuration from configs/config.yaml
    
    Returns:
        Dict containing full configuration including plotting palette
    
    TODO: Implement YAML loading with error handling
    TODO: Add validation that plotting.palette exists
    """
    # TODO: Load YAML file
    # TODO: Handle FileNotFoundError gracefully
    # TODO: Return config dict
    pass


def get_palette() -> Dict[str, str]:
    """
    Get Periospot brand color palette as a dictionary.
    
    Returns:
        Dict mapping color names to hex codes:
            periospot_blue: #15365a
            mystic_blue: #003049
            periospot_red: #6c1410
            crimson_blaze: #a92a2a
            vanilla_cream: #f7f0da
            black: #000000
            white: #ffffff
    
    TODO: Extract palette from config
    TODO: Return as dict with color_name -> hex_code
    """
    # TODO: Call load_config()
    # TODO: Extract config['plotting']['palette']
    # TODO: Return palette dict
    pass


def get_default_colors() -> List[str]:
    """
    Get default color sequence for categorical plots.
    
    Returns:
        List of hex color codes in priority order
    
    TODO: Extract default_colors from config
    """
    # TODO: Load config
    # TODO: Return config['plotting']['default_colors']
    pass


def set_style() -> None:
    """
    Apply Periospot matplotlib style globally.
    
    Sets:
        - Font family (DejaVu Sans)
        - Font sizes (title=16, label=12, tick=10)
        - Figure DPI (300 for publication quality)
        - Seaborn context and palette
    
    Call this once at the start of your notebook.
    
    TODO: Load font settings from config
    TODO: Apply via plt.rcParams and sns.set_context()
    """
    # TODO: Load config['plotting']['matplotlib']
    # TODO: Set plt.rcParams['font.family']
    # TODO: Set plt.rcParams['font.size'], axes.titlesize, axes.labelsize, xtick.labelsize
    # TODO: Set plt.rcParams['figure.dpi']
    # TODO: Call sns.set_context() with appropriate settings
    # TODO: Set seaborn palette to default_colors
    pass


def styled_fig_ax(figsize: tuple = (10, 6), **kwargs):
    """
    Create a styled figure and axes with Periospot defaults.
    
    Args:
        figsize: Tuple of (width, height) in inches
        **kwargs: Additional arguments passed to plt.subplots()
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    
    TODO: Call plt.subplots with appropriate styling
    TODO: Optionally apply grid, spines, etc.
    """
    # TODO: Create fig, ax = plt.subplots(figsize=figsize, **kwargs)
    # TODO: Apply any default styling (grid, spines)
    # TODO: Return fig, ax
    pass


def save_figure(
    fig,
    filepath: str,
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
    **kwargs
) -> None:
    """
    Save figure to disk with Periospot defaults.
    
    Args:
        fig: Matplotlib figure object
        filepath: Output path (e.g., "figures/my_plot.png")
        dpi: DPI override (defaults to config value, typically 300)
        bbox_inches: Bounding box mode (default "tight")
        **kwargs: Additional arguments for fig.savefig()
    
    TODO: Load DPI from config if not provided
    TODO: Create parent directory if missing
    TODO: Save with fig.savefig()
    TODO: Print confirmation message
    """
    # TODO: If dpi is None, load from config
    # TODO: Ensure Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # TODO: fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    # TODO: print(f"Saved figure: {filepath}")
    pass


def apply_periospot_colors_to_ax(ax, color_key: str = "periospot_blue") -> None:
    """
    Apply a single Periospot brand color to an axes object.
    
    Useful for:
        - Bar plots: apply to all bars
        - Line plots: apply to line color
        - Scatter plots: apply to marker color
    
    Args:
        ax: Matplotlib axes object
        color_key: Name of color from palette (e.g., "periospot_blue")
    
    TODO: Get color hex from palette
    TODO: Apply to ax children (bars, lines, collections)
    """
    # TODO: palette = get_palette()
    # TODO: color = palette[color_key]
    # TODO: Iterate over ax.patches (bars), ax.lines, ax.collections
    # TODO: Set facecolor or color to chosen color
    pass


# =============================================================================
# Example Usage (for testing in a notebook)
# =============================================================================

def demo():
    """
    Demo function showing how to use these helpers.
    Run this in a notebook cell to verify style setup.
    """
    # TODO: Uncomment and test
    # set_style()
    # palette = get_palette()
    # print("Periospot Palette:")
    # for name, hex_code in palette.items():
    #     print(f"  {name}: {hex_code}")
    #
    # fig, ax = styled_fig_ax(figsize=(8, 5))
    # ax.plot([1, 2, 3], [1, 4, 2], linewidth=2, color=palette['periospot_blue'])
    # ax.set_title("Demo Plot with Periospot Blue")
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # save_figure(fig, "figures/demo_plot.png")
    pass

