"""
Periospot Plotting Style Helpers
Author: Francisco Teixeira Barbosa (Cisco)

Purpose: Enforce Periospot brand colors and consistent matplotlib styling
         across all figures generated in this project.

Usage:
    # Method 1: Import color constants directly
    from ps_plot import (
        set_style, save_figure, 
        PERIOSPOT_BLUE, CRIMSON_BLAZE, VANILLA_CREAM
    )
    
    set_style()  # Apply once at notebook start
    
    fig, ax = plt.subplots()
    ax.plot(x, y, color=PERIOSPOT_BLUE)
    save_figure(fig, "figures/my_plot.png")
    
    # Method 2: Load palette as dictionary
    from ps_plot import set_style, get_palette, save_figure
    
    set_style()
    colors = get_palette()  # Returns dict: {'periospot_blue': '#15365a', ...}
    
    fig, ax = plt.subplots()
    ax.bar(x, y, color=colors['periospot_blue'])
    save_figure(fig, "figures/my_plot.png")

Available Color Constants:
    HEX (strings):
        PERIOSPOT_BLUE, MYSTIC_BLUE, PERIOSPOT_RED, CRIMSON_BLAZE,
        VANILLA_CREAM, BLACK, WHITE, CLASSIC_PERIOSPOTBLUE,
        PERIOSPOT_LIGHT_BLUE, PERIOSPOT_DARK_BLUE, PERIOSPOT_YELLOW,
        PERIOSPOT_BRIGHT_BLUE
    
    RGB (tuples for libraries like missingno):
        PERIOSPOT_BLUE_RGB, MYSTIC_BLUE_RGB, PERIOSPOT_RED_RGB,
        CRIMSON_BLAZE_RGB, VANILLA_CREAM_RGB, BLACK_RGB, WHITE_RGB,
        PERIOSPOT_YELLOW_RGB
    
    Utility:
        hex_to_rgb() - Convert any hex color to RGB tuple
"""

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Color Constants (loaded from config at module import time)
# =============================================================================

def _load_color_constants():
    """
    Load Periospot color constants at module import time.
    This allows direct usage like: from ps_plot import PERIOSPOT_BLUE
    """
    try:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        palette = config.get('plotting', {}).get('palette', {})
        return palette
    except Exception:
        # Fallback colors if config not available
        return {
            'periospot_blue': '#15365a',
            'mystic_blue': '#003049',
            'periospot_red': '#6c1410',
            'crimson_blaze': '#a92a2a',
            'vanilla_cream': '#f7f0da',
            'black': '#000000',
            'white': '#ffffff',
        }

# Load palette and create module-level constants
_PALETTE = _load_color_constants()

# Export color constants as HEX strings (can be imported directly)
PERIOSPOT_BLUE = _PALETTE.get('periospot_blue', '#15365a')
MYSTIC_BLUE = _PALETTE.get('mystic_blue', '#003049')
PERIOSPOT_RED = _PALETTE.get('periospot_red', '#6c1410')
CRIMSON_BLAZE = _PALETTE.get('crimson_blaze', '#a92a2a')
VANILLA_CREAM = _PALETTE.get('vanilla_cream', '#f7f0da')
BLACK = _PALETTE.get('black', '#000000')
WHITE = _PALETTE.get('white', '#ffffff')
CLASSIC_PERIOSPOTBLUE = _PALETTE.get('classic_periospotblue', '#0031af')
PERIOSPOT_LIGHT_BLUE = _PALETTE.get('periospot_light_blue', '#0297ed')
PERIOSPOT_DARK_BLUE = _PALETTE.get('periospot_dark_blue', '#02011e')
PERIOSPOT_YELLOW = _PALETTE.get('periospot_yellow', '#ffc430')
PERIOSPOT_BRIGHT_BLUE = _PALETTE.get('periospot_bright_blue', '#1040dd')

# Export RGB versions (for libraries that require RGB tuples like missingno)
PERIOSPOT_BLUE_RGB = (0.082, 0.212, 0.353)  # #15365a
MYSTIC_BLUE_RGB = (0.000, 0.188, 0.286)     # #003049
PERIOSPOT_RED_RGB = (0.424, 0.078, 0.063)   # #6c1410
CRIMSON_BLAZE_RGB = (0.663, 0.165, 0.165)   # #a92a2a
VANILLA_CREAM_RGB = (0.969, 0.941, 0.855)   # #f7f0da
BLACK_RGB = (0.000, 0.000, 0.000)           # #000000
WHITE_RGB = (1.000, 1.000, 1.000)           # #ffffff
PERIOSPOT_YELLOW_RGB = (1.000, 0.769, 0.188)  # #ffc430


def load_config() -> Dict:
    """
    Load configuration from configs/config.yaml
    
    Returns:
        Dict containing full configuration including plotting palette
    """
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please ensure configs/config.yaml exists."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color string to RGB tuple (normalized 0-1).
    
    Args:
        hex_color: Hex color string (e.g., '#15365a' or '15365a')
    
    Returns:
        Tuple of (r, g, b) with values in [0, 1] range
    
    Example:
        >>> hex_to_rgb('#15365a')
        (0.082, 0.212, 0.353)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB (0-255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Normalize to 0-1
    return (r / 255.0, g / 255.0, b / 255.0)


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
            classic_periospotblue: #0031af
            periospot_light_blue: #0297ed
            periospot_dark_blue: #02011e
            periospot_yellow: #ffc430
            periospot_bright_blue: #1040dd
    TODO: Extract palette from config
    TODO: Return as dict with color_name -> hex_code
    """
    config = load_config()
    palette = config.get('plotting', {}).get('palette', {})
    
    if not palette:
        raise ValueError(
            "No palette found in config. "
            "Ensure config.yaml contains 'plotting.palette' section."
        )
    
    return palette


def get_default_colors() -> List[str]:
    """
    Get default color sequence for categorical plots.
    
    Returns:
        List of hex color codes in priority order
    """
    config = load_config()
    default_colors = config.get('plotting', {}).get('default_colors', [])
    
    if not default_colors:
        # Fallback to palette values if default_colors not specified
        palette = get_palette()
        default_colors = list(palette.values())
    
    return default_colors


def set_style() -> None:
    """
    Apply Periospot matplotlib style globally.
    
    Sets:
        - Font family (DejaVu Sans)
        - Font sizes (title=16, label=12, tick=10)
        - Figure DPI (300 for publication quality)
        - Seaborn context and palette
    
    Call this once at the start of your notebook.
    """
    config = load_config()
    mpl_config = config.get('plotting', {}).get('matplotlib', {})
    
    # Font settings
    plt.rcParams['font.family'] = mpl_config.get('font_family', 'DejaVu Sans')
    plt.rcParams['axes.titlesize'] = mpl_config.get('title_size', 16)
    plt.rcParams['axes.labelsize'] = mpl_config.get('label_size', 12)
    plt.rcParams['xtick.labelsize'] = mpl_config.get('tick_size', 10)
    plt.rcParams['ytick.labelsize'] = mpl_config.get('tick_size', 10)
    
    # Figure quality
    plt.rcParams['figure.dpi'] = mpl_config.get('figure_dpi', 300)
    plt.rcParams['savefig.dpi'] = mpl_config.get('figure_dpi', 300)
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Set seaborn style and palette
    sns.set_context("notebook", font_scale=1.0)
    sns.set_style("whitegrid")
    
    # Apply Periospot color palette
    default_colors = get_default_colors()
    sns.set_palette(default_colors)


def styled_fig_ax(figsize: tuple = (10, 6), **kwargs):
    """
    Create a styled figure and axes with Periospot defaults.
    
    Args:
        figsize: Tuple of (width, height) in inches
        **kwargs: Additional arguments passed to plt.subplots()
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    
    # Apply minimal styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax


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
    """
    # Load DPI from config if not provided
    if dpi is None:
        config = load_config()
        dpi = config.get('plotting', {}).get('matplotlib', {}).get('figure_dpi', 300)
    
    # Create parent directory if it doesn't exist
    filepath_obj = Path(filepath)
    filepath_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"📊 Saved figure: {filepath}")


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
    """
    palette = get_palette()
    
    if color_key not in palette:
        raise ValueError(
            f"Color '{color_key}' not found in palette. "
            f"Available colors: {list(palette.keys())}"
        )
    
    color = palette[color_key]
    
    # Apply to bar plots
    for patch in ax.patches:
        patch.set_facecolor(color)
    
    # Apply to line plots
    for line in ax.lines:
        line.set_color(color)
    
    # Apply to scatter plots and other collections
    for collection in ax.collections:
        collection.set_color(color)


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

