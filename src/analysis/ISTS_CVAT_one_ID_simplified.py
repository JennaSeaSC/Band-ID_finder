# ISTS_CVAT_one_ID_simplified.py
 

"""
Simplified Pipeline Figure: 2-panel version showing RESULTS
Trajectory (colored by speed) + Speed Profile

Usage:
    python ISTS_simple_pipeline_CVAT.py cvat_flow_tracking.csv cvat_flow_swim.csv [turtle_id]
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib.collections import LineCollection
import os

# Presentation colors
COLORS = {
    'background': '#1a1a1a',
    'gold': '#D4A574',
    'orange': '#E07A39',
    'text': '#FFFFFF',
    'grid': '#333333',
}

def set_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'savefig.facecolor': COLORS['background'],
        'text.color': COLORS['text'],
        'axes.labelcolor': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'font.size': 16,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'legend.fontsize': 16,
    })


def create_simple_pipeline(tracking_df, swim_df, turtle_id, output_path):
    """Create simplified 2-panel pipeline figure"""
    
    print(f"\n📊 Creating simplified pipeline for Turtle {turtle_id}...")
    
    # Get this turtle's data
    track_df = tracking_df[tracking_df['turtle_id'] == turtle_id].sort_values('frame').copy()
    swim_track = swim_df[swim_df['turtle_id'] == turtle_id].sort_values('frame').copy()
    
    if len(track_df) == 0:
        print(f"❌ No data found for turtle {turtle_id}")
        return
    
    print(f"   Track length: {len(track_df)} points")
    print(f"   Duration: {track_df['timestamp'].max() - track_df['timestamp'].min():.1f}s")
    if len(swim_track) > 0:
        print(f"   Mean speed: {swim_track['speed'].mean():.3f} m/s")
        print(f"   Max speed: {swim_track['speed'].max():.3f} m/s")
    
    set_plot_style()
    
    # Create figure with 2 panels side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Determine which coordinate columns to use
    if 'x_stab_m' in track_df.columns:
        x_m_col, y_m_col = 'x_stab_m', 'y_stab_m'
    else:
        x_m_col, y_m_col = 'x_meters', 'y_meters'
    
    x_m = track_df[x_m_col].values
    y_m = track_df[y_m_col].values
    
    # ========================================================================
    # PANEL 1: Trajectory colored by speed
    # ========================================================================
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['background'])
    
    # Get speeds from swim data - match by frame number
    if len(swim_track) > 0:
        # Create dict mapping frame -> speed
        speed_dict = dict(zip(swim_track['frame'].values, swim_track['speed'].values))
        
        # Match speeds to tracking frames
        speeds_matched = np.array([speed_dict.get(f, np.nan) for f in track_df['frame'].values])
        
        # Fill NaN with 0 for visualization
        speeds_matched = np.nan_to_num(speeds_matched, nan=0.0)
        
        # Debug info
        valid_speeds = speeds_matched[speeds_matched > 0]
        if len(valid_speeds) > 0:
            print(f"   Speed range in swim data: {swim_track['speed'].min():.3f} - {swim_track['speed'].max():.3f} m/s")
            print(f"   Valid speeds matched: {len(valid_speeds)}/{len(speeds_matched)}")
            print(f"   Matched speed range: {valid_speeds.min():.3f} - {valid_speeds.max():.3f} m/s")
        else:
            print(f"   ⚠️  No frame overlap between tracking and swim data!")
    else:
        speeds_matched = np.zeros(len(x_m))
    
    # Create line segments colored by speed
    points = np.array([x_m, y_m]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Use actual speed range for colormap
    speed_max = max(speeds_matched.max(), 0.1)
    
    lc = LineCollection(segments, cmap='plasma', linewidth=6)
    lc.set_array(speeds_matched[:-1])
    lc.set_clim(0, speed_max)
    ax1.add_collection(lc)
    
    # Start/end markers
    ax1.plot(x_m[0], y_m[0], 'o', color='grey', markersize=18, 
             markeredgecolor='white', markeredgewidth=2, zorder=10, label='Start')
    ax1.plot(x_m[-1], y_m[-1], 's', color='black', markersize=16,
             markeredgecolor='white', markeredgewidth=2, zorder=10, label='End')
    
    # Set limits with inverted Y-axis (shore at bottom, swimming upward)
    x_margin = (x_m.max() - x_m.min()) * 0.15
    y_margin = (y_m.max() - y_m.min()) * 0.15
    ax1.set_xlim(x_m.min() - x_margin, x_m.max() + x_margin)
    ax1.set_ylim(y_m.max() + y_margin, y_m.min() - y_margin)  # Inverted Y
    ax1.set_aspect('equal')
    
    ax1.set_title(f'Turtle {turtle_id} Trajectory', fontsize=28, color=COLORS['gold'], 
                  pad=20, weight='bold')
    ax1.set_xlabel('X (meters)', fontsize=22, weight='bold')
    ax1.set_ylabel('Y (meters)', fontsize=22, weight='bold')
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=2)
    ax1.tick_params(labelsize=18)
    ax1.legend(fontsize=14, loc='upper right', framealpha=0.95, 
              edgecolor='white', fancybox=False)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma', 
                               norm=plt.Normalize(vmin=0, vmax=speed_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Speed (m/s)', fontsize=20, weight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    # ========================================================================
    # PANEL 2: Speed profile over time
    # ========================================================================
    ax2 = axes[1]
    ax2.set_facecolor(COLORS['background'])
    
    if len(swim_track) > 0:
        time_rel = swim_track['timestamp'] - swim_track['timestamp'].iloc[0]
        speeds_plot = swim_track['speed'].values
        
        ax2.plot(time_rel, speeds_plot, 
                 color=COLORS['orange'], linewidth=4, zorder=2)
        ax2.fill_between(time_rel, 0, speeds_plot, 
                         color=COLORS['orange'], alpha=0.5, zorder=1)
        
        # Reference line
        ax2.axhline(0.21, color='red', linestyle='--', linewidth=3) 
                    # label='Published mean\n(Chung et al. 2009)', zorder=3)
        
        # Stats box
        mean_speed = swim_track['speed'].mean()
        max_speed = swim_track['speed'].max()
        duration = track_df['timestamp'].max() - track_df['timestamp'].min()
        
        stats_text = (f"Duration: {duration:.1f}s\n"
                      f"Mean: {mean_speed:.3f} m/s\n"
                      f"Max: {max_speed:.3f} m/s")
        
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                 fontsize=18, va='top', ha='right', weight='bold',
                 bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
                          alpha=0.9, edgecolor=COLORS['orange'], linewidth=3))
    else:
        ax2.text(0.5, 0.5, "Speed data\nnot available", 
                ha='center', va='center', fontsize=24, color='white')
    
    ax2.set_title('Swimming Speed', fontsize=28, color=COLORS['gold'], 
                  pad=20, weight='bold')
    ax2.set_xlabel('Time (s)', fontsize=22, weight='bold')
    ax2.set_ylabel('Speed (m/s)', fontsize=22, weight='bold')
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=2)
    ax2.tick_params(labelsize=18)
    if len(swim_track) > 0:
        ax2.legend(fontsize=16, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=COLORS['background'], 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def main(tracking_csv, swim_csv, turtle_id=None, output_dir="ISTS_figures_CVAT"):
    """Generate simplified 2-panel pipeline figure"""
    
    print("=" * 80)
    print("SIMPLIFIED PIPELINE FIGURE - 2 PANELS (CVAT DATA)")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading CVAT data...")
    tracking_df = pd.read_csv(tracking_csv)
    swim_df = pd.read_csv(swim_csv)
    
    print(f"✅ {len(tracking_df)} detections, {tracking_df['turtle_id'].nunique()} unique IDs")
    
    # Select featured turtle
    if turtle_id is not None:
        print(f"\n🎯 Using specified turtle: {turtle_id}")
    else:
        # Use longest track
        track_lengths = tracking_df.groupby('turtle_id').size()
        turtle_id = track_lengths.idxmax()
        print(f"\n🎯 Using longest track: Turtle {turtle_id} ({track_lengths.max()} points)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"07_simple_pipeline_turtle_{turtle_id}.png")
    
    # Create figure
    create_simple_pipeline(tracking_df, swim_df, turtle_id, output_path)
    
    print("\n" + "=" * 80)
    print("✨ SIMPLIFIED PIPELINE FIGURE COMPLETE!")
    print("=" * 80)
    print(f"\nSaved to: {output_path}")
    print("\n💡 Perfect for 7-min ISTS talk - shows THE RESULT, not the process")
    print("\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ISTS_simple_pipeline_CVAT.py tracking.csv swim.csv [turtle_id]")
        sys.exit(1)
    
    tracking_csv = sys.argv[1]
    swim_csv = sys.argv[2]
    turtle_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    main(tracking_csv, swim_csv, turtle_id)

# ========== the one below works, but formatting shifts oddly/individual ========

# """
# Simplified Pipeline Figure: 2-panel version showing RESULTS
# Trajectory (colored by speed) + Speed Profile

# Usage:
#     python ISTS_simple_pipeline_CVAT.py cvat_flow_tracking.csv cvat_flow_swim.csv [turtle_id]
# """

# import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# import numpy as np
# from matplotlib.collections import LineCollection
# import os

# # Presentation colors
# COLORS = {
#     'background': '#1a1a1a',
#     'gold': '#D4A574',
#     'orange': '#E07A39',
#     'text': '#FFFFFF',
#     'grid': '#333333',
# }

# def set_plot_style():
#     plt.style.use('dark_background')
#     plt.rcParams.update({
#         'figure.facecolor': COLORS['background'],
#         'axes.facecolor': COLORS['background'],
#         'savefig.facecolor': COLORS['background'],
#         'text.color': COLORS['text'],
#         'axes.labelcolor': COLORS['text'],
#         'xtick.color': COLORS['text'],
#         'ytick.color': COLORS['text'],
#         'grid.color': COLORS['grid'],
#         'font.size': 16,
#         'axes.titlesize': 24,
#         'axes.labelsize': 20,
#         'legend.fontsize': 16,
#     })


# def create_simple_pipeline(tracking_df, swim_df, turtle_id, output_path):
#     """Create simplified 2-panel pipeline figure"""
    
#     print(f"\n📊 Creating simplified pipeline for Turtle {turtle_id}...")
    
#     # Get this turtle's data
#     track_df = tracking_df[tracking_df['turtle_id'] == turtle_id].sort_values('frame').copy()
#     swim_track = swim_df[swim_df['turtle_id'] == turtle_id].sort_values('frame').copy()
    
#     if len(track_df) == 0:
#         print(f"❌ No data found for turtle {turtle_id}")
#         return
    
#     print(f"   Track length: {len(track_df)} points")
#     print(f"   Duration: {track_df['timestamp'].max() - track_df['timestamp'].min():.1f}s")
#     if len(swim_track) > 0:
#         print(f"   Mean speed: {swim_track['speed'].mean():.3f} m/s")
#         print(f"   Max speed: {swim_track['speed'].max():.3f} m/s")
    
#     set_plot_style()
    
#     # Create figure with 2 panels side-by-side
#     fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
#     # Determine which coordinate columns to use
#     if 'x_stab_m' in track_df.columns:
#         x_m_col, y_m_col = 'x_stab_m', 'y_stab_m'
#     else:
#         x_m_col, y_m_col = 'x_meters', 'y_meters'
    
#     x_m = track_df[x_m_col].values
#     y_m = track_df[y_m_col].values
    
#     # ========================================================================
#     # PANEL 1: Trajectory colored by speed
#     # ========================================================================
#     ax1 = axes[0]
#     ax1.set_facecolor(COLORS['background'])
    
#     # Get speeds from swim data - match by frame number
#     if len(swim_track) > 0:
#         # Create dict mapping frame -> speed
#         speed_dict = dict(zip(swim_track['frame'].values, swim_track['speed'].values))
        
#         # Match speeds to tracking frames
#         speeds_matched = np.array([speed_dict.get(f, np.nan) for f in track_df['frame'].values])
        
#         # Fill NaN with 0 for visualization
#         speeds_matched = np.nan_to_num(speeds_matched, nan=0.0)
        
#         # Debug info
#         valid_speeds = speeds_matched[speeds_matched > 0]
#         if len(valid_speeds) > 0:
#             print(f"   Speed range in swim data: {swim_track['speed'].min():.3f} - {swim_track['speed'].max():.3f} m/s")
#             print(f"   Valid speeds matched: {len(valid_speeds)}/{len(speeds_matched)}")
#             print(f"   Matched speed range: {valid_speeds.min():.3f} - {valid_speeds.max():.3f} m/s")
#         else:
#             print(f"   ⚠️  No frame overlap between tracking and swim data!")
#     else:
#         speeds_matched = np.zeros(len(x_m))
    
#     # Create line segments colored by speed
#     points = np.array([x_m, y_m]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
#     # Use actual speed range for colormap
#     speed_max = max(speeds_matched.max(), 0.1)
    
#     lc = LineCollection(segments, cmap='plasma', linewidth=6)
#     lc.set_array(speeds_matched[:-1])
#     lc.set_clim(0, speed_max)
#     ax1.add_collection(lc)
    
#     # Start/end markers
#     ax1.plot(x_m[0], y_m[0], 'o', color='grey', markersize=24, 
#              markeredgecolor='white', markeredgewidth=3, zorder=10, label='Start')
#     ax1.plot(x_m[-1], y_m[-1], 's', color='black', markersize=24,
#              markeredgecolor='white', markeredgewidth=3, zorder=10, label='End')
    
#     # Set limits with inverted Y-axis (shore at bottom, swimming upward)
#     x_margin = (x_m.max() - x_m.min()) * 0.15
#     y_margin = (y_m.max() - y_m.min()) * 0.15
#     ax1.set_xlim(x_m.min() - x_margin, x_m.max() + x_margin)
#     ax1.set_ylim(y_m.max() + y_margin, y_m.min() - y_margin)  # Inverted Y
#     ax1.set_aspect('equal')
    
#     ax1.set_title(f'Turtle {turtle_id} Trajectory', fontsize=28, color=COLORS['gold'], 
#                   pad=20, weight='bold')
#     ax1.set_xlabel('X (meters)', fontsize=22, weight='bold')
#     ax1.set_ylabel('Y (meters)', fontsize=22, weight='bold')
#     ax1.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=2)
#     ax1.tick_params(labelsize=18)
#     ax1.legend(fontsize=18, loc='upper left', framealpha=0.9)
    
#     # Colorbar
#     sm = plt.cm.ScalarMappable(cmap='plasma', 
#                                norm=plt.Normalize(vmin=0, vmax=speed_max))
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
#     cbar.set_label('Speed (m/s)', fontsize=20, weight='bold')
#     cbar.ax.tick_params(labelsize=16)
    
#     # ========================================================================
#     # PANEL 2: Speed profile over time
#     # ========================================================================
#     ax2 = axes[1]
#     ax2.set_facecolor(COLORS['background'])
    
#     if len(swim_track) > 0:
#         time_rel = swim_track['timestamp'] - swim_track['timestamp'].iloc[0]
#         speeds_plot = swim_track['speed'].values
        
#         ax2.plot(time_rel, speeds_plot, 
#                  color=COLORS['orange'], linewidth=4, zorder=2)
#         ax2.fill_between(time_rel, 0, speeds_plot, 
#                          color=COLORS['orange'], alpha=0.5, zorder=1)
        
#         # Reference line
#         ax2.axhline(0.21, color='red', linestyle='--', linewidth=3) 
#                     # label='Published mean\n(Chung et al. 2009)', zorder=3)
        
#         # Stats box
#         mean_speed = swim_track['speed'].mean()
#         max_speed = swim_track['speed'].max()
#         duration = track_df['timestamp'].max() - track_df['timestamp'].min()
        
#         stats_text = (f"Duration: {duration:.1f}s\n"
#                       f"Mean: {mean_speed:.3f} m/s\n"
#                       f"Max: {max_speed:.3f} m/s")
        
#         ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
#                  fontsize=18, va='top', ha='right', weight='bold',
#                  bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
#                           alpha=0.9, edgecolor=COLORS['orange'], linewidth=3))
#     else:
#         ax2.text(0.5, 0.5, "Speed data\nnot available", 
#                 ha='center', va='center', fontsize=24, color='white')
    
#     ax2.set_title('Swimming Speed', fontsize=28, color=COLORS['gold'], 
#                   pad=20, weight='bold')
#     ax2.set_xlabel('Time (s)', fontsize=22, weight='bold')
#     ax2.set_ylabel('Speed (m/s)', fontsize=22, weight='bold')
#     ax2.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=2)
#     ax2.tick_params(labelsize=18)
#     if len(swim_track) > 0:
#         ax2.legend(fontsize=16, loc='upper right', framealpha=0.9)
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, facecolor=COLORS['background'], 
#                 edgecolor='none', bbox_inches='tight')
#     plt.close()
    
#     print(f"✅ Saved: {output_path}")


# def main(tracking_csv, swim_csv, turtle_id=None, output_dir="ISTS_figures_CVAT"):
#     """Generate simplified 2-panel pipeline figure"""
    
#     print("=" * 80)
#     print("SIMPLIFIED PIPELINE FIGURE - 2 PANELS (CVAT DATA)")
#     print("=" * 80)
    
#     # Load data
#     print(f"\n📂 Loading CVAT data...")
#     tracking_df = pd.read_csv(tracking_csv)
#     swim_df = pd.read_csv(swim_csv)
    
#     print(f"✅ {len(tracking_df)} detections, {tracking_df['turtle_id'].nunique()} unique IDs")
    
#     # Select featured turtle
#     if turtle_id is not None:
#         print(f"\n🎯 Using specified turtle: {turtle_id}")
#     else:
#         # Use longest track
#         track_lengths = tracking_df.groupby('turtle_id').size()
#         turtle_id = track_lengths.idxmax()
#         print(f"\n🎯 Using longest track: Turtle {turtle_id} ({track_lengths.max()} points)")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"07_simple_pipeline_turtle_{turtle_id}.png")
    
#     # Create figure
#     create_simple_pipeline(tracking_df, swim_df, turtle_id, output_path)
    
#     print("\n" + "=" * 80)
#     print("✨ SIMPLIFIED PIPELINE FIGURE COMPLETE!")
#     print("=" * 80)
#     print(f"\nSaved to: {output_path}")
#     print("\n💡 Perfect for 7-min ISTS talk - shows THE RESULT, not the process")
#     print("\n")


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python ISTS_simple_pipeline_CVAT.py tracking.csv swim.csv [turtle_id]")
#         sys.exit(1)
    
#     tracking_csv = sys.argv[1]
#     swim_csv = sys.argv[2]
#     turtle_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
#     main(tracking_csv, swim_csv, turtle_id)