# ISTS_figs_CVAT_cap_speed_displacement.py

"""
Create ISTS presentation figures from CVAT flow-corrected tracking data

Usage:
    python ISTS_figs_from_CVAT.py /path/to/cvat_flow_tracking.csv /path/to/cvat_flow_swim.csv /path/to/cvat_flow_behaviors.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import seaborn as sns
import os

# Set style
plt.style.use('dark_background')
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 10

# Color schemes
COLORS = {
    'floater': '#4A90E2',      # Blue
    'meanderer': '#D4A574',    # Tan/gold
    'jammer': '#E67E22',       # Orange
    'background': '#2B2B2B'
}

def classify_behavior(row):
    """
    Classify turtle behavior based on NET DISPLACEMENT
    
    This is THE KEY METRIC that separates:
    - Floaters: move WITH the water (low net displacement, may have high path length from waves)
    - Jammers: swim THROUGH the water (high net displacement = actually going somewhere)
    - Meanderers: active but not directional (medium net displacement)
    """
    net_disp = row['net_displacement']
    max_speed = row['max_speed']
    
    # Filter out impossible speeds (annotation errors/teleporters)
    # But don't use this for classification - use displacement!
    if max_speed > 1.5:
        print(f"   ⚠️  T{int(row['turtle_id'])}: max speed {max_speed:.3f} m/s looks like teleporter")
    
    # Classification based on net displacement (how far they actually WENT)
    # Floaters: barely went anywhere (drifting with waves)
    if net_disp < 0.4:
        return 'floater'
    
    # Jammers: covered serious ground (swimming away from shore)
    elif net_disp > 0.8:
        return 'jammer'
    
    # Meanderers: active wandering (in between)
    else:
        return 'meanderer'


def create_behavioral_classification_figure(behaviors, swim, save_path="01_behavioral_classes_cap_speed_displace.png"):
    """
    Create the 3-panel behavioral classification figure
    Shows example tracks for floater, meanderer, jammer
    """
    print("\n📊 Creating behavioral classification figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Add behavioral class to behaviors
    behaviors['behavior_class'] = behaviors.apply(classify_behavior, axis=1)
    
    # Find examples of each class
    floater_id = behaviors[behaviors['behavior_class'] == 'floater']['turtle_id'].iloc[0] if len(behaviors[behaviors['behavior_class'] == 'floater']) > 0 else None
    meanderer_id = behaviors[behaviors['behavior_class'] == 'meanderer']['turtle_id'].iloc[0] if len(behaviors[behaviors['behavior_class'] == 'meanderer']) > 0 else None
    jammer_id = behaviors[behaviors['behavior_class'] == 'jammer']['turtle_id'].iloc[0] if len(behaviors[behaviors['behavior_class'] == 'jammer']) > 0 else None
    
    examples = [
        (floater_id, 'FLOATERS\n(Low Activity)', COLORS['floater'], 0),
        (meanderer_id, 'MEANDERERS\n(Active, Non-Directional)', COLORS['meanderer'], 1),
        (jammer_id, 'JAMMERS\n(Directional, Fast)', COLORS['jammer'], 2)
    ]
    
    for tid, title, color, idx in examples:
        ax = axes[idx]
        ax.set_facecolor(COLORS['background'])
        
        if tid is None:
            ax.text(0.5, 0.5, 'No examples\nfound', 
                   ha='center', va='center', fontsize=20, color='white')
            ax.set_title(title, fontsize=18, color=color, fontweight='bold')
            continue
            
        # Get track data (use stabilized positions)
        track = swim[swim['turtle_id'] == tid].copy()
        track_info = behaviors[behaviors['turtle_id'] == tid].iloc[0]
        
        # Use stabilized coordinates if available
        if 'x_stab_m' in track.columns:
            x_col, y_col = 'x_stab_m', 'y_stab_m'
        else:
            x_col, y_col = 'x_meters', 'y_meters'
        
        # Plot track colored by speed
        speeds = track['speed'].values
        points = ax.scatter(track[x_col], track[y_col], 
                           c=speeds, cmap='plasma', s=50, alpha=0.8,
                           vmin=0, vmax=max(speeds.max(), 0.01))
        
        # Plot line
        ax.plot(track[x_col], track[y_col], 
               '-', color=color, alpha=0.3, linewidth=2)
        
        # Mark start/end
        ax.plot(track[x_col].iloc[0], track[y_col].iloc[0], 
               's', color='lightgreen', markersize=12, markeredgecolor='white', 
               markeredgewidth=2, label='Start')
        ax.plot(track[x_col].iloc[-1], track[y_col].iloc[-1], 
               's', color='lightpink', markersize=12, markeredgecolor='white',
               markeredgewidth=2, label='End')
        
        # Add stats box
        stats_text = f"Turtle {int(tid)}\n"
        stats_text += f"Max Speed: {track_info['max_speed']:.3f} m/s\n"
        stats_text += f"Distance: {track_info['total_distance']:.1f} m\n"
        stats_text += f"Duration: {track_info['tracking_duration']:.1f} s"
        
        ax.text(0.05, 0.95, stats_text,
               transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, 
                        edgecolor=color, linewidth=2))
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(title, fontsize=18, color=color, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        ax.legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_speed_distributions(behaviors, swim, save_path="02_speed_analysis_cap_speed_displace.png"):
    """
    Create 4-panel speed analysis figure
    """
    print("\n📊 Creating speed distribution figure...")
    
    # Add behavioral class
    behaviors['behavior_class'] = behaviors.apply(classify_behavior, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # 1. Speed distribution histogram (using MAX speeds)
    ax = axes[0, 0]
    ax.set_facecolor(COLORS['background'])
    max_speeds = behaviors['max_speed'].values
    ax.hist(max_speeds, bins=30, color='#4A90E2', alpha=0.7, edgecolor='white')
    ax.axvline(0.21, color='#E74C3C', linestyle='--', linewidth=2, label='Chung et al. (2009) mean')
    ax.set_xlabel('Maximum Speed (m/s)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Speed Distribution (Flow-Corrected)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Speed by behavioral class (using MAX speeds)
    ax = axes[0, 1]
    ax.set_facecolor(COLORS['background'])
    class_order = ['floater', 'meanderer', 'jammer']
    class_data = [behaviors[behaviors['behavior_class'] == c]['max_speed'].values 
                  for c in class_order]
    class_colors = [COLORS[c] for c in class_order]
    
    bp = ax.boxplot(class_data, labels=['Floaters', 'Meanderers', 'Jammers'],
                    patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], class_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='white', linewidth=1.5)
    
    ax.set_ylabel('Maximum Speed (m/s)', fontsize=12)
    ax.set_title('Speed by Behavioral Class', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Total distance traveled by class
    ax = axes[1, 0]
    ax.set_facecolor(COLORS['background'])
    for cls in class_order:
        data = behaviors[behaviors['behavior_class'] == cls]['total_distance'].values
        if len(data) > 0:
            ax.hist(data, bins=10, alpha=0.6, label=cls.capitalize(), 
                   color=COLORS[cls], edgecolor='white')
    
    ax.set_xlabel('Total Distance (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distance Traveled Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Behavioral class pie chart
    ax = axes[1, 1]
    ax.set_facecolor(COLORS['background'])
    class_counts = behaviors['behavior_class'].value_counts()
    colors_list = [COLORS[c] for c in class_counts.index]
    
    wedges, texts, autotexts = ax.pie(class_counts.values, 
                                       labels=[c.capitalize() for c in class_counts.index],
                                       colors=colors_list,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 12, 'weight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
    
    ax.set_title('Behavioral Class Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_all_tracks_speed_colored(swim, save_path="03_all_tracks_speed_cap_speed_displace.png"):
    """All tracks colored by instantaneous speed"""
    print("\n📊 Creating all tracks speed-colored figure...")
    
    import matplotlib.cm as cm
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Use stabilized coordinates if available
    if 'x_stab_m' in swim.columns:
        x_col, y_col = 'x_stab_m', 'y_stab_m'
        title_suffix = " (Stabilized + Flow-Corrected)"
    else:
        x_col, y_col = 'x_meters', 'y_meters'
        title_suffix = ""
    
    turtle_ids = sorted(swim['turtle_id'].unique())
    
    for tid in turtle_ids:
        track = swim[swim['turtle_id'] == tid].sort_values('frame')
        if len(track) < 2:
            continue
        
        # Create line segments colored by speed
        x_vals = track[x_col].values
        y_vals = track[y_col].values
        speeds = track['speed'].values
        
        points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap='plasma', linewidth=2, alpha=0.7)
        lc.set_array(speeds[:-1])
        lc.set_clim(0, 0.5)  # Cap for better color spread
        ax.add_collection(lc)
        
        # Mark start
        ax.plot(x_vals[0], y_vals[0], 's', color='lightgreen', 
               markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    # Set limits
    all_x = swim[x_col].values
    all_y = swim[y_col].values
    x_margin = (all_x.max() - all_x.min()) * 0.05
    y_margin = (all_y.max() - all_y.min()) * 0.05
    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=0.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Speed (m/s)')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel('X (meters)', fontsize=14, color='white', weight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, color='white', weight='bold')
    ax.set_title(f'Individual Tracks Colored by Speed (n={len(turtle_ids)}){title_suffix}', 
                fontsize=16, fontweight='bold', color='#D4AF37')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_summary_stats_table(behaviors, save_path="04_summary_stats_cap_speed_displace.png"):
    """Create a summary statistics figure"""
    print("\n📊 Creating summary stats table...")
    
    behaviors['behavior_class'] = behaviors.apply(classify_behavior, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    ax.axis('off')
    
    # Calculate stats
    n_total = len(behaviors)
    n_floaters = len(behaviors[behaviors['behavior_class'] == 'floater'])
    n_meanderers = len(behaviors[behaviors['behavior_class'] == 'meanderer'])
    n_jammers = len(behaviors[behaviors['behavior_class'] == 'jammer'])
    
    overall_max_mean = behaviors['max_speed'].mean()
    overall_max_std = behaviors['max_speed'].std()
    overall_max_max = behaviors['max_speed'].max()
    
    # Create text
    title_text = "TRACKING SUMMARY (MANUAL CVAT + FLOW-CORRECTED)"
    stats_text = f"""
Total Individuals Tracked: {n_total}

Behavioral Classification:
  • Floaters (max speed < 0.3 m/s): {n_floaters} ({100*n_floaters/n_total:.1f}%)
  • Meanderers (0.3-1.0 m/s): {n_meanderers} ({100*n_meanderers/n_total:.1f}%)
  • Jammers (max speed > 1.0 m/s): {n_jammers} ({100*n_jammers/n_total:.1f}%)

Swimming Performance (Maximum Speeds):
  • Mean: {overall_max_mean:.3f} ± {overall_max_std:.3f} m/s
  • Peak: {overall_max_max:.3f} m/s
  • Range: {behaviors['max_speed'].min():.3f} - {overall_max_max:.3f} m/s

Published Comparison:
  • Chung et al. (2009): 0.05-0.21 m/s mean
  • Our peak performance: {overall_max_max:.3f} m/s (flow-corrected)
    """
    
    ax.text(0.5, 0.95, title_text, ha='center', va='top', 
           fontsize=20, fontweight='bold', color='#D4AF37',
           transform=ax.transAxes)
    
    ax.text(0.1, 0.80, stats_text, ha='left', va='top',
           fontsize=14, family='monospace', color='white',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, 
                    edgecolor='#D4AF37', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_all_tracks_overview(tracking, behaviors, save_path="05_all_tracks_validated_cap_speed_displace.png"):
    """All tracks in one figure with individual colors - CLEANED UP"""
    print("\n📊 Creating all tracks overview...")
    
    import matplotlib.cm as cm
    
    # Filter out teleporters (max speed > 1.5 m/s)
    valid_turtles = behaviors[behaviors['max_speed'] < 1.5]['turtle_id'].values
    tracking_filtered = tracking[tracking['turtle_id'].isin(valid_turtles)]
    
    print(f"   • Showing {len(valid_turtles)} valid tracks (filtered {len(behaviors) - len(valid_turtles)} teleporters)")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Use stabilized coordinates if available
    if 'x_stab_m' in tracking_filtered.columns:
        x_col, y_col = 'x_stab_m', 'y_stab_m'
        title_suffix = " (Stabilized)"
    else:
        x_col, y_col = 'x_meters', 'y_meters'
        title_suffix = ""
    
    colors = cm.tab20(np.linspace(0, 1, len(valid_turtles)))
    turtle_ids = sorted(valid_turtles)
    
    for idx, tid in enumerate(turtle_ids):
        track = tracking_filtered[tracking_filtered['turtle_id'] == tid].sort_values('frame')
        if len(track) < 2:
            continue
        
        color = colors[idx % 20]
        # THINNER LINES, no intermediate markers
        ax.plot(track[x_col], track[y_col], '-', color=color, 
               linewidth=1.2, alpha=0.75)
        
        # Start markers only (smaller, less clutter)
        ax.plot(track[x_col].iloc[0], track[y_col].iloc[0], 
               'o', color=color, markersize=5, 
               markeredgecolor='white', markeredgewidth=0.5, zorder=10, alpha=0.9)
    
    ax.set_xlabel('X (meters)', fontsize=16, color='white', weight='bold')
    ax.set_ylabel('Y (meters)', fontsize=16, color='white', weight='bold')
    ax.set_title(f'All Tracks (n={len(valid_turtles)}){title_suffix}', 
                fontsize=20, color='#D4AF37', weight='bold', pad=20)
    ax.grid(True, alpha=0.3, color='#505050')
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=12)
    
    # NO LEGEND - too cluttered with 29 tracks!
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"✅ Saved: {save_path}")
    
    ax.set_xlabel('X (meters)', fontsize=16, color='white', weight='bold')
    ax.set_ylabel('Y (meters)', fontsize=16, color='white', weight='bold')
    ax.set_title(f'Validated Tracks (n={len(turtle_ids)}){title_suffix}', 
                fontsize=20, color='#D4AF37', weight='bold', pad=20)
    ax.grid(True, alpha=0.3, color='#505050')
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
             fontsize=11, framealpha=0.9, facecolor='#2B2B2B', edgecolor='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"✅ Saved: {save_path}")


def main(tracking_csv, swim_csv, behaviors_csv, output_dir="ISTS_figures_CVAT"):
    """Generate all ISTS presentation figures from CVAT data"""
    
    print("="*70)
    print("ISTS PRESENTATION FIGURES FROM CVAT MANUAL TRACKING")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading CVAT data...")
    tracking = pd.read_csv(tracking_csv)
    swim = pd.read_csv(swim_csv)
    behaviors = pd.read_csv(behaviors_csv)
    
    print(f"   • {len(tracking)} detections")
    print(f"   • {behaviors['turtle_id'].nunique()} unique turtles")
    print(f"   • Max speed: {behaviors['max_speed'].max():.3f} m/s")
    
    # Calculate net displacement (straight-line start to end)
    print("\n📏 Calculating net displacement...")
    displacements = []
    
    for tid in tracking['turtle_id'].unique():
        track = tracking[tracking['turtle_id'] == tid].sort_values('frame')
        
        # Use stabilized coordinates if available
        if 'x_stab_m' in track.columns:
            x_col, y_col = 'x_stab_m', 'y_stab_m'
        else:
            x_col, y_col = 'x_meters', 'y_meters'
        
        # Straight line from start to end
        x_start, y_start = track[x_col].iloc[0], track[y_col].iloc[0]
        x_end, y_end = track[x_col].iloc[-1], track[y_col].iloc[-1]
        
        net_disp = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        displacements.append({'turtle_id': tid, 'net_displacement': net_disp})
    
    disp_df = pd.DataFrame(displacements)
    behaviors = behaviors.merge(disp_df, on='turtle_id')
    
    print(f"   ✓ Net displacement calculated for {len(behaviors)} turtles")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate figures
    print(f"\n🎨 Generating figures in {output_dir}...")
    
    create_behavioral_classification_figure(
        behaviors, swim, 
        save_path=os.path.join(output_dir, "01_behavioral_classes_cap_speed_displace.png")
    )
    
    create_speed_distributions(
        behaviors, swim,
        save_path=os.path.join(output_dir, "02_speed_analysis_cap_speed_displace.png")
    )
    
    create_all_tracks_speed_colored(
        swim,
        save_path=os.path.join(output_dir, "03_all_tracks_speed_cap_speed_displace.png")
    )
    
    create_summary_stats_table(
        behaviors,
        save_path=os.path.join(output_dir, "04_summary_stats_cap_speed_displace.png")
    )
    
    create_all_tracks_overview(
        tracking, behaviors,
        save_path=os.path.join(output_dir, "05_all_tracks_cap_speed_displace.png")
    )
    
    print("\n" + "="*70)
    print("✅ ALL FIGURES CREATED!")
    print("="*70)
    print(f"\n📁 Saved to: {output_dir}")
    print("\n🐢 Figures ready for ISTS presentation! 🎤")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python ISTS_figs_from_CVAT.py tracking.csv swim.csv behaviors.csv")
        sys.exit(1)
    
    tracking_csv = sys.argv[1]
    swim_csv = sys.argv[2]
    behaviors_csv = sys.argv[3]
    
    main(tracking_csv, swim_csv, behaviors_csv)