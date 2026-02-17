# ISTS_CVAT_annotaiton_individuals_flow.py

"""
Convert CVAT MOT 1.1 export to swim metrics with STABILIZATION + FLOW CORRECTION

This version applies the same stabilization and flow correction used in the automated
tracker to remove wave drift and camera jitter from manually-tracked CVAT data.

Usage:
    python cvat_to_swim_metrics_flow.py /path/to/gt.txt /path/to/video.mp4
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Video parameters (adjust these for your video!)
FPS = 29.55
ALTITUDE_M = 2.4  # from your SRT
GIMBAL_PITCH_DEG = -40.0
HORIZONTAL_FOV = 82.1  # Mini 4 Pro

# Flow correction parameters
FLOW_WIN = 25  # window size for sampling local flow
FLOW_SKIP = 3  # calculate flow every N frames for speed
FLOW_SCALE = 0.1  # downsample scale for flow calculation


def calculate_pixel_to_meter(frame_width, altitude_m, pitch_deg, fov_deg=82.1):
    """Calculate pixel to meter ratio"""
    eff_alt = altitude_m / max(np.cos(np.radians(abs(pitch_deg))), 1e-6)
    ground_width = 2 * eff_alt * np.tan(np.radians(fov_deg / 2))
    return ground_width / max(frame_width, 1)


def load_mot_format(mot_file):
    """Load MOT 1.1 format tracking data"""
    df = pd.read_csv(mot_file, header=None, 
                     names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height',
                           'conf', 'x', 'y', 'z'])
    
    # Calculate centroid from bounding box
    df['x_pixel'] = df['bb_left'] + df['bb_width'] / 2
    df['y_pixel'] = df['bb_top'] + df['bb_height'] / 2
    
    # Add timestamp
    df['timestamp'] = df['frame'] / FPS
    
    # Rename id to turtle_id
    df['turtle_id'] = df['id']
    
    return df[['frame', 'timestamp', 'turtle_id', 'x_pixel', 'y_pixel']].sort_values('frame')


def apply_stabilization_and_flow(df, video_path):
    """
    Apply stabilization and flow correction to CVAT tracks
    
    Returns dataframe with added columns:
    - x_stab_px, y_stab_px: stabilized pixel coordinates
    - flow_u_px, flow_v_px: local flow vectors (pixels/frame)
    - x_meters, y_meters: raw meter coordinates
    - x_stab_m, y_stab_m: stabilized meter coordinates  
    - flow_u_m, flow_v_m: flow in meters/frame
    """
    print("🌀 Applying stabilization and flow correction...")
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Video: {frame_width}x{frame_height}, {total_frames} frames")
    
    # Calculate scale
    px_to_m = calculate_pixel_to_meter(frame_width, ALTITUDE_M, GIMBAL_PITCH_DEG)
    print(f"   Pixel to meter: {px_to_m:.6f} m/px")
    
    # Stabilization state
    stab_prev_gray = None
    stab_cum_dx = 0.0
    stab_cum_dy = 0.0
    
    # Flow state
    flow_prev_gray = None
    last_dense_flow = None
    flow_frame_counter = 0
    
    # Build frame lookup for detections
    frame_data = {}
    for _, row in df.iterrows():
        fn = int(row['frame'])
        if fn not in frame_data:
            frame_data[fn] = []
        frame_data[fn].append({
            'turtle_id': int(row['turtle_id']),
            'x_pixel': row['x_pixel'],
            'y_pixel': row['y_pixel']
        })
    
    # Process video frame by frame
    results = []
    frame_num = 0
    
    while cap.isOpened() and frame_num < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update stabilization (global camera motion)
        stab_prev_gray, stab_cum_dx, stab_cum_dy = update_stabilizer(
            gray, stab_prev_gray, stab_cum_dx, stab_cum_dy
        )
        
        # Update dense flow (local wave motion)
        flow_prev_gray, last_dense_flow, flow_frame_counter = update_dense_flow(
            gray, flow_prev_gray, last_dense_flow, flow_frame_counter,
            frame_width, frame_height
        )
        
        # Process detections for this frame
        if frame_num in frame_data:
            for det in frame_data[frame_num]:
                xpx = det['x_pixel']
                ypx = det['y_pixel']
                tid = det['turtle_id']
                
                # Stabilize (remove global camera motion)
                x_stab = xpx - stab_cum_dx
                y_stab = ypx - stab_cum_dy
                
                # Sample local flow (wave motion at this location)
                flow_u, flow_v = sample_local_flow(
                    xpx, ypx, last_dense_flow, FLOW_WIN
                )
                
                # Convert to meters
                xm_raw = xpx * px_to_m
                ym_raw = ypx * px_to_m
                xm_stab = x_stab * px_to_m
                ym_stab = y_stab * px_to_m
                flow_um = flow_u * px_to_m
                flow_vm = flow_v * px_to_m
                
                results.append({
                    'frame': frame_num,
                    'timestamp': frame_num / FPS,
                    'turtle_id': tid,
                    'x_pixel': xpx,
                    'y_pixel': ypx,
                    'x_stab_px': x_stab,
                    'y_stab_px': y_stab,
                    'flow_u_px': flow_u,
                    'flow_v_px': flow_v,
                    'x_meters': xm_raw,
                    'y_meters': ym_raw,
                    'x_stab_m': xm_stab,
                    'y_stab_m': ym_stab,
                    'flow_u_m': flow_um,
                    'flow_v_m': flow_vm,
                })
        
        if frame_num % 100 == 0:
            pct = 100 * frame_num / total_frames
            print(f"   Frame {frame_num}/{total_frames} ({pct:.1f}%)")
        
        frame_num += 1
    
    cap.release()
    
    result_df = pd.DataFrame(results)
    print(f"✅ Stabilization complete: {len(result_df)} detections processed")
    return result_df


def update_stabilizer(frame_gray, prev_gray, cum_dx, cum_dy):
    """Update cumulative global translation (camera jitter removal)"""
    if prev_gray is None:
        return frame_gray, cum_dx, cum_dy
    
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=15)
    if pts is not None:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        ok = (st.flatten() == 1) if st is not None else np.array([], dtype=bool)
        if ok.any():
            d = (nxt[ok] - pts[ok]).reshape(-1, 2)
            dx = np.median(d[:, 0])
            dy = np.median(d[:, 1])
            cum_dx += float(dx)
            cum_dy += float(dy)
    
    return frame_gray, cum_dx, cum_dy


def update_dense_flow(frame_gray, prev_gray, last_flow, counter, width, height):
    """Update dense optical flow (wave motion detection)"""
    counter += 1
    
    # Skip flow calculation most frames - reuse last one
    if counter % FLOW_SKIP != 0:
        return frame_gray, last_flow, counter
    
    if prev_gray is None:
        return frame_gray, None, counter
    
    # Downsample aggressively for speed
    small_prev = cv2.resize(prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, 
                           interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(frame_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, 
                           interpolation=cv2.INTER_AREA)
    
    flow_small = cv2.calcOpticalFlowFarneback(
        small_prev, small_curr, None,
        pyr_scale=0.5, levels=2, winsize=10, iterations=2,
        poly_n=5, poly_sigma=1.1, flags=0
    )
    
    # Upscale flow back to original resolution
    flow = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR)
    flow = flow / FLOW_SCALE
    
    return frame_gray, flow, counter


def sample_local_flow(x, y, flow, win_size):
    """Sample median flow around detection point"""
    if flow is None:
        return 0.0, 0.0
    
    h, w = flow.shape[:2]
    x0 = max(0, int(x) - win_size)
    x1 = min(w, int(x) + win_size + 1)
    y0 = max(0, int(y) - win_size)
    y1 = min(h, int(y) + win_size + 1)
    
    roi = flow[y0:y1, x0:x1, :]
    if roi.size == 0:
        return 0.0, 0.0
    
    return float(np.median(roi[..., 0])), float(np.median(roi[..., 1]))


def calculate_swim_metrics(df):
    """
    Calculate speed, distance, bearing with FLOW CORRECTION
    
    Uses stabilized positions and subtracts local flow vectors
    """
    pieces = []
    
    for tid in df['turtle_id'].unique():
        track = df[df['turtle_id'] == tid].sort_values('frame').copy()
        
        if len(track) < 2:
            continue
        
        # Use stabilized positions
        xcol, ycol = 'x_stab_m', 'y_stab_m'
        
        track['prev_x'] = track[xcol].shift(1)
        track['prev_y'] = track[ycol].shift(1)
        track['prev_t'] = track['timestamp'].shift(1)
        
        track['time_diff'] = (track['timestamp'] - track['prev_t']).fillna(1.0 / FPS).replace(0, 1.0 / FPS)
        
        # Subtract local water flow from movement
        du = track['flow_u_m'].fillna(0.0)
        dv = track['flow_v_m'].fillna(0.0)
        
        dx = (track[xcol] - track['prev_x']) - du
        dy = (track[ycol] - track['prev_y']) - dv
        
        track['distance'] = np.sqrt(dx**2 + dy**2)
        track['speed'] = track['distance'] / track['time_diff']
        
        # Bearing (flow-corrected)
        track['bearing'] = np.degrees(np.arctan2(dy, dx))
        track['bearing'] = (track['bearing'] + 360) % 360
        
        # Turn angle
        turn = track['bearing'].diff()
        turn = np.where(turn > 180, turn - 360, turn)
        turn = np.where(turn < -180, turn + 360, turn)
        track['turn_angle'] = np.abs(turn)
        
        pieces.append(track)
    
    return pd.concat(pieces, ignore_index=True) if pieces else None


def summarize_behaviors(swim_data):
    """Summarize per-turtle behaviors"""
    summaries = []
    
    for tid in swim_data['turtle_id'].unique():
        track = swim_data[swim_data['turtle_id'] == tid].dropna(subset=['speed'])
        
        if len(track) < 3:
            continue
        
        summaries.append({
            'turtle_id': tid,
            'n_observations': len(track),
            'tracking_duration': track['timestamp'].max() - track['timestamp'].min(),
            'mean_speed': track['speed'].mean(),
            'max_speed': track['speed'].max(),
            'total_distance': track['distance'].sum(),
        })
    
    return pd.DataFrame(summaries)


def plot_tracks(df, save_path='cvat_flow_tracks.png'):
    """Plot turtle trajectories (using stabilized coordinates)"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ids = sorted(df['turtle_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(ids)))
    
    for i, tid in enumerate(ids):
        track = df[df['turtle_id'] == tid].sort_values('frame')
        
        # Use stabilized positions
        ax.plot(track['x_stab_m'], track['y_stab_m'], 'o-', 
                color=colors[i], label=f'T{tid}', 
                linewidth=2, markersize=4, alpha=0.8)
        
        # Mark start
        ax.plot(track['x_stab_m'].iloc[0], track['y_stab_m'].iloc[0], 's',
                color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Hatchling Tracks (CVAT + Stabilized + Flow-Corrected)', 
                 fontsize=14, fontweight='bold')
    ax.legend(ncol=2, fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved track plot: {save_path}")
    plt.close()


def plot_behaviors(swim_data, save_path='cvat_flow_behaviors.png'):
    """Plot behavioral metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ids = sorted(swim_data['turtle_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(ids)))
    
    # Speed over time
    for i, tid in enumerate(ids):
        track = swim_data[swim_data['turtle_id'] == tid]
        axes[0, 0].plot(track['timestamp'], track['speed'], 'o-', 
                       color=colors[i], alpha=0.7, label=f'T{tid}', markersize=3)
    axes[0, 0].set_title('Speed Over Time (Flow-Corrected)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed (m/s)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(ncol=2, fontsize=8)
    
    # Speed distribution
    speed_groups = [swim_data[swim_data['turtle_id'] == tid]['speed'].dropna() 
                    for tid in ids]
    axes[0, 1].boxplot(speed_groups, tick_labels=[f'T{tid}' for tid in ids])
    axes[0, 1].set_title('Speed Distribution', fontweight='bold')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Turn angle histogram
    if 'turn_angle' in swim_data.columns:
        for i, tid in enumerate(ids):
            track = swim_data[swim_data['turtle_id'] == tid]['turn_angle'].dropna()
            if len(track) > 0:
                axes[1, 0].hist(track, bins=20, alpha=0.5, color=colors[i], label=f'T{tid}')
        axes[1, 0].set_title('Turn Angle Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Degrees')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(ncol=2, fontsize=8)
    
    # Speed vs Turn
    if 'turn_angle' in swim_data.columns:
        for i, tid in enumerate(ids):
            track = swim_data[swim_data['turtle_id'] == tid]
            axes[1, 1].scatter(track['turn_angle'], track['speed'], 
                             alpha=0.6, color=colors[i], label=f'T{tid}', s=20)
        axes[1, 1].set_title('Speed vs Turn Angle', fontweight='bold')
        axes[1, 1].set_xlabel('Turn Angle (deg)')
        axes[1, 1].set_ylabel('Speed (m/s)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved behavior plot: {save_path}")
    plt.close()


def main(mot_file, video_file, output_prefix='cvat_flow'):
    """Main analysis pipeline with stabilization + flow correction"""
    print("="*70)
    print("CVAT MANUAL TRACKING → STABILIZED + FLOW-CORRECTED SWIM METRICS")
    print("="*70)
    
    # Load CVAT data
    print(f"\n📂 Loading MOT data: {mot_file}")
    df = load_mot_format(mot_file)
    print(f"   Loaded {len(df)} detections, {df['turtle_id'].nunique()} unique IDs")
    
    # Apply stabilization + flow correction
    print(f"\n🌊 Processing video with stabilization + flow: {video_file}")
    df_corrected = apply_stabilization_and_flow(df, video_file)
    
    # Save tracking data
    tracking_csv = f"{output_prefix}_tracking.csv"
    df_corrected.to_csv(tracking_csv, index=False)
    print(f"\n✅ Saved tracking data: {tracking_csv}")
    
    # Calculate swim metrics
    print(f"\n🏊 Calculating flow-corrected swim metrics...")
    swim_data = calculate_swim_metrics(df_corrected)
    
    if swim_data is not None:
        swim_csv = f"{output_prefix}_swim.csv"
        swim_data.to_csv(swim_csv, index=False)
        print(f"✅ Saved swim metrics: {swim_csv}")
        
        # Summarize behaviors
        behaviors = summarize_behaviors(swim_data)
        behavior_csv = f"{output_prefix}_behaviors.csv"
        behaviors.to_csv(behavior_csv, index=False)
        print(f"✅ Saved behavior summary: {behavior_csv}")
        
        # Print summary
        print("\n" + "="*70)
        print("BEHAVIORAL SUMMARY (FLOW-CORRECTED SPEEDS)")
        print("="*70)
        for _, row in behaviors.sort_values('mean_speed', ascending=False).iterrows():
            print(f"  T{int(row.turtle_id)}: mean {row.mean_speed:.3f} m/s, "
                  f"max {row.max_speed:.3f} m/s, dist {row.total_distance:.2f} m")
        
        # Generate plots
        print(f"\n📊 Generating visualizations...")
        plot_tracks(df_corrected, f"{output_prefix}_tracks.png")
        plot_behaviors(swim_data, f"{output_prefix}_behaviors.png")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cvat_to_swim_metrics_flow.py /path/to/gt.txt /path/to/video.mp4")
        sys.exit(1)
    
    mot_file = sys.argv[1]
    video_file = sys.argv[2]
    
    main(mot_file, video_file)