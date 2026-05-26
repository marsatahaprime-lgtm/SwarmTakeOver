#!/usr/bin/env python3
"""
Control Authority Score Calculator - Drone 2 Only

This script calculates and visualizes the authority contestation timeline for a
single drone during a command injection attack. The authority score is defined
as e^(-deviation / threshold), where deviation is the drone's distance from its
planned grid path.

The plot shows:
- Authority score over time (1.0 = following planned path, <0.37 = hijacked)
- Attack onset detection based on authority drop below threshold
- Recovery detection based on sustained authority above threshold
- Telemetry gap detection
- Takeoff tolerance to ignore initial positioning errors

Used to generate Figure 4 in the paper (Authority contestation timeline).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# HARDCODED CONFIGURATION VALUES
# These are specific to Trial 1, Drone 2 analysis
# ============================================================================

# Drone configuration (HARDCODED for Drone 2, Trial 1)
DRONE_CONFIG = {
    'uri': 'radio://0/80/2M/E7E7E7E7E8',     # Drone 2 URI
    'id': 2,
    'name': 'Drone 2',
    'start': (-0.5, -1.85),                  # HARDCODED start position
    'waypoints': [(-1.04, -0.93), (-1.04, 1.09)],  # HARDCODED delivery points
    'color': '#ff7f0e',                      # Orange (matches paper)
    'log_file': 'drone_position_radio_0_80_2M_E7E7E7E7E8.jsonl'  # Input data
}

# Detection threshold (HARDCODED from paper's 0.3m deviation threshold)
DEVIATION_THRESHOLD = 0.3   # meters - when deviation exceeds this, consider "attacker leaning"

# Whether to show text callouts on the plot (True for publication version)
SHOW_CALLOUTS = True

# Takeoff tolerance settings (HARDCODED based on flight characteristics)
TAKEOFF_DURATION = 15       # seconds - ignore initial positioning errors during takeoff
TAKEOFF_TOLERANCE = 0.5     # meters - higher threshold during takeoff (0.5m vs 0.3m)
CALIBRATE_START = True      # Whether to calibrate start position from first data point
STABLE_DURATION = 30        # seconds of maintained authority before showing recovery line

# Data trimming window (HARDCODED to focus on relevant flight period)
# Keeps data from 0 to 100 seconds (attack occurs at ~25s, recovery by ~80s)
TRIM_START = 0    # seconds - remove data before this
TRIM_END = 100    # seconds - remove data after this

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_grid_path(start, waypoints, calibrated_start=None):
    """
    Generate the expected grid path for deviation calculation.
    
    The path follows the delivery pattern: vertical along street to waypoint Y,
    then horizontal to waypoint X, returning to street after each waypoint.
    
    Args:
        start: Tuple of (x, y) start coordinates
        waypoints: List of (x, y) target waypoints
        calibrated_start: Optional calibrated start position (from first data points)
    
    Returns:
        List of (x, y) path points
    """
    path = []
    height = 0.6
    
    # Use calibrated start if provided, otherwise use the defined start
    if calibrated_start:
        current_x, current_y = calibrated_start
        streetX = start[0]  # Still use original streetX for the grid pattern
    else:
        current_x, current_y = start
        streetX = start[0]
    
    # Add the starting position (calibrated if available)
    path.append((current_x, current_y, height, 0))
    
    # Add transition from start to first grid point if needed
    if calibrated_start and (abs(current_x - streetX) > 0.05 or abs(current_y - start[1]) > 0.05):
        path.append((streetX, current_y, height, 0))
    
    for target in waypoints:
        yaw = 0
        # Move vertically along street to target's Y
        path.append((streetX, target[1], height, yaw))
        
        # Move horizontally to target's X, then return to street
        if streetX != target[0]:
            path.append((target[0], target[1], height, yaw))
            path.append((streetX, target[1], height, yaw))
    
    # Return just the 2D points for deviation calculation
    return [(p[0], p[1]) for p in path]


def precompute_path_points(drone_config, calibrated_start=None):
    """
    Precompute all path points and line segments for efficient deviation calculation.
    
    Args:
        drone_config: Dictionary with drone configuration
        calibrated_start: Optional calibrated start position
    
    Returns:
        Dictionary with 'points' and 'segments' keys
    """
    path_points = generate_grid_path(drone_config['start'], drone_config['waypoints'], calibrated_start)
    
    segments = []
    for i in range(len(path_points) - 1):
        segments.append((path_points[i], path_points[i + 1]))
    
    return {
        'points': path_points,
        'segments': segments
    }


def distance_to_line_segment(px, py, p1, p2):
    """
    Calculate minimum distance from point to line segment.
    
    Args:
        px, py: Point coordinates
        p1, p2: Line segment endpoints (x, y)
    
    Returns:
        Minimum Euclidean distance
    """
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    # If segment is a point
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to segment range [0, 1]
    if t < 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    elif t > 1:
        return np.sqrt((px - x2)**2 + (py - y2)**2)
    else:
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def calculate_deviation_with_takeoff_tolerance(x, y, time, path_data, in_takeoff_phase):
    """
    Calculate deviation with special handling for takeoff phase.
    
    During takeoff, small deviations (< TAKEOFF_TOLERANCE) are ignored (return 0)
    to prevent false positives from initial positioning errors.
    
    Args:
        x, y: Current position
        time: Current time (seconds)
        path_data: Precomputed path data
        in_takeoff_phase: True if still in takeoff tolerance period
    
    Returns:
        Raw deviation (0.0 during takeoff if within tolerance)
    """
    points = path_data['points']
    segments = path_data['segments']
    
    min_distance = float('inf')
    
    # Check distance to all path points
    for px, py in points:
        dist_sq = (x - px)**2 + (y - py)**2
        if dist_sq < min_distance:
            min_distance = dist_sq
    
    # Check distance to line segments
    for p1, p2 in segments:
        dist = distance_to_line_segment(x, y, p1, p2)
        dist_sq = dist * dist
        if dist_sq < min_distance:
            min_distance = dist_sq
    
    raw_deviation = np.sqrt(min_distance)
    
    # During takeoff phase, return 0 for small deviations (so authority = 1.0)
    if in_takeoff_phase and raw_deviation < TAKEOFF_TOLERANCE:
        return 0.0
    
    return raw_deviation


def calibrate_start_position(df, defined_start, calibration_window=10):
    """
    Calibrate the actual start position from the first few data points.
    
    This accounts for slight variations in actual takeoff position vs planned.
    
    Args:
        df: DataFrame with position data
        defined_start: Planned start position (x, y)
        calibration_window: Number of initial data points to average
    
    Returns:
        Calibrated start coordinates, or None if calibration fails
    """
    if len(df) < 3:
        return None
    
    # Average the first few positions to get a stable start point
    first_rows = df.head(min(calibration_window, len(df)))
    avg_x = first_rows['x'].mean()
    avg_y = first_rows['y'].mean()
    
    # Check if the calibration is reasonable (within 1m of defined start)
    dist = np.sqrt((avg_x - defined_start[0])**2 + (avg_y - defined_start[1])**2)
    
    if dist < 1.0:
        print(f"    Calibrated start position: ({avg_x:.3f}, {avg_y:.3f}) (was ({defined_start[0]:.3f}, {defined_start[1]:.3f}))")
        return (avg_x, avg_y)
    else:
        print(f"    Calibration failed - too far from defined start ({dist:.2f}m), using defined start")
        return None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(path):
    """
    Load JSONL (JSON Lines) format log file.
    
    Args:
        path: Path to log file
    
    Returns:
        DataFrame with all records
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def process_drone_data(drone_config):
    """
    Process drone data: load, trim, calibrate, calculate deviations and authority.
    
    Args:
        drone_config: Dictionary with drone configuration
    
    Returns:
        Tuple of (DataFrame with processed data, calibrated_start_position)
    """
    print(f"\n{'='*60}")
    print(f"Processing {drone_config['name']}")
    print(f"{'='*60}")
    
    # Load data
    df = load_jsonl(drone_config['log_file'])
    
    # Sort by timestamp and add time column
    df = df.sort_values("timestamp")
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    
    # Trim data to analysis window
    original_len = len(df)
    df = df[(df["t"] >= TRIM_START) & (df["t"] <= TRIM_END)].copy()
    df["t"] = df["t"] - TRIM_START  # Rebase time to zero
    
    trimmed_len = len(df)
    print(f"  Loaded {trimmed_len} positions (removed {original_len - trimmed_len})")
    print(f"  Time range: {df['t'].min():.1f}s to {df['t'].max():.1f}s")
    
    if len(df) == 0:
        return None, None
    
    # Calibrate start position from actual data
    calibrated_start = None
    if CALIBRATE_START:
        calibrated_start = calibrate_start_position(df, drone_config['start'])
    
    # Precompute path data with calibrated start
    path_data = precompute_path_points(drone_config, calibrated_start)
    
    # Calculate deviations with takeoff tolerance
    print(f"  Calculating deviations")
    deviations = []
    for _, row in df.iterrows():
        in_takeoff = row['t'] < TAKEOFF_DURATION
        dev = calculate_deviation_with_takeoff_tolerance(
            row['x'], row['y'], row['t'], path_data, in_takeoff
        )
        deviations.append(dev)
    
    df["deviation"] = deviations
    print(f"  Deviation range: {df['deviation'].min():.3f}m to {df['deviation'].max():.3f}m")
    
    # Calculate authority score (1 = on path, 0 = far from path)
    df["authority"] = np.exp(-df["deviation"] / DEVIATION_THRESHOLD)
    print(f"  Authority range: {df['authority'].min():.3f} to {df['authority'].max():.3f}")
    
    return df, calibrated_start


# ============================================================================
# EVENT DETECTION
# ============================================================================

def detect_telemetry_gaps(df, threshold=3.0):
    """
    Detect gaps in telemetry data.
    
    Args:
        df: DataFrame with 't' column
        threshold: Multiplier for median interval to consider a gap
    
    Returns:
        Indices where gaps occur
    """
    dt = df["t"].diff()
    gap_indices = np.where(dt > dt.median() * 5)[0]
    return gap_indices


def detect_attack_and_recovery(df, threshold_authority, takeoff_duration):
    """
    Detect attack onset and recovery based on authority score.
    
    Attack onset is the first time authority drops below threshold after takeoff.
    Recovery is detected when authority stays above threshold for STABLE_DURATION seconds.
    
    Args:
        df: DataFrame with 't' and 'authority' columns
        threshold_authority: Authority threshold (0.3679 = deviation = 0.3m)
        takeoff_duration: Seconds to ignore for takeoff tolerance
    
    Returns:
        Tuple of (deviation_time, recovery_line_time, takeoff_crossing)
    """
    # Filter out takeoff period for attack detection
    post_takeoff_df = df[df["t"] >= takeoff_duration].copy()
    
    if len(post_takeoff_df) == 0:
        return None, None, False
    
    # Detect first time authority drops below threshold (attacker influence)
    deviation_time = None
    below = np.where(post_takeoff_df["authority"] < threshold_authority)[0]
    if len(below) > 0:
        deviation_time = post_takeoff_df["t"].iloc[below[0]]
    
    # Check for threshold crossings during takeoff (to be ignored)
    takeoff_df = df[df["t"] < takeoff_duration]
    takeoff_crossing = False
    if len(takeoff_df) > 0:
        takeoff_below = np.where(takeoff_df["authority"] < threshold_authority)[0]
        if len(takeoff_below) > 0:
            takeoff_crossing = True
            print(f"    Note: Authority dropped below threshold during takeoff (ignored)")
    
    # Detect stable recovery period
    recovery_line_time = None
    
    if deviation_time is not None:
        post_attack = df[df["t"] > deviation_time]
        
        # Find periods where authority is consistently above threshold
        in_stable = False
        stable_start = None
        
        for i in range(len(post_attack)):
            if post_attack["authority"].iloc[i] > threshold_authority:
                if not in_stable:
                    in_stable = True
                    stable_start = post_attack["t"].iloc[i]
            else:
                if in_stable:
                    stable_duration = post_attack["t"].iloc[i-1] - stable_start
                    if stable_duration >= STABLE_DURATION:
                        recovery_line_time = stable_start + STABLE_DURATION
                        break
                    in_stable = False
        
        # Check for stable period at the end
        if in_stable and stable_start is not None:
            stable_duration = post_attack["t"].iloc[-1] - stable_start
            if stable_duration >= STABLE_DURATION:
                recovery_line_time = stable_start + STABLE_DURATION
    
    return deviation_time, recovery_line_time, takeoff_crossing


# ============================================================================
# PLOTTING
# ============================================================================

def create_authority_plot(df, drone_config, deviation_time, recovery_line_time, 
                          gap_indices, threshold_authority, takeoff_duration, 
                          takeoff_crossing, save_path=None):
    """
    Create publication-ready authority contestation plot (Figure 4 in paper).
    
    Args:
        df: DataFrame with 't' and 'authority' columns
        drone_config: Drone configuration dictionary
        deviation_time: Time when authority first dropped below threshold
        recovery_line_time: Time when recovery was detected
        gap_indices: Indices of telemetry gaps
        threshold_authority: Authority threshold line value
        takeoff_duration: Duration of takeoff tolerance period
        takeoff_crossing: Whether threshold was crossed during takeoff
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12,5))
    
    # Main curve - authority score
    ax.plot(df["t"], df["authority"], color="black", linewidth=2)
    
    # Fill regions: above threshold = authorized, below = attacker leaning
    ax.fill_between(df["t"], df["authority"], threshold_authority,
                    where=df["authority"] > threshold_authority,
                    alpha=0.15, label="Authorized leaning")
                    
    ax.fill_between(df["t"], df["authority"], threshold_authority,
                    where=df["authority"] <= threshold_authority,
                    alpha=0.3, label="Attacker leaning")
    
    # Threshold line
    ax.axhline(threshold_authority, linestyle='--', linewidth=1, color='gray')
    
    # Telemetry gaps
    for idx in gap_indices:
        ax.axvspan(df["t"].iloc[idx-1], df["t"].iloc[idx],
                   color="gray", alpha=0.25, label="Telemetry gap" if idx == gap_indices[0] else "")
    
    # Attack onset marker
    if deviation_time is not None:
        ax.axvline(deviation_time, color='red', linestyle='--',
                   label=f"Attacker influence begins ({deviation_time:.1f}s)")
    
    # Recovery marker
    if recovery_line_time is not None and recovery_line_time <= df["t"].max():
        ax.axvline(recovery_line_time, color='green', linestyle='--', linewidth=1.5,
                   label=f"Stabilizing after recovery")
    
    # Text annotations (callouts)
    if SHOW_CALLOUTS and len(df) > 10:
        # "Authorized behavior" - after takeoff
        post_takeoff_time = takeoff_duration + 5
        if post_takeoff_time < df["t"].max():
            ax.text(post_takeoff_time, 0.65, "Authorized behavior", fontsize=9)
        
        # "Attacker influence" - after attack line
        if deviation_time is not None:
            post_attack = df[df["t"] > deviation_time]
            if len(post_attack) > 5:
                red_points = post_attack[post_attack["authority"] < threshold_authority - 0.1]
                if len(red_points) > 0:
                    text_idx = min(3, len(red_points)-1)
                    ax.text(red_points["t"].iloc[text_idx], 0.1, 
                           "Attacker influence", fontsize=9, color='red')
        
        # "Fluctuation / contest" - after first return to authorized
        if deviation_time is not None:
            post_attack = df[df["t"] > deviation_time]
            first_above = post_attack[post_attack["authority"] > threshold_authority]
            
            if len(first_above) > 0:
                first_return_time = first_above["t"].iloc[0]
                after_return = df[df["t"] > first_return_time]
                if len(after_return) > 3:
                    text_time = after_return["t"].iloc[min(2, len(after_return)-1)]
                    ax.text(text_time, 0.3, "Fluctuation / contest", 
                           fontsize=9, ha='left')
        
        # "Stabilizing" - near the end
        if len(df) > 10:
            ax.text(df["t"].iloc[-10], 0.65, "Stabilizing", fontsize=9)
        
        # Takeoff period note
        if takeoff_crossing:
            ax.text(takeoff_duration/2, 0.8, 
                   "Takeoff period\n(ignored)", 
                   fontsize=8, ha='center', color='gray',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Formatting
    ax.set_ylim(0, 1)
    ax.set_xlim(df["t"].min(), df["t"].max())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Authority score")
    ax.set_title(f"Authority Contestation and Recovery Behavior\n", fontweight='bold')
    
    # Remove duplicate legend items
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# SUMMARY OUTPUT
# ============================================================================

def print_summary(df, drone_config, deviation_time, recovery_line_time, 
                  gap_indices, takeoff_duration, takeoff_crossing):
    """
    Print summary statistics to console.
    
    Args:
        df: DataFrame with processed data
        drone_config: Drone configuration dictionary
        deviation_time: Attack onset time
        recovery_line_time: Recovery detection time
        gap_indices: Telemetry gap indices
        takeoff_duration: Takeoff tolerance duration
        takeoff_crossing: Whether threshold was crossed during takeoff
    """
    print("\n" + "="*60)
    print(f"DEVIATION SUMMARY - {drone_config['name']}")
    print("="*60)
    print(f"Analysis window: 0s to {df['t'].max():.1f}s (original {TRIM_START}s-{TRIM_END}s)")
    print(f"Takeoff tolerance: first {takeoff_duration}s (ignored)")
    print(f"Maximum deviation: {df['deviation'].max():.3f}m")
    print(f"Minimum authority: {df['authority'].min():.3f}")
    
    if takeoff_crossing:
        print(f"\nAuthority dropped below threshold during takeoff (ignored)")
    
    if deviation_time is not None:
        print(f"\nFirst attacker influence: {deviation_time:.1f}s (original {deviation_time + TRIM_START:.1f}s)")
        attack_row = df[df['t'] == deviation_time]
        if len(attack_row) > 0:
            dev_at_attack = attack_row['deviation'].values[0]
            auth_at_attack = attack_row['authority'].values[0]
            print(f"   Deviation at that time: {dev_at_attack:.3f}m")
            print(f"   Authority at that time: {auth_at_attack:.3f}")
    
    if recovery_line_time is not None:
        print(f"\nStabilizing after recovery: {recovery_line_time:.1f}s (original {recovery_line_time + TRIM_START:.1f}s)")
        if deviation_time is not None:
            print(f"   Duration of influence: {recovery_line_time - deviation_time:.1f}s")
    
    print(f"\nTelemetry gaps detected: {len(gap_indices)}")
    print("="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point - runs the authority score analysis."""
    print("\n" + "="*60)
    print("   CONTROL AUTHORITY SCORE - DRONE 2 ONLY")
    print(f"   • Log file: {DRONE_CONFIG['log_file']}")
    print(f"   • Takeoff tolerance: {TAKEOFF_DURATION}s (ignoring deviations < {TAKEOFF_TOLERANCE}m)")
    print(f"   • Deviation threshold: {DEVIATION_THRESHOLD}m")
    print(f"   • Authority threshold: {np.exp(-1.0):.3f}")
    print(f"   • Stable recovery period: {STABLE_DURATION}s")
    print("="*60 + "\n")
    
    threshold_authority = np.exp(-1.0)  # ~0.3679 when deviation = DEVIATION_THRESHOLD
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process drone data
    df, calibrated_start = process_drone_data(DRONE_CONFIG)
    
    if df is None:
        print(f"\nNo data available!")
        return
    
    # Detect events
    gap_indices = detect_telemetry_gaps(df)
    deviation_time, recovery_line_time, takeoff_crossing = detect_attack_and_recovery(
        df, threshold_authority, TAKEOFF_DURATION
    )
    
    # Create plot
    plot_filename = f"authority_drone2_takeoff_{TAKEOFF_DURATION}s_{timestamp}.png"
    create_authority_plot(
        df, DRONE_CONFIG, deviation_time, recovery_line_time,
        gap_indices, threshold_authority, TAKEOFF_DURATION,
        takeoff_crossing, save_path=plot_filename
    )
    
    # Print summary
    print_summary(df, DRONE_CONFIG, deviation_time, recovery_line_time, 
                 gap_indices, TAKEOFF_DURATION, takeoff_crossing)
    
    print(f"\nProcessing complete!")


if __name__ == '__main__':
    main()
