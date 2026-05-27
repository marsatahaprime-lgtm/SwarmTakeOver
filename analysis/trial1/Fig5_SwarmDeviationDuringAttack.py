#!/usr/bin/env python3
"""
Swarm Deviation Plotter with Takeoff Tolerance

This script generates stacked deviation plots for multiple drones
during command injection attacks. It creates two-panel plots per drone:
- Top panel: Deviation from planned path over time
- Bottom panel: Binary state (Normal/Hijacked)

Features:
- Data trimming to focus on relevant time window
- Takeoff tolerance to ignore initial positioning errors
- Automatic start position calibration
- Hijack event detection with configurable thresholds
- Telemetry gap detection
- Color-coded deviation regions and event markers

Used to generate Figure 5 in the paper (Swarm deviation during parallel attack).
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import time
import os

# ============================================================================
# HARDCODED CONFIGURATION - TRIAL 1 SPECIFIC
# ============================================================================

# Mission definitions for both drones (HARDCODED for Trial 1)
MISSIONS = {
    'radio://0/80/2M/E7E7E7E7E9': {  # Drone 1
        'id': 1,
        'name': 'Drone 1',
        'uri_short': 'E7E7E7E7E9',
        'start': (0.4, -1.85),                              # HARDCODED start
        'waypoints': [(0.85, -1.24), (0.85, -0.78), (0.85, 0.95)],  # HARDCODED waypoints
        'color': '#1f77b4',  # Blue
    },
    'radio://0/80/2M/E7E7E7E7E8': {  # Drone 2
        'id': 2,
        'name': 'Drone 2',
        'uri_short': 'E7E7E7E7E8',
        'start': (-0.5, -1.85),                             # HARDCODED start
        'waypoints': [(-1.04, -0.93), (-1.04, 1.09)],       # HARDCODED waypoints
        'color': '#ff7f0e',  # Orange
    }
}

# Data trimming window (HARDCODED for Trial 1, 0-100 seconds)
TRIM_START = 0      # Remove data before this time (seconds)
TRIM_END = 100      # Remove data after this time (seconds)

# Detection thresholds (from paper)
DEVIATION_THRESHOLD = 0.3   # meters - hijack detection threshold
TELEMETRY_THRESHOLD = 3.0   # seconds - telemetry loss detection

# Takeoff tolerance settings (HARDCODED based on flight characteristics)
TAKEOFF_DURATION = 15       # seconds - ignore initial positioning errors
TAKEOFF_TOLERANCE = 0.5     # meters - higher threshold during takeoff
CALIBRATE_START = True      # calibrate start position from first data point

# Plot colors (HARDCODED for consistency with paper)
COLORS = {
    'hijack': '#d62728',     # Red for hijacked periods
    'recovery': '#2ca02c',   # Green for recovery
    'threshold': '#7f7f7f',  # Gray for threshold line
    'warning': '#ffbb78',    # Light orange for warnings
    'takeoff': '#c0c0c0',    # Gray for takeoff period
}

# ============================================================================
# PATH GENERATION AND GEOMETRY
# ============================================================================

def generate_grid_path(start, waypoints, calibrated_start=None):
    """
    Generate the expected grid path with optional calibrated start position.
    
    Path follows: vertical along street to waypoint Y, then horizontal to waypoint X,
    returning to street after each waypoint.
    
    Args:
        start: Tuple of (x, y) defined start coordinates
        waypoints: List of (x, y) target waypoints
        calibrated_start: Optional calibrated start from actual data
    
    Returns:
        List of (x, y, z, yaw) path points
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
    
    return path


def precompute_path_points(drone_info, calibrated_start=None):
    """
    Precompute all path points and line segments for efficient deviation calculation.
    
    Args:
        drone_info: Dictionary with drone configuration
        calibrated_start: Optional calibrated start position
    
    Returns:
        Dictionary with 'points' and 'segments' keys
    """
    path_with_z = generate_grid_path(drone_info['start'], drone_info['waypoints'], calibrated_start)
    path_points = [(p[0], p[1]) for p in path_with_z]
    
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
    
    During takeoff, deviations below TAKEOFF_TOLERANCE are ignored (return 0.0)
    to prevent false positives from initial positioning errors.
    
    Args:
        x, y: Current position
        time: Current time (seconds)
        path_data: Precomputed path data
        in_takeoff_phase: True if still in takeoff tolerance period
    
    Returns:
        Deviation value (0.0 if ignored during takeoff)
    """
    points = path_data['points']
    segments = path_data['segments']
    
    min_distance = float('inf')
    
    # Check distance to all path points
    for px, py in points:
        dist = (x - px)**2 + (y - py)**2
        if dist < min_distance:
            min_distance = dist
    
    # Check distance to line segments
    for p1, p2 in segments:
        dist = distance_to_line_segment(x, y, p1, p2)
        dist_sq = dist * dist
        if dist_sq < min_distance:
            min_distance = dist_sq
    
    raw_deviation = np.sqrt(min_distance)
    
    # During takeoff phase, use higher tolerance
    if in_takeoff_phase:
        if raw_deviation < TAKEOFF_TOLERANCE:
            return 0.0  # Ignore small deviations during takeoff
        else:
            return raw_deviation  # Report large deviations even during takeoff
    
    return raw_deviation


def calibrate_start_position(positions, defined_start, calibration_window=10):
    """
    Calibrate the actual start position from the first few data points.
    
    Args:
        positions: List of position dictionaries
        defined_start: Planned start position (x, y)
        calibration_window: Number of initial data points to average
    
    Returns:
        Calibrated start coordinates, or None if calibration fails
    """
    if not positions or len(positions) < 3:
        return None
    
    # Average the first few positions to get a stable start point
    avg_x = np.mean([p['x'] for p in positions[:min(calibration_window, len(positions))]])
    avg_y = np.mean([p['y'] for p in positions[:min(calibration_window, len(positions))]])
    
    # Check if the calibration is reasonable (within 1m of defined start)
    dist = np.sqrt((avg_x - defined_start[0])**2 + (avg_y - defined_start[1])**2)
    
    if dist < 1.0:
        print(f"    Calibrated start position: ({avg_x:.3f}, {avg_y:.3f}) (was ({defined_start[0]:.3f}, {defined_start[1]:.3f}))")
        return (avg_x, avg_y)
    else:
        print(f"    Calibration failed - too far from defined start ({dist:.2f}m), using defined start")
        return None


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def find_drone_log_files():
    """
    Find log files for all drones defined in MISSIONS.
    
    Searches for files matching pattern 'drone_position_*{drone_id}*.jsonl'
    
    Returns:
        Dictionary mapping URIs to filename paths
    """
    log_files = glob.glob("drone_position_*.jsonl")
    print(f"\nFound {len(log_files)} log files:")
    for f in log_files:
        print(f"  - {f}")
    
    drone_files = {}
    
    for uri, drone_info in MISSIONS.items():
        uri_short = drone_info['uri_short']
        matching = [f for f in log_files if uri_short in f]
        
        if matching:
            drone_files[uri] = matching[0]
            print(f"Matched {drone_info['name']} -> {matching[0]}")
        else:
            print(f"No log file found for {drone_info['name']}")
    
    return drone_files


def load_and_trim_data(uri, filename, trim_start, trim_end):
    """
    Load data, trim to [trim_start, trim_end], and rebase time so new time 0 = trim_start.
    
    Args:
        uri: Drone URI
        filename: Path to log file
        trim_start: Start of trimming window (seconds)
        trim_end: End of trimming window (seconds)
    
    Returns:
        Tuple of (positions_list, start_epoch)
    """
    all_positions = []
    start_epoch = None
    
    print(f"  Loading and trimming data...")
    print(f"  Keeping only data from {trim_start}s to {trim_end}s (original time)")
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#HEADER'):
                    try:
                        data = json.loads(line)
                        
                        if start_epoch is None:
                            start_epoch = data['timestamp']
                        
                        orig_time = data['timestamp'] - start_epoch
                        
                        # Only keep data within trim window
                        if trim_start <= orig_time <= trim_end:
                            # Rebase time: new_time = orig_time - trim_start
                            new_time = orig_time - trim_start
                            
                            all_positions.append({
                                'orig_time': orig_time,
                                'time': new_time,  # Plotted time (starts at 0)
                                'x': data['x'],
                                'y': data['y'],
                                'z': data.get('z', 0),
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        print(f"  Loaded {len(all_positions)} positions after trimming")
        if all_positions:
            print(f"  Original time range: {all_positions[0]['orig_time']:.1f}s to {all_positions[-1]['orig_time']:.1f}s")
            print(f"  New time range (plotted): 0s to {all_positions[-1]['time']:.1f}s")
            print(f"  Duration after trimming: {all_positions[-1]['time']:.1f}s")
        
        return all_positions, start_epoch
    
    except Exception as e:
        print(f"  Error: {e}")
        return [], None


def detect_hijack_events_with_takeoff_tolerance(positions, path_data, takeoff_duration):
    """
    Detect hijack events with takeoff phase tolerance.
    
    Args:
        positions: List of position dictionaries
        path_data: Precomputed path data
        takeoff_duration: Duration of takeoff tolerance period (seconds)
    
    Returns:
        List of event dictionaries
    """
    events = []
    in_hijack = False
    hijack_start = None
    max_dev_in_current = 0
    
    print(f"  Scanning {len(positions)} positions...")
    print(f"  Takeoff tolerance: first {takeoff_duration}s (ignoring deviations < {TAKEOFF_TOLERANCE}m)")
    start_time = time.time()
    
    for i, pos in enumerate(positions):
        # Determine if we're in takeoff phase
        in_takeoff = pos['time'] < takeoff_duration
        
        # Calculate deviation with takeoff tolerance
        deviation = calculate_deviation_with_takeoff_tolerance(
            pos['x'], pos['y'], pos['time'], path_data, in_takeoff
        )
        
        # Show progress every 20 positions
        if i % 20 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"    Progress: {i}/{len(positions)} positions ({i/len(positions)*100:.1f}%)")
        
        if in_hijack:
            max_dev_in_current = max(max_dev_in_current, deviation)
        
        # Determine trigger condition
        trigger_condition = deviation > DEVIATION_THRESHOLD
        if in_takeoff:
            # During takeoff, require much larger deviation to trigger
            trigger_condition = deviation > TAKEOFF_TOLERANCE
        
        if trigger_condition and not in_hijack:
            in_hijack = True
            hijack_start = pos['time']
            max_dev_in_current = deviation
            events.append({
                'type': 'HIJACK_START',
                'time': pos['time'],
                'orig_time': pos['orig_time'],
                'deviation': deviation,
                'index': i,
                'in_takeoff': in_takeoff
            })
            takeoff_note = " (during takeoff)" if in_takeoff else ""
            print(f"    HIJACK START at {pos['time']:.1f}s (orig {pos['orig_time']:.1f}s), dev={deviation:.2f}m{takeoff_note}")
        
        elif deviation <= DEVIATION_THRESHOLD and in_hijack and not in_takeoff:
            # Don't count recovery during takeoff
            in_hijack = False
            events.append({
                'type': 'RECOVERY',
                'time': pos['time'],
                'orig_time': pos['orig_time'],
                'deviation': deviation,
                'duration': pos['time'] - hijack_start,
                'orig_duration': pos['orig_time'] - (hijack_start + TRIM_START),
                'max_deviation': max_dev_in_current,
                'index': i
            })
            print(f"    RECOVERY at {pos['time']:.1f}s, duration={pos['time']-hijack_start:.1f}s")
            max_dev_in_current = 0
    
    if in_hijack and hijack_start:
        events.append({
            'type': 'HIJACK_ONGOING',
            'time': positions[-1]['time'],
            'orig_time': positions[-1]['orig_time'],
            'start_time': hijack_start,
            'orig_start': hijack_start + TRIM_START,
            'duration': positions[-1]['time'] - hijack_start,
            'orig_duration': positions[-1]['orig_time'] - (hijack_start + TRIM_START),
            'max_deviation': max_dev_in_current,
        })
        print(f"    HIJACK ONGOING at end, duration={positions[-1]['time']-hijack_start:.1f}s")
    
    total_time = time.time() - start_time
    print(f"  Event detection completed in {total_time:.1f}s")
    
    return events


def detect_telemetry_gaps(positions, threshold=TELEMETRY_THRESHOLD):
    """
    Detect gaps in telemetry data.
    
    Args:
        positions: List of position dictionaries
        threshold: Minimum gap duration to report (seconds)
    
    Returns:
        List of gap dictionaries
    """
    gaps = []
    for i in range(1, len(positions)):
        gap = positions[i]['time'] - positions[i-1]['time']
        if gap > threshold:
            gaps.append({
                'start': positions[i-1]['time'],
                'end': positions[i]['time'],
                'orig_start': positions[i-1]['orig_time'],
                'orig_end': positions[i]['orig_time'],
                'duration': gap,
            })
            print(f"    Telemetry gap: {gap:.1f}s at {positions[i-1]['time']:.1f}s (orig {positions[i-1]['orig_time']:.1f}s)")
    return gaps


# ============================================================================
# PLOTTING
# ============================================================================

def create_stacked_plot(all_drone_data, all_events, all_path_data, takeoff_duration, save_path=None):
    """
    Create stacked plot with rebased time and takeoff period highlighted.
    
    Generates a two-panel plot per drone:
    - Top: Deviation from planned path over time
    - Bottom: Binary state (Normal=0, Hijacked=1)
    
    Args:
        all_drone_data: Dictionary with drone position data
        all_events: Dictionary with event data per drone
        all_path_data: Dictionary with precomputed path data
        takeoff_duration: Duration of takeoff tolerance period
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure object
    """
    n_drones = len(all_drone_data)
    if n_drones == 0:
        return None
    
    fig, axes = plt.subplots(n_drones * 2, 1, figsize=(16, 10), dpi=150, 
                             sharex=True, gridspec_kw={'height_ratios': [3, 1, 3, 1]})
    
    # Find max time after trimming
    max_time = 0
    for uri, data in all_drone_data.items():
        if data['positions']:
            max_time = max(max_time, data['positions'][-1]['time'])
    
    plot_idx = 0
    for uri, data in all_drone_data.items():
        drone_info = data['info']
        positions = data['positions']
        events = all_events.get(uri, [])
        gaps = data['gaps']
        path_data = all_path_data.get(uri)
        
        if not positions:
            continue
        
        # Calculate deviations with takeoff tolerance
        print(f"  Calculating deviations for {drone_info['name']}...")
        deviations = []
        times = []
        for p in positions:
            in_takeoff = p['time'] < takeoff_duration
            dev = calculate_deviation_with_takeoff_tolerance(p['x'], p['y'], p['time'], path_data, in_takeoff)
            deviations.append(dev)
            times.append(p['time'])
        
        # Top plot: Deviation
        ax_dev = axes[plot_idx * 2]
        
        # Highlight takeoff period
        if takeoff_duration > 0:
            ax_dev.axvspan(0, min(takeoff_duration, max_time), alpha=0.1, color=COLORS['takeoff'], 
                          label='Takeoff period (ignored)')
        
        ax_dev.plot(times, deviations, '-', color=drone_info['color'], 
                   linewidth=1.5, alpha=0.8, zorder=3)
        
        # Color the area under the curve
        for i in range(len(times) - 1):
            if deviations[i] > DEVIATION_THRESHOLD:
                ax_dev.fill_between(times[i:i+2], 0, [deviations[i], deviations[i+1]], 
                                   color=COLORS['hijack'], alpha=0.2, zorder=2)
            else:
                ax_dev.fill_between(times[i:i+2], 0, [deviations[i], deviations[i+1]], 
                                   color=drone_info['color'], alpha=0.2, zorder=2)
        
        # Threshold line
        ax_dev.axhline(y=DEVIATION_THRESHOLD, color=COLORS['threshold'], 
                      linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Mark events
        for event in events:
            if event['type'] == 'HIJACK_START':
                marker = 'v' if not event.get('in_takeoff', False) else 'X'
                color = COLORS['hijack'] if not event.get('in_takeoff', False) else COLORS['warning']
                ax_dev.scatter(event['time'], event['deviation'], 
                              s=150, c=color, marker=marker, 
                              edgecolors='white', linewidth=2, zorder=5)
            elif event['type'] == 'RECOVERY':
                ax_dev.scatter(event['time'], event['deviation'], 
                              s=150, c=COLORS['recovery'], marker='^', 
                              edgecolors='white', linewidth=2, zorder=5)
        
        # Telemetry gaps
        for gap in gaps:
            ax_dev.axvspan(gap['start'], gap['end'], alpha=0.2, color=COLORS['warning'])
        
        ax_dev.set_ylabel(f'{drone_info["name"]}\nDeviation (m)', fontsize=11, fontweight='bold')
        ax_dev.set_ylim(bottom=0)
        ax_dev.set_xlim(0, max_time)
        ax_dev.grid(True, alpha=0.3)
        
        ax_dev.text(0.02, 0.95, drone_info['name'], transform=ax_dev.transAxes,
                   fontsize=12, fontweight='bold', color=drone_info['color'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bottom plot: State (Normal/Hijacked)
        ax_state = axes[plot_idx * 2 + 1]
        
        # Highlight takeoff period
        if takeoff_duration > 0:
            ax_state.axvspan(0, min(takeoff_duration, max_time), alpha=0.1, color=COLORS['takeoff'])
        
        states = [1 if d > DEVIATION_THRESHOLD else 0 for d in deviations]
        ax_state.fill_between(times, 0, states, where=np.array(states)==1, 
                             color=COLORS['hijack'], alpha=0.5, step='post')
        ax_state.fill_between(times, 0, states, where=np.array(states)==0, 
                             color=drone_info['color'], alpha=0.3, step='post')
        
        ax_state.set_ylabel('State', fontsize=10)
        ax_state.set_yticks([0, 1])
        ax_state.set_yticklabels(['Normal', 'Hijacked'], fontsize=8)
        ax_state.set_ylim(-0.1, 1.1)
        ax_state.set_xlim(0, max_time)
        ax_state.grid(True, alpha=0.3, axis='x')
        
        for gap in gaps:
            ax_state.axvspan(gap['start'], gap['end'], alpha=0.2, color=COLORS['warning'])
        
        plot_idx += 1
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['threshold'], linestyle='--', label=f'Threshold ({DEVIATION_THRESHOLD}m)'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['hijack'], markersize=8, label='Hijack start'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['recovery'], markersize=8, label='Recovery')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
              bbox_to_anchor=(0.5, 0.98), fontsize=10)
    
    fig.suptitle('Swarm Response to Command Injection: Path Deviation Over Time', 
            fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved stacked plot: {save_path}")
    
    plt.show()
    return fig


def print_summary(all_drone_data, all_events, takeoff_duration):
    """
    Print summary statistics to console.
    
    Args:
        all_drone_data: Dictionary with drone position data
        all_events: Dictionary with event data per drone
        takeoff_duration: Duration of takeoff tolerance period
    """
    print("\n" + "="*80)
    print(f"DEVIATION SUMMARY - AFTER TRIMMING")
    print(f"Trimmed: original {TRIM_START}s to {TRIM_END}s, time 0 = {TRIM_START}s")
    print(f"Takeoff tolerance: first {takeoff_duration}s (deviations < {TAKEOFF_TOLERANCE}m ignored)")
    print("="*80)
    
    for uri, data in all_drone_data.items():
        drone_info = data['info']
        events = all_events.get(uri, [])
        
        # Filter out events that occurred during takeoff
        hijack_starts = [e for e in events if e['type'] == 'HIJACK_START' and not e.get('in_takeoff', False)]
        takeoff_events = [e for e in events if e['type'] == 'HIJACK_START' and e.get('in_takeoff', False)]
        recoveries = [e for e in events if e['type'] == 'RECOVERY']
        
        print(f"\n{drone_info['name']}:")
        print(f"  Data points after trimming: {len(data['positions'])}")
        print(f"  Duration after trimming: {data['positions'][-1]['time']:.1f}s")
        print(f"  Takeoff-period events (ignored): {len(takeoff_events)}")
        print(f"  Actual Hijack Events: {len(hijack_starts)}")
        print(f"  Recoveries: {len(recoveries)}")
        
        if hijack_starts:
            print(f"  First Hijack: {hijack_starts[0]['time']:.1f}s (orig {hijack_starts[0]['orig_time']:.1f}s)")
            if len(hijack_starts) > 1:
                print(f"  Last Hijack: {hijack_starts[-1]['time']:.1f}s")
        
        if recoveries:
            durations = [r['duration'] for r in recoveries]
            print(f"  Avg Hijack Duration: {np.mean(durations):.1f}s")
            print(f"  Max Hijack Duration: {max(durations):.1f}s")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point - runs the deviation plotter."""
    print("\n" + "="*60)
    print("   DEVIATION PLOTTER - WITH TAKEOFF TOLERANCE")
    print(f"   - Removing data before {TRIM_START}s and after {TRIM_END}s")
    print(f"   - New time 0 = original {TRIM_START}s")
    print(f"   - Takeoff tolerance: {TAKEOFF_DURATION}s (ignoring deviations < {TAKEOFF_TOLERANCE}m)")
    print(f"   - Threshold for actual hijack: {DEVIATION_THRESHOLD}m")
    print("="*60 + "\n")
    
    overall_start = time.time()
    
    # Find log files
    drone_files = find_drone_log_files()
    
    if not drone_files:
        print("\nNo matching log files found!")
        print("Expected files: drone_position_*.jsonl")
        return
    
    # Process each drone
    all_drone_data = {}
    all_events = {}
    all_path_data = {}
    calibrated_starts = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for uri, filename in drone_files.items():
        drone_info = MISSIONS[uri]
        print(f"\n{'='*60}")
        print(f"Processing {drone_info['name']}")
        print(f"{'='*60}")
        
        drone_start = time.time()
        
        # Load and trim data
        positions, _ = load_and_trim_data(uri, filename, TRIM_START, TRIM_END)
        if not positions:
            print(f"  No data after trimming!")
            continue
        
        # Calibrate start position from actual data
        calibrated_start = None
        if CALIBRATE_START:
            calibrated_start = calibrate_start_position(positions, drone_info['start'])
            calibrated_starts[uri] = calibrated_start
        
        # Precompute path data with calibrated start if available
        print(f"  Precomputing path data...")
        path_data = precompute_path_points(drone_info, calibrated_start)
        all_path_data[uri] = path_data
        
        # Detect events with takeoff tolerance
        events = detect_hijack_events_with_takeoff_tolerance(positions, path_data, TAKEOFF_DURATION)
        gaps = detect_telemetry_gaps(positions)
        
        all_drone_data[uri] = {
            'positions': positions,
            'info': drone_info,
            'gaps': gaps
        }
        all_events[uri] = events
        
        drone_time = time.time() - drone_start
        print(f"  {drone_info['name']} processed in {drone_time:.1f}s")
    
    # Create stacked plot
    if all_drone_data:
        print("\nCreating stacked plot...")
        plot_start = time.time()
        
        filename = f"swarm_takeoff_tolerance_{TAKEOFF_DURATION}s_{timestamp}.png"
        create_stacked_plot(
            all_drone_data, 
            all_events,
            all_path_data,
            TAKEOFF_DURATION,
            save_path=filename
        )
        
        plot_time = time.time() - plot_start
        print(f"  Plot created in {plot_time:.1f}s")
        
        # Print summary
        print_summary(all_drone_data, all_events, TAKEOFF_DURATION)
        
        overall_time = time.time() - overall_start
        print(f"\nComplete! Total processing time: {overall_time:.1f}s")
        print(f"   File: {filename}")
    else:
        print(f"\nNo drone data after trimming!")


if __name__ == '__main__':
    main()
