#!/usr/bin/env python3
"""
Time-Lapse Plot Generator for Drone Swarm Attack Visualization

This script generates a time-lapse figure showing drone
trajectories during a command injection attack (Trial 1).

Panel timestamps (15, 25.4, 26.6, 40.4, 50, 80, 120, 160 seconds) are
used to create 8 panels (4x2 grid) showing key moments.

Key event timestamps (25.4s, 26.6s, 40.4s, 50s) are derived from prior
analysis of experimental log data (see analysis scripts and CSV outputs).
- 25.4s: Attack injection time (from attack log)
- 26.6s: Drone 1 first deviation (TTD = 1.2s)
- 40.4s: Drone 2 first deviation (TTD = 15.0s)
- 50s: Attack cessation (observed from command logs)

The remaining timestamps (15s, 80s, 120s, 160s) illustrate mission
progression before attack and during recovery.
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# GLOBAL STYLE (Publication Ready)
# ============================================================

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})

# ============================================================
# TRIAL TIMING CONSTANTS
# Derived from prior analysis of experimental logs
# (see: successful_takeovers_corrected.csv, hijack_results_*.csv)
# ============================================================

ATTACK_INJECTED = 25.4     # Time when attack was injected (from attack log)
DRONE1_RESPONSE = 26.6     # Time when Drone 1 first deviated (TTD = 1.2s)
DRONE2_RESPONSE = 40.4     # Time when Drone 2 first deviated (TTD = 15.0s)
ATTACK_END = 50            # Time when attack ceased (from command logs)

# Panel timestamps - used to create 8 panels showing key moments
# First 5 timestamps derived from experimental data
# Remaining 3 timestamps show recovery progression
PANELS = [
    (15, "Before attack"),                         # Baseline (pre-attack)
    (ATTACK_INJECTED, "Attack injected"),         # From analysis
    (DRONE1_RESPONSE, "Drone 1 deviates"),        # From analysis
    (DRONE2_RESPONSE, "Drone 2 deviates"),        # From analysis
    (ATTACK_END, "Attack ends"),                  # From analysis
    (80, "Recovery in progress"),                 # Recovery begins
    (120, "Continuing mission"),                  # Mid-recovery
    (160, "Mission complete"),                    # Full recovery
]

# Mission definitions for both drones (HARDCODED for Trial 1)
missions = {
    'radio://0/80/2M/E7E7E7E7E9': {
        'name': 'Drone 1',
        'start': (0.4, -1.85),                      # HARDCODED start position
        'waypoints': [(0.85, -1.24), (0.85, -0.78), (0.85, 0.95)],  # HARDCODED waypoints
        'marker': 'o',           # circle marker
        'facecolor': '#1f77b4',  # blue
        'edgecolor': 'black',
        'attack_target': (1.0, 0.0),                # HARDCODED attacker target
    },
    'radio://0/80/2M/E7E7E7E7E8': {
        'name': 'Drone 2',
        'start': (-0.5, -1.85),                    # HARDCODED start position
        'waypoints': [(-1.04, -0.93), (-1.04, 1.09)],  # HARDCODED waypoints
        'marker': 's',           # square marker
        'facecolor': '#ff7f0e',  # orange
        'edgecolor': 'black',
        'attack_target': (-1.0, 0.0),              # HARDCODED attacker target
    }
}


# Data trimming window - only keep positions between 0 and 180 seconds
# This removes pre-takeoff idle time and post-landing data
# All panel timestamps for trial1 (0s to 160s) fall within this window
TRIM_START = 0      # Start trimming at this time (seconds)
TRIM_END = 180      # End trimming at this time (seconds)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_grid_path(start, waypoints):
    """
    Generate the nominal grid path for a drone.
    
    This creates a path that moves vertically along the street X coordinate,
    then horizontally to each waypoint, returning to the street after each.
    
    Args:
        start: Tuple of (x, y) start coordinates
        waypoints: List of (x, y) target waypoints
    
    Returns:
        List of (x, y) path points
    """
    path = []
    streetX = start[0]

    for target in waypoints:
        # Move vertically along street to target's Y
        path.append((streetX, target[1]))
        # Move horizontally to target's X, then back to street
        if streetX != target[0]:
            path.append((target[0], target[1]))
            path.append((streetX, target[1]))

    return path


def find_drone_log_files():
    """
    Find log files for all drones defined in missions.
    
    Searches for files matching pattern 'drone_position_*{drone_id}*.jsonl'
    where drone_id is the last part of the URI.
    
    Returns:
        Dictionary mapping URIs to filename paths
    """
    log_files = glob.glob("drone_position_*.jsonl")
    drone_files = {}

    for uri in missions.keys():
        uri_short = uri.split('/')[-1]  # Extract drone ID from URI
        matching = [f for f in log_files if uri_short in f]
        if matching:
            drone_files[uri] = matching[0]
            print(f"Found {missions[uri]['name']}: {matching[0]}")

    return drone_files


def load_drone_data(uri, filename):
    """
    Load and trim position data for a drone from JSONL log file.
    
    Args:
        uri: Drone URI (for identification)
        filename: Path to log file
    
    Returns:
        List of position dictionaries with 'time', 'x', 'y' keys
    """
    positions = []
    start_time = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines
            if line and not line.startswith('#HEADER'):
                try:
                    data = json.loads(line)

                    # Record the first timestamp as reference
                    if start_time is None:
                        start_time = data['timestamp']

                    # Calculate relative time
                    rel_time = data['timestamp'] - start_time

                    # Keep only data within trimming window
                    if TRIM_START <= rel_time <= TRIM_END:
                        positions.append({
                            'time': rel_time - TRIM_START,  # Rebase time to 0
                            'x': data['x'],
                            'y': data['y'],
                        })
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

    return positions


def get_positions_up_to_time(positions, target_time):
    """
    Return all positions with time <= target_time.
    
    Args:
        positions: List of position dictionaries
        target_time: Maximum time to include
    
    Returns:
        Filtered list of positions
    """
    return [p for p in positions if p['time'] <= target_time]


def find_closest_position(positions, target_time):
    """
    Find the position record closest to a target time.
    
    Args:
        positions: List of position dictionaries
        target_time: Target timestamp
    
    Returns:
        Position dictionary closest to target_time, or None if empty
    """
    if not positions:
        return None
    # Find index with minimum absolute time difference
    idx = min(range(len(positions)), key=lambda i: abs(positions[i]['time'] - target_time))
    return positions[idx]


def add_mission_elements(ax, drone_info):
    """
    Add start point markers and waypoints to a subplot.
    
    Args:
        ax: Matplotlib axes object
        drone_info: Dictionary with drone configuration
    """
    start = drone_info['start']
    waypoints = drone_info['waypoints']
    
    # Start position (open circle)
    ax.scatter(start[0], start[1], 
              s=60, color='black', marker='o',
              facecolors='none', linewidth=1, zorder=10)
    ax.annotate('Start', (start[0], start[1]),
               xytext=(3, 6), textcoords='offset points',
               fontsize=6, ha='center')
    
    # Waypoints (star markers)
    for i, (wx, wy) in enumerate(waypoints):
        ax.scatter(wx, wy, s=80, color='black', marker='*',
                  linewidth=0.8, zorder=10)
        ax.annotate(f'WP{i+1}', (wx, wy),
                   xytext=(3, 6), textcoords='offset points',
                   fontsize=6, ha='center')


# ============================================================
# MAIN PLOTTING FUNCTION
# ============================================================

def create_timelapse_plot(all_drone_data, save_path=None):
    """
    Create a time-lapse figure with 8 panels (4 rows x 2 columns).
    
    Shows drone trajectories at key moments during the attack.
    
    Args:
        all_drone_data: Dictionary with drone position data
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure object
    """
    # Create 4x2 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(9, 12))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Calculate consistent axis limits from all data
    all_x = []
    all_y = []
    for uri, data in all_drone_data.items():
        for p in data['positions']:
            all_x.append(p['x'])
            all_y.append(p['y'])
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    # Generate each panel
    for idx, (panel_time, title_suffix) in enumerate(PANELS):
        ax = axes[idx]
        
        # ---- Plot nominal (expected) paths ----
        for uri, drone_info in missions.items():
            start = drone_info['start']
            waypoints = drone_info['waypoints']
            path_points = generate_grid_path(start, waypoints)
            path_x = [p[0] for p in path_points]
            path_y = [p[1] for p in path_points]
            
            ax.plot(path_x, path_y,
                    linestyle='--',
                    color='gray',
                    linewidth=1,
                    alpha=0.5,
                    label="Nominal path" if idx == 0 else "")
        
        # ---- Plot actual trajectories up to current time ----
        for uri, drone_info in missions.items():
            data = all_drone_data.get(uri)
            if not data:
                continue
            
            positions = data['positions']
            positions_up_to_t = get_positions_up_to_time(positions, panel_time)
            
            if positions_up_to_t:
                x_vals = [p['x'] for p in positions_up_to_t]
                y_vals = [p['y'] for p in positions_up_to_t]
                
                # Trajectory line (light, semi-transparent)
                ax.plot(x_vals, y_vals,
                        color=drone_info['facecolor'],
                        linestyle='-',
                        linewidth=0.8,
                        alpha=0.4,
                        label=f"{drone_info['name']} path" if idx == 0 else "")
                
                # Position dots (all recorded positions)
                ax.scatter(x_vals, y_vals,
                          s=10,
                          marker=drone_info['marker'],
                          facecolor=drone_info['facecolor'],
                          edgecolor='black',
                          linewidth=0.3,
                          alpha=0.6,
                          zorder=3)
                
                # Highlight position at exact panel time
                pos_at_t = find_closest_position(positions, panel_time)
                if pos_at_t:
                    ax.scatter(pos_at_t['x'], pos_at_t['y'],
                              s=100,
                              marker=drone_info['marker'],
                              facecolor=drone_info['facecolor'],
                              edgecolor='black',
                              linewidth=1.5,
                              zorder=5)
                    # Add timestamp label
                    ax.annotate(f'{pos_at_t["time"]:.0f}s', 
                               (pos_at_t['x'], pos_at_t['y']),
                               xytext=(4, 4), textcoords='offset points',
                               fontsize=6, fontweight='bold')
        
        # ---- Mark attacker's target destinations ----
        for uri, drone_info in missions.items():
            target = drone_info['attack_target']
            ax.scatter(target[0], target[1],
                      s=80, marker='x', color='red',
                      linewidth=1.5, zorder=8)
            # Annotate on Drone 1 deviation panel only
            if idx == 2:
                ax.annotate("Attacker\ntarget", 
                           (target[0], target[1]),
                           xytext=(3, -10), textcoords='offset points',
                           fontsize=5, color='red')
        
        # ---- Mark attack injection point ----
        if panel_time >= ATTACK_INJECTED:
            for uri, drone_info in missions.items():
                data = all_drone_data.get(uri)
                if data:
                    attack_pos = find_closest_position(data['positions'], ATTACK_INJECTED)
                    if attack_pos:
                        ax.scatter(attack_pos['x'], attack_pos['y'],
                                  s=120, marker='v', color='red',
                                  edgecolors='black', linewidth=1,
                                  zorder=6, alpha=0.9)
                        # Annotate on attack injected panel only
                        if idx == 1:
                            ax.annotate("Attack\ninjected", 
                                       (attack_pos['x'], attack_pos['y']),
                                       xytext=(3, 8), textcoords='offset points',
                                       fontsize=5, color='red')
        
        # ---- Add mission elements (waypoints, start points) ----
        for uri, drone_info in missions.items():
            add_mission_elements(ax, drone_info)
        
        # ---- Panel formatting ----
        ax.set_title(title_suffix + f"\n(t = {panel_time:.0f}s)", fontsize=9, fontweight='bold')
        
        # X-axis label only on bottom row panels (indices 6 and 7)
        if idx >= 6:
            ax.set_xlabel("X (m)", fontsize=7)
        else:
            ax.set_xlabel("")
        
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.15)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.tick_params(labelsize=6)
    
    # Hide any unused subplots (if PANELS < 8)
    for idx in range(len(PANELS), len(axes)):
        axes[idx].set_visible(False)
    
    # ---- Add shared legend at bottom of figure ----
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1, label='Nominal path'),
        plt.Line2D([0], [0], color=missions['radio://0/80/2M/E7E7E7E7E9']['facecolor'], 
                   linestyle='-', linewidth=0.8, marker='o', markersize=4, label='Drone 1'),
        plt.Line2D([0], [0], color=missions['radio://0/80/2M/E7E7E7E7E8']['facecolor'], 
                   linestyle='-', linewidth=0.8, marker='s', markersize=4, label='Drone 2'),
        plt.Line2D([0], [0], color='red', marker='v', linestyle='None', markersize=8, label='Attack start'),
        plt.Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8, label="Attacker target"),
        plt.Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=8, label='Waypoint'),
        plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=6, 
                   markerfacecolor='none', label='Start'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=7, 
               bbox_to_anchor=(0.5, 0.02))
    
    # Figure title
    fig.suptitle("Time‑Lapse of Drone Trajectories During Attack (Trial 1)", 
                fontsize=12, fontweight='bold', y=0.98)
    
    # Adjust layout spacing
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, 
                       wspace=0.15, hspace=0.25)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main entry point for the script."""
    print("\n" + "="*60)
    print("TIME-LAPSE PLOT - TRIAL 1")
    print("="*60)

    # Find all drone log files
    drone_files = find_drone_log_files()

    if not drone_files:
        print("No log files found.")
        print("Expected files: drone_position_*.jsonl")
        print("Make sure the victim mission has been run first.")
        return

    # Load data for all drones
    all_drone_data = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for uri, filename in drone_files.items():
        print(f"\nLoading {missions[uri]['name']}...")
        positions = load_drone_data(uri, filename)
        if positions:
            print(f"   Loaded {len(positions)} positions")
            all_drone_data[uri] = {'positions': positions}
        else:
            print(f"   Warning: No positions loaded for {missions[uri]['name']}")

    if not all_drone_data:
        print("No data loaded. Check log files and trimming settings.")
        return

    # Generate the time-lapse plot
    filename = f"timelapse_trial1_{timestamp}.png"
    create_timelapse_plot(all_drone_data, save_path=filename)

    print(f"\nComplete! Saved to: {filename}")


if __name__ == '__main__':
    main()
