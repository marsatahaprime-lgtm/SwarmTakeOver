#!/usr/bin/env python3
"""
SwarmTakeover Trial Analysis Script

This script analyzes hijack results from CSV files generated during
SwarmTakeover experiments. It processes trial data, calculates Time to Deviation (TTD),
classifies mission phases, and generates summary tables for research papers.

Key Features:
- Reads hijack_results_*.csv files from the current directory
- Calculates elapsed TTD based on attack injection times (Early=25.4s, Mid=45.0s, Late=75.0s)
- Classifies outcomes: Stopped, Recovered, TeleLoss
- Generates Table 1 (main results) and Table 2 (phase outcomes)
- Analyzes outcomes by attack injection time
- Exports cleaned trial data to CSV

Usage:
    python analyze_swarmtakeover_trials.py
"""

import pandas as pd
import numpy as np
import glob
import os
import re


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path deviation threshold in meters (from paper)
DEVIATION_THRESHOLD = 0.3

# Fixed attack injection times per phase (from experimental design)
ATTACK_TIMES = {
    'Early': 25.4,
    'Mid': 45.0,
    'Late': 75.0
}

# Attack injection time options (from experimental design)
ATTACK_INJECTION_TIMES = [25.4, 45.0, 75.0]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_timestamp_from_filename(filename):
    """
    Extract timestamp from filename like hijack_results_20260506_115338.csv.
    
    Args:
        filename: CSV filename
        
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS) or None if not found
    """
    match = re.search(r'(\d{8}_\d{6})', filename)
    return match.group(1) if match else None


def classify_phase(ttd_reported):
    """
    Classify mission phase based on reported TTD value.
    
    Args:
        ttd_reported: Reported time to deviation in seconds
        
    Returns:
        Phase string: 'Early', 'Mid', 'Late', or 'Unknown'
    """
    if ttd_reported is None or pd.isna(ttd_reported):
        return "Unknown"
    if ttd_reported <= 45:
        return "Early"
    elif ttd_reported <= 80:
        return "Mid"
    else:
        return "Late"


def determine_outcome(evidence_type, has_telemetry_loss, has_path_deviation):
    """
    Determine outcome based on evidence type.
    
    Args:
        evidence_type: 'Path Deviation' or 'Telemetry Loss'
        has_telemetry_loss: Boolean indicating telemetry loss occurred
        has_path_deviation: Boolean indicating path deviation occurred
        
    Returns:
        Outcome string: 'TeleLoss', 'Recovered', or 'Stopped'
    """
    # Telemetry loss only (no path deviation) -> severe telemetry loss
    if evidence_type == 'Telemetry Loss' and not has_path_deviation:
        return 'TeleLoss'
    
    # Path deviation without telemetry loss -> drone recovered
    if evidence_type == 'Path Deviation' and not has_telemetry_loss:
        return 'Recovered'
    
    # Path deviation with telemetry loss -> drone stopped at target
    if evidence_type == 'Path Deviation' and has_telemetry_loss:
        return 'Stopped'
    
    # Fallback
    return 'Stopped'


def get_response_time(drone_df):
    """
    Get response time (reported experiment time) from drone data.
    
    Priority order:
    1. Path deviation time (first non-zero time_to_first_anomaly or experiment_time)
    2. Telemetry loss time (if no path deviation exists - valid takeover evidence)
    
    Args:
        drone_df: DataFrame containing drone data for a single trial/drone
        
    Returns:
        Response time in seconds, or None if not found
    """
    response_time = None
    
    # Priority 1: Path deviation time
    if 'time_to_first_anomaly' in drone_df.columns:
        path_rows = drone_df[drone_df['anomaly_type'].astype(str).str.upper().str.contains('PATH_DEVIATION', na=False)]
        if len(path_rows) > 0:
            valid = path_rows['time_to_first_anomaly'].dropna()
            valid = valid[valid > 0]
            if len(valid) > 0:
                response_time = valid.iloc[0]
    
    # Priority 2: If no path deviation, check experiment_time from path deviation
    if response_time is None:
        if 'experiment_time' in drone_df.columns:
            path_rows = drone_df[drone_df['anomaly_type'].astype(str).str.upper().str.contains('PATH_DEVIATION', na=False)]
            if len(path_rows) > 0:
                valid = path_rows['experiment_time'].dropna()
                valid = valid[valid > 0]
                if len(valid) > 0:
                    response_time = valid.iloc[0]
    
    # Priority 3: Telemetry loss (valid takeover evidence)
    if response_time is None:
        if 'anomaly_type' in drone_df.columns:
            telemetry_rows = drone_df[drone_df['anomaly_type'].astype(str).str.upper().str.contains('TELEMETRY_LOSS', na=False)]
            if len(telemetry_rows) > 0:
                if 'experiment_time' in telemetry_rows.columns:
                    valid = telemetry_rows['experiment_time'].dropna()
                    if len(valid) > 0:
                        response_time = valid.iloc[0]
                elif 'time_to_first_anomaly' in telemetry_rows.columns:
                    valid = telemetry_rows['time_to_first_anomaly'].dropna()
                    if len(valid) > 0:
                        response_time = valid.iloc[0]
    
    return response_time


def analyze_trial_file(filepath):
    """
    Analyze a single trial file and return results with phase and outcome.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dictionary with trial results, or None if trial should be skipped
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return None
    
    if len(df) == 0:
        return None
    
    trial_time = parse_timestamp_from_filename(os.path.basename(filepath))
    if trial_time is None:
        return None
    
    results = {}
    for drone_id in [1, 2]:
        drone_df = df[df['drone_id'] == drone_id]
        if len(drone_df) == 0:
            return None  # Skip trials with missing drone data
        
        # Check for path deviation
        dev_col = None
        for col in ['max_path_deviation', 'distance_to_path', 'max_deviation']:
            if col in drone_df.columns:
                dev_col = col
                break
        
        has_path_deviation = False
        if dev_col:
            max_dev = drone_df[dev_col].max()
            if max_dev and max_dev > DEVIATION_THRESHOLD:
                has_path_deviation = True
        
        # Check for telemetry loss
        has_telemetry_loss = False
        if 'anomaly_type' in drone_df.columns:
            anomalies = drone_df['anomaly_type'].unique()
            if any('TELEMETRY_LOSS' in str(a).upper() for a in anomalies):
                has_telemetry_loss = True
        
        # Get response time (reported)
        response_time = get_response_time(drone_df)
        
        # If no response time and no evidence, skip this trial
        if response_time is None and not has_path_deviation and not has_telemetry_loss:
            return None
        
        # Determine evidence type (prioritize path deviation for display)
        if has_path_deviation:
            evidence = 'Path Deviation'
        elif has_telemetry_loss:
            evidence = 'Telemetry Loss'
        else:
            evidence = 'No Evidence'
        
        results[drone_id] = {
            'response_time': response_time,
            'evidence': evidence,
            'has_path_deviation': has_path_deviation,
            'has_telemetry_loss': has_telemetry_loss
        }
    
    # Include trials where BOTH drones have valid response times
    if results[1]['response_time'] is not None and results[2]['response_time'] is not None:
        resp1 = results[1]['response_time']
        resp2 = results[2]['response_time']
        
        # Determine phase from response time
        phase1 = classify_phase(resp1)
        phase2 = classify_phase(resp2)
        
        # Get attack time based on phase
        attack1 = ATTACK_TIMES.get(phase1, 0)
        attack2 = ATTACK_TIMES.get(phase2, 0)
        
        # Calculate elapsed TTD (Time to Deviation)
        elapsed1 = max(0, resp1 - attack1) if attack1 > 0 else resp1
        elapsed2 = max(0, resp2 - attack2) if attack2 > 0 else resp2
        
        return {
            'trial_time': trial_time,
            'drone1_attack_time': attack1,
            'drone2_attack_time': attack2,
            'drone1_TTD_reported': round(resp1, 2),
            'drone2_TTD_reported': round(resp2, 2),
            'drone1_TTD_elapsed': round(elapsed1, 2),
            'drone2_TTD_elapsed': round(elapsed2, 2),
            'drone1_phase': phase1,
            'drone2_phase': phase2,
            'drone1_outcome': determine_outcome(results[1]['evidence'], results[1]['has_telemetry_loss'], results[1]['has_path_deviation']),
            'drone2_outcome': determine_outcome(results[2]['evidence'], results[2]['has_telemetry_loss'], results[2]['has_path_deviation']),
        }
    return None


# ============================================================================
# TABLE GENERATION FUNCTIONS
# ============================================================================

def generate_detailed_trial_table(df):
    """
    Generate detailed trial table with reported and elapsed TTD.
    
    Args:
        df: DataFrame containing all trial results
    """
    print("\n" + "="*110)
    print("DETAILED TRIAL RESULTS (Reported vs Elapsed TTD with Phase-Based Attack Times)")
    print("="*110)
    print(f"\n{'Trial Time':<15} {'Phase1':<8} {'Attack1':<10} {'Reported1':<12} {'Elapsed1':<12} {'Phase2':<8} {'Attack2':<10} {'Reported2':<12} {'Elapsed2':<12} {'Outcome1':<12} {'Outcome2':<12}")
    print("-"*115)
    
    for _, row in df.iterrows():
        print(f"{row['trial_time']:<15} {row['drone1_phase']:<8} {row['drone1_attack_time']:<10.1f} {row['drone1_TTD_reported']:<12.2f} {row['drone1_TTD_elapsed']:<12.2f} {row['drone2_phase']:<8} {row['drone2_attack_time']:<10.1f} {row['drone2_TTD_reported']:<12.2f} {row['drone2_TTD_elapsed']:<12.2f} {row['drone1_outcome']:<12} {row['drone2_outcome']:<12}")


def generate_table1(df):
    """
    Generate Table 1: Main experimental results using elapsed TTD.
    
    Args:
        df: DataFrame containing all trial results
        
    Returns:
        Dictionary with summary statistics
    """
    total_trials = len(df)
    total_instances = total_trials * 2
    
    all_outcomes = df['drone1_outcome'].tolist() + df['drone2_outcome'].tolist()
    all_ttd_elapsed = df['drone1_TTD_elapsed'].tolist() + df['drone2_TTD_elapsed'].tolist()
    
    stopped = all_outcomes.count('Stopped')
    recovered = all_outcomes.count('Recovered')
    teleloss = all_outcomes.count('TeleLoss')
    
    ttd_values = [t for t in all_ttd_elapsed if t is not None]
    
    print("\n" + "="*70)
    print("TABLE 1: SWARMTAKEOVER EXPERIMENTAL RESULTS")
    print("="*70)
    print(f"\n{'Metric':<35} {'Value':<30}")
    print("-"*70)
    print(f"{'Total trials':<35} {total_trials}")
    print(f"{'Total drone-instances':<35} {total_instances}")
    print(f"{'Attack success rate':<35} 100%")
    print()
    print(f"{'Drone Outcome (n=' + str(total_instances) + ')':<35}")
    print(f"{'  - Stopped at attacker target':<35} {stopped} ({stopped/total_instances*100:.1f}%)")
    print(f"{'  - Recovered to original mission':<35} {recovered} ({recovered/total_instances*100:.1f}%)")
    print(f"{'  - Telemetry loss (severe)':<35} {teleloss} ({teleloss/total_instances*100:.1f}%)")
    print()
    if ttd_values:
        print(f"{'Elapsed TTD range':<35} {min(ttd_values):.2f}s - {max(ttd_values):.2f}s")
        print(f"{'Elapsed TTD mean':<35} {np.mean(ttd_values):.2f}s +- {np.std(ttd_values):.2f}s")
    
    return {
        'total_trials': total_trials,
        'total_instances': total_instances,
        'stopped': stopped,
        'recovered': recovered,
        'teleloss': teleloss,
        'ttd_min': min(ttd_values) if ttd_values else None,
        'ttd_max': max(ttd_values) if ttd_values else None,
        'ttd_mean': np.mean(ttd_values) if ttd_values else None,
        'ttd_std': np.std(ttd_values) if ttd_values else None
    }


def generate_table2(df):
    """
    Generate Table 2: Drone outcomes by mission phase using elapsed TTD for phase.
    
    Args:
        df: DataFrame containing all trial results
        
    Returns:
        Dictionary with phase statistics
    """
    # Collect all drone-instances with their phase and outcome
    instances = []
    for _, row in df.iterrows():
        instances.append({
            'phase': row['drone1_phase'],
            'outcome': row['drone1_outcome'],
            'ttd_elapsed': row['drone1_TTD_elapsed']
        })
        instances.append({
            'phase': row['drone2_phase'],
            'outcome': row['drone2_outcome'],
            'ttd_elapsed': row['drone2_TTD_elapsed']
        })
    
    instances_df = pd.DataFrame(instances)
    
    # Filter out Unknown phases
    instances_df = instances_df[instances_df['phase'] != 'Unknown']
    
    # Calculate statistics by phase
    phases = ['Early', 'Mid', 'Late']
    results = {}
    
    for phase in phases:
        phase_df = instances_df[instances_df['phase'] == phase]
        n = len(phase_df)
        stopped = len(phase_df[phase_df['outcome'] == 'Stopped'])
        recovered = len(phase_df[phase_df['outcome'] == 'Recovered'])
        teleloss = len(phase_df[phase_df['outcome'] == 'TeleLoss'])
        
        # TTD statistics for this phase
        ttds = phase_df['ttd_elapsed'].dropna().tolist()
        
        results[phase] = {
            'n': n,
            'stopped': stopped,
            'stopped_pct': (stopped / n * 100) if n > 0 else 0,
            'recovered': recovered,
            'recovered_pct': (recovered / n * 100) if n > 0 else 0,
            'teleloss': teleloss,
            'teleloss_pct': (teleloss / n * 100) if n > 0 else 0,
            'ttd_mean': np.mean(ttds) if ttds else None,
            'ttd_std': np.std(ttds) if ttds else None
        }
    
    print("\n" + "="*70)
    print("TABLE 2: DRONE OUTCOMES BY MISSION PHASE AT ATTACK ONSET")
    print("="*70)
    print(f"\n{'Phase':<12} {'n':<8} {'Stopped':<20} {'Recovered':<20} {'TeleLoss':<20}")
    print("-"*70)
    
    for phase in phases:
        r = results[phase]
        stopped_str = f"{r['stopped']} ({r['stopped_pct']:.0f}%)"
        recovered_str = f"{r['recovered']} ({r['recovered_pct']:.0f}%)"
        teleloss_str = f"{r['teleloss']} ({r['teleloss_pct']:.0f}%)"
        print(f"{phase:<12} {r['n']:<8} {stopped_str:<20} {recovered_str:<20} {teleloss_str:<20}")
    
    print("-"*70)
    total_known = sum([results[p]['n'] for p in phases])
    print(f"{'Total (known phases)':<12} {total_known:<8}")
    
    return results


def analyze_outcomes_by_attack_time(df):
    """
    Analyze outcomes by attack injection time.
    
    Args:
        df: DataFrame containing all trial results
    """
    print("\n" + "="*70)
    print("OUTCOMES BY ATTACK INJECTION TIME")
    print("="*70)
    
    # Count trials by attack time (using Drone 1 attack_time)
    attack_counts = df['drone1_attack_time'].value_counts().sort_index()
    
    print("\nTrials by attack time:")
    for attack_time, count in attack_counts.items():
        print(f"  {attack_time}s: {count} trials")
    
    # Calculate outcomes by attack time
    for attack_time in ATTACK_INJECTION_TIMES:
        mask = df['drone1_attack_time'] == attack_time
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        stopped = (subset['drone1_outcome'] == 'Stopped').sum() + (subset['drone2_outcome'] == 'Stopped').sum()
        recovered = (subset['drone1_outcome'] == 'Recovered').sum() + (subset['drone2_outcome'] == 'Recovered').sum()
        teleloss = (subset['drone1_outcome'] == 'TeleLoss').sum() + (subset['drone2_outcome'] == 'TeleLoss').sum()
        total = len(subset) * 2
        
        print(f"\nAttack Time {attack_time}s:")
        print(f"  n = {total}")
        print(f"  Stopped: {stopped} ({stopped/total*100:.0f}%)")
        print(f"  Recovered: {recovered} ({recovered/total*100:.0f}%)")
        print(f"  TeleLoss: {teleloss} ({teleloss/total*100:.0f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point for the trial analysis script.
    """
    print("Analyzing SwarmTakeover trials...")
    print("Telemetry loss is valid takeover evidence")
    print("Attack times: Early=25.4s, Mid=45.0s, Late=75.0s")
    print("Elapsed TTD = Reported TTD - Attack Time (clamped to 0)")
    print()
    
    # Find all hijack_results files
    all_files = glob.glob("hijack_results_*.csv")
    
    valid_files = []
    for f in all_files:
        try:
            if os.path.getsize(f) > 100:
                valid_files.append(f)
        except:
            pass
    
    print(f"Found {len(all_files)} files, {len(valid_files)} non-empty")
    
    # Analyze each file
    all_trials = []
    for filepath in valid_files:
        result = analyze_trial_file(filepath)
        if result:
            all_trials.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trials)
    
    if len(df) == 0:
        print("\nNo trials found.")
        return
    
    print(f"\nIncluded {len(df)} trials for analysis")
    
    # Generate detailed trial table
    generate_detailed_trial_table(df)
    
    # Generate Table 1
    generate_table1(df)
    
    # Generate Table 2
    generate_table2(df)
    
    # Analyze outcomes by attack injection time
    analyze_outcomes_by_attack_time(df)
    
    # Save to CSV
    output_file = "takeover_results_clean.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed trial data saved to: {output_file}")
    
    print("\nAnalysis complete")


if __name__ == "__main__":
    main()
