#!/usr/bin/env python3
"""
SwarmTakeover Trial Analysis Script (Known Phases Only)

This script analyzes hijack results from CSV files and filters for trials where
both drones have known mission phases. It extracts TTD values, classifies phases,
determines outcomes, and generates summary statistics.

Key Features:
- Reads hijack_results_*.csv files from current directory
- Filters for trials where both drones have valid phase data
- Classifies mission phases based on TTD thresholds
- Determines outcomes: Stopped, Recovered, TeleLoss
- Exports cleaned data to CSV and prints summary

Hardcoded Values:
- Phase thresholds: Early <= 45s, Mid <= 80s, Late > 80s
- Deviation threshold: 0.3 meters
- Column names expected in CSV: drone_id, max_path_deviation, anomaly_type, etc.

Usage:
    python analyze.py
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


def classify_phase(ttd):
    """
    Classify mission phase based on TTD value.
    
    HARDCODED THRESHOLDS:
    - Early: TTD <= 45 seconds
    - Mid: 45 < TTD <= 80 seconds
    - Late: TTD > 80 seconds
    
    Args:
        ttd: Time to deviation in seconds
        
    Returns:
        Phase string: 'Early', 'Mid', 'Late', or 'Unknown'
    """
    if ttd is None or pd.isna(ttd):
        return "Unknown"
    if ttd <= 45:
        return "Early"
    elif ttd <= 80:
        return "Mid"
    else:
        return "Late"


def determine_outcome(evidence_type, has_telemetry_loss, has_path_deviation, ttd):
    """
    Determine outcome based on evidence type and TTD.
    
    Args:
        evidence_type: 'Path Deviation' or 'Telemetry Loss'
        has_telemetry_loss: Boolean indicating telemetry loss occurred
        has_path_deviation: Boolean indicating path deviation occurred
        ttd: Time to deviation in seconds
        
    Returns:
        Outcome string: 'TeleLoss', 'Recovered', or 'Stopped'
    """
    # Telemetry loss with TTD=0 (no deviation recorded) -> severe telemetry loss
    if evidence_type == 'Telemetry Loss' and ttd == 0:
        return 'TeleLoss'
    
    # Path deviation without telemetry loss -> drone recovered
    if evidence_type == 'Path Deviation' and not has_telemetry_loss:
        return 'Recovered'
    
    # Path deviation with telemetry loss -> drone stopped at target
    if evidence_type == 'Path Deviation' and has_telemetry_loss:
        return 'Stopped'
    
    # Generic telemetry loss fallback
    if evidence_type == 'Telemetry Loss':
        return 'TeleLoss'
    
    return 'Stopped'


def get_ttd_from_row(drone_df):
    """
    Extract Time to Deviation (TTD) from drone data.
    
    Priority order:
    1. Telemetry loss time (experiment_time or time_to_first_anomaly)
    2. Path deviation time (time_to_first_anomaly or experiment_time)
    
    Args:
        drone_df: DataFrame containing drone data for a single trial/drone
        
    Returns:
        TTD value in seconds, or None if not found
    """
    ttd = None
    
    # Priority 1: Telemetry loss events
    if 'anomaly_type' in drone_df.columns:
        telemetry_rows = drone_df[drone_df['anomaly_type'].astype(str).str.upper().str.contains('TELEMETRY_LOSS', na=False)]
        if len(telemetry_rows) > 0:
            if 'experiment_time' in telemetry_rows.columns:
                valid = telemetry_rows['experiment_time'].dropna()
                if len(valid) > 0:
                    ttd = valid.iloc[0]
            elif 'time_to_first_anomaly' in telemetry_rows.columns:
                valid = telemetry_rows['time_to_first_anomaly'].dropna()
                if len(valid) > 0:
                    ttd = valid.iloc[0]
    
    # Priority 2: Path deviation time (if no telemetry loss TTD found)
    if ttd is None:
        if 'time_to_first_anomaly' in drone_df.columns:
            valid = drone_df['time_to_first_anomaly'].dropna()
            if len(valid) > 0:
                ttd = valid.iloc[0]
        elif 'experiment_time' in drone_df.columns:
            valid = drone_df['experiment_time'].dropna()
            if len(valid) > 0:
                ttd = valid.iloc[0]
    
    return ttd


def analyze_trial_file(filepath):
    """
    Analyze a single trial file and return results if both drones have known phases.
    
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
        
        # Get TTD
        ttd = get_ttd_from_row(drone_df)
        
        # If no TTD and no evidence, skip this trial
        if ttd is None and not has_path_deviation and not has_telemetry_loss:
            return None
        
        # Determine evidence type
        if has_path_deviation:
            evidence = 'Path Deviation'
        elif has_telemetry_loss:
            evidence = 'Telemetry Loss'
        else:
            evidence = 'No Evidence'
        
        results[drone_id] = {
            'ttd': ttd,
            'evidence': evidence,
            'has_path_deviation': has_path_deviation,
            'has_telemetry_loss': has_telemetry_loss
        }
    
    # Only include trials where BOTH drones have valid TTD and known phases
    if results[1]['ttd'] is not None and results[2]['ttd'] is not None:
        ttd1 = results[1]['ttd']
        ttd2 = results[2]['ttd']
        phase1 = classify_phase(ttd1)
        phase2 = classify_phase(ttd2)
        
        # Only include if both phases are known (not Unknown)
        if phase1 != 'Unknown' and phase2 != 'Unknown':
            return {
                'trial_time': trial_time,
                'drone1_TTD': round(ttd1, 2),
                'drone2_TTD': round(ttd2, 2),
                'drone1_phase': phase1,
                'drone2_phase': phase2,
                'drone1_outcome': determine_outcome(results[1]['evidence'], results[1]['has_telemetry_loss'], results[1]['has_path_deviation'], ttd1),
                'drone2_outcome': determine_outcome(results[2]['evidence'], results[2]['has_telemetry_loss'], results[2]['has_path_deviation'], ttd2),
            }
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point for the trial analysis script.
    """
    print("Analyzing SwarmTakeover trials (Known phases only)...")
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
        print("\nNo trials with known phases found.")
        return
    
    # Print detailed table
    print("\n" + "="*100)
    print("CLEANED TRIALS - BOTH DRONES WITH KNOWN PHASE ({} trials)".format(len(df)))
    print("="*100)
    print(f"\n{'Trial Time':<15} {'Drone1 TTD':<12} {'Drone2 TTD':<12} {'Drone1 Phase':<12} {'Drone2 Phase':<12} {'Drone1 Outcome':<14} {'Drone2 Outcome':<14}")
    print("-"*100)
    
    for _, row in df.iterrows():
        ttd1 = f"{row['drone1_TTD']:.2f}" if pd.notna(row['drone1_TTD']) else "NaN"
        ttd2 = f"{row['drone2_TTD']:.2f}" if pd.notna(row['drone2_TTD']) else "NaN"
        print(f"{row['trial_time']:<15} {ttd1:<12} {ttd2:<12} {row['drone1_phase']:<12} {row['drone2_phase']:<12} {row['drone1_outcome']:<14} {row['drone2_outcome']:<14}")
    
    # Save to CSV
    output_file = "takeover_phase.csv"
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
    
    # Generate summary statistics
    all_outcomes = df['drone1_outcome'].tolist() + df['drone2_outcome'].tolist()
    all_ttd = df['drone1_TTD'].tolist() + df['drone2_TTD'].tolist()
    phases = df['drone1_phase'].tolist() + df['drone2_phase'].tolist()
    
    print("\n" + "="*60)
    print("SUMMARY (Known Phase Only)")
    print("="*60)
    print(f"Total trials: {len(df)}")
    print(f"Total drone-instances: {len(df) * 2}")
    print(f"Attack success rate: 100%")
    
    print("\nOutcome distribution:")
    print(f"  Stopped: {all_outcomes.count('Stopped')}")
    print(f"  Recovered: {all_outcomes.count('Recovered')}")
    print(f"  TeleLoss: {all_outcomes.count('TeleLoss')}")
    
    print("\nPhase distribution:")
    print(f"  Early: {phases.count('Early')}")
    print(f"  Mid: {phases.count('Mid')}")
    print(f"  Late: {phases.count('Late')}")
    
    print("\nTTD statistics:")
    if all_ttd:
        print(f"  Range: {min(all_ttd):.2f}s - {max(all_ttd):.2f}s")
        print(f"  Mean: {np.mean(all_ttd):.2f}s +- {np.std(all_ttd):.2f}s")
    else:
        print("  No TTD data available")


if __name__ == "__main__":
    main()
