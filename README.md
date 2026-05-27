# SwarmTakeOver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Research framework for studying architectural vulnerabilities in micro-UAV 
command stacks under concurrent adversarial control.

**Hardware Platform:** Bitcraze Crazyflie 2.1 Brushless with Lighthouse Positioning System

---

## Overview

Characterizes how performance-driven design choices in UAV command architectures— 
trusting connection state, serializing control authority, stateless packet 
processing—behave under concurrent adversarial interaction. The attack preserves 
the legitimate ground control station (GCS) connection throughout, creating 
authority contention resolved by packet timing alone.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `attack/` | SwarmTakeover framework: parallel scanner, modified commander, multi-drone takeover |
| `missions/` | Autonomous delivery missions for Drone 1 (3-waypoint) and Drone 2 (2-waypoint) |
| `detection/` | Real-time anomaly detection: path deviation monitoring, telemetry loss detection |
| `analysis/` | Statistical analysis and figure generation |

## Quick Start

### Prerequisites

- Python 3.8+
- Bitcraze cflib (`pip install cflib`)

### Running Analysis

```bash
git clone [this-repository-url]
cd SwarmTakeOver

# Full dataset analysis (all 49 trials)
python3 analysis/all_trials/analyze_trials.py

# Trial 1 detailed analysis
python3 analysis/trial1/analyze.py

# Generate figures
python3 analysis/trial1/Fig3_TimeLapseDroneTrajectories.py
python3 analysis/trial1/Fig4_AuthorityContestation.py
python3 analysis/trial1/Fig5_SwarmDeviationDuringAttack.py
