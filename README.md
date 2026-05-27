# SwarmTakeover: Authority Contention in Micro-UAV Swarm Command Architectures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository accompanies a paper under review. It provides the complete
implementation, experimental dataset, and analysis code for studying
architectural vulnerabilities in micro-UAV command stacks under concurrent
adversarial control.

**Hardware Platform:** 
Bitcraze Crazyflie 2.1 Brushless
Lighhouse Positioning System

---

## Overview

This framework characterizes how performance-driven design choices in UAV
command architectures—trusting connection state, serializing control authority,
stateless packet processing—behave under concurrent adversarial interaction.
The attack preserves the legitimate ground control station (GCS) connection
throughout, creating authority contention resolved by packet timing alone.

## Repository Structure

| Directory | Description | Paper Section |
|-----------|-------------|---------------|
| `attack/` | SwarmTakeover framework: parallel scanner, modified commander, multi-drone takeover
| `missions/` | Autonomous delivery missions for Drone 1 (3-waypoint) and Drone 2 (2-waypoint)
| `detection/` | Real-time anomaly detection: path deviation monitoring, telemetry loss detection 
| `analysis/` | Statistical analysis and figure generation for all paper results 
|

## Quick Start

### Prerequisites

- Python 3.8+
- Bitcraze cflib (`pip install cflib`)


### Reproducing Paper Results

```bash
git clone [this-repository-url]
cd swarmtakeover
# Regenerate all statistics and figures from the raw trial data
python3 SwarmTakeOver/analysis/trial1/analyze.py
python3 SwarmTakeOver/analysis/all_trials/analyze_trials.py
python3 SwarmTakeOver/analysis/trial1/Fig5_SwarmDeviationDuringAttack.py
python3 SwarmTakeOver/analysis/trial1/Fig4_AuthorityContestation.py
python3 SwarmTakeOver/analysis/trial1/Fig3_TimeLapseDroneTrajectories.py
             






