# SwarmTakeover: Authority Contention in Micro-UAV Swarm Command Architectures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository accompanies a paper under review. It provides the complete
implementation, experimental dataset, and analysis code for studying
architectural vulnerabilities in micro-UAV command stacks under concurrent
adversarial control.

**Hardware Platform:** Bitcraze Crazyflie 2.1 Brushless

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
| `attack/` | SwarmTakeover framework: parallel scanner, modified commander, multi-drone takeover | §3 |
| `missions/` | Autonomous delivery missions for Drone 1 (3-waypoint) and Drone 2 (2-waypoint) | §4.1 |
| `detection/` | Real-time anomaly detection: path deviation monitoring, telemetry loss detection | §4.2, App. A |
| `analysis/` | Statistical analysis and figure generation for all paper results | §4.3–4.6 |
| `data/` | Complete 49-trial dataset with TTD, phase, and outcome labels | App. B |
| `docs/` | Experimental setup guide, hardware requirements, replication notes | — |

## Quick Start

### Prerequisites

- Python 3.8+
- Bitcraze cflib (`pip install cflib`)
- See `requirements.txt` for full dependency list

### Reproducing Paper Results

```bash
git clone [this-repository-url]
cd swarmtakeover
pip install -r requirements.txt

# Regenerate all statistics and figures from the raw trial data
python analysis/analyze_results.py
python analysis/generate_figures.py
