#!/usr/bin/env python3
"""
Crazyflie Victim Mission with Buffered Position Logging

This script executes a coordinated grid-based delivery mission for multiple Crazyflie drones.
Each drone follows a predefined path of waypoints, moving vertically along a fixed street
X-coordinate, then horizontally to each delivery point. The mission includes:
- Takeoff to 1.0m altitude
- Grid path following with yaw fixed at 0 degrees
- Delivery simulation (descent/ascent at waypoints)
- Return to start point and landing

The script uses buffered asynchronous logging to record position data without affecting
flight performance. It also includes:
- Battery monitoring for low-voltage detection
- Emergency stop handling (Ctrl+C)
- Graceful shutdown with log flushing
"""

import time
import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.crazyflie.log import LogConfig
import signal
import sys
import threading
from battery_Monitoring import BatteryMonitor
import json
from datetime import datetime
import queue


# Global state variables
emergency_stop_triggered = False
active_drones = {}
position_loggers = {}
position_files = {}
log_running = True

log_queues = {}
log_threads = {}
log_stats = {}

# Mission parameters
height = 1
rotateTime = 8
testTime = 8
deliveryTime = testTime
returnTime = 10
buffer = 1.5
controller = "CTR-7F3A9C"


def log_writer_worker(uri, filename):
    """
    Background thread that writes buffered log entries to file asynchronously.
    
    Args:
        uri: Drone URI (used for queue identification)
        filename: Path to log file
    """
    buffer = []
    buffer_size = 5
    last_flush = time.time()
    flush_interval = 0.2
    
    # Write header with metadata
    with open(filename, 'w') as f:
        header_info = {
            'drone_uri': uri,
            'start_time': datetime.now().isoformat(),
            'controller': 'victim',
            'mission': missions.get(uri, ('unknown', []))[1],
            'log_type': 'buffered_async'
        }
        f.write(f"#HEADER: {json.dumps(header_info)}\n")
    
    # Main write loop
    with open(filename, 'a') as f:
        while log_running or not log_queues[uri].empty():
            try:
                log_data = log_queues[uri].get(timeout=1.0)
                buffer.append(log_data)
                
                current_time = time.time()
                if (len(buffer) >= buffer_size or 
                    current_time - last_flush >= flush_interval):
                    for data in buffer:
                        f.write(data + '\n')
                    f.flush()
                    buffer.clear()
                    last_flush = current_time
                    
                log_queues[uri].task_done()
                
            except queue.Empty:
                # Flush remaining buffer on timeout
                if buffer:
                    for data in buffer:
                        f.write(data + '\n')
                    f.flush()
                    buffer.clear()
                    last_flush = time.time()
                    
        # Final flush when stopping
        if buffer:
            for data in buffer:
                f.write(data + '\n')
            f.flush()


def setup_position_logging(cf, uri):
    """
    Configure real-time position logging with buffered async writer.
    
    Args:
        cf: Crazyflie connection object
        uri: Drone URI for log file naming
    
    Returns:
        LogConfig object if successful, None otherwise
    """
    stats = {
        'message_count': 0,
        'dropped_count': 0,
        'last_report': time.time(),
        'last_timestamp': time.time()
    }
    
    def log_pos_callback(timestamp, data, logconf):
        """Callback for position data from Crazyflie state estimator"""
        current_time = time.time()
        stats['message_count'] += 1
        
        # Report performance every 30 seconds
        if current_time - stats['last_report'] > 30:
            time_since_last = current_time - stats['last_timestamp']
            print(f"📊 {uri} logging: {stats['message_count']} msgs, "
                  f"{stats['dropped_count']} dropped, "
                  f"interval: {time_since_last:.3f}s")
            stats['message_count'] = 0
            stats['dropped_count'] = 0
            stats['last_report'] = current_time
        
        stats['last_timestamp'] = current_time
        
        try:
            x = data['stateEstimate.x']
            y = data['stateEstimate.y']
            z = data['stateEstimate.z']
            
            position_data = {
                'timestamp': current_time,
                'datetime': datetime.now().isoformat(),
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'controller': controller
            }
            
            if uri in log_queues:
                try:
                    log_queues[uri].put_nowait(json.dumps(position_data))
                except queue.Full:
                    stats['dropped_count'] += 1
                    
        except Exception as e:
            print(f"Error in log callback for {uri}: {e}")
    
    log_stats[uri] = stats
    
    # Configure log variables
    log_conf = LogConfig(name='Position', period_in_ms=600)
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')
    
    try:
        cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(log_pos_callback)
        log_conf.start()
        position_loggers[uri] = log_conf
        
        filename = f"drone_position_{uri.replace('://', '_').replace('/', '_')}.jsonl"
        position_files[uri] = filename
        
        log_queues[uri] = queue.Queue(maxsize=200)
        log_thread = threading.Thread(
            target=log_writer_worker, 
            args=(uri, filename),
            daemon=True
        )
        log_thread.start()
        log_threads[uri] = log_thread
        
        print(f"Buffered position logging started for {uri} -> {filename}")
        return log_conf
        
    except Exception as e:
        print(f"Failed to start position logging for {uri}: {e}")
        return None


def stop_position_logging(uri):
    """
    Stop position logging and flush remaining buffered data.
    
    Args:
        uri: Drone URI to stop logging for
    """
    if uri in position_loggers:
        try:
            position_loggers[uri].stop()
            
            # Wait for queue to flush
            if uri in log_queues:
                print(f"Flushing log queue for {uri}...")
                log_queues[uri].join()
                
            # Write end marker
            if uri in position_files:
                try:
                    end_marker = {
                        'timestamp': time.time(),
                        'datetime': datetime.now().isoformat(),
                        'event': 'logging_stopped',
                        'controller': controller,
                        'final_stats': log_stats.get(uri, {})
                    }
                    with open(position_files[uri], 'a') as f:
                        f.write(json.dumps(end_marker) + '\n')
                except Exception as e:
                    print(f"Error writing end marker for {uri}: {e}")
                    
            print(f"Position logging stopped for {uri}")
            
        except Exception as e:
            print(f"Error stopping position logger for {uri}: {e}")


def emergency_stop(sig=None, frame=None):
    """
    Emergency stop handler for Ctrl+C.
    
    Stops all drone motors, flushes logs, and exits cleanly.
    """
    global emergency_stop_triggered, log_running
    print("EMERGENCY STOP TRIGGERED! Stopping all drones...")
    emergency_stop_triggered = True
    log_running = False
    
    # Stop all loggers
    for uri in list(position_loggers.keys()):
        print(f"Stopping logger for {uri}")
        stop_position_logging(uri)
    
    # Stop all drones
    for uri, cf in active_drones.items():
        try:
            print(f"Stopping motors for {uri}")
            cf.commander.send_stop_setpoint()
        except Exception as e:
            print(f"Error stopping {uri}: {e}")
    
    print("Final log flush...")
    time.sleep(2.0)
    sys.exit(0)


signal.signal(signal.SIGINT, emergency_stop)


def arm(scf):
    """Arm the drone (enable motors)."""
    scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)


def generate_grid_path(start, waypoints):
    """
    Generate grid-aligned path from start point to waypoints.
    
    Movement pattern: vertical along street X to target Y, then horizontal to target X.
    
    Args:
        start: Tuple of (x, y) start coordinates
        waypoints: List of (x, y) target waypoints
    
    Returns:
        List of (x, y, z, yaw) path points
    """
    path = []
    streetX = start[0]

    for target in waypoints:
        yaw = 0
        path.append((streetX, target[1], height, yaw))
        
        if streetX != target[0]:                              
            path.append((target[0], target[1], height, yaw))
            path.append((streetX, target[1], height, yaw))
    
    return path


def check_position(key, x, y):
    """
    Check if given coordinates match a waypoint for the specified drone.
    
    Args:
        key: Drone URI
        x: X coordinate
        y: Y coordinate
    
    Returns:
        True if (x,y) is a waypoint for this drone, False otherwise
    """
    targets = missions.get(key, (None, []))[1]
    return any((x == mx and y == my) for mx, my in targets)


def check_emergency_stop(cf, commander):
    """
    Check if emergency stop was triggered and stop motors if so.
    
    Args:
        cf: Crazyflie connection object
        commander: High level commander instance
    
    Returns:
        True if emergency stop was triggered, False otherwise
    """
    global emergency_stop_triggered
    if emergency_stop_triggered:
        print("Emergency stop - stopping motors")
        commander.send_stop_setpoint()
        time.sleep(0.1)
        return True
    return False


def execute_grid_path(scf, path, start_point):
    """
    Execute grid path mission for a single drone.
    
    This is the main mission execution function called by the swarm parallel runner.
    
    Args:
        scf: SwarmCrazyflie object
        path: List of (x, y, z, yaw) points to follow
        start_point: Tuple of (x, y) start coordinates for return
    """
    global emergency_stop_triggered, active_drones
    cf = scf.cf
    commander = cf.high_level_commander
    uri = scf._link_uri
     
    active_drones[uri] = cf
    
    # Setup position logging
    pos_log = setup_position_logging(cf, uri)
    if not pos_log:
        print(f"Failed to start position logging for {uri}")
    
    if check_emergency_stop(cf, commander):
        return
    
    # Initialize battery monitor
    monitor = BatteryMonitor(cf)
    monitor.setup_logging()
    time.sleep(1) 

    descend = 0.7
    
    if check_emergency_stop(cf, commander):
        return
        
    # Check battery before takeoff
    decision = monitor.flight_decision() 
    if decision > 0:
        print("Takeoff Aborted")
        return

    # Takeoff
    commander.takeoff(1.0, 2.0)
    time.sleep(3)
    
    # Execute path
    for point in path:
        if check_emergency_stop(cf, commander):
            return
            
        x, y, z, yaw = point
        
        print(f" Moving to: {point}" + uri)

        # Battery check
        decision = monitor.flight_decision()
        if decision == 1:
            commander.land(0.0, 3.0)
            time.sleep(4)
            return
        elif decision == 2:
            print(uri + " returning to station")
            if start_point:
                x, y = start_point[0], start_point[1]
                commander.go_to(x, y, height, 0, returnTime)  
                time.sleep(returnTime + 1)
                commander.land(0.0, 6.0)
                time.sleep(9)
                return
            else:
                return

        # Move to next waypoint
        commander.go_to(x, y, z, 0, rotateTime)
        time.sleep(rotateTime + buffer)
        
        if check_emergency_stop(cf, commander):
            return
            
        # Check battery again after movement
        decision = monitor.flight_decision()
        if decision == 1:
            commander.land(0.0, 3.0)
            time.sleep(4)
            return
        elif decision == 2:
            print(uri + " returning to station")
            if start_point:
                x, y = start_point[0], start_point[1]
                commander.go_to(x, y, height, 0, returnTime)
                time.sleep(returnTime + 1)
                commander.land(0.0, 6.0)
                time.sleep(9)
                return
            else:
                return

        # Simulate package delivery at waypoint
        if check_position(uri, x, y):
            if check_emergency_stop(cf, commander):
                return
                
            # Descend to simulate delivery
            commander.go_to(x, y, z - descend, 0, testTime)
            time.sleep(testTime + buffer)
            
            if check_emergency_stop(cf, commander):
                return
                
            # Ascend back to cruising height
            commander.go_to(x, y, z, 0, testTime)
            time.sleep(testTime + buffer)

    if check_emergency_stop(cf, commander):
        return
        
    # Return to start
    if start_point:
        x, y = start_point[0], start_point[1]
        commander.go_to(x, y, height, 0, returnTime)
        time.sleep(returnTime + 1)
        commander.land(0.0, 6.0)
        time.sleep(9)
    
        if uri in active_drones:
            del active_drones[uri]


# Mission definitions: URI -> (start_position, waypoints)
missions = {
    'radio://0/80/2M/E7E7E7E7E9': (
        (0.23, -2.1),
        [(0.85, -1.24), (0.85, -0.78), (0.85, 0.85)]  
    ),
    'radio://0/80/2M/E7E7E7E7E8': (
        (-0.29, -2.1), 
        [(-1.04, -0.93), (-1.04, 1.09)] 
    )
}


if __name__ == '__main__':
    cflib.crtp.init_drivers()

    print("Starting victim mission with buffered position logging...")
    
    # Generate paths for all drones
    swarm_paths = {
        uri: generate_grid_path(data[0], data[1])
        for uri, data in missions.items()
    }
    
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(missions.keys(), factory=factory) as swarm:
        if emergency_stop_triggered:
            print("Emergency stop triggered before flight")
            sys.exit(0)

        # Reset estimators and arm
        swarm.reset_estimators() 
        swarm.parallel_safe(arm)
        time.sleep(1)
        
        if emergency_stop_triggered:
            print("Emergency stop triggered after arming")
            for scf in swarm._cfs.values():
                try:
                    scf.cf.commander.send_stop_setpoint()
                except:
                    pass
            sys.exit(0)
            
        print("Victim mission started - buffered logging active")
        
        # Execute missions in parallel
        swarm.parallel_safe(execute_grid_path, args_dict={
            uri: [swarm_paths[uri], missions[uri][0]] for uri in swarm_paths
        })
