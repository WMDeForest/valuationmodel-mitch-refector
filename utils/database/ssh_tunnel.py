"""
SSH Tunnel Management

This module provides functions for establishing and managing SSH tunnels
for secure database connections to remote servers.
"""

import os
import time
import threading
import logging
import sshtunnel
from utils.database.config import SSH_PARAMS

# Set up logging
logger = logging.getLogger('database.ssh_tunnel')

# Global SSH tunnel and lock
ssh_tunnel = None
ssh_tunnel_lock = threading.Lock()

def safe_log(level, message):
    """Thread/process-safe logging function."""
    process_id = os.getpid()
    if level == 'INFO':
        logger.info(f"[Process {process_id}] {message}")
    elif level == 'WARNING':
        logger.warning(f"[Process {process_id}] {message}")
    elif level == 'ERROR':
        logger.error(f"[Process {process_id}] {message}")
    elif level == 'DEBUG':
        logger.debug(f"[Process {process_id}] {message}")

def setup_ssh_tunnel(ssh_params=None):
    """
    Set up SSH tunnel for database connections.
    
    Args:
        ssh_params: Optional dictionary with SSH parameters. If not provided,
                   the default parameters from config.py will be used.
    
    Returns:
        The SSH tunnel instance that was created or already exists.
    """
    global ssh_tunnel
    
    # Use provided params or fall back to defaults
    params = ssh_params or SSH_PARAMS
    
    # Use a lock to prevent multiple threads from creating tunnels simultaneously
    with ssh_tunnel_lock:
        if ssh_tunnel is None or not ssh_tunnel.is_active:
            try:
                # Close any existing tunnel
                if ssh_tunnel is not None:
                    try:
                        ssh_tunnel.close()
                    except:
                        pass
                
                safe_log('INFO', "Setting up new SSH tunnel...")
                
                # Create SSH tunnel with conservative connection settings
                tunnel = sshtunnel.SSHTunnelForwarder(
                    (params['ssh_host']),
                    ssh_username=params['ssh_username'],
                    ssh_password=params['ssh_password'],
                    remote_bind_address=params['remote_bind_address'],
                    local_bind_address=('localhost', 0),  # Use random local port
                    set_keepalive=5,                     # Keep tunnel alive with packets every 5 seconds
                    compression=True,                    # Enable compression for better performance
                    allow_agent=False                    # Don't use SSH agent
                )
                
                # Start the tunnel
                tunnel.start()
                
                # Wait a moment to ensure it's fully established
                time.sleep(1)
                
                if not tunnel.is_active:
                    raise Exception("Failed to establish SSH tunnel")
                
                ssh_tunnel = tunnel
                safe_log('INFO', f"SSH tunnel established on local port {tunnel.local_bind_port}")
                
                # Add a heartbeat thread to keep the tunnel alive
                def heartbeat():
                    while ssh_tunnel and ssh_tunnel.is_active:
                        try:
                            # Send a keep-alive packet
                            if hasattr(ssh_tunnel.ssh_transport, 'send_ignore'):
                                ssh_tunnel.ssh_transport.send_ignore()
                            time.sleep(10)  # Send heartbeat every 10 seconds
                        except:
                            # If there's an error, break out of the loop
                            break
                
                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()
                
                return tunnel
            except Exception as e:
                safe_log('ERROR', f"Error setting up SSH tunnel: {e}")
                if 'tunnel' in locals() and tunnel.is_active:
                    try:
                        tunnel.close()
                    except:
                        pass
                raise
    
    return ssh_tunnel

def close_ssh_tunnel():
    """
    Close the SSH tunnel if it exists and is active.
    
    Returns:
        True if the tunnel was successfully closed, False otherwise.
    """
    global ssh_tunnel
    
    if ssh_tunnel is not None and ssh_tunnel.is_active:
        try:
            ssh_tunnel.close()
            ssh_tunnel = None
            safe_log('INFO', "SSH tunnel closed successfully")
            return True
        except Exception as e:
            safe_log('ERROR', f"Error closing SSH tunnel: {e}")
    
    return False

def get_tunnel_port():
    """
    Get the local port of the SSH tunnel.
    
    Returns:
        The local port number if the tunnel is active, None otherwise.
    """
    global ssh_tunnel
    
    if ssh_tunnel is not None and ssh_tunnel.is_active:
        return ssh_tunnel.local_bind_port
    
    return None 