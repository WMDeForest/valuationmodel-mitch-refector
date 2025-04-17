"""
Configuration settings for database connections.

This module contains default configuration for:
- SSH tunnel parameters
- Database connection parameters
- Connection pool settings
"""

import multiprocessing

# SSH connection parameters
SSH_PARAMS = {
    'ssh_host': '135.181.17.84',
    'ssh_username': 'root',
    'ssh_password': 'E7Kigxn3UtQiTx',
    'remote_bind_address': ('localhost', 5432)
}

# Database connection parameters
DB_PARAMS = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'Zafe5Ph353',
    'host': 'localhost',  # Connect via SSH tunnel
    'port': '5432'
}

# Default parallelism and connection pool settings
# Use a conservative worker count to avoid overloading the SSH tunnel
DEFAULT_WORKERS = min(4, max(1, multiprocessing.cpu_count() // 2))  # Lower worker count
MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 10  # Reduced max connections to avoid overwhelming the tunnel 