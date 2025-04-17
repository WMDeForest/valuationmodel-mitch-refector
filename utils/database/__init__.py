"""
Database utilities for connecting to and querying the database.

This package provides common functionality for:
- Setting up SSH tunnels to remote databases
- Managing database connection pools
- Handling SQLAlchemy engines
- Helper functions for querying and updating data
"""

from utils.database.ssh_tunnel import setup_ssh_tunnel, close_ssh_tunnel
from utils.database.connection import (
    initialize_connection_pool,
    initialize_sqlalchemy_engine,
    get_db_connection,
    release_db_connection,
    get_sqlalchemy_engine,
    close_all_connections
)
from utils.database.query import (
    execute_query,
    fetch_all,
    fetch_one,
    execute_batch,
    query_to_dataframe,
    dataframe_to_table
)

# Import default configuration settings
from utils.database.config import SSH_PARAMS, DB_PARAMS, DEFAULT_WORKERS, MIN_CONNECTIONS, MAX_CONNECTIONS

__all__ = [
    # SSH tunnel management
    'setup_ssh_tunnel',
    'close_ssh_tunnel',
    
    # Connection management
    'initialize_connection_pool',
    'initialize_sqlalchemy_engine',
    'get_db_connection',
    'release_db_connection',
    'get_sqlalchemy_engine',
    'close_all_connections',
    
    # Query execution
    'execute_query',
    'fetch_all',
    'fetch_one',
    'execute_batch',
    'query_to_dataframe',
    'dataframe_to_table',
    
    # Configuration
    'SSH_PARAMS',
    'DB_PARAMS',
    'DEFAULT_WORKERS',
    'MIN_CONNECTIONS',
    'MAX_CONNECTIONS'
] 