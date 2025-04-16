"""
Database Connection Management

This module provides functions for establishing and managing database connections
through connection pools and SQLAlchemy engines.
"""

import os
import time
import logging
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine, pool as sa_pool
import atexit

# Import from other utility modules
from utils.database.ssh_tunnel import setup_ssh_tunnel, safe_log, ssh_tunnel
from utils.database.config import DB_PARAMS, MIN_CONNECTIONS, MAX_CONNECTIONS

# Set up logging
logger = logging.getLogger('database.connection')

# Global connection pools
db_connection_pool = None
sqlalchemy_engine = None

def initialize_connection_pool(db_params=None, min_conn=None, max_conn=None):
    """
    Initialize the database connection pool using SSH tunnel.
    
    Args:
        db_params: Optional dictionary with database parameters. If not provided,
                  the default parameters from config.py will be used.
        min_conn: Minimum number of connections in the pool.
        max_conn: Maximum number of connections in the pool.
    
    Returns:
        The initialized connection pool.
    """
    global db_connection_pool, ssh_tunnel
    
    # Use provided params or fall back to defaults
    params = db_params or DB_PARAMS
    min_connections = min_conn or MIN_CONNECTIONS
    max_connections = max_conn or MAX_CONNECTIONS
    
    if db_connection_pool is not None:
        return db_connection_pool
    
    # Ensure we have a working SSH tunnel
    retry_count = 0
    while retry_count < 3:
        try:
            # Make sure we have an SSH tunnel
            tunnel = setup_ssh_tunnel()
            
            # Create connection parameters with tunnel's local port
            conn_params = params.copy()
            conn_params['port'] = tunnel.local_bind_port
            
            # Create connection pool
            connection_pool = pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                **conn_params
            )
            
            # Test a connection to make sure it works
            test_conn = connection_pool.getconn()
            connection_pool.putconn(test_conn)
            
            db_connection_pool = connection_pool
            safe_log('INFO', f"Database connection pool initialized with {min_connections}-{max_connections} connections")
            return connection_pool
        except Exception as e:
            retry_count += 1
            safe_log('ERROR', f"Error initializing connection pool (attempt {retry_count}/3): {e}")
            
            # If we have a tunnel issue, reset it and retry
            if ssh_tunnel is not None:
                try:
                    ssh_tunnel.close()
                except:
                    pass
                
            # Wait before retrying
            time.sleep(2)
    
    # If we exhaust all retries, raise the error
    raise Exception("Failed to initialize connection pool after multiple attempts")

def initialize_sqlalchemy_engine(db_params=None, max_conn=None):
    """
    Initialize SQLAlchemy engine with connection pooling.
    
    Args:
        db_params: Optional dictionary with database parameters. If not provided,
                  the default parameters from config.py will be used.
        max_conn: Maximum number of connections in the pool.
    
    Returns:
        The initialized SQLAlchemy engine.
    """
    global sqlalchemy_engine, ssh_tunnel
    
    # Use provided params or fall back to defaults
    params = db_params or DB_PARAMS
    max_connections = max_conn or MAX_CONNECTIONS
    
    if sqlalchemy_engine is not None:
        return sqlalchemy_engine
    
    # Ensure we have a working SSH tunnel
    retry_count = 0
    while retry_count < 3:
        try:
            # Make sure we have an SSH tunnel
            tunnel = setup_ssh_tunnel()
            
            # Create connection string using tunnel's local port
            conn_string = f"postgresql://{params['user']}:{params['password']}@localhost:{tunnel.local_bind_port}/{params['dbname']}"
            
            # Create engine with connection pooling
            engine = create_engine(
                conn_string,
                poolclass=sa_pool.QueuePool,
                pool_size=max_connections,
                max_overflow=2,
                pool_timeout=30,
                pool_recycle=1800  # Recycle connections after 30 minutes
            )
            
            # Test the connection
            with engine.connect() as conn:
                pass  # Just open and close the connection to test it
            
            sqlalchemy_engine = engine
            safe_log('INFO', "SQLAlchemy engine initialized with connection pooling")
            return engine
        except Exception as e:
            retry_count += 1
            safe_log('ERROR', f"Error creating SQLAlchemy engine (attempt {retry_count}/3): {e}")
            
            # If we have a tunnel issue, reset it and retry
            if ssh_tunnel is not None:
                try:
                    ssh_tunnel.close()
                except:
                    pass
                
            # Wait before retrying
            time.sleep(2)
    
    # If we exhaust all retries, raise the error
    raise Exception("Failed to initialize SQLAlchemy engine after multiple attempts")

def get_db_connection():
    """
    Get a connection from the connection pool.
    
    Returns:
        A database connection from the pool.
    """
    global db_connection_pool
    
    # Initialize pool if not already done
    if db_connection_pool is None:
        initialize_connection_pool()
    
    # Retry logic for getting a connection
    retry_count = 0
    while retry_count < 3:
        try:
            # Get connection from pool
            connection = db_connection_pool.getconn()
            
            # Test the connection with a simple query
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                
            return connection
        except Exception as e:
            retry_count += 1
            safe_log('ERROR', f"Error getting connection from pool (attempt {retry_count}/3): {e}")
            
            # If the connection failed, reinitialize everything
            if retry_count >= 2:
                safe_log('WARNING', "Attempting to rebuild connection pool...")
                try:
                    if db_connection_pool is not None:
                        db_connection_pool.closeall()
                except:
                    pass
                    
                db_connection_pool = None
                
                # Reset SSH tunnel
                if ssh_tunnel is not None:
                    try:
                        ssh_tunnel.close()
                    except:
                        pass
                
                # Reinitialize
                initialize_connection_pool()
            
            time.sleep(1)  # Wait before retrying
    
    raise Exception("Failed to get a database connection after multiple attempts")

def release_db_connection(connection):
    """
    Release a connection back to the pool.
    
    Args:
        connection: The database connection to release.
    """
    global db_connection_pool
    
    if db_connection_pool is not None and connection is not None:
        try:
            db_connection_pool.putconn(connection)
        except Exception as e:
            safe_log('ERROR', f"Error returning connection to pool: {e}")

def get_sqlalchemy_engine():
    """
    Get the SQLAlchemy engine with connection pooling.
    
    Returns:
        The initialized SQLAlchemy engine.
    """
    global sqlalchemy_engine
    
    # Initialize engine if not already done
    if sqlalchemy_engine is None:
        initialize_sqlalchemy_engine()
    
    return sqlalchemy_engine

def close_all_connections():
    """
    Close all database connections and SSH tunnel.
    
    This function should be registered with atexit to ensure
    proper cleanup when the application exits.
    """
    global db_connection_pool, sqlalchemy_engine
    
    safe_log('INFO', "Closing all database connections")
    
    # Close psycopg2 connection pool
    if db_connection_pool is not None:
        try:
            db_connection_pool.closeall()
        except Exception as e:
            safe_log('ERROR', f"Error closing connection pool: {e}")
        db_connection_pool = None
    
    # Close SQLAlchemy engine
    if sqlalchemy_engine is not None:
        try:
            sqlalchemy_engine.dispose()
        except Exception as e:
            safe_log('ERROR', f"Error disposing SQLAlchemy engine: {e}")
        sqlalchemy_engine = None
    
    # Close SSH tunnel (imported from ssh_tunnel.py)
    from utils.database.ssh_tunnel import close_ssh_tunnel
    close_ssh_tunnel()

# Register cleanup function to be called when using this module
atexit.register(close_all_connections) 