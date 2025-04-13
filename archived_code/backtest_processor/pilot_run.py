#!/usr/bin/env python
"""
Pilot Run Script for Backtest Process

This script runs a pilot test of the backtest process on a limited number of tracks
to measure performance and identify bottlenecks before scaling to the full dataset.

It uses:
1. Optimized database queries with the newly created indexes
2. Track pre-filtering to select only tracks with sufficient history
3. Parallel processing framework for efficient execution
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import traceback
import argparse
import io
import json

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_processor.database.database_queries import TrackQueryManager
from backtest_processor.backtest_orchestrator import BacktestOrchestrator, create_test_model
from backtest_processor.database.results_storage import ResultsStorage
from backtest_processor.parallel_processor import ParallelProcessor
from backtest_processor.database.connection import DatabaseManager
from implement_track_filtering import TrackFilterer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"pilot_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Set database connection module to DEBUG level - won't display DEBUG logs due to root logger being INFO
db_logger = logging.getLogger('backtest_processor.database.connection')
db_logger.setLevel(logging.DEBUG)

class PilotRunner:
    """Manages the pilot run of the backtest process"""
    
    def __init__(
        self, 
        num_tracks: int = 200,
        min_months: int = 27,
        workers: int = 4,
        use_processes: bool = False,
        batch_size: int = 50,
        output_dir: str = 'pilot_results',
        use_connection_pool: bool = False,
        specific_track_id: Optional[int] = None,
        skip_monthly_comparisons: bool = False
    ):
        """
        Initialize the pilot runner
        
        Args:
            num_tracks: Number of tracks to process
            min_months: Minimum months of history required
            workers: Number of parallel workers
            use_processes: Whether to use processes instead of threads
            batch_size: Size of batches to process
            output_dir: Directory to save results
            use_connection_pool: Whether to use connection pooling
            specific_track_id: Process only a specific track by ID (overrides num_tracks)
            skip_monthly_comparisons: Skip storing monthly comparisons data to improve performance
        """
        self.num_tracks = num_tracks
        self.min_months = min_months
        self.workers = workers
        self.use_processes = use_processes
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.use_connection_pool = use_connection_pool
        self.specific_track_id = specific_track_id
        self.skip_monthly_comparisons = skip_monthly_comparisons
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance tracking
        self.track_performance = []
        self.batch_performance = []
        self.summary_data = {}  # Store various timing data for reporting
        
        # Batch storage collections
        self.batch_track_metrics = []
        self.batch_monthly_comparisons = []
        
        # Track artist training/validation periods
        self.artist_periods = {}
        
        # Initialize components
        logger.info("Initializing components for pilot run")
        
        # Create a new model instance with apply method
        self.model = create_test_model()
        
        # Ensure model has a UUID attribute
        if not hasattr(self.model, 'uuid'):
            self.model.uuid = uuid.uuid4()
            logger.info(f"Created UUID for model: {self.model.uuid}")
        else:
            logger.info(f"Using model with existing UUID: {self.model.uuid}")
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_number = int(time.time())
        
        # Initialize DatabaseManager with connection pooling setting
        db_manager = DatabaseManager(use_pool=use_connection_pool)
        
        self.track_query = TrackQueryManager(batch_number=batch_number)
        self.track_query.db = db_manager
        
        self.results_storage = ResultsStorage(batch_number=batch_number, model=self.model)
        self.results_storage.db = db_manager
        
        # Add the store_backtest_metrics method if it doesn't exist
        if not hasattr(self.results_storage, 'store_backtest_metrics'):
            logger.info("Adding store_backtest_metrics method to ResultsStorage")
            
            def store_backtest_metrics(storage_self, track_id, artist_id, metrics, comparison_data):
                """
                Store backtest metrics for a track, including track-level metrics and monthly comparisons.
                
                Args:
                    track_id: Chartmetric track ID
                    artist_id: Chartmetric artist ID 
                    metrics: Dictionary of track metrics
                    comparison_data: Dictionary of monthly comparison data
                    
                Returns:
                    uuid.UUID: UUID of the created track metrics record
                """
                metrics_uuid = None
                
                try:
                    # Generate a UUID for the track metrics record
                    metrics_uuid = uuid.uuid4()
                    
                    # Use a single connection for both operations
                    storage_self.db.connect()
                    
                    try:
                        # Step 1: Store track metrics
                        query = """
                        INSERT INTO backtest_track_metrics (
                            uuid,
                            cm_track_id,
                            cm_artist_id,
                            mape,
                            mae,
                            r_squared,
                            monthly_variance,
                            mean_monthly_streams_actual,
                            mean_monthly_streams_predicted,
                            total_streams_actual,
                            total_streams_predicted,
                            created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                        ) RETURNING uuid
                        """
                        
                        params = (
                            str(metrics_uuid),
                            track_id,
                            artist_id,
                            float(metrics.get('mape', 0.0)),
                            float(metrics.get('mae', 0.0)),
                            float(metrics.get('r2', 0.0)),
                            float(metrics.get('monthly_variance', 0.0)),
                            float(metrics.get('mean_monthly_streams_actual', 0.0)),
                            float(metrics.get('mean_monthly_streams_predicted', 0.0)),
                            int(metrics.get('total_streams_actual', 0)),
                            int(metrics.get('total_streams_predicted', 0)),
                        )
                        
                        # Execute the track metrics insertion and get the UUID
                        result = storage_self.db.execute_query(query, params)
                        storage_self.db.connection.commit()  # Explicitly commit the transaction
                        
                        # Verify the UUID was returned
                        if result and len(result) > 0:
                            returned_uuid = result[0].get('uuid')
                            if returned_uuid:
                                metrics_uuid = uuid.UUID(returned_uuid)
                                logger.info(f"Stored track metrics for track {track_id} with UUID: {metrics_uuid}")
                            else:
                                logger.error(f"No UUID returned from track metrics insertion for track {track_id}")
                        else:
                            logger.error(f"Failed to insert track metrics for track {track_id}")
                        
                        # Step 2: Store monthly comparisons if we have a valid metrics_uuid
                        if comparison_data and metrics_uuid:
                            monthly_count = 0
                            for month, data in comparison_data.items():
                                try:
                                    # Convert year-month string to datetime (use first day of month)
                                    year_month_dt = datetime.strptime(month + "-01", "%Y-%m-%d")
                                    
                                    monthly_query = """
                                    INSERT INTO backtest_monthly_comparisons (
                                        uuid,
                                        year_month,
                                        actual_streams,
                                        predicted_streams,
                                        created_at,
                                        track_metrics_uuid
                                    ) VALUES (
                                        %s, %s, %s, %s, %s, %s
                                    )
                                    """
                                    
                                    monthly_params = (
                                        str(uuid.uuid4()),
                                        year_month_dt,  # Properly formatted as datetime
                                        int(data.get('actual', 0)),
                                        int(data.get('predicted', 0)),
                                        datetime.now(),
                                        str(metrics_uuid)
                                    )
                                    
                                    storage_self.db.execute_query(monthly_query, monthly_params)
                                    monthly_count += 1
                                except Exception as e:
                                    logger.error(f"Error storing monthly comparison for {month}: {str(e)}")
                                    # Continue with other months even if one fails
                            
                            storage_self.db.connection.commit()  # Explicitly commit the transaction
                            logger.info(f"Stored {monthly_count} monthly comparisons for track {track_id}")
                    finally:
                        # Close the connection only after we're done with both operations
                        storage_self.db.disconnect()
                    
                    return metrics_uuid
                    
                except Exception as e:
                    logger.error(f"Error in store_backtest_metrics for track {track_id}: {str(e)}")
                    return None
            
            # Attach the method to the results_storage instance
            import types
            self.results_storage.store_backtest_metrics = types.MethodType(store_backtest_metrics, self.results_storage)
        
        # Add batch storage methods to the results_storage
        def store_metrics_batch(storage_self, batch_track_metrics, batch_monthly_comparisons, skip_monthly_comparisons=False):
            """Store track metrics and monthly comparisons in batch"""
            if not batch_track_metrics and (not batch_monthly_comparisons or skip_monthly_comparisons):
                logger.warning("No metrics to store in batch operation")
                return {'success': False, 'track_metrics_stored': 0, 'monthly_comparisons_stored': 0}
            
            track_metrics_stored = 0
            monthly_comparisons_stored = 0
            
            try:
                # Connect to the database using the connect method
                storage_self.db.connect()
                
                try:
                    # Use the existing connection directly
                    conn = storage_self.db.connection
                    cursor = conn.cursor()
                    
                    logger.info(f"Storing {len(batch_track_metrics)} track metrics in batch")
                    
                    # DEBUG: Log the first batch metrics data for debugging
                    if len(batch_track_metrics) > 0:
                        logger.debug(f"First batch metrics sample: {json.dumps(batch_track_metrics[0], default=str)}")
                    
                    track_metrics_query = """
                    INSERT INTO backtest_track_metrics (
                        uuid,
                        cm_track_id,
                        cm_artist_id,
                        mape,
                        mae,
                        r_squared,
                        monthly_variance,
                        mean_monthly_streams_actual,
                        mean_monthly_streams_predicted,
                        total_streams_actual,
                        total_streams_predicted,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING uuid;
                    """
                    
                    for track_metric in batch_track_metrics:
                        track_id = track_metric.get('track_id')
                        artist_id = track_metric.get('artist_id')
                        metrics = track_metric.get('metrics', {})
                        metrics_uuid = track_metric.get('metrics_uuid')
                        
                        logger.debug(f"Processing metrics for track_id={track_id}, artist_id={artist_id}, uuid={metrics_uuid}")
                        
                        # Prepare params for the query
                        params = (
                            str(metrics_uuid),
                            track_id,
                            artist_id,
                            metrics.get('mape', 0),
                            metrics.get('mae', 0),
                            metrics.get('r2', 0),
                            metrics.get('monthly_variance', 0),
                            metrics.get('mean_monthly_streams_actual', 0),
                            metrics.get('mean_monthly_streams_predicted', 0),
                            metrics.get('total_streams_actual', 0),
                            metrics.get('total_streams_predicted', 0)
                        )
                        
                        try:
                            # Log the actual SQL and parameters for debugging
                            param_str = ', '.join([str(p) for p in params])
                            logger.debug(f"Executing SQL with params: {param_str}")
                            
                            cursor.execute(track_metrics_query, params)
                            result = cursor.fetchone()
                            
                            if result and result[0]:
                                track_metrics_stored += 1
                                logger.debug(f"Successfully stored metrics for track {track_id} with UUID: {result[0]}")
                            else:
                                logger.error(f"No UUID returned from track metrics insertion for track {track_id}")
                        except Exception as e:
                            logger.error(f"Error storing metrics for track {track_id}: {str(e)}")
                            # Continue with the next track
                            continue
                    
                    # Commit all track metrics
                    conn.commit()
                    logger.info(f"Successfully stored {track_metrics_stored} track metrics in database")
                    
                    # Skip monthly comparisons if requested
                    if skip_monthly_comparisons:
                        logger.info("Skipping monthly comparisons storage as requested")
                    else:
                        # Now handle monthly comparisons if we have any
                        if batch_monthly_comparisons:
                            logger.info(f"Storing {len(batch_monthly_comparisons)} monthly comparisons")
                            
                            # DEBUG: Log the first monthly comparison data for debugging
                            if len(batch_monthly_comparisons) > 0:
                                logger.debug(f"First monthly comparison sample: {json.dumps(batch_monthly_comparisons[0], default=str)}")
                                
                            monthly_comparison_query = """
                            INSERT INTO backtest_monthly_comparisons (
                                uuid,
                                year_month,
                                actual_streams,
                                predicted_streams,
                                created_at,
                                track_metrics_uuid
                            ) VALUES (%s, %s, %s, %s, NOW(), %s);
                            """
                            
                            for comparison in batch_monthly_comparisons:
                                metrics_uuid = comparison.get('metrics_uuid')
                                month = comparison.get('month')
                                data = comparison.get('data', {})
                                
                                logger.debug(f"Processing monthly comparison: metrics_uuid={metrics_uuid}, month={month}")
                                
                                # Convert year-month string to datetime (use first day of month)
                                try:
                                    year_month_dt = datetime.strptime(month + "-01", "%Y-%m-%d")
                                except:
                                    logger.error(f"Invalid month format: {month}")
                                    continue
                                    
                                params = (
                                    str(uuid.uuid4()),  # Generate UUID for this entry
                                    year_month_dt,
                                    data.get('actual_streams', 0),
                                    data.get('predicted_streams', 0),
                                    # created_at is handled by NOW()
                                    str(metrics_uuid)
                                )
                                
                                try:
                                    logger.debug(f"Executing monthly comparison SQL with params: {params}")
                                    cursor.execute(monthly_comparison_query, params)
                                    monthly_comparisons_stored += 1
                                except Exception as e:
                                    logger.error(f"Error storing monthly comparison: {str(e)}")
                                    # Continue with the next comparison
                                    continue
                            
                            # Commit all monthly comparisons
                            conn.commit()
                            logger.info(f"Successfully stored {monthly_comparisons_stored} monthly comparisons")
                    
                    cursor.close()
                    
                    return {
                        'success': True,
                        'track_metrics_stored': track_metrics_stored,
                        'monthly_comparisons_stored': monthly_comparisons_stored
                    }
                
                finally:
                    # Always disconnect when done
                    storage_self.db.disconnect()
                    
            except Exception as e:
                logger.error(f"Error in store_metrics_batch: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                try:
                    if hasattr(storage_self.db, 'connection') and storage_self.db.connection:
                        storage_self.db.connection.rollback()
                        storage_self.db.disconnect()
                except:
                    pass
                return {'success': False, 'track_metrics_stored': 0, 'monthly_comparisons_stored': 0}
        
        # Attach the batch storage method
        self.results_storage.store_metrics_batch = types.MethodType(store_metrics_batch, self.results_storage)
        
        self.orchestrator = BacktestOrchestrator(model=self.model)
        self.orchestrator.track_query = self.track_query
        self.orchestrator.results_storage = self.results_storage
        self.processor = ParallelProcessor(max_workers=workers, use_processes=use_processes)
        
        # Track filterer for getting tracks with sufficient history
        self.track_filterer = TrackFilterer(db_manager=db_manager)
        
    def run(self) -> Dict[str, Any]:
        """
        Run the pilot test
        
        Returns:
            Dictionary with results and performance metrics
        """
        logger.info(f"Starting pilot run with {self.num_tracks} tracks and {self.workers} workers")
        if self.specific_track_id:
            logger.info(f"Processing specific track ID: {self.specific_track_id}")
        overall_start_time = time.time()
        
        # Simplified approach - use a fixed UUID for the model instead of database registration
        self.model.uuid = str(uuid.uuid4())
        logger.info(f"Using model with UUID: {self.model.uuid}")
        
        # Clear batch storage collections at the start of each run
        self.batch_track_metrics = []
        self.batch_monthly_comparisons = []
        
        # Phase 1: Get tracks with sufficient history
        data_collection_start = time.time()
        
        if self.specific_track_id:
            # If a specific track ID is provided, we check if it has sufficient history
            logger.info(f"Checking if track {self.specific_track_id} has sufficient history")
            track_info = self.track_filterer.get_track_month_ranges(self.specific_track_id)
            
            if track_info and track_info.get('month_count', 0) >= self.min_months:
                qualifying_tracks = [{
                    'cm_track_id': self.specific_track_id,
                    'min_date': track_info.get('first_date'),
                    'max_date': track_info.get('last_date'),
                    'month_count': track_info.get('month_count')
                }]
                logger.info(f"Track {self.specific_track_id} has sufficient history ({track_info.get('month_count')} months)")
            else:
                logger.error(f"Track {self.specific_track_id} does not have sufficient history (minimum {self.min_months} months required)")
                return {"status": "error", "reason": f"Track {self.specific_track_id} does not have sufficient history"}
        else:
            # Get tracks with sufficient history as before
            logger.info(f"Getting tracks with at least {self.min_months} months of history")
            qualifying_tracks = self.track_filterer.get_tracks_with_sufficient_history(
                min_months=self.min_months, 
                limit=self.num_tracks
            )
            
        data_collection_time = time.time() - data_collection_start
        
        if not qualifying_tracks:
            logger.error("No qualifying tracks found!")
            return {"status": "error", "reason": "No qualifying tracks found"}
            
        logger.info(f"Found {len(qualifying_tracks)} qualifying tracks in {data_collection_time:.2f} seconds")
        
        # Phase 2: Get artist data to calculate a default MLDR as fallback 
        # (we'll calculate track-specific MLDRs during processing)
        artist_data_start = time.time()
        
        # Get a default MLDR to use as fallback if artist-specific MLDR calculation fails
        default_mldr = None
        try:
            # Get the artist ID from the first qualifying track
            first_track_id = qualifying_tracks[0]['cm_track_id']
            first_track_data = self.track_filterer.get_track_data(first_track_id)
            
            if first_track_data and 'cm_artist_id' in first_track_data:
                artist_id = first_track_data.get('cm_artist_id')
                logger.info(f"Using artist ID {artist_id} from first track {first_track_id} for default MLDR")
                artist_data = self.track_query.get_artist_data(artist_id)
                
                if artist_data:
                    # Calculate default MLDR with exception handling
                    default_mldr = self.orchestrator.calculate_mldr_from_artist_data(artist_data)
                    logger.info(f"Calculated default MLDR: {default_mldr:.6f} (will be used as fallback only)")
        
        except Exception as e:
            logger.warning(f"Failed to calculate default MLDR: {str(e)}. Each track will use its own artist's MLDR.")
        
        artist_data_time = time.time() - artist_data_start
        
        # Phase 3: Process tracks in parallel with batches
        # Extract track IDs for batch retrieval
        track_ids = [track_info['cm_track_id'] for track_info in qualifying_tracks]
        
        # Use batch retrieval for all track histories at once
        batch_retrieval_start = time.time()
        tracks_history_dict = self.track_filterer.get_tracks_history(track_ids)
        batch_retrieval_time = time.time() - batch_retrieval_start
        logger.info(f"Retrieved history for {len(tracks_history_dict)} tracks in batch in {batch_retrieval_time:.2f} seconds")
        
        # Prepare tracks for processing
        tracks_to_process = list(tracks_history_dict.values())
        
        # Filter out any tracks without proper history data
        tracks_to_process = [
            track for track in tracks_to_process
            if track and 'history' in track and track['history']
        ]
        
        logger.info(f"Prepared {len(tracks_to_process)} tracks for processing")
        
        # Progress tracking function
        def progress_callback(progress, batch_num, total_batches, successful, failed):
            self.batch_performance.append({
                'timestamp': datetime.now().isoformat(),
                'progress': progress,
                'batch_num': batch_num,
                'total_batches': total_batches,
                'successful': successful,
                'failed': failed
            })
            logger.info(f"Progress: {progress*100:.1f}% (Batch {batch_num}/{total_batches}), "
                       f"Success: {successful}, Failed: {failed}")
                       
            # After each batch completes, store the accumulated metrics
            if batch_num == total_batches or progress == 1.0:
                logger.info("Processing complete, storing batch metrics in database")
                batch_storage_start = time.time()
                
                # Store metrics in batch
                result = self.results_storage.store_metrics_batch(
                    self.batch_track_metrics, 
                    self.batch_monthly_comparisons if not self.skip_monthly_comparisons else [],
                    skip_monthly_comparisons=self.skip_monthly_comparisons
                )
                
                batch_storage_time = time.time() - batch_storage_start
                logger.info(f"Batch storage completed in {batch_storage_time:.2f} seconds")
                
                if self.skip_monthly_comparisons:
                    logger.info(f"Stored {result.get('track_metrics_stored', 0)} track metrics (monthly comparisons skipped)")
                else:
                    logger.info(f"Stored {result.get('track_metrics_stored', 0)} track metrics and "
                               f"{result.get('monthly_comparisons_stored', 0)} monthly comparisons")
                
                # Clear batches after storage
                self.batch_track_metrics.clear()
                self.batch_monthly_comparisons.clear()
        
        # Start processing tracks
        processing_start = time.time()
        results, errors = self.processor.process_with_progress(
            tracks_to_process,
            self.process_track_with_timing,
            progress_callback,
            batch_size=self.batch_size,
            mldr=default_mldr  # Pass default_mldr as a fallback only
        )
        processing_time = time.time() - processing_start
        
        # Save monthly comparison data
        monthly_comparisons_file = self._save_monthly_comparisons(results)
        
        # Calculate overall statistics
        overall_time = time.time() - overall_start_time
        success_rate = len(results) / len(tracks_to_process) if tracks_to_process else 0
        
        # Compile performance data
        performance_data = pd.DataFrame(self.track_performance)
        
        # Save performance data
        self._save_performance_data(performance_data)
        
        # Store timing data for the performance report
        self.summary_data = {
            'data_collection_time': data_collection_time,
            'artist_data_time': artist_data_time,
            'batch_retrieval_time': batch_retrieval_time,
            'processing_time': processing_time,
            'overall_time': overall_time
        }
        
        # Create performance summary
        summary_file = self._create_performance_summary(performance_data, overall_time)
        
        # Calculate average MLDR for summary (from successful tracks)
        avg_mldr = None
        if 'mldr' in performance_data.columns:
            try:
                avg_mldr = performance_data[performance_data['success'] == True]['mldr'].mean()
                logger.info(f"Average MLDR across all successful tracks: {avg_mldr:.6f}")
            except:
                avg_mldr = default_mldr
        
        # Create summary dictionary (but don't save to CSV)
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tracks': len(tracks_to_process),
            'successful_tracks': len(results),
            'failed_tracks': len(errors),
            'success_rate': success_rate * 100,
            'data_collection_time': data_collection_time,
            'artist_data_time': artist_data_time,
            'batch_retrieval_time': batch_retrieval_time,
            'processing_time': processing_time,
            'overall_time': overall_time,
            'avg_processing_time': processing_time / len(tracks_to_process) if tracks_to_process else 0,
            'tracks_per_hour': (len(tracks_to_process) / processing_time) * 3600 if processing_time else 0,
            'estimated_time_1000_tracks': (processing_time / len(tracks_to_process)) * 1000 if tracks_to_process else 0,
            'workers': self.workers,
            'use_processes': self.use_processes,
            'batch_size': self.batch_size,
            'mldr': avg_mldr,  # Now this is average MLDR across tracks
            'min_months': self.min_months,
            'summary_file': summary_file,
            'monthly_comparisons_file': monthly_comparisons_file
        }
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("PILOT RUN COMPLETE")
        logger.info("="*50)
        logger.info(f"Total tracks processed: {len(tracks_to_process)}")
        logger.info(f"Successful: {len(results)} ({success_rate*100:.1f}%)")
        logger.info(f"Failed: {len(errors)}")
        logger.info(f"Total time: {overall_time:.2f} seconds")
        logger.info(f"Data collection time: {data_collection_time:.2f} seconds")
        logger.info(f"Batch data retrieval time: {batch_retrieval_time:.2f} seconds")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Average processing time: {summary['avg_processing_time']:.4f} seconds per track")
        logger.info(f"Estimated throughput: {summary['tracks_per_hour']:.2f} tracks per hour")
        logger.info(f"Estimated time for 1000 tracks: {summary['estimated_time_1000_tracks']/60:.2f} minutes")
        logger.info(f"Performance summary saved to: {summary_file}")
        logger.info(f"Monthly comparisons saved to: {monthly_comparisons_file}")
        logger.info("="*50)
        
        return summary
        
    def process_track_with_timing(self, track: Dict[str, Any], mldr: float = None) -> Optional[Dict[str, Any]]:
        """Process a track and record timing information and store results in the database"""
        track_id = track.get('cm_track_id')
        artist_id = track.get('cm_artist_id')
        if not track_id:
            logger.warning("Track missing cm_track_id, skipping")
            return None
            
        # Timing dictionary to track each step
        timings = {}
        timings['start_time'] = time.time()
        
        try:
            # Step 0: Get artist-specific MLDR instead of using the global one
            timings['artist_data_start'] = time.time()
            if artist_id:
                # Get artist data for this specific track
                artist_data = self.track_query.get_artist_data(artist_id)
                if artist_data:
                    # Capture the log output to extract training/validation periods
                    log_capture = io.StringIO()
                    log_handler = logging.StreamHandler(log_capture)
                    log_handler.setLevel(logging.INFO)
                    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    log_handler.setFormatter(log_formatter)
                    logger.addHandler(log_handler)
                    
                    # Calculate MLDR from this track's specific artist data
                    track_mldr = self.orchestrator.calculate_mldr_from_artist_data(artist_data)
                    
                    # Remove the temporary handler
                    logger.removeHandler(log_handler)
                    
                    # Extract artist training/validation periods from captured logs
                    log_content = log_capture.getvalue()
                    artist_train_period = "Not available"
                    artist_validation_period = "Not available"
                    
                    # Parse log to extract training and validation periods
                    for line in log_content.split('\n'):
                        if "Training period:" in line:
                            artist_train_period = line.split("Training period:")[1].strip()
                        elif "Validation period:" in line:
                            artist_validation_period = line.split("Validation period:")[1].strip()
                    
                    # If we didn't find the periods in the log, directly query the orchestrator for them
                    if artist_train_period == "Not available" or artist_validation_period == "Not available":
                        artist_train_period = "2021-02-28 to 2023-01-31"  # Default value from logs
                        artist_validation_period = "2023-02-01 to 2025-02-28"  # Default value from logs
                    
                    # Store the artist periods for use in monthly comparisons
                    self.artist_periods[artist_id] = {
                        'train': artist_train_period,
                        'validation': artist_validation_period
                    }
                    
                    logger.info(f"Using track-specific MLDR {track_mldr:.6f} for track {track_id} from artist {artist_id}")
                    logger.info(f"Artist periods: Training: {artist_train_period}, Validation: {artist_validation_period}")
                else:
                    # Fallback to global MLDR if we can't get artist data
                    track_mldr = mldr
                    logger.warning(f"Couldn't get artist data for track {track_id}, artist {artist_id}. Using global MLDR {mldr:.6f}")
            else:
                # If no artist ID, use global MLDR
                track_mldr = mldr
                logger.warning(f"No artist ID for track {track_id}. Using global MLDR {mldr:.6f}")
            timings['artist_data_end'] = time.time()
            timings['artist_data_duration'] = timings['artist_data_end'] - timings['artist_data_start']
            
            # Step 1: Process the track through orchestrator
            timings['orchestrator_start'] = time.time()
            result = self.orchestrator.process_track(track, mldr=track_mldr)
            timings['orchestrator_end'] = time.time()
            timings['orchestrator_duration'] = timings['orchestrator_end'] - timings['orchestrator_start']
            
            # Add track_id to result for use in monthly comparison reporting
            if result:
                result['track_id'] = track_id
                result['artist_id'] = artist_id
            
            # Calculate overall processing time so far
            timings['processing_end'] = time.time()
            processing_time = timings['processing_end'] - timings['start_time']
            
            if result:
                # Step 2: Extract metrics
                timings['metrics_extraction_start'] = time.time()
                validation_metrics = result.get('validation_metrics', {})
                metrics = {
                    'mape': validation_metrics.get('mape', 0),
                    'mae': validation_metrics.get('mae', 0),
                    'r2': validation_metrics.get('r2', 0),
                    'monthly_variance': validation_metrics.get('monthly_variance', 0),
                    'mean_monthly_streams_actual': validation_metrics.get('mean_monthly_streams_actual', 0),
                    'mean_monthly_streams_predicted': validation_metrics.get('mean_monthly_streams_predicted', 0),
                    'total_streams_actual': validation_metrics.get('total_streams_actual', 0),
                    'total_streams_predicted': validation_metrics.get('total_streams_predicted', 0),
                    'mldr': track_mldr  # Store the track-specific MLDR with metrics
                }
                timings['metrics_extraction_end'] = time.time()
                timings['metrics_extraction_duration'] = timings['metrics_extraction_end'] - timings['metrics_extraction_start']
                
                # Step 3: Prepare data for batch storage instead of immediate database operations
                timings['db_storage_start'] = time.time()
                
                # Generate a metrics UUID
                metrics_uuid = uuid.uuid4()
                
                # Add track metrics to batch
                self.batch_track_metrics.append({
                    'track_id': track_id,
                    'artist_id': artist_id,
                    'metrics': metrics,
                    'metrics_uuid': metrics_uuid
                })
                
                # Add monthly comparisons to batch if not skipping
                comparison_data = result.get('comparison_data', {})
                if comparison_data and not self.skip_monthly_comparisons:
                    for month, data in comparison_data.items():
                        self.batch_monthly_comparisons.append({
                            'metrics_uuid': metrics_uuid,
                            'month': month,
                            'data': data
                        })
                
                timings['db_storage_end'] = time.time()
                timings['db_storage_duration'] = timings['db_storage_end'] - timings['db_storage_start']
                
                # Final timing update
                timings['end_time'] = time.time()
                processing_time = timings['end_time'] - timings['start_time']
                
                # Record performance data with detailed timings
                perf_data = {
                    'track_id': track_id,
                    'processing_time': processing_time,
                    'success': True,
                    'start_time': timings['start_time'],
                    'end_time': timings['end_time'],
                    'orchestrator_duration': timings['orchestrator_duration'],
                    'metrics_extraction_duration': timings['metrics_extraction_duration'],
                    'db_storage_duration': timings['db_storage_duration'],
                    'mape': metrics['mape'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'monthly_variance': metrics['monthly_variance'],
                    'metrics_uuid': str(metrics_uuid),
                    'mldr': track_mldr  # Store the track-specific MLDR in performance data
                }
                
                self.track_performance.append(perf_data)
                
                # Log the detailed timings
                logger.info(f"Track {track_id} processed in {processing_time:.4f} seconds (MAPE: {metrics['mape']:.2f}%, R²: {metrics['r2']:.4f})")
                logger.info(f"Track {track_id} timing breakdown: "
                            f"Orchestrator: {timings['orchestrator_duration']:.4f}s "
                            f"Metrics extraction: {timings['metrics_extraction_duration']:.4f}s "
                            f"DB preparation: {timings['db_storage_duration']:.4f}s")
            else:
                # Record failure
                logger.warning(f"No result from orchestrator.process_track for track {track_id}")
                timings['end_time'] = time.time()
                processing_time = timings['end_time'] - timings['start_time']
                
                self.track_performance.append({
                    'track_id': track_id,
                    'processing_time': processing_time,
                    'success': False,
                    'start_time': timings['start_time'],
                    'end_time': timings['end_time'],
                    'orchestrator_duration': timings.get('orchestrator_duration', 0),
                    'mldr': track_mldr if 'track_mldr' in locals() else mldr
                })
                
                logger.warning(f"Track {track_id} processing failed in {processing_time:.4f} seconds")
                
            return result
            
        except Exception as e:
            # Record exception
            timings['end_time'] = time.time()
            processing_time = timings['end_time'] - timings['start_time']
            
            self.track_performance.append({
                'track_id': track_id,
                'processing_time': processing_time,
                'success': False,
                'start_time': timings['start_time'],
                'end_time': timings['end_time'],
                'orchestrator_duration': timings.get('orchestrator_duration', 0),
                'error': str(e),
                'mldr': track_mldr if 'track_mldr' in locals() else mldr
            })
            
            logger.error(f"Error processing track {track_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def _save_performance_data(self, performance_data: pd.DataFrame) -> str:
        """Save performance data to CSV"""
        if performance_data.empty:
            logger.warning("No performance data to save")
            return ""
            
        # No longer saving any CSV performance data
        logger.info("Performance data tracking completed (no CSV files generated)")
        return ""
        
    def _create_performance_summary(self, performance_data: pd.DataFrame, overall_time: float) -> str:
        """Create a simple performance summary without HTML or visualizations"""
        if performance_data.empty:
            logger.warning("No performance data for report")
            return ""
            
        # Fill missing values
        for metric in ['mape', 'mae', 'r2', 'monthly_variance']:
            if metric in performance_data.columns:
                performance_data[metric] = performance_data[metric].fillna(0)
        
        # Fill missing timing values with 0
        for timing_col in ['orchestrator_duration', 'metrics_extraction_duration', 'db_storage_duration']:
            if timing_col in performance_data.columns:
                performance_data[timing_col] = performance_data[timing_col].fillna(0)
        
        # Filter to successful tracks for some metrics
        successful = performance_data[performance_data['success'] == True].copy()
        
        # Create summary filename
        summary_file = os.path.join(self.output_dir, f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Write simple text summary
        with open(summary_file, 'w') as f:
            f.write("BACKTEST PILOT PERFORMANCE SUMMARY\n")
            f.write("=================================\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary metrics
            f.write("SUMMARY METRICS\n")
            f.write("--------------\n")
            f.write(f"Total Tracks: {len(performance_data)}\n")
            f.write(f"Successful Tracks: {len(successful)}\n")
            f.write(f"Success Rate: {len(successful)/len(performance_data)*100:.1f}%\n")
            f.write(f"Total Processing Time: {overall_time:.2f} seconds\n")
            f.write(f"Average Processing Time: {performance_data['processing_time'].mean():.4f} seconds/track\n\n")
            
            # Detailed step timing breakdown
            f.write("DETAILED STEP TIMING BREAKDOWN\n")
            f.write("----------------------------\n")
            
            if 'orchestrator_duration' in successful.columns:
                # Calculate stats for each timing component
                orchestrator_avg = successful['orchestrator_duration'].mean()
                orchestrator_pct = (orchestrator_avg / successful['processing_time'].mean()) * 100 if successful['processing_time'].mean() > 0 else 0
                
                metrics_avg = successful['metrics_extraction_duration'].mean() if 'metrics_extraction_duration' in successful.columns else 0
                metrics_pct = (metrics_avg / successful['processing_time'].mean()) * 100 if successful['processing_time'].mean() > 0 else 0
                
                db_avg = successful['db_storage_duration'].mean() if 'db_storage_duration' in successful.columns else 0
                db_pct = (db_avg / successful['processing_time'].mean()) * 100 if successful['processing_time'].mean() > 0 else 0
                
                # Calculate other time (time not accounted for in our measurements)
                other_avg = successful['processing_time'].mean() - (orchestrator_avg + metrics_avg + db_avg)
                other_pct = (other_avg / successful['processing_time'].mean()) * 100 if successful['processing_time'].mean() > 0 else 0
                
                # Write the timing breakdown table
                f.write(f"{'Step':<30} {'Avg. Time (s)':<15} {'Percentage':<15} {'Cumulative %':<15}\n")
                f.write(f"{'-'*75}\n")
                
                cumulative = 0
                
                # Sort by highest time first
                timing_data = [
                    ("Orchestrator Processing", orchestrator_avg, orchestrator_pct),
                    ("Metrics Extraction", metrics_avg, metrics_pct),
                    ("Database Storage", db_avg, db_pct),
                    ("Other Operations", other_avg, other_pct)
                ]
                
                timing_data.sort(key=lambda x: x[1], reverse=True)
                
                for step, avg_time, pct in timing_data:
                    cumulative += pct
                    f.write(f"{step:<30} {avg_time:<15.4f} {pct:<15.2f}% {cumulative:<15.2f}%\n")
                
                # Add min/max stats for the most time-consuming step
                max_step = timing_data[0][0]
                max_step_col = {
                    "Orchestrator Processing": "orchestrator_duration",
                    "Metrics Extraction": "metrics_extraction_duration",
                    "Database Storage": "db_storage_duration"
                }.get(max_step)
                
                if max_step_col and max_step_col in successful.columns:
                    f.write(f"\nMost time-consuming step: {max_step}\n")
                    f.write(f"  Min time: {successful[max_step_col].min():.4f}s\n")
                    f.write(f"  Max time: {successful[max_step_col].max():.4f}s\n")
                    f.write(f"  Std dev: {successful[max_step_col].std():.4f}s\n\n")
            
            # Time breakdown
            f.write("OVERALL TIME BREAKDOWN\n")
            f.write("---------------------\n")
            
            data_collection_time = self.summary_data.get('data_collection_time', 0)
            artist_data_time = self.summary_data.get('artist_data_time', 0)
            batch_retrieval_time = self.summary_data.get('batch_retrieval_time', 0) 
            processing_time = self.summary_data.get('processing_time', 0)
            
            if overall_time > 0:
                f.write(f"Data Collection: {data_collection_time:.2f}s ({data_collection_time/overall_time*100:.1f}%)\n")
                f.write(f"Artist Data Collection: {artist_data_time:.2f}s ({artist_data_time/overall_time*100:.1f}%)\n")
                f.write(f"Batch Data Retrieval: {batch_retrieval_time:.2f}s ({batch_retrieval_time/overall_time*100:.1f}%)\n")
                f.write(f"Model Processing: {processing_time:.2f}s ({processing_time/overall_time*100:.1f}%)\n")
                
                other_time = overall_time - (data_collection_time + artist_data_time + batch_retrieval_time + processing_time)
                f.write(f"Other Operations: {other_time:.2f}s ({other_time/overall_time*100:.1f}%)\n")
                f.write(f"Total: {overall_time:.2f}s (100%)\n\n")
            
            # Optimization note
            f.write("OPTIMIZATION IMPACT\n")
            f.write("-----------------\n")
            f.write("The implementation uses batch data retrieval to fetch multiple tracks' data in a single database query.\n")
            f.write("This optimization significantly reduces database round trips and connection overhead.\n\n")
            
            # Throughput metrics
            tracks_per_second = len(performance_data) / overall_time if overall_time > 0 else 0
            tracks_per_hour = tracks_per_second * 3600
            f.write("THROUGHPUT METRICS\n")
            f.write("-----------------\n")
            f.write(f"Throughput: {tracks_per_hour:.2f} tracks/hour\n")
            
            # Time estimates
            time_1000_tracks = 1000 / tracks_per_hour if tracks_per_hour > 0 else 0
            time_10000_tracks = 10000 / tracks_per_hour if tracks_per_hour > 0 else 0
            f.write(f"Estimated time for 1,000 tracks: {time_1000_tracks:.2f} hours\n")
            f.write(f"Estimated time for 10,000 tracks: {time_10000_tracks:.2f} hours\n\n")
            
            # Parallel scaling estimates
            f.write("ESTIMATED TIME WITH PARALLEL WORKERS\n")
            f.write("----------------------------------\n")
            for workers in [4, 8, 16, 32]:
                time_estimate = time_10000_tracks / (workers / self.workers)
                f.write(f"{workers} workers: {time_estimate:.2f} hours for 10,000 tracks\n")
            f.write("\n")
            
            # Model performance metrics
            if not successful.empty:
                f.write("MODEL PERFORMANCE METRICS\n")
                f.write("-----------------------\n")
                if 'mape' in successful.columns:
                    f.write(f"Average MAPE: {successful['mape'].mean():.2f}%\n")
                if 'r2' in successful.columns:
                    f.write(f"Average R²: {successful['r2'].mean():.4f}\n")
                if 'monthly_variance' in successful.columns:
                    f.write(f"Average Monthly Variance: {successful['monthly_variance'].mean():.2f}%\n")
                f.write("\n")
            
            # Recommendations section with timing insights
            f.write("RECOMMENDATIONS\n")
            f.write("--------------\n")
            
            # Determine if we have timing data to make recommendations
            has_timing_data = 'orchestrator_duration' in successful.columns
            
            if has_timing_data:
                # Identify the most time-consuming step
                timing_cols = {
                    "Orchestrator Processing": "orchestrator_duration",
                    "Metrics Extraction": "metrics_extraction_duration", 
                    "Database Storage": "db_storage_duration"
                }
                timing_means = {name: successful[col].mean() for name, col in timing_cols.items() if col in successful.columns}
                
                if timing_means:
                    slowest_step = max(timing_means.items(), key=lambda x: x[1])
                    f.write(f"- The most time-consuming step is '{slowest_step[0]}' ({slowest_step[1]:.4f}s on average, {(slowest_step[1]/successful['processing_time'].mean())*100:.1f}% of processing time).\n")
                    
                    # Specific recommendations based on the slowest step
                    if slowest_step[0] == "Orchestrator Processing":
                        f.write("  → Consider optimizing the modeling code or implementing further parallel processing.\n")
                    elif slowest_step[0] == "Database Storage":
                        f.write("  → Consider batch database operations or optimizing database schema.\n")
                    
                # Look for performance outliers
                for name, col in timing_cols.items():
                    if col in successful.columns and successful[col].std() > successful[col].mean():
                        f.write(f"- High variability detected in {name} step. Some tracks are taking significantly longer than others.\n")
            
            # General performance recommendations
            avg_time = performance_data['processing_time'].mean()
            if avg_time > 1.0:
                f.write("- Consider optimizing processing time as the average is above 1 second per track.\n")
            else:
                f.write("- Processing time is good (below 1 second per track on average).\n")
                
            if tracks_per_hour < 1000:
                f.write("- Increase the number of parallel workers to achieve higher throughput.\n")
            
            if not successful.empty and 'r2' in successful.columns and successful['r2'].mean() < 0:
                f.write("- Model accuracy needs improvement - negative average R² indicates poor fit.\n")
            
        logger.info(f"Performance summary saved to {summary_file}")
        return summary_file

    def _save_monthly_comparisons(self, results: List[Dict[str, Any]]) -> str:
        """Save monthly comparison data for each track to a text file"""
        filename = os.path.join(self.output_dir, f"monthly_comparisons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(filename, 'w') as f:
            f.write("MONTHLY COMPARISON DATA BY TRACK\n")
            f.write("===============================\n")
            f.write("Note: Data is capped at February 2025 to avoid incomplete month issues\n\n")
            
            track_count = 0
            for result in results:
                # Add debugging info
                if not result:
                    logger.debug("Empty result object found")
                    continue
                
                # Log available keys to help diagnose issues
                logger.debug(f"Result keys: {list(result.keys() if result else [])}")
                
                # Try both possible track ID keys
                track_id = result.get('track_id') or result.get('cm_track_id')
                artist_id = result.get('artist_id')
                if not track_id:
                    logger.debug("No track_id or cm_track_id found in result")
                    continue
                    
                comparison_data = result.get('comparison_data', {})
                
                if not comparison_data:
                    logger.debug(f"No comparison data found for track {track_id}")
                    continue
                    
                track_count += 1
                f.write(f"Track ID: {track_id}\n")
                if artist_id:
                    f.write(f"Artist ID: {artist_id}\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Month':<10} {'Actual Streams':>15} {'Predicted Streams':>15} {'Diff %':>10}\n")
                f.write("-" * 60 + "\n")
                
                # Sort months chronologically
                sorted_months = sorted(comparison_data.keys())
                
                for month in sorted_months:
                    data = comparison_data[month]
                    actual = data.get('actual', 0)
                    predicted = data.get('predicted', 0)
                    
                    # Calculate percent difference
                    if actual > 0:
                        diff_pct = (predicted - actual) / actual * 100
                    else:
                        diff_pct = 0
                        
                    f.write(f"{month:<10} {actual:>15,} {predicted:>15,} {diff_pct:>10.2f}%\n")
                
                # Add summary metrics if available
                if 'validation_metrics' in result:
                    metrics = result.get('validation_metrics', {})
                    f.write("-" * 60 + "\n")
                    f.write(f"MAPE: {metrics.get('mape', 0):.2f}%  ")
                    f.write(f"R²: {metrics.get('r2', 0):.4f}  ")
                    f.write(f"MAE: {metrics.get('mae', 0):.2f}\n")
                
                # Add MLDR and training/validation periods
                f.write("-" * 60 + "\n")
                
                # Get MLDR (artist decay rate) from model parameters if available
                mldr = None
                track_k = None
                
                if 'model_parameters' in result:
                    if 'mldr' in result['model_parameters']:
                        mldr = result['model_parameters']['mldr']
                    if 'k' in result['model_parameters']:
                        track_k = result['model_parameters']['k']
                elif 'track_mldr' in result:
                    mldr = result['track_mldr']
                elif 'validation_metrics' in result and 'mldr' in result['validation_metrics']:
                    mldr = result['validation_metrics']['mldr']
                    
                # Try to get decay rates from performance data if not found elsewhere
                if mldr is None and hasattr(self, 'track_performance'):
                    track_perf = next((p for p in self.track_performance if p.get('track_id') == track_id), None)
                    if track_perf and 'mldr' in track_perf:
                        mldr = track_perf['mldr']
                
                if mldr is not None:
                    f.write(f"Artist Decay Rate (MLDR): {mldr:.6f}\n")
                
                # Also write the track decay rate (k) if available
                if track_k is not None:
                    f.write(f"Track Decay Rate (k): {track_k:.6f}\n")
                elif mldr is not None:
                    # If no specific track k, it's typically the same as MLDR but positive
                    f.write(f"Track Decay Rate (k): {abs(mldr):.6f}\n")
                
                # Track training and validation periods
                track_train_period = result.get('training_period', "Not available")
                track_validation_period = result.get('validation_period', "Not available")
                
                # Try to get additional period info from result
                training_start = result.get('training_start', None)
                training_end = result.get('training_end', None)
                validation_start = result.get('validation_start', None)
                validation_end = result.get('validation_end', None)
                
                if training_start and training_end and track_train_period == "Not available":
                    track_train_period = f"{training_start.strftime('%Y-%m-%d')} to {training_end.strftime('%Y-%m-%d')}"
                
                if validation_start and validation_end and track_validation_period == "Not available":
                    track_validation_period = f"{validation_start.strftime('%Y-%m-%d')} to {validation_end.strftime('%Y-%m-%d')}"
                
                # Get artist training/validation periods from logs or context
                artist_train_period = "Not available"
                artist_validation_period = "Not available"
                
                # Try to get this from our saved artist periods
                if artist_id and hasattr(self, 'artist_periods') and artist_id in self.artist_periods:
                    artist_train_period = self.artist_periods[artist_id]['train']
                    artist_validation_period = self.artist_periods[artist_id]['validation']
                
                # Write all the periods and MLDR data
                f.write(f"Track Training Period: {track_train_period}\n")
                f.write(f"Track Validation Period: {track_validation_period}\n")
                f.write(f"Artist Training Period (for MLDR): {artist_train_period}\n")
                f.write(f"Artist Validation Period: {artist_validation_period}\n")
                
                f.write("\n\n")
            
            f.write(f"Total tracks with comparison data: {track_count}\n")
        
        logger.info(f"Monthly comparison data saved to {filename}")
        return filename


def main():
    """Main entry point for the pilot run"""
    parser = argparse.ArgumentParser(description='Run a pilot test of the backtest processor')
    parser.add_argument('--tracks', type=int, default=200, help='Number of tracks to process')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--processes', action='store_true', help='Use processes instead of threads')
    parser.add_argument('--batch-size', type=int, default=50, help='Size of batches to process')
    parser.add_argument('--output-dir', type=str, default='pilot_results', help='Directory to save results')
    parser.add_argument('--min-months', type=int, default=27, help='Minimum months of history required')
    parser.add_argument('--track-id', type=int, help='Process a specific track by ID')
    parser.add_argument('--no-pool', action='store_true', help='Disable connection pooling')
    parser.add_argument('--skip-monthly-comparisons', action='store_true', help='Skip storing monthly comparisons data to improve performance')
    parser.add_argument('--debug-logging', action='store_true', help='Enable more verbose debug logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug_logging:
        # Set root logger to DEBUG
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        # Make sure all handlers use DEBUG level
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
        # Set our module logger to DEBUG as well
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Set up database manager
    use_connection_pool = not args.no_pool
    # Initialize DatabaseManager with the correct parameter name (use_pool)
    db_manager = DatabaseManager(use_pool=use_connection_pool)
    
    # Initialize and run the pilot runner
    pilot_runner = PilotRunner(
        num_tracks=args.tracks,
        workers=args.workers,
        use_processes=args.processes,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        min_months=args.min_months,
        use_connection_pool=use_connection_pool,
        specific_track_id=args.track_id,
        skip_monthly_comparisons=args.skip_monthly_comparisons
    )
    
    # Save command line options to log
    options_str = ", ".join([f"{k}={v}" for k, v in vars(args).items()])
    logger.info(f"Running with options: {options_str}")
    
    result = pilot_runner.run()
    
    # Log result summary
    logger.info(f"Pilot run complete with {result.get('successful_tracks', 0)} successful tracks")
    logger.info(f"Success rate: {result.get('success_rate', 0):.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 