# Import necessary modules and classes
import logging
import threading
import sys
from datetime import datetime


from fds import (
    RealTimeDataIngestionManager, 
    seed_sample_data, 
    HistoricalDataCollector, 
    EnhancedAIFraudDashboard, 
    add_novelty_detection_page
)

def main():
    """
    Main function to run the AI Fraud Detection System
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('ai_fraud_detection.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        print("Starting AI Fraud Detection System...")
        
        # Create data ingestion manager
        print("Initializing data manager...")
        data_manager = RealTimeDataIngestionManager(
            database_path='ai_fraud_reports.db'
        )
        
        # Initialize database with default data
        print("Initializing database...")
        data_manager.initialize_database()
        
        # Pre-seed database with sample data if needed
        print("Seeding sample data...")
        seed_sample_data(data_manager)
        
        # Collect historical data
        print("Collecting historical data from January 2025...")
        historical_collector = HistoricalDataCollector(data_manager)
        historical_reports = historical_collector.collect_historical_data()
        print(f"Collected {historical_reports} historical reports")
        
        # Create dashboard
        print("Setting up dashboard...")
        dashboard = EnhancedAIFraudDashboard(data_manager)
        
        print("Setting up novelty detection...")
        add_novelty_detection_page(dashboard, data_manager) 
        
        # Connect dashboard to data manager
        data_manager.dashboard = dashboard
        
        # Start data collection in a separate thread
        print("Starting data collection thread...")
        collection_thread = threading.Thread(
            target=data_manager.start_periodic_collection,
            kwargs={'collection_interval_minutes': 15, 'forecast_interval_hours': 6},
            daemon=True
        )
        collection_thread.start()
        
        # Start dashboard in main thread
        print("Starting dashboard at http://localhost:8050")
        print("Press Ctrl+C to stop the server")
        dashboard.run(debug=True, port=8050)
        
    except Exception as e:
        logger.error(f"Critical error in AI Fraud Detection System: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Check if we're running in historical mode
    if len(sys.argv) > 1 and sys.argv[1] == "--historical":
        print("Running in historical data collection mode")
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('historical_collection.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize data manager
            data_manager = RealTimeDataIngestionManager(
                database_path='ai_fraud_reports.db'
            )
            
            # Initialize database
            data_manager.initialize_database()
            
            # Run historical collection
            historical_collector = HistoricalDataCollector(data_manager)
            collected = historical_collector.collect_historical_data()
            
            print(f"Successfully collected {collected} historical reports")
            sys.exit(0)
        
        except Exception as e:
            logger.error(f"Error in historical collection: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Run normal main function
        main()