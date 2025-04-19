import sys
import logging
from pathlib import Path
from web_server import app, socketio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    try:
        # Create necessary directories
        Path('data').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        
        # Start the web server
        socketio.run(
            app,
            debug=False,
            port=5000,
            allow_unsafe_werkzeug=True,
            host='0.0.0.0'
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")

if __name__ == '__main__':
    main()
