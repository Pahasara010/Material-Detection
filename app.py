from flask import Flask, render_template, request, jsonify  # Import Flask web framework components
import firebase_admin  # Import Firebase Admin SDK for Python
from firebase_admin import credentials, db  # Import credentials and database modules
import datetime  # For handling timestamps
import json  # For parsing configuration file
import logging  # For logging system events and errors

# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Load config.json file containing Firebase configuration
try:
    with open('/home/ravindu/Desktop/final/saved_models/config.json', 'r') as f:  # Open and read the config file
        config = json.load(f)  # Parse JSON into a Python dictionary
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")  # Log error if file loading fails
    config = {}  # Set config to empty dictionary as fallback

# Initialize Firebase Admin SDK with credentials and database URL
try:
    cred = credentials.Certificate("/home/ravindu/Desktop/final/saved_models/homesecurity-4731c-firebase-adminsdk-fbsvc-5a941ab075.json")  # Load service account credentials
    firebase_admin.initialize_app(cred, {
        'databaseURL': config.get('firebase_config', {}).get('databaseURL', '')  # Set database URL from config or empty string
    })
    logger.info("[Wave_Walker] Firebase initialized successfully")  # Log successful initialization
except Exception as e:
    logger.error(f"[Wave_Walker] Firebase initialization failed: {e}")  # Log error if initialization fails

@app.route('/', methods=['GET', 'POST'])  # Define route for the homepage, handling GET and POST requests
def index():
    # Handle the index page, allowing username submission and displaying user data
    try:
        if request.method == 'POST':  # Check if the request is a POST (form submission)
            username = request.form.get('username')  # Get username from form data
            if username:  # Proceed only if username is provided
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
                user_data_ref = db.reference('user_data')  # Reference to the user_data node in Firebase
                user_data_ref.push({  # Push new user data to Firebase
                    'username': username,
                    'timestamp': current_time
                })
                logger.info(f"[Wave_Walker] Pushed username: {username}")  # Log successful data push

        user_data = db.reference('user_data').get() or {}  # Fetch all user data or empty dict if none
        return render_template('index.html', user_data=user_data)  # Render the index template with user data
    except Exception as e:
        logger.error(f"[Wave_Walker] Error in index route: {e}")  # Log any errors
        return "Error loading page", 500  # Return error response with status code 500

@app.route('/live')  # Define route for live activity data, handling GET requests
def live():
    # Fetch and return the latest activity detection data from Firebase
    try:
        # Fetch the latest activity log for room1 and room2
        logs = {}
        for room in ['room1', 'room2']:  # Iterate over the two rooms
            room_logs = db.reference(f'activity_logs/{room}').order_by_child('timestamp').limit_to_last(1).get()  # Get the latest log
            if room_logs:  # If logs exist for this room
                latest_key = list(room_logs.keys())[0]  # Get the key of the latest entry
                logs[room] = room_logs[latest_key]  # Store the latest log

        if logs:  # If any logs were found
            latest_room = max(logs, key=lambda r: logs[r].get('timestamp', ''))  # Find the room with the latest timestamp
            latest_activity = logs[latest_room]  # Get the latest activity data
            return jsonify({  # Return JSON response with activity details
                'activity': latest_activity.get('activity', 'empty'),  # Default to 'empty' if no activity
                'rx_sources': latest_activity.get('rx_sources', [latest_room]),  # Default to room if no sources
                'timestamp': latest_activity.get('timestamp', '')  # Default to empty string if no timestamp
            })
        return jsonify({'activity': 'empty', 'rx_sources': [], 'timestamp': ''})  # Return empty response if no logs
    except Exception as e:
        logger.error(f"[Wave_Walker] Error fetching live data: {e}")  # Log any errors
        return jsonify({'activity': 'empty', 'rx_sources': [], 'timestamp': ''}), 500  # Return error response with status code 500

@app.route('/toggle_arm', methods=['POST'])  # Define route to toggle the system armed state, handling POST requests
def toggle_arm():
    # Toggle the armed state of the system in Firebase
    try:
        system_ref = db.reference('system')  # Reference to the system node in Firebase
        current_state = system_ref.get().get('armed', True)  # Get current armed state, default to True
        new_state = not current_state  # Toggle the state
        system_ref.set({'armed': new_state})  # Update the armed state in Firebase
        logger.info(f"[Wave_Walker] System armed: {new_state}")  # Log the new state
        return jsonify({'armed': new_state})  # Return JSON response with the new state
    except Exception as e:
        logger.error(f"[Wave_Walker] Error toggling arm state: {e}")  # Log any errors
        return jsonify({'armed': current_state}), 500  # Return error response with current state and status code 500

if __name__ == '__main__':
    # Run the Flask application in debug mode, accessible from any host on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)  # Start the server
