from flask import Flask, render_template
import pyrebase
from datetime import datetime, timedelta
from collections import Counter
import json

app = Flask(__name__)

config = {
    "apiKey": "your_api_key",
    "authDomain": "your_project.firebaseapp.com",
    "databaseURL": "https://your_project.firebaseio.com",
    "storageBucket": "your_project.appspot.com"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()

def get_activity_logs(period='all'):
    logs = db.child("activity_logs").get().val()
    if not logs:
        return []

    filtered_logs = []
    now = datetime.utcnow()
    if period == 'day':
        cutoff = now - timedelta(days=1)
    elif period == 'week':
        cutoff = now - timedelta(weeks=1)
    else:
        cutoff = None

    for log_id, log in logs.items():
        log_time = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
        if cutoff is None or log_time >= cutoff:
            filtered_logs.append({
                'timestamp': log['timestamp'],
                'activity': log['activity'],
                'rx_sources': log.get('rx_sources', [])
            })

    return sorted(filtered_logs, key=lambda x: x['timestamp'], reverse=True)

def get_activity_stats(logs):
    activities = [log['activity'] for log in logs]
    counts = Counter(activities)
    return {
        'labels': list(counts.keys()),
        'data': list(counts.values())
    }

@app.route('/')
def index():
    period = 'all'
    logs = get_activity_logs(period)
    stats = get_activity_stats(logs)
    return render_template('report_index.html', logs=logs, stats=json.dumps(stats), period=period)

@app.route('/<period>')
def filter_by_period(period):
    if period not in ['day', 'week', 'all']:
        period = 'all'
    logs = get_activity_logs(period)
    stats = get_activity_stats(logs)
    return render_template('report_index.html', logs=logs, stats=json.dumps(stats), period=period)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)