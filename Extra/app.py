from flask import Flask, render_template, jsonify
import pyrebase
import websockets
import asyncio
import json

app = Flask(__name__)

# Firebase config
config = {
    "apiKey": "your_api_key",
    "authDomain": "your_project.firebaseapp.com",
    "databaseURL": "https://your_project.firebaseio.com",
    "storageBucket": "your_project.appspot.com"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()

async def get_websocket_data():
    async with websockets.connect('ws://localhost:9999') as websocket:
        data = await websocket.recv()
        return json.loads(data)

def get_activity_log():
    logs = db.child("activity_logs").order_by_child("timestamp").limit_to_last(10).get().val()
    return [{"timestamp": log["timestamp"], "activity": log["activity"], "rx_sources": log.get("rx_sources", [])} for log in logs.values()] if logs else []

@app.route('/')
def home():
    return render_template('index.html', logs=get_activity_log())

@app.route('/live')
async def live_data():
    try:
        data = await get_websocket_data()
        return jsonify({"activity": data["hypothesis"], "rx_sources": data["rx_sources"]})
    except:
        return jsonify({"activity": "empty", "rx_sources": []})

@app.route('/toggle_arm', methods=['POST'])
def toggle_arm():
    armed = db.child("system").get().val().get("armed", True)
    db.child("system").set({"armed": not armed})
    return jsonify({"armed": not armed})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)