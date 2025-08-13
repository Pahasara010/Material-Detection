   import pyrebase

   # Configuration matching your config.json
   config = {
       "apiKey": "AIzaSyApin9R_LCeiIaltDV1cyGpOS_saFH4UZc",
       "authDomain": "homesecurity-4731c.firebaseapp.com",
       "databaseURL": "https://homesecurity-4731c-default-rtdb.firebaseio.com/",
       "storageBucket": "homesecurity-4731c.firebasestorage.app"
   }

   try:
       # Initialize Firebase
       firebase = pyrebase.initialize_app(config)
       db = firebase.database()

       # Write a test entry
       db.child("test").set({"status": "connected", "timestamp": "2025-05-17T10:01:00"})
       print("Firebase connection successful!")
       print("Data written to /test: ", db.child("test").get().val())

       # Read back to confirm
       data = db.child("test").get().val()
       print("Data read from /test: ", data)

   except Exception as e:
       print(f"Firebase error: {e}")
   ```

3. **Save and Exit**:
   - Press `Ctrl+O`, `Enter` to save, then `Ctrl+X` to exit.

4. **Run the Script**:
   - Ensure your virtual environment is activated:
     ```bash
     source ~/Desktop/final/venv/bin/activate
     ```
   - Run:
     ```bash
     python3 test_firebase.py
     ```

5. **Expected Output**:
   - If successful:
     ```
     Firebase connection successful!
     Data written to /test: {'status': 'connected', 'timestamp': '2025-05-17T10:01:00'}
     Data read from /test: {'status': 'connected', 'timestamp': '2025-05-17T10:01:00'}
     ```
   - If it fails, you might see:
     ```
     Firebase error: 401 Client Error: Unauthorized...
     ```
     - **Fix**: Re-check Firebase rules in the Console (`https://console.firebase.google.com`):
       ```json
       {
         "rules": {
           ".read": true,
           ".write": true
         }
       }
