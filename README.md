

# Record CSI data files for different activities (60 seconds each, using /dev/ttyUSB0 at 921600 baud)
timeout 60s ./Wave_Walker/scripts/logserial.sh -d /dev/ttyUSB0 -b 921600 -l walk_1.csi
timeout 60s ./Wave_Walker/scripts/logserial.sh -d /dev/ttyUSB0 -b 921600 -l run_1.csi
timeout 60s ./Wave_Walker/scripts/logserial.sh -d /dev/ttyUSB0 -b 921600 -l idle_1.csi
timeout 60s ./Wave_Walker/scripts/logserial.sh -d /dev/ttyUSB0 -b 921600 -l jump_1.csi
timeout 60s ./Wave_Walker/scripts/logserial.sh -d /dev/ttyUSB0 -b 921600 -l empty_1.csi


# Navigate to project root and activate virtual environment
cd Wave_Walker  # Navigate to the 'final' directory under Wave_Walker
source venv/bin/activate  # Activate the virtual environment

# Install Python dependencies using the requirements file for the Wave_Walker project
pip3 install -r Wave_Walker/requires/main.txt


# Set up Raspberry Pi serial reading script
cd RPi
chmod +x serialread.sh  # Make serialread.sh executable
sudo ./serialread.sh  # Run the serial reading script with superuser privileges

# Populate CSI FIFO for real-time data streaming
chmod +x ./Wave_Walker/scripts/populate_csififo.sh  # Make populate_csififo.sh executable
./Wave_Walker/scripts/populate_csififo.sh -d /dev/ttyUSB0 -b 921600 -n /tmp/csififo -s 235 -p 0644 -u 1000  # Start FIFO population

# Run the main HAR processing script
python main.py --config Wave_Walker/saved_models/config.json --host localhost --port 9999 --frequency 2  # Start real-time HAR server

# Run Flask web applications in the background
python Wave_Walker/app.py &  # Start the main web app in the background
python Wave_Walker/report_app.py &  # Start the report web app in the background

# Update and upgrade system packages
sudo apt update && sudo apt upgrade -y  # Update package lists and upgrade all packages

# Re-run Raspberry Pi and FIFO setup (if needed after update)
cd RPi
chmod +x serialread.sh  # Ensure serialread.sh is executable
sudo ./serialread.sh  # Re-run serial reading script

./Wave_Walker/scripts/populate_csififo.sh -d /dev/ttyUSB0 -b 921600 -n /tmp/csififo -s 235 -p 0644 -u 1000  # Re-run FIFO population

# Re-run the main HAR processing script (if needed)
python main.py --config Wave_Walker/saved_models/config.json --host localhost --port 9999 --frequency 2  # Re-start HAR server

# Navigate to React app directory and start the frontend
cd Wave_Walker/my-react-app
npm start  # Start the React development server

# Open Firebase console to configure database rules (manual step)
# https://console.firebase.google.com/project/homesecurity-4731c/database/homesecurity-4731c-default-rtdb/rules