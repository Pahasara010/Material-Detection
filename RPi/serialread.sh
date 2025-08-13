#!/bin/bash -e

# Declare associative array of devices RX
declare -A DEVICES
DEVICES["/dev/ttyUSB0"]="room1"
#DEVICES["/dev/ttyUSB1"]="room2"
#DEVICES["/dev/ttyUSB2"]="room3"

# Loop through each device in the DEVICES array
for SERIAL_DEVICE in "${!DEVICES[@]}"; do
    ROOM_NAME="${DEVICES[$SERIAL_DEVICE]}"

    # Set baud rate and serial config for the device
    stty -F "$SERIAL_DEVICE" 921600 cs8 -cstopb -parenb

    # Create a unique FIFO for each room
    FIFO_PATH="/tmp/csififo_${ROOM_NAME}"
    if [[ ! -p "$FIFO_PATH" ]]; then
        mkfifo "$FIFO_PATH"  # Use mkfifo to create the named pipe
        chmod 0644 "$FIFO_PATH"  # Set permissions (rw-r--r--)
    fi

    # Stream data from serial port into the corresponding FIFO in the background
    cat < "$SERIAL_DEVICE" | awk NF > "$FIFO_PATH" &
done

# Keep the script running to maintain background processes
wait