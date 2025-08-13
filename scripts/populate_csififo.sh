#!/bin/bash -e  # Shebang to run with Bash, exit on error

# Defaults for configuration parameters
BAUDRATE=921600  # Default baud rate for serial communication
CSIFIFO_NAME=/tmp/csififo  # Default name/path for the FIFO (named pipe)
CSIFIFO_BUFSIZ=235  # Default buffer size for the FIFO
CSIFIFO_PERM=0644  # Default permissions for the FIFO (read/write for owner, read for others)
CSIFIFO_USER=1000  # Default user ID for the FIFO owner

# Array to store devices and their corresponding room tags
declare -A DEVICES  # Associative array for device-to-room mapping
DEVICES["/dev/ttyUSB0"]="room1"  # Map first USB device to room1
#DEVICES["/dev/ttyUSB1"]="room2"  # Uncomment to map second USB device to room2
# Add more devices as needed, e.g., DEVICES["/dev/ttyUSB2"]="room3"  # Uncomment to map additional devices

# Parse command-line arguments to override defaults (optional)
while [[ $# -gt 0 ]]; do  # Loop while there are arguments
    key="$1"  # Get the current argument
    case $key in
        -b|--baudrate)  # Option to set baud rate
        BAUDRATE="$2"  # Set baud rate to the next argument
        shift; shift  # Move past the option and its value
        ;;
        -n|--csififo-name)  # Option to set FIFO name
        CSIFIFO_NAME="$2"  # Set FIFO name to the next argument
        shift; shift  # Move past the option and its value
        ;;
        -s|--csififo-bufsize)  # Option to set FIFO buffer size
        CSIFIFO_BUFSIZ="$2"  # Set buffer size to the next argument
        shift; shift  # Move past the option and its value
        ;;
        -p|--csififo-perm)  # Option to set FIFO permissions
        CSIFIFO_PERM="$2"  # Set permissions to the next argument
        shift; shift  # Move past the option and its value
        ;;
        -u|--csififo-user)  # Option to set FIFO user ID
        CSIFIFO_USER="$2"  # Set user ID to the next argument
        shift; shift  # Move past the option and its value
        ;;
        *)    # Handle unknown options
        shift  # Move to the next argument
        ;;
    esac
done

# Print the current configuration for verification
echo "[*] Baud rate      : $BAUDRATE"  # Display configured baud rate
echo "[*] CSI FIFO       : $CSIFIFO_NAME"  # Display configured FIFO path
echo "[*] CSI FIFO size  : $CSIFIFO_BUFSIZ"  # Display configured buffer size
echo "[*] CSI FIFO perm  : $CSIFIFO_PERM"  # Display configured permissions
echo "[*] CSI FIFO user  : $CSIFIFO_USER"  # Display configured user ID
for dev in "${!DEVICES[@]}"; do  # Loop through device keys
    echo "[*] Device $dev tagged as ${DEVICES[$dev]}"  # Display each device and its room tag
done
echo  # Add a newline for readability

# Load the emlog kernel module if not already loaded
if grep --quiet 'emlog' /proc/modules; then  # Check if emlog module is loaded
    echo "[+] Kernel module loaded"  # Confirm module is already loaded
else
    echo "[-] Kernel module not detected. Loading..."  # Indicate module needs loading
    sudo modprobe emlog  # Load the emlog kernel module with superuser privileges
fi

# Create or verify the FIFO (named pipe) for data streaming
if [[ -c $CSIFIFO_NAME ]]; then  # Check if FIFO exists as a character device
    echo "[+] Detected FIFO $CSIFIFO_NAME"  # Confirm FIFO is present
else
    echo "[-] Log device not found. Creating..."  # Indicate FIFO needs creation
    sudo mkemlog $CSIFIFO_NAME $CSIFIFO_BUFSIZ $CSIFIFO_PERM $CSIFIFO_USER  # Create FIFO with specified parameters
fi

# Configure serial ports for each device
for dev in "${!DEVICES[@]}"; do  # Loop through device keys
    echo "[+] Programming serial port $dev"  # Indicate configuration of the current device
    stty -F "$dev" $BAUDRATE cs8 -cstopb -parenb  # Set baud rate, 8 data bits, no parity, 1 stop bit
done

# Function to read from a device and tag data with the corresponding area
read_and_tag() {
    local device=$1  # Device path (e.g., /dev/ttyUSB0)
    local area=$2  # Room tag (e.g., room1)
    while true; do  # Infinite loop to continuously read data
        if read -r line < "$device"; then  # Attempt to read a line from the device
            echo "${area}: ${line}" >> $CSIFIFO_NAME  # Tag the line with the area and append to FIFO
        else
            echo "[-] Error reading from $device, retrying..."  # Log read error
            sleep 1  # Wait 1 second before retrying
        fi
    done
}

# Run read_and_tag for each device in the background
for dev in "${!DEVICES[@]}"; do  # Loop through device keys
    read_and_tag "$dev" "${DEVICES[$dev]}" &  # Start the function in the background
done

# Wait for all background processes to complete (runs indefinitely)
echo "[+] Populating FIFO for all devices in background"  # Indicate background operation
wait  # Wait for all background processes to finish