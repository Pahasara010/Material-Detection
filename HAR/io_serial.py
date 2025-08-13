import serial

def read_csi_from_serial(port='/dev/ttyUSB0', baudrate=921600, expected_lines=256) -> list[str]:
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
    lines = []

    while len(lines) < expected_lines:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("CSI_DATA"):
                lines.append(line)
        except Exception as e:
            print("Serial Read Error:", e)
            break

    ser.close()
    return lines
