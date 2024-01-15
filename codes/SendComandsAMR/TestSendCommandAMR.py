#It test that a command can be send 2 the AMR via serial

import serial

# Replace 'COM1' with the actual name of your serial port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)

# String to be sent
message = "83 1 100"

# Encode the string to bytes before sending
encoded_message = message.encode('utf-8')

# Send the message
ser.write(encoded_message)
print(encoded_message)

# Close the serial port
ser.close()
