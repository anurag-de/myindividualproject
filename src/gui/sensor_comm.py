# ==============================================================================
#
# Sensor Communication Class
#
# This module encapsulates all low-level communication with the
# Red Pitaya device. It handles establishing SSH connections for sending
# commands and managing the UDP socket for high-speed data acquisition.
# This abstracts the hardware details away from the main application logic.
#
# ==============================================================================

# --- Core Python and System Libraries ---
import struct

# --- Network Communication Libraries ---
import socket
import paramiko

# --- Machine Learning and Signal Processing Libraries ---
import numpy as np

class RedPitayaSensor:
    """Handles all low-level communication with the Red Pitaya device."""
    def __init__(self):
        # --- Network Configuration ---
        self.hostIP = "169.254.148.148"
        self.data_port = 61231
        self.ssh_port = 22
        self.server_address_port = (self.hostIP, self.data_port)

        # --- Data Protocol Configuration ---
        self.size_of_raw_adc = 25000
        self.buffer_size = (25000 + 17) * 4 # Max expected UDP packet size

        # --- State and Clients ---
        self.sensor_status_message = "Waiting to Connect..."
        self.udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_client_socket.settimeout(2.0) # Timeout to prevent indefinite blocking
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Auto-accept SSH key
        
    def connect_ssh(self):
        """Establishes an SSH connection to the device if not already active."""
        if not self.is_ssh_connected():
            try: self.ssh_client.connect(self.hostIP, self.ssh_port, "root", "root", timeout=5)
            except Exception as e: raise ConnectionError(f"SSH connection failed: {e}")
            
    def is_ssh_connected(self):
        """Checks if the SSH transport is active."""
        return self.ssh_client.get_transport() and self.ssh_client.get_transport().is_active()
        
    def disconnect_ssh(self):
        """Closes the SSH connection if it's open."""
        if self.is_ssh_connected(): self.ssh_client.close()
            
    def give_ssh_command(self, command):
        """Executes a command on the Red Pitaya via SSH, auto-connecting if needed."""
        try:
            if not self.is_ssh_connected(): self.connect_ssh()
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            if (error := stderr.read().decode().strip()) and stdout.channel.recv_exit_status() != 0: 
                print(f"SSH Error: {error}")
            return stdout.read().decode()
        except Exception as e: raise ConnectionError(f"SSH command failed ('{command}'): {e}")
            
    def set_sensor_message(self, message):
        """Sets the status message to be displayed in the GUI."""
        self.sensor_status_message = message
        
    def get_sensor_status_message(self):
        """Retrieves the current status message."""
        return self.sensor_status_message
        
    def send_msg_to_server(self, msg):
        """Sends a UDP message to the sensor."""
        self.udp_client_socket.sendto(str.encode(msg), self.server_address_port)
        
    def get_data_info_from_server(self):
        """Performs the initial handshake to get data metadata from the sensor."""
        self.send_msg_to_server("-i 1") # Command to request info
        try:
            packet = self.udp_client_socket.recv(self.buffer_size)
            # Unpack binary data from the response packet
            self.header_length = int(struct.unpack('@f', packet[:4])[0])
            self.total_data_blocks = int(struct.unpack('@f', packet[56:60])[0])
            if self.total_data_blocks > 0: 
                self.set_sensor_message(f"Connected to {self.hostIP}:{self.data_port}")
                return True
            else: raise ConnectionError("Handshake failed: 0 data blocks.")
        except socket.timeout: raise ConnectionError("Handshake timed out.")
            
    def get_data_from_server(self):
        """Retrieves a full signal chunk from the sensor over UDP."""
        if self.total_data_blocks == 0: raise ConnectionError("Not connected.")
        ultrasonic_data = []
        self.set_sensor_message("Receiving...")
        # Loop to receive all data blocks that constitute one full signal
        for i in range(self.total_data_blocks):
            self.send_msg_to_server("-a 1") # Command to request a data block
            packet = self.udp_client_socket.recv(self.buffer_size)
            
            # Sanity check for packet order
            if i != int(struct.unpack('@f', packet[60:64])[0]): raise ValueError("Data sync error.")
            
            # Unpack the ADC values (signed short integers) from the packet payload
            for d in struct.iter_unpack('@h', packet[self.header_length:]): 
                ultrasonic_data.append(d[0])

        if len(ultrasonic_data) != self.size_of_raw_adc: 
            raise ValueError("Data length mismatch. Detection will not work.")
        
        return np.array(ultrasonic_data, dtype=np.float32)