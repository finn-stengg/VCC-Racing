# Autonomous Vehicle Client

Python client for connecting to autonomous vehicle simulation server.

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/phonghongs/VCCRacing.git
cd VCCRacing
```

## Demo Map

[Download Map](https://drive.google.com/drive/folders/11aVo2YRB26ctLSV_uloT1B4OUmtLQtIp?usp=sharing)



## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run with default settings (localhost:11000)
```bash
python client.py
```

### Run with custom port
```bash
python client.py 11001
```

### Run with custom host and port
```bash
python client.py --host 192.168.1.10 --port 11000
```

### Run with custom timeout
```bash
python client.py --timeout 10
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `port` | Server port (positional) | 11000 |
| `--host` | Server hostname/IP | 127.0.0.1 |
| `--timeout` | Socket timeout (seconds) | 5.0 |
| `--help` | Show help message | - |

## Adding Your Application Logic

Open `client.py` and find the main loop around **line 339**:

```python
while True:
    print("[LOOP] Fetching vehicle data...")

    # Your application logic here  � ADD YOUR CODE HERE

    # Retrieve vehicle data
    state = client.get_state_data()
    raw_image = client.get_raw_image()
    segmented_image = client.get_segmented_image()
```

### Example: Add custom processing

```python
while True:
    # Retrieve vehicle data
    state = client.get_state_data()
    raw_image = client.get_raw_image()
    segmented_image = client.get_segmented_image()

    # YOUR CUSTOM LOGIC HERE
    # Example: Process images
    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    # Example: Make decisions based on state
    if state.get('speed', 0) < 10:
        client.set_control(speed=20, angle=0)

    # Example: Detect objects
    # your_object_detection(segmented_image)

    # Display
    cv2.imshow('Processed', gray_image)

    if cv2.waitKey(1) == ord('q'):
        break
```

## Available Methods

### AVClient Methods

```python
# Set vehicle controls
client.set_control(speed=10, angle=0)  # speed: 0-90, angle: �25

# Get vehicle state (returns dict)
state = client.get_state_data()

# Get camera images (returns numpy arrays)
raw_image = client.get_raw_image()
segmented_image = client.get_segmented_image()
```

## Data Format

### State Data (JSON)
The server returns vehicle state as a dictionary. Check your server documentation for available fields.

### Images
- **Raw Image**: Original camera feed
- **Segmented Image**: Processed/segmented camera feed
- Format: OpenCV numpy arrays (BGR)

## Troubleshooting

**Connection timeout:**
```bash
python client.py --timeout 10
```

**Server not responding:**
- Check if Unity/server is running
- Verify correct port number
- Check firewall settings

**Script hangs after "Connected":**
- Debug messages show where it's stuck
- Increase timeout if server is slow
- Verify server is sending data in correct format

## Exit

Press **'q'** key to exit the client gracefully.
