# Sign Language Detection System

A real-time sign language detection and translation system using computer vision and deep learning. This application captures hand gestures through a webcam, classifies them using a trained neural network, and converts the detected signs into text and speech.

## Features

- **Real-time Hand Detection**: Uses MediaPipe and CVZone for accurate hand tracking
- **Sign Language Recognition**: Classifies hand gestures into predefined sign language words
- **Text-to-Speech**: Converts detected signs into spoken words using Google Text-to-Speech (gTTS)
- **Interactive Web Interface**: Built with Streamlit for easy-to-use interaction
- **Data Collection Tool**: Includes a script for collecting training data for new signs

## Supported Signs

The system currently recognizes the following sign language gestures:
- HI
- Please
- I LOVE YOU
- MY
- Thank You
- Yes
- Time

## Prerequisites

- Python 3.8 or higher
- Webcam
- Operating System: Windows, macOS, or Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sanjithwoxsen/SIGH-LANGUAGE-DETECTION.git
cd SIGH-LANGUAGE-DETECTION
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Sign Language Translator

To start the sign language detection application:

```bash
streamlit run Main.py
```

This will open a web interface in your default browser where you can:
1. Click the "Start" button to begin detection
2. Show hand signs to the webcam
3. The system will recognize the sign and display it on screen
4. After consistent detection, it will also speak the recognized word
5. Click "Stop" to end the session

### Collecting Training Data

To collect your own training data for new signs:

```bash
python Data_Collection.py
```

**Instructions:**
1. The script will open your webcam
2. Position your hand to make the desired sign
3. Press 'S' to start saving images
4. The script will save up to 3000 images
5. Images are saved in the `Data/yes` folder (modify the `folder` variable in the script for different signs)

## Project Structure

```
SIGH-LANGUAGE-DETECTION/
│
├── Main.py                 # Main application with Streamlit interface
├── Data_Collection.py      # Script for collecting training data
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
└── Model/
    ├── keras_model.h5     # Trained neural network model
    └── labels.txt         # Labels for sign classifications
```

## How It Works

1. **Hand Detection**: The system uses CVZone's HandDetector to identify and track hand landmarks in real-time
2. **Image Processing**: Detected hand regions are cropped, resized to 1000x1000 pixels, and normalized
3. **Classification**: The preprocessed image is fed to a trained Keras model for sign classification
4. **Confidence Check**: The system waits for consistent detection (6 frames) before confirming a sign
5. **Output**: Recognized signs are displayed as text and converted to speech

## Technical Details

- **Computer Vision**: OpenCV for image processing and webcam capture
- **Hand Tracking**: CVZone HandTrackingModule (built on MediaPipe)
- **Deep Learning**: TensorFlow/Keras for sign classification
- **Web Framework**: Streamlit for the user interface
- **Text-to-Speech**: Google Text-to-Speech (gTTS) for audio output

## Contributing

Contributions are welcome! Here are some ways you can contribute:
- Add support for more sign language gestures
- Improve the detection accuracy
- Enhance the user interface
- Add support for different sign language standards (ASL, BSL, etc.)

## License

This project is open-source and available for educational and research purposes.

## Acknowledgments

- CVZone library for simplified hand detection
- MediaPipe for robust hand tracking
- TensorFlow/Keras for the deep learning framework
- Streamlit for the interactive web interface

## Troubleshooting

### Common Issues

1. **Webcam not detected**: Ensure your webcam is properly connected and not being used by another application
2. **Import errors**: Make sure all dependencies are installed using `pip install -r requirements.txt`
3. **Model not found**: Ensure the `Model/keras_model.h5` and `Model/labels.txt` files are present
4. **Slow detection**: Try adjusting the `detectionCon` parameter in the code for faster but less accurate detection

## Contact

For questions or support, please open an issue in the GitHub repository.
