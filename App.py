import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import pafy
import streamlit_webrtc as webrtc
import tensorflow as tf
from PIL import Image


accidents = pd.DataFrame({
    "location": ["A", "B", "C", "D", "E"],
    "severity": [3, 2, 4, 1, 5],
    "timestamp": ["2024-03-09 12:00:00", "2024-03-09 12:10:00", "2024-03-09 12:20:00", "2024-03-09 12:30:00", "2024-03-09 12:40:00"]
})

youtube_url = {
    "Giresun": r"https://youtu.be/szTRjqKTT8A", # Giresun, Turkey
    "Sukiyabashi Crossing": r"https://youtu.be/EuOZeHQmg-4", # Sukiyabashi Crossing, Tokyo
    "Mandeok Tunnel": r"https://youtu.be/9-uES26sQl4", # Mandeok Tunnel, Busan
    "Kabukicho Crossing": r"https://youtu.be/6bPfX9m7wVU", # Kabukicho Crossing, Tokyo
    "Hopkins Street": r"https://youtu.be/E8LsKcVpL5A", # Hopkins Street, Texas
    "Paso Del Norte North View": r"https://youtu.be/0Pg3S6s76IE", # Paso Del Norte North View, Mexico
    "Miami": r"https://youtu.be/UBhee7b1U9s"  # Miami, Florida
}

def frame_generator(video_url):
    # Use pafy to get the video
    video_pafy = pafy.new(video_url)
    best_resolution_video = video_pafy.getbest(preftype="mp4")

    # Create a video capture object from the YouTube URL using cv2
    video_capture = cv2.VideoCapture(best_resolution_video.url)

    # Loop through the frames of the video capture object
    while True:
        success, frame = video_capture.read()

        # If the frame is valid, yield it as a numpy array
        if success:
            yield frame
        else:
            break
    
interpreter = tf.lite.Interpreter(model_path = 'tf_lite_model.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def alert(location, prediction):
    # Use the global keyword to access the global variable
    global alert_message
    global accidents
    sevirity = 0
    # Assign the alert message to the global variable
    alert_message = f"Accident detected at location {location}! Please take immediate action."
    # Get the current datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Append the location, datetime and prediction value to the accidents list
    if prediction <= 0.2:
        sevirity = 1
    elif prediction < 0.4:
        sevirity = 2
    elif prediction < 0.6:
        sevirity <= 3
    elif prediction < 0.8:
        sevirity <= 4
    else:
        sevirity = 5
    accidents.append([location, sevirity, prediction])

class AccidentDetector(webrtc.VideoTransformerBase):
    def __init__(self, frames):
        super().__init__()
        self.frames = frames

    def transform(self, custom=False):

        frame = cv2.cvtColor(self.frames, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((250, 250))
        frame = np.array(frame)
        input_tensor = tf.convert_to_tensor(frame)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_tensor = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_tensor[0][1]
        if prediction > 0.66:
            alert(self.location, prediction)
        else:
            pass



st.set_page_config(page_title="Intelligent Traffic Safety System", layout="wide")

accidents = pd.DataFrame({
    "location": ["A", "B", "C", "D", "E"],
    "severity": [3, 2, 4, 1, 5],
    "timestamp": ["2024-03-09 12:00:00", "2024-03-09 12:10:00", "2024-03-09 12:20:00", "2024-03-09 12:30:00", "2024-03-09 12:40:00"]
})

traffic = pd.DataFrame({
    "location": ["A", "B", "C", "D", "E"],
    "speed": [60, 50, 40, 70, 30],
    "density": [0.8, 0.6, 0.9, 0.4, 1.0]
})

st.sidebar.title("Intelligent Traffic Safety System")
st.sidebar.subheader("Select your preferences")

time_range = st.sidebar.slider("Select the time range", min_value=0, max_value=59, value=(0, 59))

locations = st.sidebar.multiselect("Select the locations", options=["A","B","C","D","E","Giresun", "Kabukicho Crossing", "Sukiyabashi Crossing", "Mandeok Tunnel", "Hopkins Street", "Miami"], default=["A", "B", "C"])

alerts = st.sidebar.checkbox("Enable alerts", value=True)

accidents = accidents[(accidents["timestamp"] >= f"2024-03-09 {time_range[0]:02d}:00:00") & (accidents["timestamp"] <= f"2024-03-09 {time_range[1]:02d}:59:59")]
accidents = accidents[accidents["location"].isin(locations)]

traffic = traffic[traffic["location"].isin(locations)]

st.title("Intelligent Traffic Safety System")
st.subheader("Accident Map")
coords = {
    "A": (19.2290, 72.8573),
    "B": (19.2183, 72.9781),
    "C": (19.0760, 72.8777),
    "D": (19.1642, 72.8561),
    "E": (19.1197, 72.9051),

}

accidents["lat"] = accidents["location"].apply(lambda x: coords[x][0])
accidents["lon"] = accidents["location"].apply(lambda x: coords[x][1])

fig = px.scatter_mapbox(accidents, lat="lat", lon="lon", color="severity", size="severity", hover_data=["severity", "timestamp"], zoom=10)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig)
st.write("Disclaimer: The locations have been predefined for demonstration purposes.")

st.subheader("Accident Details")
st.dataframe(accidents)

st.subheader("Traffic Speed and Density")
fig = px.bar(traffic, x="location", y=["speed", "density"], barmode="group")
st.plotly_chart(fig)


st.subheader("Live Feed")
for location in locations:
    if location not in youtube_url.keys():
        st.write(f"Live video feed from location {location} is not available.")
    else:
        st.write(f"Live video feed from location {location}")
        video_url = youtube_url[location]
        st.video(video_url)
        frames = frame_generator(video_url)
        factory = AccidentDetector(frames).transform()

if alerts:
    st.subheader("Alerts")
    critical_accidents = accidents[accidents["severity"] >= 4]
    for i, row in critical_accidents.iterrows():
        st.error(f"Critical accident detected at location {row['location']} with severity {row['severity']} at {row['timestamp']}. Please take immediate action.")
