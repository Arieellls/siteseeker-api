
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
import torch


from ultralytics import YOLO

import cloudinary
from cloudinary.uploader import upload
from dotenv import load_dotenv

from flask_cors import CORS


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'})

    try:
        result = upload(file, folder='siteseeker/temp')
        return jsonify({'message': 'File successfully uploaded', 'url': result['url']})
    except Exception as e:
        return jsonify({'message': str(e)})
    

@app.route("/hahaha/wait", methods=["GET"])
def test_function():
    print("you got this ariel!!!!!!!")
    return jsonify({'message': 'Welcome to the API'})

model = YOLO('yolov9c.pt')

@app.route('/predict', methods=['POST'])
def predict_img():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'})
        
        f = request.files['file']
        if f.filename == '':
            return jsonify({'message': 'No file selected for uploading'})
        
        # Save uploaded file temporarily
        basepath = os.path.dirname(__file__)
        temp_filepath = os.path.join(basepath, 'temp', secure_filename(f.filename))
        os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
        f.save(temp_filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension == 'jpg':
            img = cv2.imread(temp_filepath)
            img = cv2.resize(img, (640, 640))  # Resize to reduce memory usage

            detections = model(img, save=False)  # Run detection without saving locally

            # Convert result image to Cloudinary-compatible format
            result_img = detections[0].plot()  # Visualize detections
            _, buffer = cv2.imencode('.jpg', result_img)
            result_bytes = buffer.tobytes()

            try:
                # Upload result image to Cloudinary
                result = upload(result_bytes, folder='siteseeker/results', resource_type='image')
                
                # Clean up temporary file
                os.remove(temp_filepath)

                return jsonify({
                    "message": "Image uploaded and prediction done successfully!",
                    "url": result['url']
                })
            except Exception as e:
                return jsonify({"message": str(e)})

        elif file_extension == 'mp4':
            cap = cv2.VideoCapture(temp_filepath)
            
            # Get video dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create a temporary output video file
            output_path = os.path.join(basepath, 'temp', 'output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frames before inference
                frame = cv2.resize(frame, (640, 640))  # Resize to reduce memory usage
                results = model(frame, save=False)  # Run detection on the frame
                result_frame = results[0].plot()
                out.write(result_frame)

                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()
            out.release()

            try:
                # Upload the resulting video to Cloudinary
                result = upload(output_path, folder='siteseeker/results', resource_type='video')

                # Clean up temporary files
                os.remove(temp_filepath)
                os.remove(output_path)

                return jsonify({
                    "message": "Video prediction done successfully!",
                    "url": result['url']
                })
            except Exception as e:
                return jsonify({"message": str(e)})

    return jsonify({"message": "Failed to process the file."})



# #The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder    
    print("printing directory: ",directory) 
    files = os.listdir(directory)
    latest_file = files[0]
    
    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

    else:
        return "Invalid file format"
        
        

def get_frame():
    folder_path = os.getcwd()
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


@app.route("/video_feed")
def video_feed():
    print("function called")

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('yolov9c.pt')
    # app.run(host="0.0.0.0", port=8080, debug=True) 
    app.run() 




