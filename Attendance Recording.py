import cv2  # module import name for opencv_python
import numpy as np  # library for adding support for large, multi-dimensional arrays and matrices
import face_recognition  # This library is used to detect face locations in an Image
import os
from datetime import datetime

path = 'AttendanceImages'  # The Folder where images will be taken from, typically inside the project folder
images = []  # Array Of Images, to store the registered data that will be used to compare
imageNames = []  # Array Of Names, storing the names that belong to the person for each image saved in images array
TheList = os.listdir(path)  # Gets the files in the folder, this will hold all the data we will use to compare in a list

for cl in TheList:  # aka For i=1,i<10,i++
    CurImage = cv2.imread(f'{path}/{cl}')  # Assign variable to each image
    images.append(CurImage)  # Attaching CurrentImage to Images Array
    imageNames.append(os.path.splitext(cl)[0])  # using os.path.splitText attach image name without the .jpg


def findEncodings(imagez):  # Function to find the Encoding of stored images
    encode_list = []  # Empty array to fill up with encodings
    for eachimg in imagez:  # loop function iterate through each image in imagez
        eachimg = cv2.cvtColor(eachimg, cv2.COLOR_BGR2RGB)  # Converting image from BGR to RGB
        encode = face_recognition.face_encodings(eachimg)[0]  # Gets the encoding
        encode_list.append(encode)  # Adds encoding to the array
    return encode_list  # Return the Array


def MarkAttendance(name):  # Function to record the attendance
    with open('AttendanceList.csv', 'r+') as AttendSheet:  # Assigning the attendance list to a variable
        CsvList = AttendSheet.readlines()  # Assigning information inside the attendance list to a variable
        NameList = []  # declaring an array variable
        for line in CsvList:  # To check sheet format and columns
            entry = line.split(',')  # To split the Name and Date with a comma
            NameList.append(entry[0])  # To add it to the Array
        if name not in NameList:  # To check if the name exists already
            RecordedTime = datetime.now()  # To get preset time
            Dateformat = RecordedTime.strftime('%H:%M:%S')  # To choose time format
            AttendSheet.writelines(f'\n{name},{Dateformat}')  # To add the Name and Date of arrival in the Attendance Sheet


encodeListKnown = findEncodings(images)  # Calling the Function FindEncodings

capturing = cv2.VideoCapture(0)  # To initialize the camera

while True:  # while loop for Face Capturing
    cptureimg, img = capturing.read()  # To read from the Camera
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the detected Face to RGB from BGR
    FaceLocation = face_recognition.face_locations(img)  # Detect The Face From the webcam through Captured face
    FaceEncodings = face_recognition.face_encodings(img, FaceLocation)  # Gets the encoding from the Webcam

    for EncodeFace, FaceLoc in zip(FaceEncodings, FaceLocation):  # To find the Detected Person
        matches = face_recognition.compare_faces(encodeListKnown, EncodeFace)  # To compare the image with saved Data
        faceDis = face_recognition.face_distance(encodeListKnown, EncodeFace)  # To get the Distance of image to data
        print(faceDis[0])  # print the distance values
        Matched = np.argmin(faceDis)  # Gets the value in the array at which the image is saved

        if matches[Matched]:  # Searching in matches for the closest detected image
            name = imageNames[Matched].upper()  # Gets the person's name
            print(name)  # prints the Name
        MarkAttendance(name)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Webcam', img)  # Showing the Camera
        cv2.waitKey(1)
