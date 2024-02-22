from collections import defaultdict
import datetime
import csv
import os
from matplotlib import pyplot as plt
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

from cv2 import VideoCapture
import cv2
import numpy as np
from keras.preprocessing import image

from Evaluation import PerformanceEvaluation
from Pickling import PickleHelper
from User import UserEnrollment
from scipy.spatial.distance import cosine



class AttendanceRecording:
    def constant_img_capture(self):
        """
        Constantly Captures images until input is recieved. 
        Then returns the last image

        Returns:
            frame: last captured frame
        """
        cam = VideoCapture(0)   # Get webcam

        while True:
            ret, frame = cam.read() #Read the input
            
            cv2.imshow("Captured Frame", frame)
            key = cv2.waitKey(1)
            #Any input occured
            if key != -1:
                last_captured_frame = frame
                break
        return frame
    
    def sift_match_faces(self, test_descriptors, db_descriptors):
        bf = cv2.BFMatcher()
        matches = bf.match(test_descriptors, db_descriptors)
        distances = [m.distance for m in matches]
        score = sum(1 / (distance + 1e-6) for distance in distances)
        return score

    def pt_cnn_compare_features(self, feature1, feature2):
        """
        Compare two sets of features using cosine similarity.

        :param feature1: The features of the first image.
        :param feature2: The features of the second image.
        :return: The similarity score between the two sets of features.
        """
        # Ensure the features are 1D
        feature1 = feature1.flatten()
        feature2 = feature2.flatten()

        similarity = 1 - cosine(feature1, feature2)
        return similarity
    
    def get_pickle_files(self, directory_path):
        """
        Get all files with the .pkl extension in a given directory.

        :param directory_path: The path to the directory.
        :return: A list of file objects with the .pkl extension.
        CODE TAKEN FROM CHATGPT
        """
        pickle_files = []
        
        p = PickleHelper()
        
        # Iterate through the files in the directory
        for filename in os.listdir(directory_path):
            # Check if the file has a .pkl extension
            if filename.endswith(".pkl"):
                file_path = os.path.join(directory_path, filename)
                pickle_files.append(p.load_back(str(file_path)))

        return pickle_files
    
    
    def update_attendance_file(self, username, attendance_file):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(attendance_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([username, current_time])

    
    def record_highest_score(self, highest_score_username):
        # Assuming highest_score_username is the username with the highest score

        # Check if "attendance.csv" file exists
        attendance_file = "attendance.csv"
        file_exists = os.path.isfile(attendance_file)

        if not file_exists:
            # If the file doesn't exist, create it and write header
            with open(attendance_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Username", "Date and Time"])

        # Update the attendance file with the highest score username
        self.update_attendance_file(highest_score_username, attendance_file)
        print(f"Attendance is recorded for {highest_score_username}.")
        
    def process_img_for_pt(self, detected_face):
        """Due to problems with processing an image for a opretrained model this function garentess it will be correctly created

        Args:
            detected_face (_type_): _description_
        """
        detected_face = cv2.resize(detected_face, (224, 224))  # Resize the image to (224, 224)
        detected_face = image.img_to_array(detected_face)       # Convert to NumPy array
        detected_face = np.expand_dims(detected_face, axis=0)   # Add batch dimension
        return detected_face
    
    def record(self):
        #Create an instance of enrollment in order to reuse the same models
        enrollment = UserEnrollment()
        #Create an instance of the PickleHelper class to load back the saved pickled objects
        p = PickleHelper()
        # 1. Capture images continuously unless interrupted.
        
        detected_face = enrollment.detect_face(self.constant_img_capture())
        
        sift_keypoints, sift_descriptors = enrollment.sift_extract_face_features(detected_face)

        cnn_features = enrollment.cnn_extract_features(enrollment.cnn_model, enrollment.cnn_preprocess_image('temp.jpg'))
        
        pt_img = enrollment.pt_extract_features(enrollment.pt_model, self.process_img_for_pt(detected_face))
        
        #This will be used to determine the highest score and update the coresponding attendance chart
        user_signing_in = [0, '']
        
        #Iterate through each user in the database
        for user in self.get_pickle_files('database'):
            username, user_detected_face, SIFT_Data, CNN_Data, Pretrained_Model = user 
            
            total_score = self.cumulaitve_model_score(sift_descriptors, cnn_features, pt_img, SIFT_Data, CNN_Data, Pretrained_Model)
            
            #If the total of all the model scores are greater than the current highest update a possible mark for attandence
            if total_score > user_signing_in[0]:
                user_signing_in = [total_score, username]
                
        os.remove('temp.jpg')
        self.record_highest_score(user_signing_in[1])


    def cumulaitve_model_score(self, sift_descriptors, cnn_features, pt_img, SIFT_Data, CNN_Data, Pretrained_Model):
        """
        Using the three differnet models calculate the overall face detection score

        Returns:
            floa: the overall detection score
        """
        sift_score = self.sift_match_faces(SIFT_Data.descriptors, sift_descriptors)

        cnn_score = self.pt_cnn_compare_features(cnn_features, CNN_Data.features)
            
        pt_score = self.pt_cnn_compare_features(Pretrained_Model.features, pt_img )
            
        total_score = sift_score + cnn_score + pt_score
        return total_score

class AttendanceSystem:
    def __init__(self):
        self.user_enrollment = UserEnrollment()
        self.attendance_recording = AttendanceRecording()
        self.performance_evaluation = PerformanceEvaluation()
        self.attendance_plotting = AttendancePlotting()
        self.attendance_file = "attendance.csv"

    def main_menu(self):
        print("1. Enroll New User")
        print("2. Record Attendance")
        print("3. Plot Attendance")
        choice = input("Enter your choice: ")
        if choice == "1":
            self.user_enrollment.enroll()
        elif choice == "2":
            self.attendance_recording.record()
        elif choice == "3":
            self.attendance_plotting.plot()


class AttendancePlotting:
    def plot(self):
        """
        Plot the attendance based on the data in the attendance file.

        :param attendance_file: The path to the attendance.csv file.
        CODE TAKEN FROM CHATGPT
        """
        # Dictionary to store the presence count for each student
        presence_count = defaultdict(int)

        # Read the attendance file and update the presence count
        with open('attendance.csv', 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the header row
            next(reader, None)
            for row in reader:
                if len(row) == 2:  # Ensure the row has the expected format
                    username, _ = row
                    presence_count[username] += 1

        # Plot the attendance
        usernames, counts = zip(*presence_count.items())

        plt.bar(usernames, counts)
        plt.xlabel('Students')
        plt.ylabel('Attendance')
        plt.title('Attendance Analysis')
        plt.show()