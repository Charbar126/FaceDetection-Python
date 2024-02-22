import numpy as np
from User import UserEnrollment
import os
import cv2
from Pickling import PickleHelper
from keras.preprocessing import image
from scipy.spatial.distance import cosine


class PerformanceEvaluation:
    
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

    def process_img_for_pt(self, detected_face):
        """Due to problems with processing an image for a opretrained model this function garentess it will be correctly created

        Args:
            detected_face (_type_): _description_
        """
        detected_face = cv2.resize(detected_face, (224, 224))  # Resize the image to (224, 224)
        detected_face = image.img_to_array(detected_face)       # Convert to NumPy array
        detected_face = np.expand_dims(detected_face, axis=0)   # Add batch dimension
        return detected_face
    
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
            
    def evaluate(self, chosen_model):
        # Get the other users from the database
        other_users = self.get_pickle_files('database')
        
        #Gather the target user for comparrisons
        target_user_username = input("Enter the username of your target user.\n").lower()
        target_user = self.find_target_usr(other_users, target_user_username)
            
        
        #Establish a connection with the webcamera and initalize variables
        cam = cv2.VideoCapture(0)
        num_of_matches = 0
        num_of_total_attempts = 0
        enrollment = UserEnrollment()
        
        while num_of_matches < 10:
            
            result, image = cam.read()
            
            detected_face = enrollment.detect_face(image)
            
            #This will be used to determine the highest score
            user_signing_in = [0, '']
            
            if chosen_model == "SIFT":
                #Gather features from the captured image
                sift_keypoints, sift_descriptors = enrollment.sift_extract_face_features(detected_face)
                
                #Iterate through each user in the database
                for user in self.get_pickle_files('database'):
                    username, user_detected_face, SIFT_Data, CNN_Data, Pretrained_Model = user 
                    
                    score = self.sift_match_faces(sift_descriptors, SIFT_Data.descriptors)
                    
                    #If the total of all the model scores are greater than the current highest update a possible mark for attandence
                    if score > user_signing_in[0]:
                        user_signing_in = [score, username]
                #Update the number of attempts
                #No Refactoring in order to keep variables gloabl
                if user_signing_in[1].lower() == target_user_username:
                    num_of_matches += 1
                num_of_total_attempts += 1
                print(f"{num_of_matches}/{num_of_total_attempts}")
                
            if chosen_model == "CNN":
                #Gather feautres from the caputred image
                cnn_features = enrollment.cnn_extract_features(enrollment.cnn_model, enrollment.cnn_preprocess_image('temp.jpg'))
                
                for user in self.get_pickle_files('database'):
                    username, user_detected_face, SIFT_Data, CNN_Data, Pretrained_Model = user 
                    
                    score = self.pt_cnn_compare_features(cnn_features, CNN_Data.features)
                    #If the total of all the model scores are greater than the current highest update a possible mark for attandence
                    if score > user_signing_in[0]:
                        user_signing_in = [score, username]
                #Update the number of attempts
                if user_signing_in[1].lower() == target_user_username:
                    num_of_matches += 1
                num_of_total_attempts += 1
                print(f"{num_of_matches}/{num_of_total_attempts}")
                
            if chosen_model == "Pre-Trained":
                #Make sure image can be used for pre trained model
                pt_img = enrollment.pt_extract_features(enrollment.pt_model, self.process_img_for_pt(detected_face))
                
                for user in self.get_pickle_files('database'):
                    username, user_detected_face, SIFT_Data, CNN_Data, Pretrained_Model = user
                
                score = self.pt_cnn_compare_features(pt_img, Pretrained_Model.features)
                #If the total of all the model scores are greater than the current highest update a possible mark for attandence
                if score > user_signing_in[0]:
                    user_signing_in = [score, username]
            #Update the number of attempts
            if user_signing_in[1].lower() == target_user_username:
                num_of_matches += 1
            num_of_total_attempts += 1
            print(f"{num_of_matches}/{num_of_total_attempts}")
        os.remove('temp.jpg')
                
                
                

    # def update_correct_attempts(self, target_user_username, num_of_matches, num_of_total_attempts, user_signing_in):
        
           

    def find_target_usr(self, other_users, target_usr):
        """
        Finds the target user from the database 
        
        """
        for user in other_users:
            username, user_detected_face, SIFT_Data, CNN_Data, Pretrained_Model = user
            username = username.lower()
            if target_usr == username:
                print(f"You will be matched with {user[0]}\n")
                return user
        print("The target user couldn't be found within the database")
        exit()
    
        
p = PerformanceEvaluation()

chosen_model_num = int(input("Which model would you like to test.\n1. SIFT\n2. CNN\n3. Pretrained\n").strip())
chosen_model = ""
#Establish a chosen_model
if(chosen_model_num == 1):
    chosen_model = "SIFT"
elif(chosen_model_num == 2):
    chosen_model = "CNN"
elif(chosen_model_num == 3):
    chosen_model = "Pre-Trained"
p.evaluate(chosen_model)