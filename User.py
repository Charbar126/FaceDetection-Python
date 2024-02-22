import os
from cv2 import VideoCapture, imshow, waitKey, destroyWindow, cvtColor, CascadeClassifier, rectangle, COLOR_BGR2GRAY, COLOR_BGR2RGB
import cv2
from matplotlib import image, pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import Pickling
class UserEnrollment:
    
    def __init__(self):
        """
        Initalize the enrollment with a CNN, SIFT and Pretrained model
        """
        self.cnn_model = self.cnn_create_feature_extractor()
        self.sift_model = cv2.SIFT_create()
        vgg16 = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.pt_model =  Model(inputs=vgg16.input, outputs=vgg16.output)


    def take_photo(self):
        """
        Uses the first camera  configured on the device to caputre a single image.
        Raises:
            Exception: If the image was not captured. Typically the camera isn't  set up correctly
        Returns:
            Image(numpy array): A RGB numpy array displaying the image of a face
        """
        # Create a connection to the camera
        cam = VideoCapture(0)
        result, image = cam.read()
        # The image was captured
        if result:  
            imshow("Captured Image", image)
            # When a key is pressed, destroy the window
            waitKey(0) 
            destroyWindow("Captured Image") 
            return image
        # The image wasn't captured 
        else:
            raise Exception("Couldn't capture image, webcam is likely not set up correctly")

    def detect_face(self, image):
        # Convert the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the pre-trained model for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # For each face detected, extract the face region
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            face_output_path = f'temp.jpg'
            cv2.imwrite(face_output_path, face)

            # Display the extracted face
            face_image = cv2.cvtColor(cv2.imread(face_output_path), cv2.COLOR_BGR2RGB)
            #NOTE This only works with one face since it goes through the loop once
            return face_image

    def sift_extract_face_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift_model.detectAndCompute(gray, None)
        return keypoints, descriptors 
    
    def convert_sift_keypoints(self, keypoints):
        """
        This function takes keypoints given by the SIFT algorithum 
        and converts them into dictionary format, so they may be tupled.
        This code was generated via ChatGPT.

        Args:
            keypoints (_type_): _description_

        Returns:
            List: Contains a description for all the keypoints
        """
        converted_keypoints = []
        for kp in keypoints:
            converted_keypoints.append({
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave,
                'class_id': kp.class_id
            })
        return converted_keypoints

    
    # 1. Create a Simple CNN Model
    def cnn_create_feature_extractor(self, input_shape=(64, 64, 1), embedding_dim=128):
        input = Input(shape=input_shape, name='input')
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(input)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = MaxPooling2D((2, 2), name='pool3')(x)
        x = Flatten(name='flatten')(x)
        output = Dense(embedding_dim, activation='relu', name='embedding')(x)
        model = Model(inputs=input, outputs=output, name='feature_extractor')
        return model 
    
        # 2. Load and Preprocess the Image
    def cnn_preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=[0, -1])
        return img
    
    def cnn_extract_features(self, model, image):
        """
        Extract features from the given image using the provided model.

        :param model: The CNN model used for feature extraction.
        :param image: The preprocessed image to extract features from.
        :return: The extracted features (embeddings) for the image.
        """
        # Ensure the image is preprocessed (normalized, resized, etc.)
        # If additional preprocessing is required, it should be done here.

        # Extract features using the model
        embeddings = model.predict(image)
        return embeddings
    
    
    def pt_preprocess_image_vgg(self,image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def pt_extract_features(self, model, image):
        return model.predict(image)


    def enroll(self):
        # 1. Capture user details and image.
        username = input("Enter your full name\n")
        # Take a photo, extract any faces and then segment them
        self_img = self.take_photo()
        detected_face = self.detect_face(self_img)
        
        # SIFT
        sift_keypoints, sift_descriptors = self.sift_extract_face_features(detected_face)
        converted_sift_keypoints = self.convert_sift_keypoints(sift_keypoints)

        
        #CNN
        cnn_img = self.cnn_preprocess_image('temp.jpg')
        cnn_features = self.cnn_extract_features(self.cnn_model, cnn_img)
        
        #Pretrained Model
        #Load Model Just put this inside the class constructor
        pt_image_preprocessed = self.pt_preprocess_image_vgg('temp.jpg')
        pt_embedding = self.pt_extract_features(self.pt_model, pt_image_preprocessed)
        
        #Tuple the data
        tuple = (username, detected_face,
                 SIFT_Data(keypoints= converted_sift_keypoints, descriptors= sift_descriptors, img = detected_face)
                 , CNN_Data(img = cnn_img, features = cnn_features)
                 , Pretrained_Model(img = pt_image_preprocessed, features = pt_embedding))
        pickler = Pickling.PickleHelper()
        
        pickler.save_to(f'database/{username}.pkl', tuple)
        print("Data successfully saved\n")
        os.remove('temp.jpg')
        pass


class Pretrained_Model:
    def __init__(self, img, features) -> None:
        self.img = img
        self.features = features
class CNN_Data:
    def __init__(self, img, features) -> None:
        self.img = img
        self.features = features     
class SIFT_Data:
    """
    This Class is used to store the SIFT keypoints and data since it cannot be pickled
    """
    def __init__(self, keypoints, descriptors,img):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.img = img

    