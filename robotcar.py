import cv2
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

'''
Use RaspberryPi to control the robot car.
https://bananarobotics.com/shop/How-to-use-the-HG7881-(L9110)-Dual-Channel-Motor-Driver-Module
'''
import time
import sys
import RPi.GPIO as GPIO
# Connect the control PIN to RPi's GPIO
relay1A = 2
relay1B = 3
relay2A = 4
relay2B = 17 

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False) 
GPIO.setup(relay1A, GPIO.OUT)
GPIO.setup(relay1B, GPIO.OUT)
GPIO.setup(relay2A, GPIO.OUT)
GPIO.setup(relay2B, GPIO.OUT)

# Define the car actions.
def car_move(action):
    if action == "forward":
       print("Forward")
       GPIO.output(relay1A,False)
       GPIO.output(relay1B,True)
       GPIO.output(relay2A,False)
       GPIO.output(relay2B,True)
    elif action == "backward":
       print("Backward")
       GPIO.output(relay1A,True)
       GPIO.output(relay1B,False)
       GPIO.output(relay2A,True)
       GPIO.output(relay2B,False)
    elif action == "left":
       print("Left")
       GPIO.output(relay1A,False)
       GPIO.output(relay1B,True)
       GPIO.output(relay2A,False)
       GPIO.output(relay2B,False)
    elif action == "right":
       print("Right")
       GPIO.output(relay1A,False)
       GPIO.output(relay1B,False)
       GPIO.output(relay2A,False)
       GPIO.output(relay2B,True)
    else:
       print("stop")
       GPIO.output(relay1A,False)
       GPIO.output(relay1B,False)
       GPIO.output(relay2A,False)
       GPIO.output(relay2B,False)


def autorun():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Load the model
    model = load_model("keras_car.h5", compile=False)
    
    # Load the labels
    class_names = open("labels.txt", "r").readlines()
    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    # https://forum.opencv.org/t/why-must-i-not-delay-videocapture-read-calls/11369
    img_counter = 0
    
    car_move("stop")
    while True:
        key = cv2.waitKey(1)
        if key == ord('q') :
            # q pressed
            print("q hit, closing...")
            break
        #elif k%256 == 32:
        # SPACE pressed
        #img_name = "opencv_frame_{}.png".format(img_counter)
        #cv2.imwrite(img_name, frame)
        #print("{} written!".format(img_name))
        
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
    
    
        #image = Image.open("opencv_frame_0.png").convert("RGB")
        image = Image.fromarray(frame)
    # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size)
    
    # turn the image into a numpy array
        image_array = np.asarray(image)
    
    # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
        data[0] = normalized_image_array
    
    # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
    
    # Print prediction and confidence score
        print("Class:"+ class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        if class_name == "0 right\n" and confidence_score > 0.8:
            #print ("L")
            car_move("right")
            time.sleep(0.1)
            #car_move("forward")
        elif class_name == "1 left\n" and confidence_score > 0.8:
            #print ("R")
            car_move("left")
            time.sleep(0.1)
            #car_move("forward")
        else:
            car_move("forward")
            print("none")
    
        #Purge the buffer
        # https://stackoverflow.com/questions/41412057/get-most-recent-frame-from-webcam
        for i in range(10): #Annoyingly arbitrary constant
          cam.grab()
    GPIO.cleanup()
    car_move("stop")
    cam.release()
    cv2.destroyAllWindows()

#if __name__ == "__main__":
