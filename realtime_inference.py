from tensorflow import keras
import cv2
import numpy as np
import time
import tensorflow as tf
import pyautogui as pa
model_dir='Models/best_model.h5'
model = keras.models.load_model(model_dir)
print("model loaded")
time.sleep(3)
vid = cv2.VideoCapture(0)
x,y=pa.position()
pa.click(x,y)
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    resized_frame = cv2.resize(frame, (128, 128))
    #name=Train_straight_dir+'straight'+str(count)+'.png'
    test_image=np.expand_dims(resized_frame,axis=0)
    test_image = tf.cast(test_image, tf.float32)
    test_image = test_image / 255.0
    prediction=model.predict(test_image)
    p=np.argmax(prediction[0])
    print(p)

    if p==2:
        print("watch")
    if p==1:
        pa.press('right')
    if p==0:
        pa.press('left')


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()