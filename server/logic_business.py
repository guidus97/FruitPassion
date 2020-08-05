import numpy as np
import tensorflow as tf
import cv2

class Logic_Business:

    self.model = tf.keras.models.load_model('model.h5')

    def predict_fruit(self,img):
        
        fruit = self.model.predict(img)
        return fruit

    def resize_image(self,img):
        
        img = cv2.resize(img, (100,100), cv2.IMREAD_COLOR)

        return img

    def save_pred_to_db(self, pred):
        