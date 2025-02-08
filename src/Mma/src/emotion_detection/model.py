import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Conv2D, Concatenate, Multiply
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'model.h5'

class SpatialAttentionLayer(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal'
        )

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
        )(inputs)

        max_pool = tf.keras.layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
        )(inputs)

        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention_map = self.conv(concat)
        return Multiply()([inputs, attention_map])

    def get_config(self):
        config = super(SpatialAttentionLayer, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmotionDetector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
                
            self.model = load_model(
                MODEL_PATH,
                custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer}
            )
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self._warmup_model()
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
        except Exception as e:
            raise
            
    def _warmup_model(self):
        try:
            dummy_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)
        except Exception:
            pass

    def _process_face(self, gray_image, face):
        try:
            x, y, w, h = face
            roi = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)
            
            predictions = self.model.predict(roi, verbose=0)
            emotion_index = np.argmax(predictions[0])
            
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            emotion = emotions[emotion_index]
            
            return emotion, predictions[0][emotion_index]
            
        except Exception as e:
            return str(e)

    def detect_emotion(self, image):
        try:
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                return self._process_face(gray, faces[0])
            return "No faces detected."
            
        except Exception as e:
            return str(e)
