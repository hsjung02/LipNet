from re import L
from lipnet.lipreading.videos import Video
from lipnet.lipreading.visualization import show_video_subtitle
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import threading
from time import time

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(
    CURRENT_PATH, '..', 'common', 'predictors', 'shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY = True
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(
    CURRENT_PATH, '..', 'common', 'dictionaries', 'grid.txt')

async def show_get_frames():

    for i in range(100):
        await asyncio.sleep(0)
        print('b')
    
    return (0,0)

async def predict(video_path):
    start_time = time()
    print('start predicting at:',start_time)
    global lipnet, spell, decoder
    
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    video.from_frames(video_path)
    print("Data loaded.\n")
    X_data = np.array([video.data]).astype(np.float32) / 255


    y_pred = await lipnet.predict(X_data)
    y_pred = np.array([y_pred])
    
    result = []
    for i in range(len(y_pred[0])):
        temp = y_pred[0][:i+1]
        result.append(decoder.decode([temp],np.array([i+1])))
    
    result = []
    print('finish predicting at:',time())
    print('spent',time()-start_time,'for predicting')

    return result #list

async def main(temp_frames):
    frames = temp_frames
    frames_old = []
    results = []
    result = []
    flag = 1

    while flag:
        start_time = time()
        print('\n===============start a loop========\n')
        a = await asyncio.gather(predict(frames),show_get_frames())
        result = a[0]
        frames = a[1][0]
        flag = a[1][1]
        print('spend total:',time()-start_time)
        results.append(result)
        frames_old = frames
        print('\n===============end a loop========\n')


    return results



weight_path = "./models/overlapped-weights368.h5"
video_path = "./samples/id2_vcd_swwp2s.mpg"
lipnet = LipNet(img_c=3, img_w=100, img_h=50, frames_n=25,absolute_max_string_len=32, output_size=28)
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
lipnet.model.load_weights(weight_path)
spell = Spell(path=PREDICT_DICTIONARY)
decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH, postprocessors=[labels_to_text, spell.sentence])
frames = []
cap = cv2.VideoCapture(0)
while len(frames)<25:
    ret, im = cap.read()
    if ret:
        frames.append(im)
cap.release()
asyncio.run(main(frames))                                    