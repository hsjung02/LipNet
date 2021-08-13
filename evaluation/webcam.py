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

np.random.seed(55)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(
    CURRENT_PATH, '..', 'common', 'predictors', 'shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY = True
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(
    CURRENT_PATH, '..', 'common', 'dictionaries', 'grid.txt')


def predict(video_path):  # output_size=28? 11172+2, 26+2
    #print("\nLoading data from webcam...")
    '''
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print("Data loaded.\n")
    print((video.data).shape)
    '''
        
        
    global lipnet, spell, decoder
    
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    video.from_frames(video_path)
    #print("Data loaded.\n")
    #print((video.data).shape)

    '''
    if tf.keras.backend.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape
    '''
    
    

    #print('frames_n, img_w, img_h, img_c :',frames_n, img_w, img_h, img_c)

    X_data = np.array([video.data]).astype(np.float32) / 255

    print(X_data.shape)

    input_length = np.array([len(video.data)])


    y_pred = lipnet.predict(X_data)
    y_pred = np.array([y_pred])

    #print('y_pred:',y_pred.shape,y_pred)
    '''for i in y_pred[0]:
        print(i)
        print('max idx:',np.argmax(i),'max value:',i[np.argmax(i)])
    '''

    result = decoder.decode(y_pred, input_length)
    result = result[0]


    return result


if __name__ == '__main__':

    '''
    if len(sys.argv) == 3:
        video, result = predict(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        video, result = predict(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        video, result = predict(
            sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        video, result = None, ""
    '''
    
    weight_path = "./models/overlapped-weights368.h5"
    video_path = "./samples/id2_vcd_swwp2s.mpg"

    lipnet = LipNet(img_c=3, img_w=100, img_h=50, frames_n=25,
                    absolute_max_string_len=32, output_size=28)
    #lipnet.summary()

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipnet.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)


    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    cap = cv2.VideoCapture(0)
    frames = []
    flag = 0
    results = ['']
    i = 0
    while True:
        frames = []
        for j in range(25):
            ret, im = cap.read()
            if ret:
                frames.append(im)
            print(str(25*i+j))
            cv2.putText(im,results[i],(50,50),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
            cv2.imshow('LipNet', im)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                flag = 1
                break
        result = predict(frames)
        results.append(result)
        if flag == 1:
            break
        i+=1
    cv2.destroyAllWindows()
    print(results)



    '''
    while(cap.isOpened()):
        for i in range(5):
            ret, im = cap.read()
            if ret:
                frames.append(im)
                cnt+=1
            if len(frames) == 15: #input_size를 10으로 유지
                frames = frames[5:]
            else:
                continue
        video,result = predict(weight_path, frames)
        print(result)
    '''
    '''
        cv2.putText(video,result,(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv2.imshow("Output", im)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    '''



    '''
    im = video.face[0]
    for i in range(len(video.face)):
        im = video.face[i]
        cv2.putText(im,result[i][0],(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)
        cv2.imshow("Output", im)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    '''

    #if video is not None:
    #    show_video_subtitle(video.face, result)

    stripe = "-" * len(result)

    print("")
    print(" __                   __  __          __      ")
    print("/\\ \\       __        /\\ \\/\\ \\        /\\ \\__   ")
    print("\\ \\ \\     /\\_\\  _____\\ \\ `\\\\ \\     __\\ \\ ,_\\  ")
    print(" \\ \\ \\  __\\/\\ \\/\\ '__`\\ \\ , ` \\  /'__`\\ \\ \\/  ")
    print("  \\ \\ \\L\\ \\\\ \\ \\ \\ \\L\\ \\ \\ \\`\\ \\/\\  __/\\ \\ \\_ ")
    print("   \\ \\____/ \\ \\_\\ \\ ,__/\\ \\_\\ \\_\\ \\____\\\\ \\__\\")
    print("    \\/___/   \\/_/\\ \\ \\/  \\/_/\\/_/\\/____/ \\/__/")
    print("                  \\ \\_\\                       ")
    print("                   \\/_/                       ")
    print("")
    print("             --{}- ".format(stripe))
    #print("[ DECODED ] |> {} |".format(result.encode('utf-8')))
    print("             --{}- ".format(stripe))
