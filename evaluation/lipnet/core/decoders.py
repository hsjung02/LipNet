from logging import log
from tensorflow.keras import backend as K
import numpy as np
import tensorflow.python.keras.backend as L

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):

    assert (K.backend() == 'tensorflow')

    decoded = K.ctc_decode(y_pred=y_pred,input_length=input_length,greedy=greedy, beam_width=beam_width, top_paths=top_paths)

    #print("path.eval+++++")
    

    #for path in decoded[0]:
        #print(path.numpy())
        #print(path.eval(session=L.get_session()))

    #print("+++++path.eval")

    paths = [path.numpy() for path in decoded[0]]
    #paths = [path.eval(session=L.get_session()) for path in decoded[0]]

    #print("logprobs$$$$$$$$$$$$$")


    logprobs = decoded[1].numpy()
    #logprobs = decoded[1].eval(session=L.get_session())
    #print(logprobs)
    
    return (paths, logprobs)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    '''print("@@@@@@@@@@@@@decode@@@@@@@@@@@@@")
    print(y_pred)
    print("######################")
    print(input_length)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    '''

    paths, logprobs = _decode(y_pred=y_pred, input_length=input_length, greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    #print("paths : "+str(paths))
    #print("logs : "+str(logprobs))

    if language_model is not None:
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        result = paths[0]
    return result

class Decoder(object):
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy         = greedy
        self.beam_width     = beam_width
        self.top_paths      = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', []) #labels_to_text, spell.sentence

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy, beam_width=self.beam_width,
                         top_paths=self.top_paths, language_model=self.language_model)
        preprocessed = []

        for output in decoded: #decoded 한번 돔(아마 여기서 여러번 돌아야 여러 문장이 가능)            
            out = output

            for postprocessor in self.postprocessors: # 2번 실행 labels_to_text -> spell.sentence
                out = postprocessor(out)
            preprocessed.append(out)

        return preprocessed