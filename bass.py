import pdb
import sys
from shutil import copyfile

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

################################################################
class Composition():
    def __init__(self, BIT_RATE, BPM, SIGNATURE, BARS):
        self.bit_rate  = BIT_RATE
        self.bpm       = BPM
        self.signature = SIGNATURE
        self.bars      = BARS
        
        samples = int(BIT_RATE*60./BPM*BARS*SIGNATURE[0])
        self.out = np.zeros(samples)

        self.bits_bar = int(BIT_RATE*60./BPM*SIGNATURE[1])
        self.bits_beat = int(BIT_RATE*60./BPM)
        self.time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        self.prob = np.array([0.02, 0.08, 0.5, 0.3, 0.1, 0.0, 0.0])

    def add_layer(self, layer):
        def transmogrify(nn, note):
            def normalize(xx):
                return xx/np.amax(np.abs(xx))
            
            def poisson(note, lam):
                tt = np.arange(0, lam, lam/note.shape[0])
                note = note*tt*np.exp(-tt)
                return normalize(note)
            
            def boxcar(note, smooth):
                tt = np.arange(0, np.pi, np.pi/note.shape[0])
                note = note*np.sin(tt)**(1/smooth)
                return normalize(note)
            
            shapes = [poisson, boxcar]
            shape = layer.shape[nn]
            param = layer.param[nn]
            return shapes[int(shape)](note, param)
    
        def mk_note(nn):
            freq = 440*2**(layer.pitch[nn]/12.)
            tt = np.arange(0, int(self.bits_beat * layer.value[nn]))
            return np.sin(2*np.pi*freq*tt/self.bit_rate)

        bit_cts = 0
        for nn in range(0,len(layer.value)):
            note = mk_note(nn)
            note = transmogrify(nn, note)
            
            lo = bit_cts
            up = bit_cts + int(self.bits_beat * layer.value[nn])
            self.out[lo:up] += note
            bit_cts = up

    def mk_layer(self):
        value = []
        for bar in range(0,self.bars):
            in_bar = 0
            while in_bar < self.signature[0]:
                tmp = np.random.choice(self.time, p=self.prob)
                if in_bar+tmp <= self.signature[0]:
                    in_bar += tmp
                    value.append(tmp)

        value = np.array(value)
        shape = np.ones(value.shape[0])*0
        param = np.ones(value.shape[0])*10
        pitch = -np.random.uniform(0,30,value.shape[0]).astype('int')
        return Layer(value, shape, param, pitch)

    def mk_bass(self):
        value = np.ones(self.bars*self.signature[0])
        shape = np.ones(value.shape[0])*0
        param = np.ones(value.shape[0])*10
        pitch = np.ones(value.shape[0])*-30
        return Layer(value, shape, param, pitch)
            
class Layer():
    def __init__(self, value, shape, param, pitch):
        self.value = value
        self.shape = shape
        self.param = param
        self.pitch = pitch

################################################################
def mk_cp(src, dst):
    copyfile(src, dst)
    
def float2int16(dat):
    dat = dat/np.amax(np.abs(dat))
    dat = 2**15*dat
    dat = np.where(dat-2**15 > -1e-6, 32767, dat)
    dat = np.int16(dat)
    return dat

def save_wav(path, bit_rate, dat):
    dat = float2int16(dat)
    wavfile.write(path, bit_rate, dat)

################################################################
SOURCE    = '/Users/silver/Dropbox/bin/mus_gen.py'
NAME      = sys.argv[1]
BIT_RATE  = 2**16
BPM       = 80
SIGNATURE = [4, 4]
BARS      = 32
PATH      = '/Users/silver/Dropbox/sci/pro_mus/gen/'
OUTP      =  PATH + NAME + '.wav'

mk_cp(SOURCE, PATH + NAME + '.py')
comp = Composition(BIT_RATE, BPM, SIGNATURE, BARS)
layer = comp.mk_layer()
comp.add_layer(layer)
comp.add_layer(comp.mk_bass())

save_wav(OUTP, BIT_RATE, comp.out)
