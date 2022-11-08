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

    def add_layer(self, layer):
        # This shapes each note in the time domain
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
            shape = int(layer.shape[nn])
            param = layer.param[nn]
            return shapes[shape](note, param)

        # This constructs the waveform for each note
        def mk_note(nn):
            def ift(coefs, *amp):
                if len(amp) == 0:
                    amp = 1
                else:
                    amp = amp[0][np.newaxis].T
                    
                tt = np.arange(0, int(self.bits_beat * layer.value[nn]))
                coefs = coefs[np.newaxis].T
                waveform = amp*np.sin(2*np.pi*coefs*tt/self.bit_rate)
                return np.sum(waveform, axis=0)

            def pure():
                freq = np.array([440*2**(layer.pitch[nn]/12.)])
                return ift(freq)

            def triad():
                freq = layer.pitch[nn]
                freq = freq+np.array([0, 4, 7])
                freq = 440*2**(freq/12.)
                return ift(freq)

            def pipe():
                freq = layer.pitch[nn]
                freq = freq+np.array([0, 12, 24, 36, 48])
                freq = 440*2**(freq/12.)
                amp  = np.array([3, 1, 0.5, 0.2, 0.1])
                return ift(freq, amp)

            instruments = [pure, triad, pipe]
            instr = int(layer.instr[nn])
            return instruments[instr]()

        # The layer is added here
        bit_cts = 0
        for nn in range(0,len(layer.value)):
            note = mk_note(nn)
            note = transmogrify(nn, note)
            
            lo = bit_cts
            up = bit_cts + int(self.bits_beat * layer.value[nn])
            self.out[lo:up] += layer.ampli[nn]*note
            bit_cts = up

class Generator():
    def __init__(self, BIT_RATE, BPM, SIGNATURE, BARS):
        self.bit_rate  = BIT_RATE
        self.bpm       = BPM
        self.signature = SIGNATURE
        self.bars      = BARS
        
        samples = int(BIT_RATE*60./BPM*BARS*SIGNATURE[0])
        self.out = np.zeros(samples)

        self.bits_bar = int(BIT_RATE*60./BPM*SIGNATURE[1])
        self.bits_beat = int(BIT_RATE*60./BPM)

    # Help function for note values
    def get_vals(self, time, prob):
        value = []
        for bar in range(0,self.bars):
            in_bar = 0
            while in_bar < self.signature[0]:
                tmp = np.random.choice(time, p=prob)
                if in_bar+tmp <= self.signature[0]:
                    in_bar += tmp
                    value.append(tmp)
        return np.array(value)

    # Generates different layers below this line
    def mk_layer(self):
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.02, 0.08, 0.5, 0.3, 0.1, 0.0, 0.0])
        
        value = get_vals(time, prob)
        nn = value.shape[0]
        ampli = np.random.uniform(0.6, 0.9,nn)
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
        pitch = -np.random.uniform(0,24,nn)
        instr = np.ones(nn)
        return Layer(value, ampli, shape, param, pitch, instr)

    def mk_organ(self):
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.75, 0.25, 0., 0.0, 0.0, 0.0, 0.0])
        
        value = self.get_vals(time, prob)
        nn = value.shape[0]
        ampli = np.random.uniform(0.6, 0.9,nn)
        shape = np.ones(nn)*1
        param = np.ones(nn)*1
        pitch = np.random.uniform(-36,-24,nn)
        instr = np.ones(nn)*2
        return Layer(value, ampli, shape, param, pitch, instr)

    def mk_mus_box(self):
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.4, 0.4, 0.2, 0., 0., 0.0, 0.0])
        
        value = self.get_vals(time, prob)
        nn = value.shape[0]
        ampli = np.random.uniform(0.2, 0.4,nn)
        shape = np.ones(nn)*1
        param = np.ones(nn)*1
        pitch = np.random.uniform(0,12,nn)
        instr = np.ones(nn)*2
        return Layer(value, ampli, shape, param, pitch, instr)
    
    def mk_bass(self):
        value = np.ones(self.bars*self.signature[0])
        nn = value.shape[0]
        ampli = np.ones(nn)
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
        pitch = np.ones(nn)*-34
        instr = np.ones(nn)*0
        return Layer(value, ampli, shape, param, pitch, instr)

class Layer():
    def __init__(self, value, ampli, shape, param, pitch, instr):
        self.value = value
        self.ampli = ampli
        self.shape = shape
        self.param = param
        self.pitch = pitch
        self.instr = instr

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
BPM       = 128
SIGNATURE = [4, 4]
BARS      = 32
PATH      = '/Users/silver/Dropbox/sci/pro_mus/gen/'
OUTP      =  PATH + NAME + '.wav'

mk_cp(SOURCE, PATH + NAME + '.py')
comp = Composition(BIT_RATE, BPM, SIGNATURE, BARS)
gen = Generator(BIT_RATE, BPM, SIGNATURE, BARS)
comp.add_layer(gen.mk_organ())
comp.add_layer(gen.mk_mus_box())
#comp.add_layer(gen.mk_bass())

save_wav(OUTP, BIT_RATE, comp.out)
