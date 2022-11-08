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
                amp  = np.array([1, 0.1, 0.002, 0.001, 0.0004])
                return ift(freq, amp)

            def bass():
                freq = layer.pitch[nn]
                freq = freq+np.linspace(0,12,13)
                freq = 440*2**(freq/12.)
                amp  = 1-np.linspace(0, 1, 13)
                return ift(freq, amp)

            instruments = [pure, triad, pipe, bass]
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

    def get_periodic(self, time, prob):
        leitmotif = []
        in_bar = 0
        while in_bar < self.signature[0]:
            tmp = np.random.choice(time, p=prob)
            if in_bar+tmp <= self.signature[0]:
                in_bar += tmp
                leitmotif.append(tmp)
                
        value = []
        for bar in range(0,self.bars):
            value += leitmotif

        return np.array(value), len(leitmotif)

    def get_int_mel(self, time, prob):
        leitmotif_val = [] # Values and pitches
        leitmotif_pit = []
        scale = [3,5,7,8,10,12,14,15] # C major I think/hope/pray
        in_bar = 0
        period = 2
        # This just fills a bar
        while in_bar < self.signature[0]*period:
            tmp_val = np.random.choice(time, p=prob) # Random value
            if in_bar+1 <= self.signature[0]*period:
                in_bar += 1
                for i in range(0,int(1/tmp_val)):
                    leitmotif_val.append(tmp_val)
                    tmp_pit = np.random.choice(scale)
                    leitmotif_pit.append(tmp_pit)

        # End on dominant? and tonic?
        leitmotif_pit[-2] = scale[4]
        leitmotif_pit[-1] = scale[0]
                
        # This should be the second part of magic
        value = []
        pitch = []
        for bar in range(0,self.bars//period):
            value += leitmotif_val
            tmp_pit = list(np.array(leitmotif_pit)+np.random.choice([2,0,-2]))
            tweak = np.random.choice(range(0, len(leitmotif_pit[:-2])))
            tmp_pit[tweak] += np.random.choice([-4, -2, 0, 0, 2, 4])
            pitch += tmp_pit
            leitmotif = tmp_pit
            
        return np.array(value), np.array(pitch)

    def get_pirate(self):
        leitmotif_pit = [-40, -40,   0, 3, 5,   5,   5, 7, 8,   8,   8, 10, 7,   7,   5, 3] # Values and pitches
        leitmotif_val = [  4,   2, 0.5, 0.5, 1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5,  1, 1, 0.5, 0.5]
        in_bar = 0
        period = 4
                
        # This should be the second part of magic
        value = []
        pitch = []
        for bar in range(0,self.bars//period):
            value += leitmotif_val
            pitch += leitmotif_pit

        return np.array(value), np.array(pitch)

    # Generates different layers below this line
    def mk_layer(self):
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.02, 0.08, 0.5, 0.3, 0.1, 0.0, 0.0])
        
        value = self.get_vals(time, prob)
        nn = value.shape[0]
        ampli = np.random.uniform(0.6, 0.9,nn)
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
        pitch = np.random.uniform(-12, 0,nn)
        instr = np.ones(nn)
        return Layer(value, ampli, shape, param, pitch, instr)

    def mk_melody(self):
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.0, 0.0, 0.2, 0.5, 0.3, 0.0, 0.0])
        
        value, period = self.get_periodic(time, prob)
        nn = value.shape[0]
        ampli = np.random.uniform(0.6, 0.9,nn)
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
        pitch = np.tile(np.random.uniform(-12, 0, period), int(nn/period))
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
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.5, 0.5, 0., 0., 0., 0.0, 0.0])

        value, period = self.get_periodic(time, prob)
        nn = value.shape[0]
        ampli = np.random.uniform(0.6, 1.,nn)
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
        pitch = np.random.choice([-26,-22,-18],nn)
        instr = np.ones(nn)*1
        return Layer(value, ampli, shape, param, pitch, instr)

    def mk_bkg(self):
        value = 4*np.ones(self.bars*self.signature[0]//4)
        nn = value.shape[0]
        ampli = np.ones(nn)*0.1
        shape = np.ones(nn)*1
        param = np.ones(nn)*2
        pitch = np.ones(nn)*np.random.choice([-32,-28,-24],nn)
        instr = np.ones(nn)*np.random.choice([1], nn)
        return Layer(value, ampli, shape, param, pitch, instr)
    
    def mk_beat(self):
        value = np.ones(self.bars*self.signature[0])
        nn = value.shape[0]
        ampli = np.ones(nn)*0.5
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
        pitch = np.ones(nn)*-33
        instr = np.ones(nn)*np.random.choice([0], nn)
        return Layer(value, ampli, shape, param, pitch, instr)
    
    def mk_cs_bass(self):
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.15, 0.5, 0.35, 0., 0., 0.0, 0.0])

        value, period = self.get_periodic(time, prob)
        nn = value.shape[0]
        ampli = np.random.uniform(0.3, 0.4,nn)
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
        pitch = np.random.choice([-21, -14, -19],nn)
        instr = np.ones(nn)*1
        return Layer(value, ampli, shape, param, pitch, instr)
    
    def mk_cs_mel(self):
        time = np.array([4, 2, 1, 1/2, 1/4, 1/8, 1/16])
        prob = np.array([0.0, 0.0, 0.35, 0.35, 0.3, 0., 0.0])
        
        value, pitch = self.get_int_mel(time, prob)
        nn = value.shape[0]
        ampli = np.random.choice([0.04, 0.05, 0.05],nn)
        shape = np.ones(nn)*1
        param = np.ones(nn)*2
        instr = np.ones(nn)*0
        return Layer(value, ampli, shape, param, pitch, instr)

    def mk_pirate(self):
        value, pitch = self.get_pirate()
        nn = value.shape[0]
        ampli = np.ones(nn)*1
        shape = np.ones(nn)*0
        param = np.ones(nn)*10
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
# Help functions
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
# Main
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
#comp.add_layer(gen.mk_organ())
#comp.add_layer(gen.mk_mus_box())
#comp.add_layer(gen.mk_melody())

comp.add_layer(gen.mk_cs_bass())
comp.add_layer(gen.mk_cs_mel())

#comp.add_layer(gen.mk_beat())
#comp.add_layer(gen.mk_bkg())
#comp.add_layer(gen.mk_pirate())

save_wav(OUTP, BIT_RATE, comp.out)
