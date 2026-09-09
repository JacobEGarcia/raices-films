import numpy as np, sys
sr=44100
rng=np.random.default_rng(9)

from scipy.signal import lfilter
_ks_cache={}
def ks(freq,dur,vel=1.0,bright=0.55):
    """Karplus-Strong plucked string via IIR feedback (vectorized)."""
    key=(round(freq,2),round(dur,3),bright)
    if key not in _ks_cache:
        n=int(sr*dur); N=max(2,int(sr/freq))
        a=np.zeros(N+1); a[0]=1.0; a[N-1]=-0.996*bright*0.5-0.996*(1-bright); a[N]=-0.996*bright*0.5
        x=np.zeros(n); x[:N]=(rng.random(N)*2-1)
        out=lfilter([1.0],a,x)
        out*=np.exp(-np.linspace(0,4.2,n))
        _ks_cache[key]=out
    return _ks_cache[key]*vel

def pad(freq,dur,vel=0.3):
    n=int(sr*dur); t=np.arange(n)/sr
    env=np.minimum(1,np.linspace(0,3,n))*np.exp(-np.linspace(0,1.2,n))
    w=np.sin(2*np.pi*freq*t)+0.5*np.sin(2*np.pi*freq*1.005*t+0.7)+0.25*np.sin(2*np.pi*freq*0.5*t)
    return w*env*vel/1.75

def palma(t0,total,vel=0.5):
    n=int(sr*0.09); x=rng.standard_normal(n)
    env=np.exp(-np.linspace(0,9,n))
    seg=x*env*vel
    k=int(t0*sr)
    if k+n<=len(total): total[k:k+n]+=seg[:,None]*np.array([0.7,1.0])

def place(total,sig,t0,pan=0.5):
    k=int(t0*sr); n=len(sig)
    if k+n>len(total): sig=sig[:len(total)-k]; n=len(sig)
    total[k:k+n,0]+=sig*(1-pan*0.6)
    total[k:k+n,1]+=sig*(0.4+pan*0.6)

N=lambda s: 440*2**((s-9)/12)  # semitones from A4
# A minor andalusian: Am G F E  (midi rel to A4: A=0, G=-2, F=-4, E=-5)
CHORDS=[[0,3,7],[-2,2,5],[-4,0,3],[-5,-1,2]]  # Am, G, F, E
BASS=[-12,-14,-16,-17]
MELODY=[(0,12),(1,10),(2,7),(3,10),(4,12),(6,15),(7,12),(8,10),(10,7),(11,3),(12,7),(14,8),(15,7),(16,3),(18,7),(19,10),(20,12),(22,15),(23,14),(24,12),(26,10),(27,7),(28,3)]

def build(dur, bpm=84, melody_shift=0, vol=1.0):
    total=np.zeros((int(sr*dur)+sr,2))
    bar=60/bpm*4
    nbars=int(dur/bar)+1
    t=0.0; ci=0
    while t<dur-1:
        ch=CHORDS[ci%4]; bass=BASS[ci%4]
        # pad per chord
        for n in ch:
            place(total,pad(N(n-12),bar*1.05,0.20),t,0.5)
        place(total,pad(N(bass),bar*1.0,0.30),t,0.45)
        # guitar arpeggio: bass + pattern
        place(total,ks(N(bass),1.4,0.9),t,0.42)
        pat=[ch[0],ch[2],ch[1],ch[2],ch[0],ch[2],ch[1],ch[2]]
        for i,n in enumerate(pat):
            tt=t+bar*(i+1)/9
            place(total,ks(N(n),0.9,0.5+0.1*(i%2)),tt,0.5+0.08*np.sin(i))
        # palmas on 2 and 4 after bar 4
        if ci>=4 and ci%2==0:
            palma(t+bar*0.25,total,0.35); palma(t+bar*0.75,total,0.28)
        t+=bar; ci+=1
    # melody (2 passes over 8 bars each), guitar lead
    mstart=bar*4
    for rep in range(int((dur-mstart)/(bar*8))+1):
        for beat,semi in MELODY:
            tt=mstart+rep*bar*8+beat*bar/4
            if tt<dur-2:
                place(total,ks(N(semi+melody_shift),1.6,0.75),tt,0.62)
    # wind bed
    n=len(total); x=rng.standard_normal(n)
    wind=np.convolve(x,np.ones(400)/400,mode='same')*6
    lfo=0.5+0.5*np.sin(2*np.pi*np.arange(n)/sr*0.07+1)
    total[:,0]+=wind*lfo*0.10; total[:,1]+=wind*np.roll(lfo,sr*3)*0.10
    # master
    total=np.tanh(total*1.1)*0.85*vol
    fade=int(sr*2.5)
    total[:fade]*=np.linspace(0,1,fade)[:,None]
    total[-fade:]*=np.linspace(1,0,fade)[:,None]
    return total[:int(sr*dur)]

if __name__=='__main__':
    dur=float(sys.argv[1]); out=sys.argv[2]
    import scipy.io.wavfile as wf
    mix=build(dur)
    wf.write(out,sr,(mix*32767).astype(np.int16))
    print('wrote',out,len(mix)/sr,'s')
