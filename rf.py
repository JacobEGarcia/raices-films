# rf.py - procedural film render library (numpy/PIL)
import numpy as np, math, functools
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W,H,FPS = 1920,1080,24
FD='/tmp/film/fonts/'
_g = np.random.default_rng(7)
_yy,_xx = np.mgrid[0:H,0:W]
XN = (_xx/(W-1)).astype(np.float32); YN = (_yy/(H-1)).astype(np.float32)
Xf = _xx.astype(np.float32); Yf = _yy.astype(np.float32)

def C(h):
    h=h.lstrip('#')
    return np.array([int(h[i:i+2],16) for i in (0,2,4)],np.float32)

def vgrad(stops):
    ps=[s[0] for s in stops]; cs=[np.asarray(s[1],np.float32) for s in stops]
    out=np.empty((H,W,3),np.float32)
    for c in range(3):
        out[...,c]=np.interp(YN,ps,[col[c] for col in cs]).astype(np.float32)
    return out

def lerp_stops(a,b,u):
    return [(pa,Ca*(1-u)+Cb*u) for (pa,Ca),(pb,Cb) in zip(a,b)]

def glow(img,x,y,r,color,inten=1.0,pow_=3.0):
    d=((Xf-x)**2+(Yf-y)**2)/(r*r)
    img += (np.exp(-d*pow_)*inten)[...,None]*np.asarray(color,np.float32)

def fbm1(n,octaves=4,seed=0,rough=0.55):
    r=np.random.default_rng(seed); x=np.linspace(0,1,n); h=np.zeros(n,np.float32); amp=1.;tot=0.
    for o in range(octaves):
        m=2**(o+3)+1
        h += amp*np.interp(x,np.linspace(0,1,m),r.random(m)).astype(np.float32)
        tot+=amp; amp*=rough
    return h/tot

def terrain(img, base_y, amp, seed, color, haze_color=None, haze=0.0, octaves=4):
    h = base_y - amp*fbm1(W,octaves=octaves,seed=seed)
    m = YN > h[None,:]
    col = np.asarray(color,np.float32)
    if haze_color is not None and haze>0:
        col = col*(1-haze)+np.asarray(haze_color,np.float32)*haze
    img[m]=col
    return m

def paste_rgba(canvas, sprite, x, y):
    """sprite: PIL RGBA. Paste centered at x,y onto numpy float canvas."""
    sw,sh = sprite.size
    x0,y0 = int(x-sw/2), int(y-sh/2)
    x1,y1 = x0+sw, y0+sh
    cx0,cy0 = max(0,x0),max(0,y0); cx1,cy1 = min(W,x1),min(H,y1)
    if cx1<=cx0 or cy1<=cy0: return
    sx0,sy0 = cx0-x0, cy0-y0
    a = np.asarray(sprite.crop((sx0,sy0,sx0+(cx1-cx0),sy0+(cy1-cy0))),np.float32)
    alpha = (a[...,3:4]/255.0)
    roi = canvas[cy0:cy1,cx0:cx1]
    roi[:] = roi*(1-alpha)+a[...,:3]*alpha

def add_rgba(canvas, sprite, x, y, gain=1.0):
    sw,sh = sprite.size
    x0,y0 = int(x-sw/2), int(y-sh/2)
    cx0,cy0 = max(0,x0),max(0,y0); cx1,cy1 = min(W,x0+sw),min(H,y0+sh)
    if cx1<=cx0 or cy1<=cy0: return
    sx0,sy0 = cx0-x0, cy0-y0
    a = np.asarray(sprite.crop((sx0,sy0,sx0+(cx1-cx0),sy0+(cy1-cy0))),np.float32)
    alpha=(a[...,3:4]/255.0)*gain
    canvas[cy0:cy1,cx0:cx1] += a[...,:3]*alpha

# ---------- sprites ----------
@functools.lru_cache(maxsize=8)
def _wing(size, fore):
    """One monarch wing (pointing left). RGBA."""
    s=size; img=Image.new('RGBA',(s,s),(0,0,0,0)); d=ImageDraw.Draw(img)
    cx=s*0.98  # body attach at right edge
    if fore:
        pts=[(cx,s*0.50),(s*0.62,s*0.16),(s*0.28,s*0.06),(s*0.06,s*0.14),(s*0.10,s*0.34),(s*0.30,s*0.46)]
    else:
        pts=[(cx,s*0.52),(s*0.55,s*0.52),(s*0.28,s*0.60),(s*0.12,s*0.78),(s*0.16,s*0.94),(s*0.44,s*0.90),(s*0.66,s*0.70)]
    d.polygon(pts, fill=(20,14,10,255))
    # inner orange
    def shrink(pts,f):
        mx=sum(p[0] for p in pts)/len(pts); my=sum(p[1] for p in pts)/len(pts)
        return [(cx+(p[0]-cx)*f, my+(p[1]-my)*f*0.9) for p in pts]
    d.polygon(shrink(pts,0.86), fill=(233,116,28,255))
    d.polygon(shrink(pts,0.55), fill=(244,148,46,255))
    # veins
    for i,p in enumerate(pts[1:-1]):
        d.line([(cx,s*0.50),shrink([p],0.80)[0]], fill=(20,14,10,255), width=max(2,s//70))
    # white spots on border
    for i in range(4):
        t=i/3.0
        px=pts[1][0]*(1-t)+pts[3][0]*t; py=pts[1][1]*(1-t)+pts[3][1]*t
        r=max(2,s//55)
        d.ellipse([px-r,py-r,px+r,py+r], fill=(245,240,225,255))
    return img

@functools.lru_cache(maxsize=64)
def butterfly(size, phase8):
    """Top-view monarch, wingspan 2*size. phase8: 0..7 flap phase."""
    s=size
    img=Image.new('RGBA',(2*s,s),(0,0,0,0))
    wf=_wing(s,True); wh=_wing(s,False)
    squash=0.30+0.70*abs(math.cos(phase8/8*2*math.pi))
    for wing in (wf,wh):
        ws=wing.resize((s,max(4,int(s*squash))),Image.LANCZOS)
        img.alpha_composite(ws,(0,int(s*0.5-ws.size[1]*0.5)))
        wm=ws.transpose(Image.FLIP_LEFT_RIGHT)
        img.alpha_composite(wm,(s,int(s*0.5-wm.size[1]*0.5)))
    d=ImageDraw.Draw(img)
    d.ellipse([s-0.07*s,s*0.24,s+0.07*s,s*0.78], fill=(25,18,12,255))
    d.ellipse([s-0.045*s,s*0.16,s+0.045*s,s*0.30], fill=(35,26,18,255))
    return img

@functools.lru_cache(maxsize=32)
def banner(w, color_hex, pat):
    """Papel picado banner with scalloped bottom + cutouts."""
    h=int(w*1.25)
    img=Image.new('RGBA',(w,h),(0,0,0,0)); d=ImageDraw.Draw(img)
    col=tuple(int(color_hex[i:i+2],16) for i in (0,2,4))+(255,)
    d.rectangle([0,0,w-1,h-1], fill=col)
    # scallop bottom
    n=5; r=w//(2*n)
    for i in range(n):
        cx=r+2*r*i
        d.ellipse([cx-r,h-1-2*r,cx+r,h-1+r], fill=(0,0,0,0))
    # cutout pattern
    rr=np.random.default_rng(pat)
    for i in range(3):
        for j in range(3):
            cx=w*(0.2+0.3*i); cy=h*(0.22+0.28*j)
            if rr.random()<0.5:
                rw=w*0.07
                d.polygon([(cx,cy-rw),(cx+rw,cy),(cx,cy+rw),(cx-rw,cy)], fill=(0,0,0,0))
            else:
                rw=w*0.05
                d.ellipse([cx-rw,cy-rw,cx+rw,cy+rw], fill=(0,0,0,0))
    # top fringe slits
    for i in range(6):
        x=w*(0.08+0.168*i)
        d.line([x,h*0.06,x,h*0.13], fill=(0,0,0,0), width=max(2,w//40))
    return img

@functools.lru_cache(maxsize=16)
def marigold(size, stage):
    """Marigold head, stage 0..4 bloom."""
    s=size; img=Image.new('RGBA',(s,s),(0,0,0,0)); d=ImageDraw.Draw(img)
    cx=cy=s/2
    layers=[(0.46,(176,74,10)),(0.38,(224,110,16)),(0.30,(244,150,34)),(0.22,(252,186,64)),(0.14,(255,214,110))]
    open_=0.25+0.75*(stage/4.0)
    for li,(rf_,col) in enumerate(layers):
        n=9+li*2
        for k in range(n):
            a=2*math.pi*k/n + li*0.35
            rr=s*rf_*open_
            px=cx+rr*0.55*math.cos(a); py=cy+rr*0.55*math.sin(a)
            pr=s*rf_*0.30*(0.5+0.5*open_)
            d.ellipse([px-pr,py-pr*0.8,px+pr,py+pr*0.8], fill=col+(255,))
    d.ellipse([cx-s*0.07,cy-s*0.07,cx+s*0.07,cy+s*0.07], fill=(120,50,8,255))
    return img

@functools.lru_cache(maxsize=8)
def petal(size, angle8):
    s=size; img=Image.new('RGBA',(s,s),(0,0,0,0)); d=ImageDraw.Draw(img)
    d.ellipse([s*0.2,s*0.35,s*0.8,s*0.65], fill=(244,150,34,230))
    d.ellipse([s*0.3,s*0.40,s*0.7,s*0.60], fill=(252,186,80,230))
    return img.rotate(angle8*45, expand=False, resample=Image.BICUBIC)

@functools.lru_cache(maxsize=8)
def cloud(size, seed):
    s=size; img=Image.new('RGBA',(s,s//2),(0,0,0,0)); d=ImageDraw.Draw(img)
    rr=np.random.default_rng(seed)
    for i in range(7):
        cx=s*(0.15+0.7*rr.random()); cy=s*0.25*(0.6+0.6*rr.random()); r=s*(0.10+0.12*rr.random())
        d.ellipse([cx-r,cy-r*0.6,cx+r,cy+r*0.6], fill=(255,255,255,60))
    return img.filter(ImageFilter.GaussianBlur(s*0.04))

@functools.lru_cache(maxsize=8)
def smoke(size):
    s=size; img=Image.new('RGBA',(s,s),(0,0,0,0)); d=ImageDraw.Draw(img)
    d.ellipse([s*0.15,s*0.15,s*0.85,s*0.85], fill=(200,200,210,40))
    return img.filter(ImageFilter.GaussianBlur(s*0.12))

# ---------- text ----------
def text_img(txt, font_path, size, color=(255,250,240,255), tracking=0, italic=False):
    f=ImageFont.truetype(font_path, size)
    wsum=0; imgs=[]
    tmp=Image.new('RGBA',(10,10)); td=ImageDraw.Draw(tmp)
    for ch in txt:
        bb=td.textbbox((0,0),ch,font=f)
        w=bb[2]-bb[0]; imgs.append((ch,bb,w)); wsum+=w+tracking
    wsum=max(1,wsum-tracking)
    asc,desc=f.getmetrics(); hgt=asc+desc+8
    img=Image.new('RGBA',(wsum+8,hgt),(0,0,0,0)); d=ImageDraw.Draw(img)
    x=4
    for ch,bb,w in imgs:
        d.text((x-bb[0],4-bb[1]),ch,font=f,fill=color)
        x+=w+tracking
    return img

def draw_text(canvas, txt, font, size, y, alpha=1.0, color=(255,250,240,255), tracking=6, x=None, shadow=True):
    img=text_img(txt,font,size,color,tracking)
    if alpha<1.0:
        a=img.split()[3].point(lambda p:int(p*alpha)); img.putalpha(a)
    px = (W-img.size[0])/2 if x is None else x
    if shadow:
        sh=text_img(txt,font,size,(10,8,6,255),tracking)
        a=sh.split()[3].point(lambda p:int(p*alpha*0.75)); sh.putalpha(a)
        paste_rgba(canvas,sh,px+sh.size[0]/2+3,y+sh.size[1]/2+4)
    paste_rgba(canvas,img,px+img.size[0]/2,y+img.size[1]/2)

# ---------- post ----------
VIG = (1.0-0.42*np.clip(((XN-0.5)**2*1.35+(YN-0.5)**2*2.1),0,1)**1.4).astype(np.float32)
def post(canvas, dip=1.0, grain=5.0, vig=True):
    if vig: canvas *= VIG[...,None]
    if dip<1.0: canvas *= dip
    g=_g.standard_normal((H//4,W//4,1),dtype=np.float32)
    canvas += np.kron(g,np.ones((4,4,1),np.float32))*grain
    return np.clip(canvas,0,255).astype(np.uint8)

def smooth(u): u=np.clip(u,0,1); return u*u*(3-2*u)
def fade(t, tin, tstart, dur):
    """1 before tstart, ramp to 0 at tstart+dur."""
    return float(1-np.clip((t-tstart)/max(dur,1e-6),0,1))

def glow_local(img,x,y,r,color,inten=1.0,pow_=3.0):
    x0,y0,x1,y1 = int(max(0,x-2*r)),int(max(0,y-2*r)),int(min(W,x+2*r)),int(min(H,y+2*r))
    if x1<=x0 or y1<=y0: return
    lx=Xf[y0:y1,x0:x1]; ly=Yf[y0:y1,x0:x1]
    d=((lx-x)**2+(ly-y)**2)/(r*r)
    img[y0:y1,x0:x1] += (np.exp(-d*pow_)*inten)[...,None]*np.asarray(color,np.float32)

text_img = functools.lru_cache(maxsize=128)(text_img)
