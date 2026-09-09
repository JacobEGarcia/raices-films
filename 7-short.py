import sys, subprocess, math
sys.path.insert(0,'/tmp/film')
import numpy as np
from PIL import Image, ImageDraw
from rf import *

CIN=FD+'Cinzel.ttf'; COR=FD+'Cormorant.ttf'; CORI=FD+'CormorantIt.ttf'
NIGHT=[(0.0,C('060a1c')),(0.45,C('0b1230')),(0.75,C('18244d')),(1.0,C('25335e'))]
DAWN =[(0.0,C('2a1a3e')),(0.4,C('7a3050')),(0.62,C('d96a3a')),(0.78,C('f4a54a')),(1.0,C('f8d878'))]
GOLD =[(0.0,C('8ec7e8')),(0.5,C('cfe3d8')),(0.75,C('f7cf7a')),(1.0,C('f0a84a'))]
DUSK =[(0.0,C('1b1040')),(0.45,C('5e2050')),(0.7,C('c04438')),(0.85,C('f08a30')),(1.0,C('f8b85a'))]
TWI  =[(0.0,C('0a0e28')),(0.5,C('27205a')),(0.8,C('5a3670')),(1.0,C('8a4a60'))]
WARM =[(0.0,C('120608')),(0.55,C('241009')),(1.0,C('3a1a0c'))]

def build_skyline(seed, hmin=0.52, hmax=0.92):
    r=np.random.default_rng(seed); tops=np.ones(W,np.float32); x=0
    while x<W:
        w=int(r.integers(50,150)); h=r.uniform(hmin,hmax)
        tops[x:min(W,x+w)]=1-h
        if r.random()<0.18:  # spire
            cx=x+w//2; sw=max(4,w//10)
            for i in range(sw):
                j=cx-sw//2+i
                if 0<=j<W: tops[j]=min(tops[j],1-h-0.10*(1-abs(i-sw/2)/(sw/2)))
        x+=w
    mask = YN>=tops[None,:]
    wy,wx=[],[]
    r2=np.random.default_rng(seed+1)
    for bx in range(0,W,13):
        t=tops[bx]
        if t>=1: continue
        if r2.random()<0.55: continue
        for by in range(int((t+0.03)*H),H-8,22):
            if r2.random()<0.22: wy.append(by); wx.append(bx+int(r2.integers(0,5)))
    return mask,np.array(wy),np.array(wx),np.random.default_rng(seed+2).random(len(wy))

SKY_MASK,SKYWY,SKYWX,SKYPH = build_skyline(11)
SKY_BACK,_,_,_ = build_skyline(23, hmin=0.30, hmax=0.55)

def stars(img,t,n=130,ymax=0.65,seed=5,bright=1.0):
    r=np.random.default_rng(seed)
    xs=(r.random(n)*W).astype(int); ys=(r.random(n)*H*ymax).astype(int)
    ph=r.random(n)*6.28; sp=1.5+r.random(n)*2.5
    b=(0.55+0.45*np.sin(t*sp+ph))*bright
    b=np.clip(b,0,1)[...,None]*np.array([225,232,255],np.float32)
    np.add.at(img,(ys,xs),b)

def s1(t,T):  # night city
    img=vgrad(NIGHT)
    stars(img,t)
    glow(img,W*0.78,H*0.22,260,C('8a9ac8'),0.5)
    glow_local(img,W*0.78,H*0.22,60,C('f4f0e0'),1.4,2.0)
    img[SKY_BACK]=C('0d1226')
    img[SKY_MASK]=C('05070f')
    b=0.6+0.4*np.sin(t*1.7+SKYPH*6.28)
    np.add.at(img,(SKYWY,SKYWX),(b[...,None]*np.array([190,148,84])).astype(np.float32))
    a=fade(t,0,5.4,1.6)*min(1,t/1.2)
    if a>0: draw_text(img,"cada otoño, vuelven a casa",CORI,58,H*0.46,alpha=a,tracking=4)
    return img

def s2(t,T):  # departure, night->dawn
    u=smooth(t/8.5)
    img=vgrad(lerp_stops(NIGHT,DAWN,u))
    if u<0.9: stars(img,t,bright=1-u)
    sy=H*(0.95-0.25*u)
    glow(img,W*0.5,sy,420,C('f8b860'),0.7*u)
    img[SKY_BACK]=C('141228')*(1-u)+C('4a2438')*u
    img[SKY_MASK]=C('0a0a14')*(1-u)+C('2a1420')*u
    b=(0.6+0.4*np.sin(t*1.7+SKYPH*6.28))*(1-u)
    np.add.at(img,(SKYWY,SKYWX),(b[...,None]*np.array([250,196,110])).astype(np.float32))
    uu=t/T
    bx=W*(-0.08+1.16*uu); by=H*(0.60-0.18*uu)+26*math.sin(t*2.1)
    bfly=butterfly(100,int(t*24)%8).rotate(8*math.sin(t*1.3),expand=True,resample=Image.BICUBIC)
    paste_rgba(img,bfly,bx,by)
    return img

def hills(img,base,amp,seed,col,haze_col,haze,shift):
    hfull=fbm1(W*2,octaves=4,seed=seed)
    off=int(shift)%W
    h=base-amp*hfull[off:off+W]
    m=YN>h[None,:]
    img[m]=np.asarray(col)*(1-haze)+np.asarray(haze_col)*haze
    return m

def s3(t,T):  # the crossing - golden fields
    img=vgrad(GOLD)
    glow(img,W*0.72,H*0.80,520,C('f8d888'),0.9)
    glow_local(img,W*0.72,H*0.80,90,C('fff2cc'),1.2,2.0)
    for i,(sz,sd,sp,yy) in enumerate([(700,3,6,0.30),(900,7,9,0.22),(560,11,12,0.36)]):
        cx=((t*sp+sd*300)% (W+700))-350
        paste_rgba(img,cloud(sz,sd),cx,H*yy)
    hills(img,0.78,0.10,21,C('b8934a'),C('f7cf7a'),0.45,t*14)
    hills(img,0.86,0.09,22,C('8a6c34'),C('f7cf7a'),0.22,t*22)
    hills(img,0.95,0.08,23,C('5e4a24'),C('f7cf7a'),0.05,t*34)
    uu=t/T
    bx=W*(1.08-1.16*uu); by=H*0.42+20*math.sin(t*2.4)
    paste_rgba(img,butterfly(50,int(t*24)%8),bx,by)
    a=fade(t,0,8.6,1.8)*min(1,t/1.5)
    if a>0: draw_text(img,"tres mil millas",CORI,58,H*0.24,alpha=a,tracking=4)
    return img

def s4(t,T):  # desert dusk
    img=vgrad(DUSK)
    glow(img,W*0.5,H*0.86,560,C('f89040'),1.0)
    glow_local(img,W*0.5,H*0.86,110,C('ffe0a0'),1.1,2.0)
    h=fbm1(W,octaves=2,seed=31); h=np.minimum(h,0.42)
    m=YN>(0.92-0.16*h)[None,:]; img[m]=C('3a1830')
    h2=fbm1(W,octaves=2,seed=32); h2=np.minimum(h2,0.5)
    m2=YN>(0.99-0.10*h2)[None,:]; img[m2]=C('1c0a1a')
    for i in range(3):
        uu=(t/T+i/3.2)%1.0
        bx=W*(1.05-1.1*uu); by=H*(0.40+0.08*i)+16*math.sin(t*3+i*2)
        paste_rgba(img,butterfly(24+4*i,int(t*24+i*2)%8),bx,by)
    return img

PAPEL_COLS=['e8442e','f4901e','f7c948','3aa655','2e86c1','9b59b6','e84393']
def papel_row(img,t,y0,n=8,bw=190):
    for i in range(n):
        x=W*(i+0.5)/n
        sag=30*math.sin(math.pi*(i+0.5)/n)
        yy=y0+sag+7*math.sin(t*1.8+i*0.9)
        b=banner(bw,PAPEL_COLS[i%len(PAPEL_COLS)],i).rotate(3.5*math.sin(t*1.8+i),expand=True,resample=Image.BICUBIC)
        paste_rgba(img,b,x,yy+b.size[1]/2-10)

def s5(t,T):  # gathering - papel picado + flock
    img=vgrad(TWI)
    stars(img,t,n=90,ymax=0.4,seed=6,bright=smooth(t/T))
    papel_row(img,t,H*0.10)
    for i in range(24):
        uu=(t/T*1.4+i/24)%1.0
        x=W*(1.06-1.12*uu)
        y=H*(0.52+0.10*math.sin(uu*7+i*1.7)+0.05*math.sin(i*2.2))
        sz=(15,19,25)[i%3]
        paste_rgba(img,butterfly(sz,int(t*24+i)%8),x,y)
    a=fade(t,0,7.4,1.8)*min(1,t/1.4)
    if a>0: draw_text(img,"juntas",CORI,66,H*0.80,alpha=a,tracking=10)
    return img

def build_forest(seed):
    ov=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(ov)
    r=np.random.default_rng(seed)
    for depth,(col,ymin,ymax,n,hh) in enumerate([(C('16261c'),0.55,0.62,26,0.34),(C('0e1a12'),0.68,0.78,20,0.42),(C('081008'),0.85,1.0,14,0.55)]):
        for i in range(n):
            cx=r.uniform(0,W); bw=r.uniform(120,220)*depth+80; th=H*hh*r.uniform(0.8,1.15)
            byy=H*(ymin+ (ymax-ymin)*r.random())
            d.polygon([(cx-bw/2,byy),(cx+bw/2,byy),(cx,byy-th)],fill=tuple(int(c) for c in col)+(255,))
    return ov
FOREST=build_forest(41)

def s6(t,T):  # the forest of monarchs
    img=vgrad([(0.0,C('030608')),(0.6,C('071018')),(1.0,C('0c1a14'))])
    glow(img,W*0.30,H*0.18,300,C('5a7a9a'),0.35)
    paste_rgba(img,FOREST,W/2,H/2)
    r=np.random.default_rng(43)
    n=340
    xs=r.random(n)*W; ys=H*(0.34+r.random(n)*0.50)
    jx=2.5*np.sin(t*2.2+xs*0.05); jy=2.0*np.cos(t*1.9+ys*0.04)
    b=0.45+0.55*np.sin(t*3.1+xs*0.1+ys*0.02)
    pts=(np.clip(b,0,1)[...,None]*np.array([255,150,50],np.float32)[None,:]*1.8)
    yi=(ys+jy).astype(int)%H; xi=(xs+jx).astype(int)%W
    for dy in (0,1):
        for dx in (0,1):
            np.add.at(img,((yi+dy)%H,(xi+dx)%W),pts*0.6)
    for i in range(9):  # firefly glows
        fx=W*(0.08+0.84*((i*0.618)%1)); fy=H*(0.55+0.3*math.sin(i*2.1))
        glow_local(img,fx+10*math.sin(t+i),fy+8*math.cos(t*0.8+i),26,C('d8e86a'),0.5+0.4*math.sin(t*2.4+i*1.9))
    for i in range(4):  # mist
        mst=smoke(420); ma=mst.split()[3].point(lambda p:int(p*0.5)); mst.putalpha(ma)
        paste_rgba(img,mst,W*((t*0.02+i*0.27)%1.2-0.1),H*(0.92+0.02*math.sin(i)))
    return img

def s7(t,T):  # ofrenda
    img=vgrad(WARM)
    glow(img,W*0.5,H*0.55,420,C('8a4a1a'),0.5)
    papel_row(img,t,H*0.06,n=7,bw=170)
    mstage=int(min(4,1.2+t/1.2))
    paste_rgba(img,marigold(340,mstage),W*0.5,H*0.52)
    glow_local(img,W*0.5,H*0.52,200,C('f4901e'),0.35,2.5)
    for i in range(5):  # candles
        cx=W*(0.16+0.17*i); cy=H*0.86
        ov=Image.new('RGBA',(60,120),(0,0,0,0)); dd=ImageDraw.Draw(ov)
        dd.rectangle([18,40,42,118],fill=(232,222,200,255))
        fl=2.5*math.sin(t*11+i*2.3)+1.5*math.sin(t*23+i)
        dd.polygon([(30+fl*1.4,6),(37+fl,30),(30,40),(23+fl,30)],fill=(252,180,70,255))
        dd.ellipse([26+fl*0.6,26,34+fl*0.6,40],fill=(255,240,190,255))
        paste_rgba(img,ov,cx,cy-50)
        glow_local(img,cx+fl,cy-70,60,C('f8a848'),0.8+0.3*math.sin(t*9+i*2.7),2.0)
    for i in range(10):  # falling petals
        px=W*((i*0.37+0.05)%1)+30*math.sin(t*0.9+i*2.1)
        py=H*((t*0.045+i*0.11)%1.05)
        paste_rgba(img,petal(26,int(t*2+i)%8),px,py)
    a=fade(t,0,6.6,1.8)*min(1,(t-0.6)/1.4)
    if a>0:
        draw_text(img,"para los que volvieron,",CORI,52,H*0.30,alpha=a,tracking=3)
        draw_text(img,"y los que no pudieron",CORI,52,H*0.30+70,alpha=a,tracking=3)
    return img

def s8(t,T):  # title
    img=vgrad([(0.0,C('0a0508')),(0.5,C('1c0d10')),(1.0,C('120608'))])
    glow(img,W*0.5,H*0.44,500,C('6a3010'),0.5)
    mg=marigold(700,4)
    a=mg.split()[3].point(lambda p:int(p*0.16)); mg.putalpha(a)
    paste_rgba(img,mg,W*0.5,H*0.44)
    a1=min(1,t/1.6)
    draw_text(img,"RAÍCES",CIN,150,H*0.34,alpha=a1,color=(248,230,200,255),tracking=30)
    a2=float(np.clip((t-1.6)/1.4,0,1))
    draw_text(img,"para nuestras raíces",CORI,58,H*0.56,alpha=a2,tracking=6)
    a3=float(np.clip((t-3.0)/1.4,0,1))
    draw_text(img,"LATINE FILM GANG · MES DE LA HERENCIA HISPANA · MMXXVI",CIN,30,H*0.72,alpha=a3,color=(220,190,150,255),tracking=8)
    return img

SCENES=[(s1,7,0,0),(s2,12,0,0),(s3,12,6,6),(s4,10,6,6),(s5,10,6,6),(s6,10,6,6),(s7,9,6,6),(s8,9,0,0)]
TOTAL=int(sum(s[1] for s in SCENES)*FPS)

def frame_at(F):
    t=F/FPS; acc=0
    for fn,secs,din,dout in SCENES:
        nf=int(secs*FPS)
        if F<acc+nf:
            lt=(F-acc)/FPS
            img=fn(lt,secs)
            dip=1.0
            if din and F-acc<din: dip=(F-acc)/din
            if dout and acc+nf-1-F<dout: dip=(acc+nf-1-F)/dout
            return post(img,dip=dip**0.5,grain=3.5)
        acc+=nf
    return post(s8(9,9))

def render(a,b,path):
    p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',path],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for F in range(a,b):
        p.stdin.write(frame_at(F).tobytes())
    p.stdin.close(); p.wait()

if __name__=='__main__':
    if sys.argv[1]=='probe':
        for i,F in enumerate([60,240,480,750,1000,1250,1480,1800]):
            Image.fromarray(frame_at(F)).save(f'/tmp/film/stills/probe{i}.jpg',quality=88)
        print('probes done, TOTAL frames:',TOTAL, TOTAL/FPS,'s')
    else:
        a,b=int(sys.argv[1]),int(sys.argv[2]); render(a,b,sys.argv[3])
