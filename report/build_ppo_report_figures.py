from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
p=Path(r'E:\Capstone\bevcapstone\report\figures')
p.mkdir(parents=True,exist_ok=True)
W,H=1700,1850
im=Image.new('RGB',(W,H),'#F7F9FC'); d=ImageDraw.Draw(im)
fontpath=r'C:\Windows\Fonts\arial.ttf'; boldpath=r'C:\Windows\Fonts\arialbd.ttf'
def font(sz,b=False):
    try:return ImageFont.truetype(boldpath if b else fontpath,sz)
    except:return ImageFont.load_default()
def box(x,y,w,h,title,body,fill='#FFFFFF',outline='#31516F'):
    d.rounded_rectangle((x,y,x+w,y+h),radius=22,fill=fill,outline=outline,width=4)
    d.text((x+22,y+16),title,font=font(28,True),fill='#15324B')
    yy=y+56
    for line in body.split('\n'):
        d.text((x+22,yy),line,font=font(22),fill='#273746'); yy+=31
def arrow(x1,y1,x2,y2,label=None):
    d.line((x1,y1,x2,y2),fill='#526A7B',width=5)
    import math
    a=math.atan2(y2-y1,x2-x1); L=18
    p1=(x2-L*math.cos(a-0.5),y2-L*math.sin(a-0.5));p2=(x2-L*math.cos(a+0.5),y2-L*math.sin(a+0.5))
    d.polygon([(x2,y2),p1,p2],fill='#526A7B')
    if label:d.text(((x1+x2)//2+10,(y1+y2)//2-26),label,font=font(18,True),fill='#526A7B')
d.text((55,25),'BC initialized weather conditioned BEV PPO pipeline',font=font(39,True),fill='#15324B')
d.text((55,78),'What is implemented in the evaluated MetaDrive experiment',font=font(23),fill='#526A7B')
box(90,135,690,170,'1  MetaDrive simulator','Route, lane geometry, ego state, nearby traffic\nFive camera views; low traffic density = 0.10','#E7F2FF')
box(920,135,690,170,'2  Camera sensing','Front-left + front-right stereo pair\nLeft, right and rear monocular views','#E7F2FF')
arrow(780,220,920,220,'rendered frames')
box(90,370,690,200,'3  Synthetic weather intervention','Clear / fog / rain corruption on images\nIntensity changes visual quality; vehicle friction\nand dynamics stay unchanged','#F0E9FF')
box(920,370,690,200,'4  Perception','YOLOv8n boxes on four views (320 px, conf 0.25)\nStereoSGBM depth from front pair (scale 0.5)\nSide/rear boxes projected by ground-plane model','#E8F8F0')
arrow(435,305,435,370); arrow(1265,305,1265,370); arrow(780,470,920,470,'weathered pixels')
box(90,640,690,220,'5  BEV state construction','64 × 64 ego-centric grid: free / unknown, route\ncenter + boundaries, projected object footprints, ego\nSynthetic condition-derived horizon masks distant cells','#FFF6D9')
box(920,640,690,220,'6  Scalar state (6 values)','speed • route progress • lateral deviation\nheading error • curvature ahead • distance to goal\nHD map / simulator state provides route features','#FFF6D9')
arrow(1265,570,780,720,'detections + depth'); arrow(1265,570,1265,640,'ego/map state')
box(360,925,980,210,'7  Fused policy observation','BEV CNN: 64 × 64 → 128 features\nScalar MLP: 6 → 64 → 32 features\nConcatenate → 160 features','#E4F4F2')
arrow(435,860,650,925); arrow(1265,860,1050,925)
box(360,1210,980,210,'8  BC initialized PPO policy','Actor: 160 → 128 → 2, tanh; actions = steering + throttle/brake\nCritic: 160 → 128 → value estimate\nFrozen BC action anchor regularizes PPO actor updates','#E7F2FF')
arrow(850,1135,850,1210)
box(90,1510,690,220,'9  Action + reward','MetaDrive receives continuous action in [−1, 1]\nReward = simulator reward + safety shaping\nEnergy reward weight for evaluated checkpoint = 0','#FFECE8')
box(920,1510,690,220,'10  PPO learning and evaluation','Rollouts → clipped PPO policy/value update + BC loss\nCheckpoint: 2,048 PPO steps; evaluate 20 seeds × 5\nconditions = 100 episodes; collect outcomes + YOLO counts','#F1ECFF')
arrow(650,1420,435,1510); arrow(1050,1420,1265,1510)
# feedback loop
arrow(90,1620,45,1620); d.line((45,1620,45,230,90,230),fill='#526A7B',width=4)
d.text((55,1760),'Important: weather horizon is supplied from the known test condition, not inferred from camera pixels.',font=font(21,True),fill='#9A4E1C')
im.save(p/'ppo_weather_pipeline.png',quality=95)
