import pygame, math, serial, time, tkinter as tk
from tkinter import filedialog
from svgpathtools import svg2paths2
import numpy as np
from itertools import permutations

# --- instellingen ---
L1 = 100.0
L2 = 100.0
D = 35.0
scale = 2.0
origin = (400, 250)
W, H = 800, 600
BAUD = 115200
PORT = "COM3"

# --- init ---
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("SCARA Plotter Simulator")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

# --- Arduino ---
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print(f"Verbonden met {PORT}")
except:
    ser = None
    print("⚠️ Geen Arduino gevonden - alleen simulatie actief")

# --- functies ---
def inverse_chain(x, y, L1, L2, sign=1):
    r = math.hypot(x, y)
    if r > (L1 + L2) or r < abs(L1 - L2): return None
    cphi = (r*r - L1*L1 - L2*L2) / (2 * L1 * L2)
    phi = math.acos(max(-1,min(1,cphi)))
    a = math.atan2(y, x)
    kx = L1 + L2 * math.cos(phi)
    ky = L2 * math.sin(phi)
    theta = a - sign * math.atan2(ky, kx)
    return theta, phi

def compute_positions(x, y):
    sL, sR = (-D/2, 0), (D/2, 0)
    invL = inverse_chain(x - sL[0], y - sL[1], L1, L2, +1)
    invR = inverse_chain(x - sR[0], y - sR[1], L1, L2, -1)
    if not invL or not invR: return None
    thL, phiL = invL; thR, phiR = invR
    eL = (sL[0] + L1*math.cos(thL), L1*math.sin(thL))
    eR = (sR[0] + L1*math.cos(thR), L1*math.sin(thR))
    return dict(sL=sL,sR=sR,eL=eL,eR=eR,ee=(x,y),
                tL=math.degrees(thL),tR=math.degrees(thR))

def send_angles(tL, tR):
    """Stuur hoeken +180° naar Arduino."""
    if ser:
        tL_adj = tR + 180
        tR_adj = tL + 180
        ser.write(f"{int(tL_adj)},{int(tR_adj)}\n".encode())

def conv(p): return int(origin[0]+p[0]*scale), int(origin[1]-p[1]*scale)

# --- SVG functies ---
def load_svg():
    root=tk.Tk(); root.withdraw()
    f=filedialog.askopenfilename(filetypes=[("SVG files","*.svg")])
    if not f: return []
    paths,_,_=svg2paths2(f)
    lines=[]
    for p in paths:
        pts=[]
        for t in np.linspace(0,1,40):
            q=p.point(t)
            pts.append((q.real,-q.imag))
        lines.append(pts)
    order=min(permutations(range(len(lines))),key=lambda o:path_length(o,lines))
    return [lines[i] for i in order]

def path_length(o,lines):
    d=0
    for i in range(len(o)-1):
        x1,y1=lines[o[i]][-1];x2,y2=lines[o[i+1]][0]
        d+=math.hypot(x2-x1,y2-y1)
    return d

def draw_button(txt,x,y,w,h,active):
    color=(80,180,80) if active else (100,100,100)
    r=pygame.Rect(x,y,w,h)
    pygame.draw.rect(screen,color,r,border_radius=8)
    t=font.render(txt,True,(0,0,0))
    screen.blit(t,(x+(w-t.get_width())/2,y+(h-t.get_height())/2))
    return r

# --- variabelen ---
svg_paths=[]; svg_pos=[0,0]; svg_scale=0.5
drawing=False; paused=False; slider_adjusting=False
current_path=current_point=0
moving_svg=False; mode="manual"
pen_point=None
speed = 1.0  # standaard snelheid

# --- hoofdloop ---
running=True
while running:
    slider_rect = pygame.Rect(W - 220, 60, 150, 10)

    for ev in pygame.event.get():
        if ev.type==pygame.QUIT: running=False

        elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
            # Slider aanraken (drag-and-drop)
            if slider_rect.collidepoint(ev.pos):
                slider_adjusting = True
                paused_before_slider = paused
                paused = True
            # Knoppen
            elif load_btn.collidepoint(ev.pos):
                svg_paths=load_svg()
                drawing=False; paused=False; mode="position"
                current_path=current_point=0
            elif start_btn.collidepoint(ev.pos) and svg_paths:
                drawing=True; paused=False; mode="svg"
                current_path=current_point=0
            elif pause_btn.collidepoint(ev.pos):
                paused = not paused
            elif stop_btn.collidepoint(ev.pos):
                drawing=False; paused=False; mode="manual"
                current_path=current_point=0
                svg_paths=[]  # reset alle vectoren
            elif mode=="position":
                moving_svg=True; mx0,my0=ev.pos

        elif ev.type==pygame.MOUSEBUTTONUP and ev.button==1:
            moving_svg=False
            if slider_adjusting:
                slider_adjusting = False
                paused = paused_before_slider

        elif ev.type==pygame.MOUSEMOTION:
            if moving_svg:
                mx,my=ev.pos
                svg_pos[0]+=(mx-mx0)/scale
                svg_pos[1]-=(my-my0)/scale
                mx0,my0=mx,my
            if slider_adjusting:
                rel_x = min(max(ev.pos[0] - slider_rect.x, 0), slider_rect.width)
                speed = 0.2 + 1.8 * (rel_x / slider_rect.width)

        elif ev.type==pygame.MOUSEWHEEL and mode=="position":
            svg_scale *= 1.1 if ev.y>0 else 0.9

    # --- achtergrond + UI ---
    screen.fill((20,20,30))
    load_btn=draw_button("Load SVG",10,10,120,30,False)
    start_btn=draw_button("Start",140,10,80,30,drawing)
    pause_btn=draw_button("Pause",230,10,80,30,paused)
    stop_btn=draw_button("Stop",320,10,80,30,False)

    # --- SVG tonen ---
    if svg_paths:
        for path in svg_paths:
            pts=[(svg_pos[0]+x*svg_scale,svg_pos[1]+y*svg_scale) for x,y in path]
            for i in range(len(pts)-1):
                pygame.draw.line(screen,(150,150,150),conv(pts[i]),conv(pts[i+1]),1)

    # --- SCARA beweging ---
    if mode=="manual":
        mx,my=pygame.mouse.get_pos()
        x=(mx-origin[0])/scale;y=(origin[1]-my)/scale
        pos=compute_positions(x,y)
        if pos: pen_point=(x,y)
    elif mode=="svg" and drawing and not paused and svg_paths:
        path=svg_paths[current_path]
        p=path[int(current_point)]
        px,py=svg_pos[0]+p[0]*svg_scale,svg_pos[1]+p[1]*svg_scale
        pos=compute_positions(px,py)
        if pos:
            send_angles(pos['tL'],pos['tR'])  # +180° offset hier
            pen_point=(px,py)
        current_point += speed
        if current_point >= len(path):
            current_point=0; current_path+=1
            if current_path>=len(svg_paths):
                drawing=False; mode="manual"
    elif slider_adjusting:
        pass
    elif mode=="svg" and paused:
        pass

    # --- teken armen ---
    if pen_point:
        pos=compute_positions(*pen_point)
        if pos:
            pygame.draw.line(screen,(100,200,255),conv(pos['sL']),conv(pos['eL']),4)
            pygame.draw.line(screen,(100,200,255),conv(pos['sR']),conv(pos['eR']),4)
            pygame.draw.line(screen,(255,200,100),conv(pos['eL']),conv(pos['ee']),3)
            pygame.draw.line(screen,(255,200,100),conv(pos['eR']),conv(pos['ee']),3)
            for p in [pos['sL'],pos['sR'],pos['eL'],pos['eR'],pos['ee']]:
                pygame.draw.circle(screen,(255,255,255),conv(p),5)
            screen.blit(font.render(f"θL={pos['tL']:6.1f}° θR={pos['tR']:6.1f}°", True, (255,255,255)), (W-240, H-30))
            if mode=="svg": pygame.draw.circle(screen,(255,0,0),conv(pen_point),6)

    # --- slider tekenen ---
    pygame.draw.rect(screen, (80, 80, 80), slider_rect)
    handle_x = slider_rect.x + (speed - 0.2) / 1.8 * slider_rect.width
    pygame.draw.circle(screen, (255, 200, 100), (int(handle_x), slider_rect.y + 5), 8)
    screen.blit(font.render(f"Snelheid: {speed:0.2f}x", True, (255,255,255)), (slider_rect.x, slider_rect.y - 20))

    pygame.display.flip()
    clock.tick(60)

if ser: ser.close()
pygame.quit()
