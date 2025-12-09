#Als de servo nog steeds alleen beweegt bij negatieve waarden van pos['tL'], 
# dat betekent je fysieke servo-mapping/invert/offset moet nog aangepast worden. 
# Pas +180.0 of vervang door 180 - (pos + 180) voor spiegeling.

#Als je wil dat de robot alleen fysiek beweegt tijdens tekenen (niet tijdens muis-volg), 
# set if mode=="manual": send-live disabled (current code already sends only in manual optional).

#Als een SVG nog steeds “vult”: controleer in je SVG-editor dat objecten daadwerkelijk strokes hebben 
# (stroke kleur) en fills zijn none. Het script skippt paden met atribute fill ≠ 'none'.
#Pas seg_mm=0.5 in load_svg_contours(..., seg_mm=0.5) als je hogere/lager detail wilt (kleiner = meer punten = hoger detail)



import pygame, math, time, threading, tkinter as tk
from tkinter import filedialog
from svgpathtools import svg2paths2
import numpy as np
import serial

# ---------------- USER CONFIG ----------------
PORT = "COM3"
BAUD = 115200

L1 = 100.0    # mm
L2 = 100.0    # mm
D  = 35.0     # mm between servo axes

PX_TO_MM = 1/5.0   # conversion used earlier: 5 px = 1 mm
SCALE_SCREEN = 2.0  # pixels per mm for visualization
ORIGIN = (600, 600) # screen pixel for world 0,0
# ----------------------------------------------

# Pygame init
pygame.init()
W, H = 1200, 700
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("SCARA Plotter — SVG (contour-only)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

# Try open serial (optional)
arduino = None
try:
    arduino = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    print("Connected to", PORT)
except Exception as e:
    print("No Arduino on", PORT, "-> simulation only (error:", e, ")")
    arduino = None

# UI rects
btn_load = pygame.Rect(20, 20, 120, 36)
btn_start = pygame.Rect(150, 20, 100, 36)
btn_pause = pygame.Rect(260, 20, 100, 36)
btn_stop  = pygame.Rect(370, 20, 100, 36)
slider_rect = pygame.Rect(W - 300, 60, 240, 10)

# runtime state
svg_paths = []           # list of paths: each path is list of (x_mm, y_mm)
svg_loaded = False
svg_offset_mm = [0.0, 0.0]
svg_scale = 1.0
svg_angle_deg = 0.0

mode = "manual"          # "manual", "position", "drawing"
trace_points = []        # visited points (x_mm,y_mm)
control = {"paused": False, "stopped": True, "speed": 1.0}
drawing_thread = None

# slider & interaction
speed = 1.0
dragging_slider = False
dragging_svg = False
last_mouse = (0,0)

# ---------- kinematics ----------
def inverse_chain(x, y, L1, L2, sign=1):
    r = math.hypot(x, y)
    if r > (L1 + L2) or r < abs(L1 - L2):
        return None
    cphi = (r*r - L1*L1 - L2*L2) / (2 * L1 * L2)
    cphi = max(-1.0, min(1.0, cphi))
    # choose underside solution so arms point down
    phi = -math.acos(cphi)
    a = math.atan2(y, x)
    kx = L1 + L2 * math.cos(phi)
    ky = L2 * math.sin(phi)
    theta = a - sign * math.atan2(ky, kx)
    return theta, phi

def compute_positions(x_mm, y_mm):
    sL = (-D/2.0, 0.0)
    sR = ( D/2.0, 0.0)
    invL = inverse_chain(x_mm - sL[0], y_mm - sL[1], L1, L2, +1)
    invR = inverse_chain(x_mm - sR[0], y_mm - sR[1], L1, L2, -1)
    if not invL or not invR:
        return None
    thL, _ = invL; thR, _ = invR
    eL = (sL[0] + L1*math.cos(thL), sL[1] + L1*math.sin(thL))
    eR = (sR[0] + L1*math.cos(thR), sR[1] + L1*math.sin(thR))
    return {"sL": sL, "sR": sR, "eL": eL, "eR": eR, "ee": (x_mm,y_mm),
            "tL": math.degrees(thL), "tR": math.degrees(thR)}

def world_to_screen(pt):
    x_mm, y_mm = pt
    return int(ORIGIN[0] + x_mm * SCALE_SCREEN), int(ORIGIN[1] - y_mm * SCALE_SCREEN)

# ---------- SVG loading: only strokes, skip fills ----------
def sample_path(path_obj, seg_mm=0.5):
    # sample a path with resolution seg_mm (in mm)
    try:
        length_px = path_obj.length()
    except Exception:
        length_px = 100.0
    length_mm = length_px * PX_TO_MM
    n = max(2, int(math.ceil(length_mm / seg_mm)))
    pts = []
    for t in np.linspace(0,1,n):
        z = path_obj.point(t)
        pts.append((z.real, -z.imag))
    return pts

def load_svg_contours(filepath, seg_mm=0.5):
    # return list of paths in mm and centered
    paths, attributes, svg_attr = svg2paths2(filepath)
    sampled = []
    for p, attr in zip(paths, attributes):
        # skip shapes with fill (non-empty & not 'none' or transparent)
        fill = attr.get('fill', '').strip().lower()
        if fill and fill not in ('none', 'transparent', 'rgba(0,0,0,0)'):
            # skip fills: we only want stroke-contours
            continue
        pts = sample_path(p, seg_mm=seg_mm)
        if len(pts) >= 2:
            sampled.append(pts)
    if not sampled:
        return []
    # compute bbox center of all points
    allpts = [pt for path in sampled for pt in path]
    xs = [p[0] for p in allpts]; ys = [p[1] for p in allpts]
    cx = (max(xs)+min(xs))/2.0
    cy = (max(ys)+min(ys))/2.0
    # convert px -> mm and center
    converted = []
    for path in sampled:
        conv = [((x - cx) * PX_TO_MM, (y - cy) * PX_TO_MM) for (x,y) in path]
        converted.append(conv)
    # order paths with greedy nearest neighbor on endpoints
    order = greedy_order(converted)
    ordered = [converted[i] for i in order]
    return ordered

def greedy_order(paths):
    n = len(paths)
    if n <= 1: return list(range(n))
    remaining = set(range(n))
    order = []
    cur = 0
    order.append(cur); remaining.remove(cur)
    while remaining:
        last = paths[cur][-1]
        best = None; bestd = None
        for r in remaining:
            d = math.hypot(paths[r][0][0]-last[0], paths[r][0][1]-last[1])
            if best is None or d < bestd:
                best = r; bestd = d
        order.append(best)
        remaining.remove(best)
        cur = best
    return order

# ---------- drawing worker (single continuous movement) ----------
def drawing_worker(paths, ctrl, trace_list):
    for path in paths:
        if ctrl["stopped"]:
            break
        for (px, py) in path:
            # pause handling
            while ctrl["paused"] and not ctrl["stopped"]:
                time.sleep(0.05)
            if ctrl["stopped"]:
                break
            # transform: rotate, scale, offset
            rot = math.radians(svg_angle_deg)
            xr = px * math.cos(rot) - py * math.sin(rot)
            yr = px * math.sin(rot) + py * math.cos(rot)
            X = svg_offset_mm[0] + xr * svg_scale
            Y = svg_offset_mm[1] + yr * svg_scale
            pos = compute_positions(X, Y)
            if pos:
                trace_list.append((X, Y))
                # compute servo angles and map to 0..180 (add +180 to remove negatives)
                tL = pos['tL']; tR = pos['tR']
                sendL = max(0.0, min(180.0, tL + 180.0))
                sendR = max(0.0, min(180.0, tR + 180.0))
                if arduino and not ctrl["paused"]:
                    try:
                        arduino.write(f"{int(sendL)},{int(sendR)}\n".encode())
                    except:
                        pass
            # speed-based sleep
            time.sleep(max(0.002, 0.03 / max(0.1, ctrl["speed"])))
        if ctrl["stopped"]:
            break
    ctrl["stopped"] = True
    ctrl["paused"] = False
    print("Drawing finished/stopped.")

# ---------- UI helpers ----------
def draw_button(rect, text, active=False):
    base = (80,120,180)
    if active: base = (100,200,100)
    if rect.collidepoint(pygame.mouse.get_pos()):
        base = tuple(min(255, int(c*1.12)) for c in base)
    pygame.draw.rect(screen, base, rect, border_radius=6)
    txt = font.render(text, True, (0,0,0))
    screen.blit(txt, (rect.x + (rect.width - txt.get_width())/2, rect.y + (rect.height - txt.get_height())/2))

# ---------- main loop ----------
running = True
while running:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            control["stopped"] = True
            running = False

        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                if btn_load.collidepoint(ev.pos):
                    tk.Tk().withdraw()
                    f = filedialog.askopenfilename(filetypes=[("SVG files","*.svg")])
                    if f:
                        svg_paths = load_svg_contours(f, seg_mm=0.5)  # 0.5 mm resolution
                        svg_loaded = (len(svg_paths) > 0)
                        svg_offset_mm = [0.0, 0.0]   # centered in world
                        svg_scale = 1.0
                        svg_angle_deg = 0.0
                        trace_points.clear()
                        mode = "position" if svg_loaded else "manual"
                        control["stopped"] = True
                        control["paused"] = False
                        print("SVG loaded, paths:", len(svg_paths))
                elif btn_start.collidepoint(ev.pos) and svg_loaded:
                    # start drawing whole svg (single continuous movement)
                    control["stopped"] = False
                    control["paused"] = False
                    control["speed"] = speed
                    trace_points.clear()
                    drawing_thread = threading.Thread(target=drawing_worker, args=(svg_paths, control, trace_points), daemon=True)
                    drawing_thread.start()
                    mode = "drawing"
                elif btn_pause.collidepoint(ev.pos):
                    control["paused"] = not control["paused"]
                elif btn_stop.collidepoint(ev.pos):
                    control["stopped"] = True
                    control["paused"] = False
                    mode = "manual"
                    svg_paths = []
                    svg_loaded = False
                    trace_points.clear()
                    # send home (0,0) to arduino
                    home = compute_positions(0.0, 0.0)
                    if home and arduino:
                        try:
                            sendL = max(0.0, min(180.0, home['tL'] + 180.0))
                            sendR = max(0.0, min(180.0, home['tR'] + 180.0))
                            arduino.write(f"{int(sendL)},{int(sendR)}\n".encode())
                        except:
                            pass
                else:
                    # click elsewhere: if positioning, start dragging SVG
                    if mode == "position" and svg_loaded:
                        dragging_svg = True
                        last_mouse = ev.pos

                # click on slider?
                if slider_rect.collidepoint(ev.pos):
                    dragging_slider = True

            # wheel handled separately

        elif ev.type == pygame.MOUSEBUTTONUP:
            if ev.button == 1:
                dragging_slider = False
                dragging_svg = False

        elif ev.type == pygame.MOUSEMOTION:
            if dragging_slider:
                rel_x = min(max(ev.pos[0] - slider_rect.x, 0), slider_rect.width)
                speed = 0.2 + 1.8 * (rel_x / slider_rect.width)
            if dragging_svg and mode == "position" and svg_loaded:
                dx = ev.pos[0] - last_mouse[0]
                dy = ev.pos[1] - last_mouse[1]
                svg_offset_mm[0] += dx / SCALE_SCREEN
                svg_offset_mm[1] -= dy / SCALE_SCREEN
                last_mouse = ev.pos

        elif ev.type == pygame.MOUSEWHEEL:
            if mode == "position" and svg_loaded:
                svg_scale *= (1.0 + ev.y * 0.08)
                svg_scale = max(0.05, min(10.0, svg_scale))

        elif ev.type == pygame.KEYDOWN:
            if mode == "position" and svg_loaded:
                if ev.key == pygame.K_LEFT:
                    svg_angle_deg -= 5
                elif ev.key == pygame.K_RIGHT:
                    svg_angle_deg += 5

    # update speed for worker
    control["speed"] = speed

    # background and axes
    screen.fill((18,18,24))
    pygame.draw.line(screen, (60,60,60), (0, ORIGIN[1]), (W, ORIGIN[1]), 1)
    pygame.draw.line(screen, (60,60,60), (ORIGIN[0], 0), (ORIGIN[0], H), 1)

    # draw svg contours (transformed)
    if svg_loaded and svg_paths:
        rot = math.radians(svg_angle_deg)
        cosA, sinA = math.cos(rot), math.sin(rot)
        for path in svg_paths:
            pts = []
            for (px, py) in path:
                xr = px * cosA - py * sinA
                yr = px * sinA + py * cosA
                sx_mm = svg_offset_mm[0] + xr * svg_scale
                sy_mm = svg_offset_mm[1] + yr * svg_scale
                pts.append(world_to_screen((sx_mm, sy_mm)))
            if len(pts) > 1:
                pygame.draw.lines(screen, (80,200,120), False, pts, 2)

    # determine current target for arms
    current_target = None
    if mode == "manual":
        mx, my = pygame.mouse.get_pos()
        x_mm = (mx - ORIGIN[0]) / SCALE_SCREEN
        y_mm = (ORIGIN[1] - my) / SCALE_SCREEN
        current_target = (x_mm, y_mm)
    elif mode == "position":
        current_target = (svg_offset_mm[0], svg_offset_mm[1])
    elif mode == "drawing":
        if trace_points:
            current_target = trace_points[-1]

    # draw arms for current target
    if current_target is not None:
        pos = compute_positions(current_target[0], current_target[1])
        if pos:
            pygame.draw.line(screen, (100,180,255), world_to_screen(pos["sL"]), world_to_screen(pos["eL"]), 6)
            pygame.draw.line(screen, (100,180,255), world_to_screen(pos["sR"]), world_to_screen(pos["eR"]), 6)
            pygame.draw.line(screen, (255,200,120), world_to_screen(pos["eL"]), world_to_screen(pos["ee"]), 4)
            pygame.draw.line(screen, (255,200,120), world_to_screen(pos["eR"]), world_to_screen(pos["ee"]), 4)
            for p in [pos["sL"], pos["sR"], pos["eL"], pos["eR"], pos["ee"]]:
                pygame.draw.circle(screen, (240,240,240), world_to_screen(p), 5)
            # live angles bottom-right
            pygame.draw.rect(screen, (36,36,44), (W-360, H-56, 340, 44))
            ang_text = f"θL={pos['tL']:6.1f}°   θR={pos['tR']:6.1f}°"
            screen.blit(font.render(ang_text, True, (255,255,255)), (W-350, H-46))
            # optionally send live in manual mode
            if mode == "manual" and arduino and not control["paused"]:
                tL = pos['tL']; tR = pos['tR']
                sendL = max(0.0, min(180.0, tL + 180.0))
                sendR = max(0.0, min(180.0, tR + 180.0))
                try:
                    arduino.write(f"{int(sendL)},{int(sendR)}\n".encode())
                except:
                    pass
        else:
            screen.blit(font.render("Doel buiten bereik", True, (255,120,120)), (10,60))

    # draw trace and pen cursor
    if trace_points:
        pts = [world_to_screen(p) for p in trace_points]
        if len(pts) > 1:
            pygame.draw.lines(screen, (0,220,0), False, pts, 2)
        pygame.draw.circle(screen, (255,50,50), pts[-1], 6)

    # UI buttons
    draw_button(btn_load, "Load SVG", svg_loaded)
    draw_button(btn_start, "Start", not control["stopped"] and not control["paused"])
    draw_button(btn_pause, "Pause" if not control["paused"] else "Resume", control["paused"])
    draw_button(btn_stop, "Stop", False)

    # slider
    pygame.draw.rect(screen, (60,60,60), slider_rect)
    handle_x = slider_rect.x + (speed - 0.2) / 1.8 * slider_rect.width
    pygame.draw.circle(screen, (255,200,100), (int(handle_x), slider_rect.y + 5), 8)
    screen.blit(font.render(f"Snelheid: {speed:0.2f}x", True, (220,220,220)), (slider_rect.x, slider_rect.y - 22))

    pygame.display.flip()
    clock.tick(60)

# cleanup
control["stopped"] = True
if drawing_thread and drawing_thread.is_alive():
    drawing_thread.join(timeout=1.0)
if arduino:
    arduino.close()
pygame.quit()
