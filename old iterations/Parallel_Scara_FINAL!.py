"""
scara_plotter_final.py

Gebruik / Bedieningsinstructies (kort):
- Laad een SVG: klik "Load SVG"
- Positioneren vóór tekenen:
    - Linkermuisknop slepen = verplaatsen (verschuiven)
    - Scrollwheel = zoomen (in/uit)
    - <- / -> pijltjes = roteren (rond midden van venster)
  (SVG verschijnt automatisch gecentreerd op het scherm bij laden)
- Knoppen:
    - Start  : start tekenen (hele SVG in één vloeiende beweging)
    - Pause  : pauzeert tekenen (Resume hervat precies waar gestopt)
    - Stop   : stopt tekenen, reset en stuurt robot naar home (0,0)
- Slider (rechtsboven): regelt teken/snelheid - geldt ook voor verplaatsingen tussen paden
- Muis-volgmodus: zolang er geen SVG geladen is / niet getekend wordt, volgt de SCARA de muis (live)
- Arduino: seriële output is "angleL,angleR\n" (integers). Poort en baud in CONFIG.
- Servo mapping: script stuurt (angle + 180) en clamp 0..180 — pas aan als je mechanisch anders is.

Opmerkingen:
- Script gebruikt svgpathtools en samplet paden met hoge resolutie (instelbaar).
- Het negeert paden met een fill (zodat vormen niet 'ingevuld' worden).
"""

import pygame, math, time, threading, tkinter as tk
from tkinter import filedialog
from svgpathtools import svg2paths2
import numpy as np
import serial

# ---------------- CONFIG ----------------
PORT = "COM5"
BAUD = 115200

L1 = 100.0    # mm bovenarm
L2 = 100.0    # mm onderarm
D  = 35.0     # mm afstand tussen servo-assen

PX_TO_MM = 1/5.0    # conversie: 5 px == 1 mm (pas aan indien gewenst)
SCALE_SCREEN = 2.0  # pixels per mm (visualisatie schaal)
ORIGIN = (600, 600) # schermpunt dat wereld (0,0) voorstelt (x,y)
W, H = 1200, 700

SVG_SAMPLE_SEG_MM = 0.5  # sampling resolutie in mm (kleiner = meer detail)

# ---------------- init ----------------
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("SCARA Plotter — Final")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

# Attempt to open serial
arduino = None
try:
    arduino = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    print("Connected to", PORT)
except Exception as e:
    print("Serial warning:", e)
    arduino = None

# UI rects
btn_load  = pygame.Rect(20, 20, 120, 36)
btn_start = pygame.Rect(150, 20, 100, 36)
btn_pause = pygame.Rect(260, 20, 100, 36)
btn_stop  = pygame.Rect(370, 20, 100, 36)
slider_rect = pygame.Rect(W - 320, 60, 260, 12)

# runtime state
svg_paths = []            # list of paths (each path = list of (x_mm,y_mm))
svg_loaded = False
svg_offset_mm = [0.0, 0.0]  # world mm offset applied to svg (world origin at screen center)
svg_scale = 1.0
svg_angle_deg = 0.0

mode = "manual"       # "manual", "position", "drawing"
trace_points = []     # visited points while drawing (for visual trace)

control = {
    "paused": False,
    "stopped": True,
    "speed": 1.0
}

drawing_thread = None
# indices preserved across pause/resume
draw_index = {
    "path_idx": 0,
    "pt_idx": 0
}

# slider state and dragging states
speed = 1.0
dragging_slider = False
dragging_svg = False
last_mouse = (0, 0)

# ---------------- helpers: kinematics ----------------
def inverse_chain(x, y, L1, L2, sign=1):
    r = math.hypot(x, y)
    if r > (L1 + L2) or r < abs(L1 - L2):
        return None
    cphi = (r*r - L1*L1 - L2*L2) / (2 * L1 * L2)
    cphi = max(-1.0, min(1.0, cphi))
    # choose underside solution - keeps arms downwards
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
    thL, phL = invL
    thR, phR = invR
    eL = (sL[0] + L1*math.cos(thL), sL[1] + L1*math.sin(thL))
    eR = (sR[0] + L1*math.cos(thR), sR[1] + L1*math.sin(thR))
    return {
        "sL": sL, "sR": sR,
        "eL": eL, "eR": eR,
        "ee": (x_mm, y_mm),
        "tL": math.degrees(thL), "tR": math.degrees(thR)
    }

def world_to_screen(pt):
    x_mm, y_mm = pt
    return int(ORIGIN[0] + x_mm * SCALE_SCREEN), int(ORIGIN[1] - y_mm * SCALE_SCREEN)

# ---------------- SVG load & sampling (contours only) ----------------
def sample_path_high_quality(path_obj, seg_mm=SVG_SAMPLE_SEG_MM):
    try:
        length_px = path_obj.length()
    except Exception:
        length_px = 100.0
    length_mm = length_px * PX_TO_MM
    n = max(2, int(math.ceil(length_mm / max(0.1, seg_mm))))
    pts = []
    for t in np.linspace(0.0, 1.0, n):
        z = path_obj.point(t)
        pts.append((z.real, -z.imag))
    return pts

def load_svg_contours(filepath, seg_mm=SVG_SAMPLE_SEG_MM):
    # returns list of paths (in mm) ordered by greedy nearest neighbor
    paths, attributes, svg_attr = svg2paths2(filepath)
    sampled = []
    for p, attr in zip(paths, attributes):
        # if the path has a fill (non-empty & not 'none'), skip it to avoid fills
        fill = attr.get('fill', '').strip().lower()
        if fill and fill not in ('none', 'transparent', 'rgba(0,0,0,0)'):
            continue
        pts = sample_path_high_quality(p, seg_mm=seg_mm)
        if len(pts) >= 2:
            sampled.append(pts)
    if not sampled:
        return []
    # center across all points
    allpts = [pt for path in sampled for pt in path]
    xs = [p[0] for p in allpts]; ys = [p[1] for p in allpts]
    cx = (max(xs) + min(xs)) / 2.0
    cy = (max(ys) + min(ys)) / 2.0
    converted = []
    for path in sampled:
        conv = [((x - cx) * PX_TO_MM, (y - cy) * PX_TO_MM) for (x,y) in path]
        converted.append(conv)
    # greedy ordering
    order = greedy_order(converted)
    ordered = [converted[i] for i in order]
    return ordered

def greedy_order(paths):
    n = len(paths)
    if n <= 1:
        return list(range(n))
    remaining = set(range(n))
    order = []
    cur = 0
    order.append(cur); remaining.remove(cur)
    while remaining:
        last_pt = paths[cur][-1]
        best = None; bestd = None
        for r in remaining:
            d = math.hypot(paths[r][0][0] - last_pt[0], paths[r][0][1] - last_pt[1])
            if best is None or d < bestd:
                best = r; bestd = d
        order.append(best)
        remaining.remove(best)
        cur = best
    return order

# ---------------- movement helpers ----------------
def interpolate_points(p1, p2, step_mm):
    # returns list of intermediate points from p1 to p2 spaced by approx step_mm
    x1,y1 = p1; x2,y2 = p2
    dist = math.hypot(x2-x1, y2-y1)
    if dist == 0:
        return []
    n = max(1, int(math.ceil(dist / step_mm)))
    pts = []
    for i in range(1, n+1):
        t = i / n
        pts.append((x1 + (x2-x1)*t, y1 + (y2-y1)*t))
    return pts

# ---------------- drawing worker (preserves indices for pause/resume) ----------------
def drawing_worker(paths, ctrl, trace_list, indices):
    # indices: dict with 'path_idx' and 'pt_idx' (mutable, shared)
    # We will draw continuously: follow each point in current path, and when jumping to next start, create interpolated travel
    num_paths = len(paths)
    # local copies to speed inner loop
    while indices['path_idx'] < num_paths and not ctrl['stopped']:
        pidx = indices['path_idx']
        path = paths[pidx]
        # iterate points in path
        while indices['pt_idx'] < len(path) and not ctrl['stopped']:
            # pause handling
            while ctrl['paused'] and not ctrl['stopped']:
                time.sleep(0.05)
            if ctrl['stopped']: break
            px, py = path[indices['pt_idx']]
            # transform (rotate, scale, offset)
            rot = math.radians(svg_angle_deg)
            xr = px * math.cos(rot) - py * math.sin(rot)
            yr = px * math.sin(rot) + py * math.cos(rot)
            X = svg_offset_mm[0] + xr * svg_scale
            Y = svg_offset_mm[1] + yr * svg_scale
            pos = compute_positions(X, Y)
            if pos:
                trace_list.append((X, Y))
                # prepare servo angles -> map to servo range: +180 and clamp
                tL = pos['tL']; tR = pos['tR']
                sendL = max(0.0, min(180.0, tL))
                sendR = max(0.0, min(180.0, tR))
                if arduino and not ctrl['paused']:
                    try:
                        arduino.write(f"{int(sendL)},{int(sendR)}\n".encode())
                    except:
                        pass
            # step forward
            indices['pt_idx'] += 1
            # sleep according to speed
            time.sleep(max(0.002, 0.03 / max(0.1, ctrl['speed'])))
        if ctrl['stopped']:
            break
        # finished current path: if there's a next path, move (pen-up) from end to next start using interpolation
        if pidx + 1 < num_paths:
            last_pt = path[-1]
            next_start = paths[pidx + 1][0]
            # interpolated travel points - spacing depends on speed (we use step_mm relative to 1/speed)
            step_mm = 1.0 / max(0.1, ctrl['speed'])  # smaller step when faster
            travel_pts = interpolate_points(last_pt, next_start, step_mm)
            for (tx, ty) in travel_pts:
                while ctrl['paused'] and not ctrl['stopped']:
                    time.sleep(0.05)
                if ctrl['stopped']:
                    break
                rot = math.radians(svg_angle_deg)
                xr = tx * math.cos(rot) - ty * math.sin(rot)
                yr = tx * math.sin(rot) + ty * math.cos(rot)
                X = svg_offset_mm[0] + xr * svg_scale
                Y = svg_offset_mm[1] + yr * svg_scale
                pos = compute_positions(X, Y)
                if pos:
                    trace_list.append((X, Y))
                    tL = pos['tL']; tR = pos['tR']
                    sendL = max(0.0, min(180.0, tL))
                    sendR = max(0.0, min(180.0, tR))
                    if arduino and not ctrl['paused']:
                        try:
                            arduino.write(f"{int(sendL)},{int(sendR)}\n".encode())
                        except:
                            pass
                time.sleep(max(0.002, 0.03 / max(0.1, ctrl['speed'])))
        # move to next path
        indices['path_idx'] += 1
        indices['pt_idx'] = 0
    ctrl['stopped'] = True
    ctrl['paused'] = False
    print("Drawing worker finished or stopped.")

# ---------------- UI / draw helpers ----------------
def draw_button(rect, text, active=False):
    base = (82,120,180)
    if active: base = (100,200,110)
    if rect.collidepoint(pygame.mouse.get_pos()):
        base = tuple(min(255, int(c*1.12)) for c in base)
    pygame.draw.rect(screen, base, rect, border_radius=6)
    txt = font.render(text, True, (0,0,0))
    screen.blit(txt, (rect.x + (rect.width - txt.get_width())/2, rect.y + (rect.height - txt.get_height())/2))

# ---------------- MAIN LOOP ----------------
running = True
while running:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            control['stopped'] = True
            running = False

        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                if btn_load.collidepoint(ev.pos):
                    # load svg
                    tk.Tk().withdraw()
                    fpath = filedialog.askopenfilename(filetypes=[("SVG files","*.svg")])
                    if fpath:
                        try:
                            svg_paths = load_svg_contours(fpath, seg_mm=SVG_SAMPLE_SEG_MM)
                            svg_loaded = len(svg_paths) > 0
                            svg_offset_mm = [0.0, 0.0]
                            svg_scale = 1.0
                            svg_angle_deg = 0.0
                            trace_points.clear()
                            control['stopped'] = True
                            control['paused'] = False
                            draw_index['path_idx'] = 0
                            draw_index['pt_idx'] = 0
                            mode = "position" if svg_loaded else "manual"
                            print("SVG loaded, paths:", len(svg_paths))
                        except Exception as e:
                            print("Error loading SVG:", e)
                elif btn_start.collidepoint(ev.pos):
                    if svg_loaded and (control['stopped'] or control['stopped'] is None):
                        # start drawing worker preserving indices
                        control['paused'] = False
                        control['stopped'] = False
                        control['speed'] = speed
                        trace_points.clear()
                        # spawn worker thread that uses shared draw_index, trace_points, control
                        drawing_thread = threading.Thread(target=drawing_worker, args=(svg_paths, control, trace_points, draw_index), daemon=True)
                        drawing_thread.start()
                        mode = "drawing"
                elif btn_pause.collidepoint(ev.pos):
                    # toggle paused
                    control['paused'] = not control['paused']
                elif btn_stop.collidepoint(ev.pos):
                    # stop drawing and reset
                    control['stopped'] = True
                    control['paused'] = False
                    mode = "manual"
                    svg_paths = []
                    svg_loaded = False
                    trace_points.clear()
                    draw_index['path_idx'] = 0
                    draw_index['pt_idx'] = 0
                    # send home pose to arduino (0,0 world)
                    home = compute_positions(0.0, 0.0)
                    if home and arduino:
                        try:
                            sendL = max(0.0, min(180.0, home['tL']))
                            sendR = max(0.0, min(180.0, home['tR'] ))
                            arduino.write(f"{int(sendL)},{int(sendR)}\n".encode())
                        except:
                            pass
                else:
                    # if positioning mode, start dragging svg (left-click drag)
                    if mode == "position" and svg_loaded:
                        dragging_svg = True
                        last_mouse = ev.pos

                # slider click detection
                if slider_rect.collidepoint(ev.pos):
                    dragging_slider = True

            # mouse wheel handled by MOUSEWHEEL event

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
                svg_scale = max(0.02, min(10.0, svg_scale))

        elif ev.type == pygame.KEYDOWN:
            if mode == "position" and svg_loaded:
                if ev.key == pygame.K_LEFT:
                    svg_angle_deg -= 5
                elif ev.key == pygame.K_RIGHT:
                    svg_angle_deg += 5

    # update control speed
    control['speed'] = speed

    # --- draw background and axes ---
    screen.fill((18,18,24))
    pygame.draw.line(screen, (60,60,60), (0, ORIGIN[1]), (W, ORIGIN[1]), 1)
    pygame.draw.line(screen, (60,60,60), (ORIGIN[0], 0), (ORIGIN[0], H), 1)

    # --- draw SVG preview (transformed) ---
    if svg_loaded and svg_paths:
        rot = math.radians(svg_angle_deg)
        cosA, sinA = math.cos(rot), math.sin(rot)
        for path in svg_paths:
            pts_screen = []
            for (px, py) in path:
                xr = px * cosA - py * sinA
                yr = px * sinA + py * cosA
                sx_mm = svg_offset_mm[0] + xr * svg_scale
                sy_mm = svg_offset_mm[1] + yr * svg_scale
                pts_screen.append(world_to_screen((sx_mm, sy_mm)))
            if len(pts_screen) > 1:
                pygame.draw.lines(screen, (80,200,120), False, pts_screen, 2)

    # --- determine current target for arms ---
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
        else:
            # nothing traced yet: use first path start if exists
            if svg_loaded and svg_paths:
                pidx = draw_index['path_idx']
                ptidx = draw_index['pt_idx']
                if pidx < len(svg_paths):
                    path = svg_paths[pidx]
                    if ptidx < len(path):
                        px, py = path[ptidx]
                        # transform
                        rot = math.radians(svg_angle_deg)
                        xr = px * math.cos(rot) - py * math.sin(rot)
                        yr = px * math.sin(rot) + py * math.cos(rot)
                        current_target = (svg_offset_mm[0] + xr * svg_scale, svg_offset_mm[1] + yr * svg_scale)

    # --- draw arms for current target ---
    pos = None
    if current_target:
        pos = compute_positions(current_target[0], current_target[1])
        if pos:
            # upper arms
            pygame.draw.line(screen, (100,180,255), world_to_screen(pos['sL']), world_to_screen(pos['eL']), 6)
            pygame.draw.line(screen, (100,180,255), world_to_screen(pos['sR']), world_to_screen(pos['eR']), 6)
            # lower arms
            pygame.draw.line(screen, (255,200,120), world_to_screen(pos['eL']), world_to_screen(pos['ee']), 4)
            pygame.draw.line(screen, (255,200,120), world_to_screen(pos['eR']), world_to_screen(pos['ee']), 4)
            # joints
            for p in [pos['sL'], pos['sR'], pos['eL'], pos['eR'], pos['ee']]:
                pygame.draw.circle(screen, (240,240,240), world_to_screen(p), 5)
            # live angles bottom-right
            pygame.draw.rect(screen, (36,36,44), (W-380, H-56, 360, 48))
            ang_text = f"θL={pos['tL']:6.1f}°   θR={pos['tR']:6.1f}°"
            screen.blit(font.render(ang_text, True, (255,255,255)), (W-370, H-46))
            # live send in manual mode (optional): we send mapped angles if not paused
            if mode == "manual" and arduino and not control['paused']:
                tL = pos['tL']; tR = pos['tR']
                sendL = max(0.0, min(180.0, tL))
                sendR = max(0.0, min(180.0, tR))
                try:
                    arduino.write(f"{int(sendL)},{int(sendR)}\n".encode())
                except:
                    pass
        else:
            screen.blit(font.render("Doel buiten bereik", True, (255,120,120)), (10,60))

    # --- draw trace & pen cursor ---
    if trace_points:
        pts = [world_to_screen(p) for p in trace_points]
        if len(pts) > 1:
            pygame.draw.lines(screen, (0,220,0), False, pts, 2)
        pygame.draw.circle(screen, (255,50,50), pts[-1], 6)

    # --- UI buttons ---
    draw_button(btn_load, "Load SVG", svg_loaded)
    # Start active when svg loaded and currently stopped
    start_active = svg_loaded and control['stopped']
    draw_button(btn_start, "Start", start_active)
    draw_button(btn_pause, "Pause" if not control['paused'] else "Resume", control['paused'])
    draw_button(btn_stop, "Stop", False)

    # --- slider ---
    pygame.draw.rect(screen, (60,60,60), slider_rect, border_radius=6)
    handle_x = slider_rect.x + (speed - 0.2) / 1.8 * slider_rect.width
    pygame.draw.circle(screen, (255,200,100), (int(handle_x), slider_rect.y + slider_rect.height//2), 8)
    screen.blit(font.render(f"Snelheid: {speed:0.2f}x", True, (220,220,220)), (slider_rect.x, slider_rect.y - 22))

    # --- tips / mode indicator top-left
    screen.blit(font.render(f"Mode: {mode}", True, (200,200,200)), (20, 70))
    screen.blit(font.render("Drag: left-click, Zoom: wheel, Rotate: ← →", True, (160,160,160)), (20, 90))

    pygame.display.flip()
    clock.tick(60)

# cleanup
control['stopped'] = True
if drawing_thread and drawing_thread.is_alive():
    drawing_thread.join(timeout=1.0)
if arduino:
    arduino.close()
pygame.quit()