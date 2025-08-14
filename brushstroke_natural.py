#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined Paint Simulation with:
  - User-friendly UI (Tkinter)
  - Automatic painting (segmentation, color-based passes)
  - Oil vs Water effect (dryness & thickness maps)
  - Progressive detail pass based on highest error areas
"""

import os
import cv2
import numpy as np
import pygame
from pygame.locals import *
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from skimage import segmentation
from collections import defaultdict
from math import comb as math_comb
import datetime
import random
import time
from sklearn.cluster import KMeans

# ----------------------------
# Tkinter UI for user options
# ----------------------------
def select_options():
    """Open a Tkinter window to select painting style, brush texture, reconstruction mode, and recording option."""
    painting_styles = ['Auto-detect', 'Realism', 'Impressionism', 'Expressionism', 'Abstract']
    brush_textures = [
        'Oil', 'Watercolor', 'Pastel', 'Ink', 'Charcoal', 'Acrylic', 'Sponge',
        'Dry Brush', 'Airbrush', 'Palette Knife', 'Pointillism', 'Fine Line',
        'Thick Line', 'Soft Round', 'Hard Round'
    ]
    reconstruction_modes = ['Off', 'Reconstruct Original Image', 'Paint by Color Range']

    root = tk.Tk()
    root.title("Select Options")
    root.resizable(False, False)

    selected_painting_style = tk.StringVar(value=painting_styles[0])
    selected_brush_texture = tk.StringVar(value=brush_textures[0])
    selected_reconstruction_mode = tk.StringVar(value=reconstruction_modes[0])
    record_var = tk.BooleanVar(value=False)

    # Painting Style
    lbl_style = tk.Label(root, text="Choose a painting style:")
    lbl_style.pack(pady=(10, 5))
    dd_style = ttk.OptionMenu(root, selected_painting_style, painting_styles[0], *painting_styles)
    dd_style.pack(pady=5, padx=10, fill='x')

    # Brush Texture
    lbl_brush = tk.Label(root, text="Choose a brush texture style:")
    lbl_brush.pack(pady=(10, 5))
    dd_brush = ttk.OptionMenu(root, selected_brush_texture, brush_textures[0], *brush_textures)
    dd_brush.pack(pady=5, padx=10, fill='x')

    # Reconstruction Mode
    lbl_recon = tk.Label(root, text="Reconstruction Mode:")
    lbl_recon.pack(pady=(10, 5))
    dd_recon = ttk.OptionMenu(root, selected_reconstruction_mode, reconstruction_modes[0], *reconstruction_modes)
    dd_recon.pack(pady=5, padx=10, fill='x')

    # Recording checkbox
    chk_record = tk.Checkbutton(root, text="Record Time Lapse", variable=record_var)
    chk_record.pack(pady=10)

    def confirm():
        root.destroy()

    btn_confirm = tk.Button(root, text="Confirm", command=confirm)
    btn_confirm.pack(pady=(0, 10))

    root.mainloop()

    return (selected_painting_style.get(),
            selected_brush_texture.get(),
            selected_reconstruction_mode.get(),
            record_var.get())


def select_image():
    """Open a file dialog to select an image."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    root.destroy()
    return file_path


def select_depth_image():
    """Open a file dialog to select an optional depth map image (grayscale). Cancel to skip."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Depth Map (optional)",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    root.destroy()
    return file_path or None


# ----------------------------
# Depth helpers
# ----------------------------
def load_depth_map(depth_path, target_size):
    """
    Loads a depth map (grayscale), resizes to target_size (w,h), and normalizes to [0..1].
    Auto-inverts if the top of the image is on average brighter than the bottom (sky brighter).
    Returns depth01 where 0=farthest, 1=nearest.
    """
    if not depth_path or not os.path.exists(depth_path):
        return None

    d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if d is None:
        print("Warning: could not read depth map, continuing without it.")
        return None

    w, h = target_size
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    d = d.astype(np.float32) / 255.0

    # Heuristic auto-invert: if top band brighter than bottom band, assume near=dark (invert)
    h3 = max(1, h // 3)
    top_mean = np.mean(d[:h3, :])
    bottom_mean = np.mean(d[-h3:, :])
    if top_mean > bottom_mean:
        d = 1.0 - d

    d = np.clip(d, 0.0, 1.0)
    return d

def apply_aerial_perspective(rgb, depth01, strength=0.45, atmosphere_rgb=(220, 230, 255)):
    """
    Simple aerial perspective: desaturate and mix toward atmosphere color as depth increases.
    rgb: tuple/list of 3 ints (0..255)
    """
    c = np.array(rgb, dtype=np.float32)
    atm = np.array(atmosphere_rgb, dtype=np.float32)
    t = np.power(float(depth01), 1.2) * float(strength)

    # Linear mix toward atmosphere
    mixed = (1.0 - t) * c + t * atm

    # Mild desaturation in HSV
    hsv = cv2.cvtColor(mixed.reshape(1,1,3).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] *= (1.0 - 0.6 * t)  # reduce saturation with depth
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(3)
    return tuple(int(v) for v in out)

def depth_modulators(d):
    """
    Given depth in [0..1] (0=far, 1=near), return per-stroke scalars:
      - size_scale: far = larger, near = smaller
      - length_scale: far = longer, near = shorter
      - opacity_scale: far = lower, near = higher
      - flow_scale: affects oil thickness (near lays more paint)
      - water_alpha_scale: near concentrates pigment more (less bleeding)
      - dry_rate: multiplier for dryness_step (far dries faster)
      - initial_dryness: far begins drier, near wetter
    """
    size_scale      = np.interp(d, [0, 1], [1.25, 0.75])
    length_scale    = np.interp(d, [0, 1], [1.6, 0.7])
    opacity_scale   = np.interp(d, [0, 1], [0.85, 1.10])
    flow_scale      = np.interp(d, [0, 1], [0.9, 1.3])
    water_alpha_scale = np.interp(d, [0, 1], [1.2, 0.8])
    dry_rate        = np.interp(d, [0, 1], [1.6, 0.7])
    initial_dryness = np.interp(d, [0, 1], [0.85, 0.25])
    return size_scale, length_scale, opacity_scale, flow_scale, water_alpha_scale, dry_rate, initial_dryness

# ----------------------------
# Helpers: dryness, thickness
# ----------------------------
def dryness_step(dryness_map, inc=0.01, rate_map=None):
    """
    Increments dryness, optionally scaled per-pixel by rate_map.
    """
    if rate_map is None:
        dryness_map += inc
    else:
        dryness_map += inc * rate_map
    np.clip(dryness_map, 0.0, 1.0, out=dryness_map)


def alpha_blend(bg, fg, alpha):
    """
    Simple alpha blend for demonstration.
      bg, fg are (R,G,B) in [0..255]
      alpha in [0..1]
    Returns (R,G,B) as int
    """
    r = alpha * fg[0] + (1 - alpha) * bg[0]
    g = alpha * fg[1] + (1 - alpha) * bg[1]
    b = alpha * fg[2] + (1 - alpha) * bg[2]
    return (int(r), int(g), int(b))


def mix_physically(base_rgb, blend_rgb, ratio):
    """
    Placeholder for a more advanced “physical mixing.”
    Using alpha_blend for demonstration.
    ratio in [0..1].
    """
    return alpha_blend(base_rgb, blend_rgb, ratio)


def paint_oil_pixel(canvas, x, y,
                    dryness_map, thickness_map,
                    carried_color, brush_color,
                    pickup_ratio=0.3, deposit_factor=0.7, glazing_factor=1.0):
    """
    - dryness_map[y, x] -> dryness in [0..1], 1=fully dry
    - thickness_map[y, x] -> how much paint has built up
    - carried_color -> color being carried by the brush
    - brush_color -> chosen brush color (not always used directly, we store carried_color instead)
    Returns the updated carried_color (the brush color after picking up).
    """
    existing_col = canvas.get_at((x, y))  # (R, G, B, A)
    existing_rgb = (existing_col.r, existing_col.g, existing_col.b)

    dval = dryness_map[y, x]
    wet_factor = 1.0 - dval

    # Pickup
    pick = pickup_ratio * wet_factor
    new_carried = mix_physically(carried_color, existing_rgb, pick)

    # Deposit
    deposit = (deposit_factor + (0.3 * wet_factor)) * glazing_factor
    out_color = mix_physically(existing_rgb, new_carried, deposit)

    # thickness grows
    thickness_map[y, x] += 0.05 * (1.0 + wet_factor)
    # force dryness to remain partly wet
    dryness_map[y, x] = min(dryness_map[y, x], 0.2)

    # set pixel
    canvas.set_at((x, y), (*out_color, 255))

    return new_carried


def paint_water_pixel(canvas, x, y,
                      dryness_map,
                      brush_color,
                      alpha_val=0.3):
    """
    Watercolor = simple alpha blend + dryness=0
    """
    existing_col = canvas.get_at((x, y))
    existing_rgb = (existing_col.r, existing_col.g, existing_col.b)
    out_rgb = alpha_blend(existing_rgb, brush_color, alpha_val)
    canvas.set_at((x, y), (*out_rgb, 255))
    dryness_map[y, x] = 0.0


# ----------------------------
# Original code's image ops
# ----------------------------
def compute_saliency(image):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    if not success:
        raise ValueError("Saliency computation failed.")
    saliency_map = (saliency_map * 255).astype("uint8")
    return saliency_map


def compute_edge_map(image_gray):
    return cv2.Canny(image_gray, threshold1=50, threshold2=150)


def detect_features(image_gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    features_mask = np.zeros_like(image_gray, dtype=np.uint8)

    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(features_mask, (fx, fy), (fx + fw, fy + fh), 255, -1)
        roi_gray = image_gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(features_mask, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), 255, -1)

    return features_mask > 0


def segment_image(image, num_segments=500):
    segments = segmentation.slic(image, n_segments=num_segments,
                                 compactness=10, start_label=1)
    return segments


def identify_painting_style(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    color_variance = np.var(image) / (255**2)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:,:,1]
    brightness = hsv_image[:,:,2]
    avg_saturation = np.mean(saturation) / 255
    avg_brightness = np.mean(brightness) / 255

    if edge_density > 0.1 and color_variance < 0.05:
        painting_style = 'Realism'
    elif avg_saturation > 0.5 and color_variance > 0.05:
        painting_style = 'Impressionism'
    elif color_variance > 0.1:
        painting_style = 'Expressionism'
    else:
        painting_style = 'Abstract'

    print(f"Identified painting style: {painting_style}")
    return painting_style


# ----------------------------
# Our dryness-based stroke generator
# ----------------------------
def generate_stroke_dryness(
    canvas, start_pos, end_pos,
    dryness_map, thickness_map,
    brush_texture, brush_color,
    stroke_size,
    carried_color_dict,
    water_alpha=0.3,
    glazing_factor=1.0
):
    """
    A dryness-based stroke generator that calls paint_oil_pixel or paint_water_pixel.
    We do a simple line from start_pos to end_pos with radius=stroke_size//2.
    'carried_color_dict' is a dictionary to hold the brush's "wet paint" color for oil.
    """
    # If we haven't stored a "carried_color" yet, initialize it with brush_color
    if 'stroke_carried_color' not in carried_color_dict:
        carried_color_dict['stroke_carried_color'] = brush_color
    carried_color = carried_color_dict['stroke_carried_color']

    # Step along the line
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    steps = max(abs(dx), abs(dy)) or 1

    for i in range(steps+1):
        t_linear = i / float(steps)
        t = ease_in_out(t_linear)
        x_pt = int(start_pos[0] + t*dx)
        y_pt = int(start_pos[1] + t*dy)

        radius = stroke_size // 2
        for dy_ in range(-radius, radius+1):
            for dx_ in range(-radius, radius+1):
                xx = x_pt + dx_
                yy = y_pt + dy_
                if 0 <= xx < canvas.get_width() and 0 <= yy < canvas.get_height():
                    dist = (dx_**2 + dy_**2)**0.5
                    if dist <= radius:
                        if brush_texture.lower() == 'oil':
                            carried_color = paint_oil_pixel(
                                canvas, xx, yy,
                                dryness_map, thickness_map,
                                carried_color, brush_color,
                                glazing_factor=glazing_factor
                            )
                        elif brush_texture.lower() == 'watercolor':
                            paint_water_pixel(
                                canvas, xx, yy,
                                dryness_map, brush_color,
                                alpha_val=water_alpha * glazing_factor
                            )
                        else:
                            # fallback to alpha blend (like “watercolor” for unknown brush)
                            paint_water_pixel(
                                canvas, xx, yy,
                                dryness_map, brush_color,
                                alpha_val=0.1 * glazing_factor
                            )

    carried_color_dict['stroke_carried_color'] = carried_color


# ----------------------------
# New Functions: bezier_curve and generate_stroke
# ----------------------------
def ease_in_out(t):
    """A simple ease-in-out function for non-linear interpolation."""
    return t * t * (3.0 - 2.0 * t)


def bezier_curve(points, num_steps):
    """Compute points along a Bézier curve defined by control points."""
    n = len(points) - 1
    curve = []
    for t_linear in np.linspace(0, 1, num_steps):
        t = ease_in_out(t_linear)
        point = np.zeros(2)
        for i, p in enumerate(points):
            bernstein_poly = math_comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            point += bernstein_poly * np.array(p)
        curve.append(point)
    return curve


def generate_stroke(surface, start_pos, end_pos, color, stroke_size, transparency, brush_texture):
    """Generate a stylized stroke on the given surface based on brush texture."""
    # Define control points for the Bézier curve with limited randomness
    control_points = [start_pos]

    brush_texture_lower = brush_texture.lower()
    mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)

    if brush_texture_lower == 'oil':
        mid_point = (
            mid_point[0] + random.uniform(-5, 5),
            mid_point[1] + random.uniform(-5, 5)
        )
        control_points.append(mid_point)
    elif brush_texture_lower == 'watercolor':
        control_points.append(mid_point)
    elif brush_texture_lower == 'pastel':
        mid_point = (
            mid_point[0] + random.uniform(-3, 3),
            mid_point[1] + random.uniform(-3, 3)
        )
        control_points.append(mid_point)
    elif brush_texture_lower == 'ink':
        control_points.append(mid_point)
    elif brush_texture_lower == 'charcoal':
        mid_point = (
            mid_point[0] + random.uniform(-4, 4),
            mid_point[1] + random.uniform(-4, 4)
        )
        control_points.append(mid_point)
    elif brush_texture_lower == 'acrylic':
        mid_point = (
            mid_point[0] + random.uniform(-2, 2),
            mid_point[1] + random.uniform(-2, 2)
        )
        control_points.append(mid_point)
    elif brush_texture_lower == 'sponge':
        mid_point = (
            mid_point[0] + random.uniform(-6, 6),
            mid_point[1] + random.uniform(-6, 6)
        )
        control_points.append(mid_point)
    elif brush_texture_lower == 'dry brush':
        mid_point = (
            mid_point[0] + random.uniform(-2, 2),
            mid_point[1] + random.uniform(-2, 2)
        )
        control_points.append(mid_point)
    elif brush_texture_lower == 'airbrush':
        control_points.append(mid_point)
    elif brush_texture_lower == 'palette knife':
        mid_point = (
            mid_point[0] + random.uniform(-7, 7),
            mid_point[1] + random.uniform(-7, 7)
        )
        control_points.append(mid_point)
    elif brush_texture_lower == 'pointillism':
        control_points.append(mid_point)
    elif brush_texture_lower == 'fine line':
        control_points.append(mid_point)
    elif brush_texture_lower == 'thick line':
        control_points.append(mid_point)
    elif brush_texture_lower == 'soft round':
        control_points.append(mid_point)
    elif brush_texture_lower == 'hard round':
        control_points.append(mid_point)
    else:
        control_points.append(mid_point)
    control_points.append(end_pos)

    if brush_texture_lower == 'pointillism':
        curve_points = [start_pos]
    else:
        curve_points = bezier_curve(control_points, num_steps=15)

    # Create a simple brush surface for demonstration
    brush_surface = pygame.Surface((stroke_size * 2, stroke_size * 2), pygame.SRCALPHA)
    pygame.draw.circle(brush_surface, (*color, transparency), (stroke_size, stroke_size), stroke_size)

    for point in curve_points:
        x = int(point[0] - stroke_size)
        y = int(point[1] - stroke_size)
        surface.blit(brush_surface, (x, y))


# ----------------------------
# Main painting routine
# ----------------------------
def run_painting_simulation(image_path, painting_style, brush_texture, reconstruction_mode, record, depth_path=None):
    pygame.init()

    screen_info = pygame.display.Info()
    screen_width, screen_height = screen_info.current_w, screen_info.current_h

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        pygame.quit()
        return

    # Auto-detect style if needed
    if painting_style == 'Auto-detect':
        painting_style = identify_painting_style(image)

    img_h, img_w = image.shape[:2]
    scaling_factor = min(screen_width / img_w, screen_height / img_h, 1)
    canvas_width = int(img_w * scaling_factor)
    canvas_height = int(img_h * scaling_factor)

    image = cv2.resize(image, (canvas_width, canvas_height))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ----- Depth map (optional) -----
    depth01 = None
    depth_grad_mag = None
    depth_tangent_angle = None
    dry_rate_map = None

    if depth_path:
        depth01 = load_depth_map(depth_path, (canvas_width, canvas_height))  # 0=far, 1=near
        if depth01 is not None:
            # Depth gradient for edge awareness and stroke tangents along silhouettes
            d_dx = cv2.Sobel(depth01, cv2.CV_32F, 1, 0, ksize=3)
            d_dy = cv2.Sobel(depth01, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad_mag = cv2.magnitude(d_dx, d_dy)
            depth_grad_mag = cv2.normalize(depth_grad_mag, None, 0, 1, cv2.NORM_MINMAX)
            # Tangent angle is gradient angle + 90°
            depth_angle = (cv2.phase(d_dx, d_dy, angleInDegrees=True) + 90.0) % 360.0
            depth_tangent_angle = depth_angle

            # Initialize dryness map from depth (far = drier)
            # and a per-pixel drying rate (far dries faster)
            size_scale, length_scale, opacity_scale, flow_scale, water_alpha_scale, dry_rate, initial_dryness = \
                depth_modulators(depth01)
            dryness_map = initial_dryness.astype(np.float32).copy()
            dry_rate_map = dry_rate.astype(np.float32)
        else:
            depth01 = None  # ensure consistency

    screen_size = (canvas_width, canvas_height)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Painting Simulation")

    # Our final painting surface
    canvas = pygame.Surface(screen_size, flags=pygame.SRCALPHA)
    canvas.fill((255, 255, 255, 255))  # white background
    clock = pygame.time.Clock()
    running = True

    # dryness/thickness maps (depth may have already initialized dryness_map)
    if 'dryness_map' not in locals():
        dryness_map = np.ones((canvas_height, canvas_width), dtype=np.float32)
    thickness_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    # Possibly start a video writer
    if record:
        directory = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"timelapse_{timestamp}.avi"
        video_path = os.path.join(directory, video_filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, screen_size)
        if not video_writer.isOpened():
            print("Error: Unable to open video writer.")
            record = False
        else:
            print(f"Recording enabled. Saving time-lapse to {video_path}")

    # Show initial
    screen.blit(canvas, (0,0))
    pygame.display.flip()

    def capture_frame_if_recording():
        if record:
            frame_arr = pygame.surfarray.array3d(pygame.display.get_surface())
            frame_arr = np.transpose(frame_arr, (1,0,2))
            frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

    capture_frame_if_recording()

    # -------------
    # Reconstruction?
    # -------------
    if reconstruction_mode == 'Reconstruct Original Image':
        print("Starting reconstruction of the original image...")

        pixels = []
        for yy in range(canvas_height):
            for xx in range(canvas_width):
                col = image_rgb[yy, xx]
                bright = np.mean(col)
                pixels.append((bright, (xx, yy), col))

        pixels.sort(key=lambda p: p[0])  # sort by brightness
        total_px = len(pixels)
        chunk_size = 1000
        chunks = [pixels[i:i+chunk_size] for i in range(0, total_px, chunk_size)]

        for chunk in chunks:
            if not running:
                break
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    break
            if not running:
                break

            # "Paint" each pixel directly
            for bright, (xx, yy), ccol in chunk:
                canvas.set_at((xx, yy), (*ccol, 255))

            screen.blit(canvas, (0,0))
            pygame.display.flip()
            capture_frame_if_recording()
            clock.tick(60)

        print("Reconstruction completed.")

    elif reconstruction_mode == 'Paint by Color Range':
        print("Starting painting by color range...")

        # cluster colors
        reshaped = image_rgb.reshape(-1, 3)
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        print("Clustering colors...")
        kmeans.fit(reshaped)
        cluster_centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Build cluster lists
        clusters = defaultdict(list)
        idx = 0
        for yy in range(canvas_height):
            for xx in range(canvas_width):
                label_ = labels[idx]
                clusters[label_].append((xx, yy))
                idx += 1

        # Sort cluster order by brightness
        def cluster_brightness(c):
            return np.mean(c)
        order = sorted(range(num_clusters), key=lambda c: cluster_brightness(cluster_centers[c]))

        for i, cluster_label in enumerate(order):
            if not running:
                break
            positions = clusters[cluster_label]
            chunk_size = 5000
            chunks = [positions[k:k+chunk_size] for k in range(0, len(positions), chunk_size)]
            color_ = cluster_centers[cluster_label]

            for chunk in chunks:
                if not running:
                    break
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                        break
                if not running:
                    break

                for (xx, yy) in chunk:
                    canvas.set_at((xx, yy), (*color_, 255))

                screen.blit(canvas, (0,0))
                pygame.display.flip()
                capture_frame_if_recording()
                clock.tick(60)

            print(f"Painted color range {i+1}/{num_clusters}.")

        print("Painting by color range completed.")

    else:
        # ------------------------------
        # Regular “auto” painting logic
        # ------------------------------
        sal_map = compute_saliency(image)
        sal_map = cv2.resize(sal_map, (canvas_width, canvas_height))
        edge_map = compute_edge_map(image_gray)
        edge_mask = edge_map > 0
        feature_mask = detect_features(image_gray)
        segments = segment_image(image_rgb, num_segments=1500)

        # Depth layers: paint far -> near to respect occlusion
        if depth01 is not None:
            # 5 layers is a good balance
            layer_edges = np.quantile(depth01, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            layer_edges = None

        color_tones = defaultdict(list)
        seg_ids = np.unique(segments)
        for seg_id in seg_ids:
            mask_ = (segments == seg_id)
            yy, xx = np.where(mask_)
            if len(yy) > 0:
                avgc = np.mean(image_rgb[mask_], axis=0).astype(int)
                avgc_tuple = tuple(avgc)
                color_tones[avgc_tuple].append((seg_id, mask_))

        def get_brightness(c):
            r,g,b = c
            return 0.299*r + 0.587*g + 0.114*b

        sorted_tones = sorted(color_tones.items(), key=lambda it: get_brightness(it[0]))

        grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        if painting_style == 'Realism':
            painting_passes = [
                {'chunk_size': 1000, 'brush_size': 2, 'opacity': 150, 'mask': None},
                {'chunk_size': 500, 'brush_size': 1, 'opacity': 200, 'mask': edge_mask},
            ]
        elif painting_style == 'Impressionism':
            painting_passes = [
                {'chunk_size': 800, 'brush_size': 4, 'opacity': 100, 'mask': None},
                {'chunk_size': 400, 'brush_size': 3, 'opacity': 150, 'mask': edge_mask},
            ]
        elif painting_style == 'Expressionism':
            painting_passes = [
                {'chunk_size': 600, 'brush_size': 5, 'opacity': 180, 'mask': None},
                {'chunk_size': 300, 'brush_size': 4, 'opacity': 220, 'mask': edge_mask},
            ]
        elif painting_style == 'Abstract':
            painting_passes = [
                {'chunk_size': 400, 'brush_size': 6, 'opacity': 200, 'mask': None},
                {'chunk_size': 200, 'brush_size': 5, 'opacity': 250, 'mask': None},
            ]
        else:
            painting_passes = [
                {'chunk_size': 800, 'brush_size': 4, 'opacity': 100, 'mask': None},
                {'chunk_size': 400, 'brush_size': 3, 'opacity': 100, 'mask': None},
                {'chunk_size': 200, 'brush_size': 2, 'opacity': 100, 'mask': edge_mask},
                {'chunk_size': 100, 'brush_size': 2, 'opacity': 150, 'mask': feature_mask},
            ]

        # --- NEW: Underpainting Pass ---
        print("\n--- Starting Underpainting Pass ---")
        underpainting_color_dark = (80, 50, 30)
        underpainting_color_light = (180, 150, 130)
        h, w = canvas_height, canvas_width

        # Create a pre-calculated underpainting image to sample colors from
        dark_layer = np.full(image.shape, underpainting_color_dark, dtype=np.uint8)
        light_layer = np.full(image.shape, underpainting_color_light, dtype=np.uint8)
        gray_alpha = np.stack([image_gray.astype(np.float32) / 255.0] * 3, axis=-1)
        underpainting_image = (dark_layer * (1 - gray_alpha) + light_layer * gray_alpha).astype(np.uint8)

        # Paint the canvas with broad strokes using the underpainting colors
        for y in range(0, h, 15): # Coarse grid
            if not running: break
            for x in range(0, w, 15):
                if not running: break
                paint_color = tuple(underpainting_image[y, x])
                angle = random.uniform(0, 360)
                length = random.randint(20, 35)
                end_x = max(0, min(w - 1, x + int(length * np.cos(np.deg2rad(angle)))))
                end_y = max(0, min(h - 1, y + int(length * np.sin(np.deg2rad(angle)))))
                generate_stroke(
                    canvas, start_pos=(x, y), end_pos=(end_x, end_y),
                    color=paint_color, stroke_size=random.randint(15, 25),
                    transparency=random.randint(100, 150), brush_texture='Oil'
                )
            if y % 60 == 0:
                screen.blit(canvas, (0,0))
                pygame.display.flip()
                capture_frame_if_recording()

        screen.blit(canvas, (0,0))
        pygame.display.flip()
        capture_frame_if_recording()
        print("--- Underpainting Complete ---")
        time.sleep(1.0) # Pause to appreciate the underpainting

        # --- Attention-Driven Painting Logic ---
        # 1. Find focus regions from saliency map
        _, thresh_sal = cv2.threshold(sal_map, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_sal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        focus_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        focus_contours = [c for c in focus_contours if cv2.contourArea(c) > 100]

        # Combine all focus contours into one list for iteration, plus a final 'None' for the whole canvas
        all_focus_areas = focus_contours + [None]

        for area_idx, contour in enumerate(all_focus_areas):
            if not running: break

            is_final_pass = (contour is None)
            focus_mask = None
            if not is_final_pass:
                print(f"\n--- Focusing on salient region {area_idx + 1}/{len(focus_contours)} ---")
                focus_mask = np.zeros_like(image_gray, dtype=np.uint8)
                cv2.drawContours(focus_mask, [contour], -1, 255, -1)
                # For focused passes, maybe we do fewer, more detailed passes
                passes_to_run = painting_passes
            else:
                print("\n--- Final broad pass for background and blending ---")
                # For the final pass, just do the broadest strokes
                passes_to_run = painting_passes[:1]

            for pass_i, pass_params in enumerate(passes_to_run):
                if not running: break
                print(f"Starting painting pass {pass_i + 1}")

                max_fps = 60
                update_interval = 1
                chunk_counter = 0
                layer_ranges = list(zip(layer_edges[:-1], layer_edges[1:])) if layer_edges is not None else [(0.0, 1.0)]

                for layer_i, (dlo, dhi) in enumerate(layer_ranges):
                    for color_, seglist in sorted_tones:
                        if not running: break
                        for (sid, mask_) in seglist:
                            if not running: break

                            py_, px_ = np.where(mask_)

                            if focus_mask is not None:
                                focus_points_mask = focus_mask[py_, px_] > 0
                                if not np.any(focus_points_mask):
                                    continue
                                py_, px_ = py_[focus_points_mask], px_[focus_points_mask]

                            if pass_params['mask'] is not None:
                                extra_m = pass_params['mask'][py_, px_]
                                py_, px_ = py_[extra_m], px_[extra_m]
                                if len(py_) == 0: continue

                            if depth01 is not None:
                                dvals = depth01[py_, px_]
                                keep = (dvals >= dlo) & (dvals < dhi)
                                py_, px_ = py_[keep], px_[keep]
                                if len(py_) == 0: continue

                            points = list(zip(px_, py_))
                            grid = defaultdict(list)
                            for (xx, yy) in points:
                                gkey = (xx // 50, yy // 50)
                                grid[gkey].append((xx, yy))

                            for gcell in sorted(grid.keys()):
                                cell_points = grid[gcell]
                                if depth_tangent_angle is not None and depth_grad_mag is not None:
                                    def blended_angle(y, x):
                                        img_ang, dep_tan = orientation[y, x], depth_tangent_angle[y, x]
                                        w = min(1.0, float(depth_grad_mag[y, x]) * 1.5)
                                        a, b = np.deg2rad(img_ang), np.deg2rad(dep_tan)
                                        vx, vy = (1-w)*np.cos(a) + w*np.cos(b), (1-w)*np.sin(a) + w*np.sin(b)
                                        return (np.rad2deg(np.arctan2(vy, vx)) + 360.0) % 360.0
                                    cell_points.sort(key=lambda p: blended_angle(p[1], p[0]))
                                else:
                                    cell_points.sort(key=lambda p: orientation[p[1], p[0]])

                                csize = pass_params['chunk_size']
                                chunks = [cell_points[i:i+csize] for i in range(0, len(cell_points), csize)]
                                for chunk in chunks:
                                    if not running: break
                                    for event in pygame.event.get():
                                        if event.type == QUIT: running = False; break
                                    if not running: break

                                    if dry_rate_map is not None: dryness_step(dryness_map, 0.01, dry_rate_map)
                                    else: dryness_step(dryness_map, 0.01)

                                    for (xx, yy) in chunk:
                                        d = float(depth01[yy, xx]) if depth01 is not None else 0.5
                                        sz_sc, len_sc, op_sc, _, wa_sc, _, _ = depth_modulators(d)
                                        brush_size = max(1, int(pass_params['brush_size'] * sz_sc))
                                        local_carried = {}
                                        if depth_tangent_angle is not None and depth_grad_mag is not None:
                                            ang = blended_angle(yy, xx)
                                        else:
                                            ang = orientation[yy, xx]
                                        length = max(3, int(random.randint(5, 10) * len_sc))
                                        end_x = max(0, min(canvas_width - 1, xx + int(length * np.cos(np.deg2rad(ang)))))
                                        end_y = max(0, min(canvas_height - 1, yy + int(length * np.sin(np.deg2rad(ang)))))
                                        paint_color = tuple(color_)
                                        if depth01 is not None:
                                            paint_color = apply_aerial_perspective(paint_color, d, strength=0.35)
                                        local_water_alpha = 0.3 * wa_sc
                                        generate_stroke_dryness(
                                            canvas, (xx, yy), (end_x, end_y),
                                            dryness_map, thickness_map, brush_texture, paint_color,
                                            stroke_size=brush_size, carried_color_dict=local_carried,
                                            water_alpha=local_water_alpha, glazing_factor=0.8)

                                    chunk_counter += 1
                                    if chunk_counter % update_interval == 0:
                                        screen.blit(canvas, (0, 0))
                                        pygame.display.flip()
                                        capture_frame_if_recording()
                                        clock.tick(max_fps)
                                    if chunk_counter % 5 == 0: time.sleep(random.uniform(0.05, 0.2))

            # Update screen after a region is done
            screen.blit(canvas, (0,0))
            pygame.display.flip()
            capture_frame_if_recording()
            if not is_final_pass:
                time.sleep(0.3) # Pause before next region

    if not running:
        print("Painting simulation terminated by user.")
    else:
        # -------------------------------------------------
        # Enhanced Detailed Touch-Up Pass
        # -------------------------------------------------
        print("Starting enhanced detailed touch-up pass")

        detailed_iterations = 2  # Number of detailed iterations

        for iteration in range(detailed_iterations):
            if not running:
                break
            print(f"Detailed pass iteration {iteration + 1}/{detailed_iterations}")

            # Dynamic brush size and opacity based on iteration
            brush_size = max(1, 2 - iteration)
            opacity = max(50, 100 - iteration * 20)
            chunk_size = 300  # Finer control with smaller chunk size

            for color, seglist in sorted_tones:
                if not running:
                    break
                for (sid, mask_) in seglist:
                    if not running:
                        break
                    yy, xx = np.where(mask_)
                    points = list(zip(xx, yy))
                    spatial_grid_size = 30  # Smaller grid for finer details
                    grid = defaultdict(list)
                    for x, y in points:
                        grid_key = (x // spatial_grid_size, y // spatial_grid_size)
                        grid[grid_key].append((x, y))
                    for grid_cell in sorted(grid.keys()):
                        cell_points = grid[grid_cell]
                        cell_points.sort(key=lambda p: orientation[p[1], p[0]])
                        chunks = [cell_points[i:i + chunk_size] for i in range(0, len(cell_points), chunk_size)]
                        for chunk in chunks:
                            if not running:
                                break
                            for event in pygame.event.get():
                                if event.type == QUIT:
                                    running = False
                                    break
                            if not running:
                                break
                            for x, y in chunk:
                                if depth01 is not None:
                                    d = float(depth01[y, x])
                                    sz_sc, len_sc, op_sc, _, _, _, _ = depth_modulators(d)
                                    # Skip very far points in detail pass to avoid over-detailing background
                                    if d < 0.20:
                                        continue
                                    current_brush_size = max(1, int(brush_size * min(1.0, 1.1 * (1.0/sz_sc))))
                                    current_opacity = int(opacity * np.clip(op_sc, 0.5, 1.3))
                                    # Blend angle with depth tangents at strong depth edges
                                    if depth_tangent_angle is not None and depth_grad_mag is not None:
                                        w = min(1.0, float(depth_grad_mag[y, x]) * 1.5)
                                        img_ang = orientation[y, x]
                                        dep_tan = depth_tangent_angle[y, x]
                                        a = np.deg2rad(img_ang); b = np.deg2rad(dep_tan)
                                        vx = (1-w)*np.cos(a) + w*np.cos(b); vy = (1-w)*np.sin(a) + w*np.sin(b)
                                        angle = (np.rad2deg(np.arctan2(vy, vx)) + 360.0) % 360.0
                                    else:
                                        angle = orientation[y, x]
                                    length = max(2, int(random.randint(3, 6) * len_sc))
                                    color_to_paint = apply_aerial_perspective(tuple(image_rgb[y, x]), d, strength=0.25)
                                else:
                                    current_brush_size = brush_size
                                    current_opacity = opacity
                                    color_to_paint = tuple(image_rgb[y, x])
                                    angle = orientation[y, x]
                                    length = random.randint(3, 6)

                                end_x = x + int(length * np.cos(np.deg2rad(angle)))
                                end_y = y + int(length * np.sin(np.deg2rad(angle)))
                                end_x = max(0, min(canvas_width - 1, end_x))
                                end_y = max(0, min(canvas_height - 1, end_y))

                                generate_stroke(
                                    canvas,
                                    start_pos=(x, y),
                                    end_pos=(end_x, end_y),
                                    color=color_to_paint,
                                    stroke_size=current_brush_size,
                                    transparency=current_opacity,
                                    brush_texture=brush_texture
                                )
                            # Update display after processing each chunk
                            screen.blit(canvas, (0, 0))
                            pygame.display.flip()

                            if record:
                                frame_arr = pygame.surfarray.array3d(pygame.display.get_surface())
                                frame_arr = np.transpose(frame_arr, (1, 0, 2))
                                frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
                                video_writer.write(frame_bgr)

                            clock.tick(60)

                            # Pause to simulate artist thinking
                            if random.random() < 0.1: # 10% chance of a short pause
                                time.sleep(random.uniform(0.01, 0.05))

    def save_canvas_image(surface, prefix='painting'):
        directory = os.path.dirname(os.path.abspath(__file__))
        counter = 1
        while True:
            fname = f"{prefix}_{counter}.png"
            fpath = os.path.join(directory, fname)
            if not os.path.exists(fpath):
                break
            counter += 1
        pygame.image.save(surface, fpath)
        print(f"Image saved as {fpath}")

    save_canvas_image(canvas)

    if record:
        capture_frame_if_recording()
        video_writer.release()
        print(f"Time-lapse video saved as {video_path}")

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        screen.blit(canvas, (0,0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    style, texture, mode, record_flag = select_options()
    if not style:
        print("No painting style selected. Exiting.")
    else:
        image_path = select_image()
        if image_path:
            depth_path = select_depth_image()  # <-- new (Cancel to skip)
            run_painting_simulation(image_path, style, texture, mode, record_flag, depth_path=depth_path)
        else:
            print("No image selected. Exiting.")
