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


# ----------------------------
# Helpers: dryness, thickness
# ----------------------------
def dryness_step(dryness_map, inc=0.01):
    """
    Increments dryness in the entire dryness_map, never exceeding 1.0.
    dryness_map: 2D array of shape (height, width)
      0.0 = fully wet
      1.0 = fully dry
    """
    dryness_map += inc
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
                    pickup_ratio=0.3, deposit_factor=0.7):
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
    deposit = deposit_factor + (0.3 * wet_factor)
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
    water_alpha=0.3
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
        t = i / float(steps)
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
                                carried_color, brush_color
                            )
                        elif brush_texture.lower() == 'watercolor':
                            paint_water_pixel(
                                canvas, xx, yy,
                                dryness_map, brush_color,
                                alpha_val=water_alpha
                            )
                        else:
                            # fallback to alpha blend (like “watercolor” for unknown brush)
                            paint_water_pixel(
                                canvas, xx, yy,
                                dryness_map, brush_color,
                                alpha_val=0.1
                            )

    carried_color_dict['stroke_carried_color'] = carried_color


# ----------------------------
# New Functions: bezier_curve and generate_stroke
# ----------------------------
def bezier_curve(points, num_steps):
    """Compute points along a Bézier curve defined by control points."""
    n = len(points) - 1
    curve = []
    for t in np.linspace(0, 1, num_steps):
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
def run_painting_simulation(image_path, painting_style, brush_texture, reconstruction_mode, record):
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

    screen_size = (canvas_width, canvas_height)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Painting Simulation")

    # Our final painting surface
    canvas = pygame.Surface(screen_size, flags=pygame.SRCALPHA)
    canvas.fill((255, 255, 255, 255))  # white background
    clock = pygame.time.Clock()
    running = True

    # dryness/thickness maps
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

        # dryness-based strokes for each pass
        for pass_i, pass_params in enumerate(painting_passes):
            if not running:
                break
            print(f"Starting painting pass {pass_i + 1}")

            max_fps = 60
            update_interval = 1

            chunk_counter = 0

            for color_, seglist in sorted_tones:
                if not running: 
                    break
                for (sid, mask_) in seglist:
                    if not running:
                        break
                    py_, px_ = np.where(mask_)
                    if pass_params['mask'] is not None:
                        extra_m = pass_params['mask'][py_, px_]
                        py_ = py_[extra_m]
                        px_ = px_[extra_m]
                        if len(py_) == 0:
                            continue
                    points = list(zip(px_, py_))
                    # cluster by 50x50 grid
                    grid = defaultdict(list)
                    for (xx, yy) in points:
                        gkey = (xx // 50, yy // 50)
                        grid[gkey].append((xx, yy))

                    for gcell in sorted(grid.keys()):
                        cell_points = grid[gcell]
                        # sort by orientation
                        cell_points.sort(key=lambda p: orientation[p[1], p[0]])
                        csize = pass_params['chunk_size']
                        chunks = [cell_points[i:i+csize] for i in range(0, len(cell_points), csize)]
                        for chunk in chunks:
                            if not running:
                                break
                            for event in pygame.event.get():
                                if event.type == QUIT:
                                    running = False
                                    break
                            if not running:
                                break

                            # dryness step each chunk so canvas slowly dries
                            dryness_step(dryness_map, inc=0.01)

                            # paint each pixel as a stroke
                            for (xx, yy) in chunk:
                                brush_size = pass_params['brush_size']
                                local_carried = {}

                                angle = orientation[yy, xx]
                                length = random.randint(5, 10)
                                end_x = xx + int(length * np.cos(np.deg2rad(angle)))
                                end_y = yy + int(length * np.sin(np.deg2rad(angle)))
                                end_x = max(0, min(canvas_width-1, end_x))
                                end_y = max(0, min(canvas_height-1, end_y))

                                generate_stroke_dryness(
                                    canvas,
                                    (xx, yy),
                                    (end_x, end_y),
                                    dryness_map, thickness_map,
                                    brush_texture, color_,
                                    stroke_size=brush_size,
                                    carried_color_dict=local_carried
                                )

                            chunk_counter += 1
                            if chunk_counter % update_interval == 0:
                                screen.blit(canvas, (0,0))
                                pygame.display.flip()
                                capture_frame_if_recording()
                                clock.tick(max_fps)

            screen.blit(canvas, (0,0))
            pygame.display.flip()
            capture_frame_if_recording()

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
                                current_brush_size = brush_size
                                current_opacity = opacity
                                # Use the actual color from the reference image at (y, x)
                                color_to_paint = image_rgb[y, x]
                                angle = orientation[y, x]
                                length = random.randint(3, 6)  # Shorter strokes for details
                                end_x = x + int(length * np.cos(np.deg2rad(angle)))
                                end_y = y + int(length * np.sin(np.deg2rad(angle)))
                                end_x = max(0, min(canvas_width - 1, end_x))
                                end_y = max(0, min(canvas_height - 1, end_y))
                                # Generate stroke using the selected brush texture
                                generate_stroke(
                                    canvas,
                                    start_pos=(x, y),
                                    end_pos=(end_x, end_y),
                                    color=tuple(color_to_paint),
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
            run_painting_simulation(image_path, style, texture, mode, record_flag)
        else:
            print("No image selected. Exiting.")
