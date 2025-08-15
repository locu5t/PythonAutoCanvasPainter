# This file will contain the core painting engine class and logic.
import numpy as np
import cv2
import math
import random

class Painter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.source_image = cv2.imread(self.image_path)
        if self.source_image is None:
            raise FileNotFoundError(f"Could not load image at: {self.image_path}")

        self.height, self.width, _ = self.source_image.shape
        # Create a white canvas to draw on. The canvas is BGR.
        self.canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        self.orientation_map = None

        print(f"Painter initialized with image: {self.image_path} ({self.width}x{self.height})")

    def _calculate_orientation_map(self):
        """
        Calculates the gradient orientation at each pixel of the source image.
        """
        print("Calculating orientation map...")
        gray = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        self.orientation_map = angle
        print("Orientation map calculated.")

    def _generate_strokes(self, num_strokes, min_length, max_length, min_opacity, max_opacity):
        """
        Generates a list of brush strokes with random placement and properties.
        """
        if self.orientation_map is None:
            raise Exception("Orientation map has not been calculated. Call _calculate_orientation_map first.")

        print(f"Generating {num_strokes} strokes...")
        strokes = []
        for _ in range(num_strokes):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)

            color = self.source_image[y, x]
            angle = (self.orientation_map[y, x] + 90) % 360

            stroke_length = random.randint(min_length, max_length)
            opacity = random.uniform(min_opacity, max_opacity)

            stroke = {
                "x": x, "y": y,
                "color": tuple(map(int, color)),
                "size": stroke_length,
                "angle": angle,
                "opacity": opacity
            }
            strokes.append(stroke)
        print(f"{len(strokes)} strokes generated.")
        return strokes

    def _render_stroke(self, layer, stroke):
        """
        Renders a single stroke onto a given BGRA layer.
        """
        x, y = stroke["x"], stroke["y"]
        color_bgr = stroke["color"]
        size = stroke["size"]
        angle_deg = stroke["angle"]
        opacity = stroke["opacity"]

        angle_rad = math.radians(angle_deg)
        half_size = size / 2
        dx = half_size * math.cos(angle_rad)
        dy = half_size * math.sin(angle_rad)

        start_point = (int(x - dx), int(y - dy))
        end_point = (int(x + dx), int(y + dy))

        # Color for the stroke will be BGRA
        color_bgra = (*color_bgr, int(opacity * 255))

        cv2.line(layer, start_point, end_point, color_bgra, thickness=2, lineType=cv2.LINE_AA)

    def paint(self):
        """
        The main painting process, now with multiple passes.
        """
        print("Painting process started...")
        self._calculate_orientation_map()

        # Define the parameters for each painting pass
        passes = [
            # Pass 1: Foundation with large, semi-transparent strokes
            {"name": "Foundation", "num_strokes": 5000, "min_length": 20, "max_length": 40, "min_opacity": 0.1, "max_opacity": 0.3},
            # Pass 2: Building form with medium strokes
            {"name": "Form Building", "num_strokes": 10000, "min_length": 8, "max_length": 16, "min_opacity": 0.5, "max_opacity": 0.8},
            # Pass 3: Detailing with small, more opaque strokes
            {"name": "Detailing", "num_strokes": 20000, "min_length": 2, "max_length": 6, "min_opacity": 0.8, "max_opacity": 1.0},
        ]

        for i, pass_info in enumerate(passes):
            print(f"\n--- PASS {i+1}: {pass_info['name']} ---")

            # Generate strokes for the current pass
            strokes = self._generate_strokes(
                num_strokes=pass_info["num_strokes"],
                min_length=pass_info["min_length"],
                max_length=pass_info["max_length"],
                min_opacity=pass_info["min_opacity"],
                max_opacity=pass_info["max_opacity"]
            )

            # Create a transparent layer for this pass
            layer = np.zeros((self.height, self.width, 4), dtype=np.uint8)

            # Render strokes onto the layer
            print(f"Rendering {len(strokes)} strokes for pass {i+1}...")
            for stroke in strokes:
                self._render_stroke(layer, stroke)

            # Alpha blend the layer onto the main canvas
            # Extract the RGB channels and the alpha mask
            fg_rgb = layer[:, :, :3]
            fg_alpha = layer[:, :, 3]

            # Convert alpha to 0-1 float range
            alpha = fg_alpha.astype(float) / 255.0

            # Alpha blend equation: C_out = C_fg * alpha + C_bg * (1 - alpha)
            # We need to reshape alpha to be broadcastable with the 3-channel images
            alpha = np.stack([alpha] * 3, axis=-1)

            # Blend
            self.canvas = (fg_rgb * alpha + self.canvas * (1 - alpha)).astype(np.uint8)

        print("\nPainting process finished.")

    def save(self, output_path):
        """
        Saves the canvas to the specified path.
        """
        print(f"Saving canvas to {output_path}...")
        cv2.imwrite(output_path, self.canvas)
        print("Canvas saved.")
