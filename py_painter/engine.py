# This file will contain the core painting engine class and logic.
import numpy as np
import cv2
import math

class Painter:
    def __init__(self, image_path):
        self.image_path = image_path
        self.source_image = cv2.imread(self.image_path)
        if self.source_image is None:
            raise FileNotFoundError(f"Could not load image at: {self.image_path}")

        self.height, self.width, _ = self.source_image.shape
        # Create a white canvas to draw on.
        self.canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        self.orientation_map = None
        self.strokes = []

        print(f"Painter initialized with image: {self.image_path} ({self.width}x{self.height})")

    def _calculate_orientation_map(self):
        """
        Calculates the gradient orientation at each pixel of the source image.
        This map is used to guide the direction of brush strokes.
        """
        print("Calculating orientation map...")
        gray = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

        self.orientation_map = angle
        print("Orientation map calculated.")

    def _generate_strokes(self, grid_spacing=10, stroke_length=12):
        """
        Generates a list of brush strokes based on a grid.
        """
        if self.orientation_map is None:
            raise Exception("Orientation map has not been calculated. Call _calculate_orientation_map first.")

        print(f"Generating strokes with grid spacing {grid_spacing}...")
        self.strokes = []
        for y in range(0, self.height, grid_spacing):
            for x in range(0, self.width, grid_spacing):
                color = self.source_image[y, x]
                angle = (self.orientation_map[y, x] + 90) % 360

                stroke = {
                    "x": x,
                    "y": y,
                    "color": tuple(map(int, color)),
                    "size": stroke_length,
                    "angle": angle,
                }
                self.strokes.append(stroke)
        print(f"{len(self.strokes)} strokes generated.")

    def _render_stroke(self, stroke):
        """
        Renders a single stroke onto the canvas.
        """
        x = stroke["x"]
        y = stroke["y"]
        color = stroke["color"]
        size = stroke["size"]
        angle_deg = stroke["angle"]

        # Convert angle to radians for math functions
        angle_rad = math.radians(angle_deg)

        # Calculate start and end points of the line
        half_size = size / 2
        dx = half_size * math.cos(angle_rad)
        dy = half_size * math.sin(angle_rad)

        start_point = (int(x - dx), int(y - dy))
        end_point = (int(x + dx), int(y + dy))

        # Draw the line on the canvas
        cv2.line(self.canvas, start_point, end_point, color, thickness=2, lineType=cv2.LINE_AA)

    def paint(self):
        """
        The main painting process.
        """
        print("Painting process started...")
        self._calculate_orientation_map()
        self._generate_strokes()

        print(f"Rendering {len(self.strokes)} strokes...")
        for stroke in self.strokes:
            self._render_stroke(stroke)

        print("Painting process finished.")

    def save(self, output_path):
        """
        Saves the canvas to the specified path.
        """
        print(f"Saving canvas to {output_path}...")
        cv2.imwrite(output_path, self.canvas)
        print("Canvas saved.")
