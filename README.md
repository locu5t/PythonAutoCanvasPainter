# OpenGL Accelerated Painter

This project is an advanced painting simulation that uses Python, Pygame, and OpenGL to create artistic renderings of images. It features a sophisticated two-phase painting process that leverages depth maps and image differencing to create a composite image with a distinct 3D effect.

The simulation is hardware-accelerated using OpenGL shaders, allowing for high-performance rendering of complex scenes with many brush strokes.

## Features

- **Two-Phase Painting:** The simulation first paints a background and then intelligently paints a character on top, using four different source images.
- **Character Segmentation:** The character is automatically isolated from the background using image differencing, creating a precise mask for painting.
- **Depth-Aware Rendering:** Brush strokes are modulated by depth maps, creating a sense of distance and perspective. Brush size and opacity change depending on the depth of the area being painted.
- **Hardware-Accelerated:** The entire rendering process is handled by the GPU using OpenGL and GLSL shaders, allowing for a high volume of brush strokes without performance degradation.

## Setup Instructions

This project is designed to be run in a Conda environment to ensure that all dependencies are managed correctly.

### 1. Create a Conda Environment

First, make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

Then, open your terminal and create a new Conda environment. We recommend using Python 3.9 for this project.

```bash
conda create --name painter_env python=3.9
```

### 2. Activate the Environment

Once the environment is created, activate it:

```bash
conda activate painter_env
```

### 3. Install Dependencies

The required Python packages are listed in the `requirements.txt` file. You can install them all at once using `pip`:

```bash
pip install -r requirements.txt
```

## How to Run

After you have set up the environment and installed the dependencies, you can run the painting simulation by executing the `opengl_painter.py` script:

```bash
python opengl_painter.py
```

When you run the script, you will be prompted to select four images in the following order:

1.  **Background Image:** The main background of the scene.
2.  **Background's Depth Map:** A grayscale image representing the depth of the background.
3.  **Final Image with Person:** The final composite image that includes the character.
4.  **Person's Individual Depth Map:** A grayscale image representing the depth of the character.

Once you have selected all four images, the simulation will begin, and you will see the painting process unfold in real-time. The final image will be displayed once the painting is complete.
