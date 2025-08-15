# This will be the main entry point for the application.
import os
from py_painter.engine import Painter

def main():
    print("CPU Painter Main Entry Point")

    # Define input and output paths
    # We assume the script is run from the root of the repository.
    source_image_path = "Orginal.jpg"
    output_image_path = "cpu_painted_output.png"

    if not os.path.exists(source_image_path):
        print(f"Error: Source image not found at '{source_image_path}'. Please run this script from the repository root.")
        return

    # 1. Instantiate the Painter
    try:
        painter = Painter(image_path=source_image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Run the full painting process
    painter.paint()

    # 3. Save the final canvas
    painter.save(output_path=output_image_path)

    print(f"Painting complete. Output saved to {output_image_path}")

if __name__ == "__main__":
    main()
