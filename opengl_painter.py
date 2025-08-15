import pyglet
# Try to disable the shadow window before any other pyglet imports.
pyglet.options['shadow_window'] = False
pyglet.options['headless'] = True

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import cv2
import os
import ctypes
import random

# --- Shaders ---

vertex_shader_draw = """
#version 330 core
in vec2 a_pos;
uniform vec2 u_pos;
uniform float u_size;
uniform mat4 u_projection;
void main()
{
    gl_Position = u_projection * vec4(a_pos * u_size + u_pos, 0.0, 1.0);
}
"""

fragment_shader_draw = """
#version 330 core
out vec4 FragColor;
uniform vec4 u_color;
void main()
{
    float dist = length(gl_PointCoord - vec2(0.5));
    if (dist > 0.5) discard;
    float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
    FragColor = vec4(u_color.rgb, u_color.a * alpha);
}
"""

# --- Helper Functions ---

def create_fbo(width, height):
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("FBO Error")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, texture

def save_fbo_to_image(fbo, texture, width, height, filename):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glBindTexture(GL_TEXTURE_2D, texture)

    glReadBuffer(GL_COLOR_ATTACHMENT0)
    pixels = glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE)

    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)

    image = cv2.flip(image, 0)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    cv2.imwrite(filename, image_bgr)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print(f"Image saved to {filename}")

def main():
    # Load a single source image
    source_image_path = "Orginal.jpg"
    if not os.path.exists(source_image_path):
        print(f"Error: Source image not found at {source_image_path}")
        return

    img_ref = cv2.imread(source_image_path)
    img_h, img_w = img_ref.shape[:2]
    img_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

    # Create a headless window with pyglet to get an OpenGL context
    window = pyglet.window.Window(width=1, height=1, visible=False)
    # Explicitly activate the context
    window.switch_to()

    shader_draw = compileProgram(compileShader(vertex_shader_draw, GL_VERTEX_SHADER), compileShader(fragment_shader_draw, GL_FRAGMENT_SHADER))

    vbo_brush = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_brush)
    glBufferData(GL_ARRAY_BUFFER, np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32).nbytes, np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32), GL_STATIC_DRAW)
    vao_brush = glGenVertexArrays(1)
    glBindVertexArray(vao_brush)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    fbo, canvas_texture = create_fbo(img_w, img_h)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    proj_matrix = np.array([[2.0/img_w,0,0,0], [0,-2.0/img_h,0,0], [0,0,-1,0], [-1,1,0,1]], dtype=np.float32)

    # --- Painting Process (runs once) ---
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, img_w, img_h)

    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT)

    glUseProgram(shader_draw)
    glUniformMatrix4fv(glGetUniformLocation(shader_draw, "u_projection"), 1, GL_FALSE, proj_matrix)
    glBindVertexArray(vao_brush)

    num_strokes = 50000
    print(f"Painting {num_strokes} strokes...")
    for _ in range(num_strokes):
        x = random.randint(0, img_w - 1)
        y = random.randint(0, img_h - 1)

        color = img_rgb[y, x]
        size = random.randint(5, 25)
        opacity = random.uniform(0.1, 0.5)

        glUniform2f(glGetUniformLocation(shader_draw, "u_pos"), x, y)
        glUniform1f(glGetUniformLocation(shader_draw, "u_size"), size)
        glUniform4f(glGetUniformLocation(shader_draw, "u_color"), color[0]/255.0, color[1]/255.0, color[2]/255.0, opacity)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("Painting complete!")

    save_fbo_to_image(fbo, canvas_texture, img_w, img_h, "output.png")

    window.close()

if __name__ == "__main__":
    main()
