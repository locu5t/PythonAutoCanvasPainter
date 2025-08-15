import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from skimage import segmentation
import os
import ctypes

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

vertex_shader_canvas = """
#version 330 core
in vec2 a_pos;
in vec2 a_tex_coord;
out vec2 v_tex_coord;
void main()
{
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_tex_coord = a_tex_coord;
}
"""

fragment_shader_canvas = """
#version 330 core
in vec2 v_tex_coord;
out vec4 FragColor;
uniform sampler2D u_canvas_texture;
void main()
{
    FragColor = texture(u_canvas_texture, v_tex_coord);
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

def draw_brush_stroke(shader, vao, pos, size, color, proj_matrix):
    glUseProgram(shader)
    glUniform2f(glGetUniformLocation(shader, "u_pos"), pos[0], pos[1])
    glUniform1f(glGetUniformLocation(shader, "u_size"), size)
    glUniform4f(glGetUniformLocation(shader, "u_color"), color[0]/255.0, color[1]/255.0, color[2]/255.0, color[3]/255.0)
    glUniformMatrix4fv(glGetUniformLocation(shader, "u_projection"), 1, GL_FALSE, proj_matrix)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

def select_image_paths():
    root = tk.Tk()
    root.withdraw()
    paths = {}
    prompts = {"background": "Select Background Image", "background_depth": "Select Background Depth Map", "person": "Select Person Image", "person_depth": "Select Person Depth Map"}
    for key, title in prompts.items():
        path = filedialog.askopenfilename(title=title, filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not path: return None
        paths[key] = path
    root.destroy()
    return paths

def load_depth_map(path, size):
    if not path or not os.path.exists(path): return None
    d = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if d is None: return None
    d = cv2.resize(d, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return 1.0 - d if np.mean(d[:d.shape[0]//3, :]) > np.mean(d[-d.shape[0]//3:, :]) else d

def depth_modulators(d):
    return np.interp(d, [0, 1], [1.25, 0.75]), np.interp(d, [0, 1], [0.85, 1.10])

def create_character_mask(person_path, bg_path, size):
    person_img = cv2.imread(person_path)
    bg_img = cv2.imread(bg_path)
    if person_img is None or bg_img is None: return None
    diff = cv2.absdiff(cv2.resize(person_img, size), cv2.resize(bg_img, size))
    _, mask = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
    return cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)

def main():
    image_paths = select_image_paths()
    if not image_paths: return

    img_ref = cv2.imread(image_paths["background"])
    img_h, img_w = img_ref.shape[:2]

    pygame.init()
    display = (img_w, img_h)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("OpenGL Accelerated Painter")

    shader_draw = compileProgram(compileShader(vertex_shader_draw, GL_VERTEX_SHADER), compileShader(fragment_shader_draw, GL_FRAGMENT_SHADER))
    shader_canvas = compileProgram(compileShader(vertex_shader_canvas, GL_VERTEX_SHADER), compileShader(fragment_shader_canvas, GL_FRAGMENT_SHADER))

    vbo_brush = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_brush)
    glBufferData(GL_ARRAY_BUFFER, np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32).nbytes, np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32), GL_STATIC_DRAW)
    vao_brush = glGenVertexArrays(1)
    glBindVertexArray(vao_brush)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    vbo_canvas = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_canvas)
    glBufferData(GL_ARRAY_BUFFER, np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype=np.float32).nbytes, np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype=np.float32), GL_STATIC_DRAW)
    vao_canvas = glGenVertexArrays(1)
    glBindVertexArray(vao_canvas)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    glEnableVertexAttribArray(1)

    fbo, canvas_texture = create_fbo(img_w, img_h)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    proj_matrix = np.array([[2.0/img_w,0,0,0], [0,-2.0/img_h,0,0], [0,0,-1,0], [-1,1,0,1]], dtype=np.float32)

    # --- Painting Process (runs once) ---
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT)

    # Phase 1: Background
    print("Painting Background...")
    image = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    depth_map = load_depth_map(image_paths["background_depth"], (img_w, img_h))
    segments = segmentation.slic(image, n_segments=1000, compactness=10)
    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        avg_color = (*np.mean(image[mask], axis=0).astype(int), 200)
        for y, x in np.argwhere(mask)[::20]:
            size, opacity = depth_modulators(depth_map[y, x] if depth_map is not None else 0.5)
            draw_brush_stroke(shader_draw, vao_brush, (x, y), 15*size, (*avg_color[:3], avg_color[3]*opacity), proj_matrix)

    # Phase 2: Character
    print("Painting Character...")
    char_mask = create_character_mask(image_paths["person"], image_paths["background"], (img_w, img_h))
    if char_mask is not None:
        person_image = cv2.cvtColor(cv2.imread(image_paths["person"]), cv2.COLOR_BGR2RGB)
        person_depth = load_depth_map(image_paths["person_depth"], (img_w, img_h))
        person_segments = segmentation.slic(person_image, n_segments=800, compactness=10)
        for seg_id in np.unique(person_segments):
            mask = (person_segments == seg_id) & (char_mask > 0)
            if not np.any(mask): continue
            avg_color = (*np.mean(person_image[mask], axis=0).astype(int), 220)
            for y, x in np.argwhere(mask)[::15]:
                size, opacity = depth_modulators(person_depth[y, x] if person_depth is not None else 0.8)
                draw_brush_stroke(shader_draw, vao_brush, (x, y), 10*size, (*avg_color[:3], avg_color[3]*opacity), proj_matrix)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print("Painting complete!")

    # --- Main Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader_canvas)
        glBindVertexArray(vao_canvas)
        glBindTexture(GL_TEXTURE_2D, canvas_texture)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
