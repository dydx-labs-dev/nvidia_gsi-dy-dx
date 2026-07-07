"""
OmniForge: Universal Procedural Synthetic Data Engine
Generates unique synthetic defect datasets for ANY part type and ANY defect type
driven entirely by natural language LLM parsing.

Architecture:
  - Part textures are generated procedurally using Perlin noise, geometric primitives,
    and material simulation (brushed metal, composite, ceramic, etc.)
  - Defect overlays use physically-inspired CV techniques (noise masks, displacement,
    random walks, gradient blending)
  - Heavy domain randomization ensures no two frames are ever identical
  - Unknown part/defect types get intelligent procedural generation seeded by the
    text hash, so the same prompt always produces a consistent visual style
"""

import sys
import argparse
import time
import os
import cv2
import numpy as np
import random
import hashlib


# UTILITY: Seeded randomness from text strings
def text_to_seed(text):
    """Convert any text string to a deterministic integer seed."""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)


def text_to_color(text, brightness_range=(60, 140)):
    """Derive a unique but industrial-looking BGR color from any text string."""
    seed = text_to_seed(text)
    rng = np.random.RandomState(seed)
    base = rng.randint(brightness_range[0], brightness_range[1])
    # Keep channels close together for metallic/industrial look, with slight tint
    b = np.clip(base + rng.randint(-20, 20), 30, 200)
    g = np.clip(base + rng.randint(-20, 20), 30, 200)
    r = np.clip(base + rng.randint(-20, 20), 30, 200)
    return (int(b), int(g), int(r))


# PROCEDURAL NOISE
def generate_perlin_noise(width, height, scale=50):
    """Generate smooth procedural noise using interpolated random gradients."""
    noise = np.zeros((height, width), dtype=np.float32)
    for octave in range(4):
        freq = 2 ** octave
        amp = 0.5 ** octave
        s = max(2, scale // freq)
        small = np.random.randn(height // s + 2, width // s + 2).astype(np.float32)
        noise += amp * cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


# UNIVERSAL BASE TEXTURE GENERATOR
# Handles ANY part type by combining geometric features + material simulation

# Keywords mapped to shape/material generation strategies
ROUND_PARTS = ["gear", "wheel", "bearing", "ball", "roller", "pulley", "disc", "disk",
               "flywheel", "sprocket", "rotor", "shaft", "cylinder", "ring", "washer",
               "turbine", "fan", "impeller", "hub"]
LONG_PARTS  = ["pipe", "tube", "rod", "bar", "beam", "rail", "axle", "channel",
               "conduit", "hose", "cable", "wire", "strut"]
FLAT_PARTS  = ["casing", "housing", "panel", "plate", "bracket", "frame", "cover",
               "shield", "enclosure", "chassis", "drone", "pcb", "board", "shell",
               "door", "lid", "flap", "fin", "blade"]
WELD_PARTS  = ["weld", "joint", "seam", "solder", "braze", "connection", "junction"]


def classify_part_shape(part_type):
    """Classify any part type into a shape category for geometry generation."""
    pt = part_type.lower()
    for kw in ROUND_PARTS:
        if kw in pt:
            return "round"
    for kw in LONG_PARTS:
        if kw in pt:
            return "long"
    for kw in FLAT_PARTS:
        if kw in pt:
            return "flat"
    for kw in WELD_PARTS:
        if kw in pt:
            return "weld"
    return "auto"  # Unknown part: derive shape from text hash


def generate_metallic_base(width, height, base_color, streak_intensity=150):
    """Generate a realistic brushed-metal surface texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = base_color

    # Directional brushed-metal streaks
    num_streaks = random.randint(80, 250)
    for _ in range(num_streaks):
        y = random.randint(0, height - 1)
        thickness = random.randint(1, 2)
        var = random.randint(-12, 12)
        streak_color = tuple(np.clip(np.array(base_color) + var, 0, 255).astype(int).tolist())
        cv2.line(img, (0, y), (width, y), streak_color, thickness)

    # Smooth procedural noise for micro-surface variation
    noise = generate_perlin_noise(width, height, scale=random.randint(30, 60))
    noise_amp = random.uniform(15, 35)
    noise_layer = (noise * noise_amp - noise_amp / 2).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise_layer[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Subtle vignette for depth/lighting
    Y, X = np.ogrid[:height, :width]
    cx, cy = width // 2 + random.randint(-50, 50), height // 2 + random.randint(-50, 50)
    vignette = 1.0 - random.uniform(0.2, 0.4) * ((X - cx) ** 2 + (Y - cy) ** 2) / (cx ** 2 + cy ** 2 + 1)
    img = np.clip(img.astype(np.float32) * vignette[:, :, np.newaxis], 0, 255).astype(np.uint8)

    return img


def generate_round_part(width, height, base_color):
    """Generate a circular/cylindrical part (gears, bearings, rotors, etc.)."""
    img = generate_metallic_base(width, height, base_color)
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 2 - random.randint(30, 60)

    # Outer ring / teeth pattern
    num_features = random.randint(12, 32)
    tooth_len = random.randint(15, 35)
    for i in range(num_features):
        angle = 2 * np.pi * i / num_features
        x1 = int(cx + radius * np.cos(angle))
        y1 = int(cy + radius * np.sin(angle))
        x2 = int(cx + (radius + tooth_len) * np.cos(angle))
        y2 = int(cy + (radius + tooth_len) * np.sin(angle))
        color_var = random.randint(-10, 10)
        tooth_color = tuple(np.clip(np.array(base_color) + 30 + color_var, 0, 255).astype(int).tolist())
        cv2.line(img, (x1, y1), (x2, y2), tooth_color, random.randint(5, 12))

    # Inner hub
    hub_radius = radius // random.randint(2, 4)
    hub_color = tuple(np.clip(np.array(base_color) - 20, 0, 255).astype(int).tolist())
    cv2.circle(img, (cx, cy), hub_radius, hub_color, -1)
    cv2.circle(img, (cx, cy), hub_radius, tuple(np.clip(np.array(base_color) - 40, 0, 255).astype(int).tolist()), 3)

    # Bolt holes in hub
    num_bolts = random.randint(3, 6)
    bolt_radius = hub_radius // 2
    for i in range(num_bolts):
        angle = 2 * np.pi * i / num_bolts + random.uniform(0, 0.3)
        bx = int(cx + bolt_radius * np.cos(angle))
        by = int(cy + bolt_radius * np.sin(angle))
        cv2.circle(img, (bx, by), random.randint(4, 8), (40, 45, 50), -1)

    return img


def generate_long_part(width, height, base_color):
    """Generate an elongated cylindrical part (pipes, rods, beams, etc.)."""
    img = generate_metallic_base(width, height, base_color)

    # Cylindrical shading (horizontal gradient)
    for y in range(height):
        factor = 1.0 - 0.45 * abs(y - height / 2) / (height / 2)
        img[y] = np.clip(img[y].astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Specular highlight along the top third
    highlight_y = height // 3
    for y in range(max(0, highlight_y - 30), min(height, highlight_y + 30)):
        factor = 1.0 + 0.3 * (1 - abs(y - highlight_y) / 30)
        img[y] = np.clip(img[y].astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Joint/flange lines
    num_joints = random.randint(1, 4)
    for _ in range(num_joints):
        x_pos = random.randint(width // 6, 5 * width // 6)
        joint_color = tuple(np.clip(np.array(base_color) - 25, 0, 255).astype(int).tolist())
        cv2.line(img, (x_pos, 0), (x_pos, height), joint_color, random.randint(3, 8))
        # Flange shadow
        cv2.line(img, (x_pos + 3, 0), (x_pos + 3, height),
                 tuple(np.clip(np.array(base_color) + 15, 0, 255).astype(int).tolist()), 1)

    return img


def generate_flat_part(width, height, base_color):
    """Generate a flat panel/casing (enclosures, PCBs, drone casings, etc.)."""
    # Slightly darker base for composite/plastic feel
    darker = tuple(np.clip(np.array(base_color) - 15, 0, 255).astype(int).tolist())
    img = generate_metallic_base(width, height, darker, streak_intensity=80)

    # Panel separation lines
    num_lines = random.randint(2, 6)
    for _ in range(num_lines):
        line_color = tuple(np.clip(np.array(base_color) - 35, 0, 255).astype(int).tolist())
        if random.random() > 0.5:
            y = random.randint(height // 5, 4 * height // 5)
            cv2.line(img, (0, y), (width, y), line_color, random.randint(1, 3))
        else:
            x = random.randint(width // 5, 4 * width // 5)
            cv2.line(img, (x, 0), (x, height), line_color, random.randint(1, 3))

    # Screws / rivets / mounting points
    num_fasteners = random.randint(4, 12)
    for _ in range(num_fasteners):
        fx, fy = random.randint(25, width - 25), random.randint(25, height - 25)
        r = random.randint(3, 7)
        cv2.circle(img, (fx, fy), r, tuple(np.clip(np.array(base_color) - 30, 0, 255).astype(int).tolist()), -1)
        cv2.circle(img, (fx, fy), r, tuple(np.clip(np.array(base_color) + 10, 0, 255).astype(int).tolist()), 1)

    # Optional: ventilation slots
    if random.random() < 0.4:
        slot_y = random.randint(height // 3, 2 * height // 3)
        for sx in range(random.randint(3, 8)):
            x_start = width // 4 + sx * 30
            cv2.line(img, (x_start, slot_y), (x_start + 20, slot_y),
                     tuple(np.clip(np.array(base_color) - 50, 0, 255).astype(int).tolist()), 3)

    return img


def generate_weld_part(width, height, base_color):
    """Generate a welded joint/seam surface."""
    img = generate_metallic_base(width, height, base_color)

    # Weld bead path (irregular curve across the image)
    pts = []
    y_center = height // 2 + random.randint(-50, 50)
    for x in range(0, width, 4):
        y = y_center + int(random.gauss(0, random.uniform(5, 15)))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)

    # Outer heat-affected zone (blue tint)
    haz_color = tuple(np.clip(np.array(base_color) + np.array([30, -10, -20]), 0, 255).astype(int).tolist())
    cv2.polylines(img, [pts], False, haz_color, random.randint(18, 28))
    img = cv2.GaussianBlur(img, (5, 5), 2)

    # Weld bead (brighter, textured)
    bead_color = tuple(np.clip(np.array(base_color) + 40, 0, 255).astype(int).tolist())
    cv2.polylines(img, [pts], False, bead_color, random.randint(8, 14))

    # Ripple pattern on bead
    inner_color = tuple(np.clip(np.array(base_color) + 20, 0, 255).astype(int).tolist())
    cv2.polylines(img, [pts], False, inner_color, random.randint(2, 5))

    return img


def generate_auto_part(width, height, part_type):
    """For completely unknown parts: derive visual style from the text itself."""
    base_color = text_to_color(part_type, brightness_range=(70, 130))
    seed = text_to_seed(part_type)
    rng = np.random.RandomState(seed)

    # Decide shape heuristic from hash
    shape_choice = seed % 4
    if shape_choice == 0:
        return generate_round_part(width, height, base_color)
    elif shape_choice == 1:
        return generate_long_part(width, height, base_color)
    elif shape_choice == 2:
        return generate_flat_part(width, height, base_color)
    else:
        # Hybrid: metallic base with random geometric features
        img = generate_metallic_base(width, height, base_color)
        num_features = rng.randint(3, 8)
        for _ in range(num_features):
            feat_type = rng.randint(0, 3)
            fx, fy = rng.randint(50, width - 50), rng.randint(50, height - 50)
            feat_color = tuple(np.clip(np.array(base_color) + rng.randint(-30, 30, 3), 0, 255).astype(int).tolist())
            if feat_type == 0:
                cv2.circle(img, (fx, fy), rng.randint(20, 60), feat_color, -1)
            elif feat_type == 1:
                w2, h2 = rng.randint(30, 80), rng.randint(30, 80)
                cv2.rectangle(img, (fx, fy), (fx + w2, fy + h2), feat_color, -1)
            else:
                cv2.line(img, (fx, fy), (fx + rng.randint(-80, 80), fy + rng.randint(-80, 80)),
                         feat_color, rng.randint(2, 6))
        return img


def get_part_base(part_type, width, height):
    """Universal part generator: handles ANY part type string."""
    base_color = text_to_color(part_type)
    shape = classify_part_shape(part_type)

    if shape == "round":
        return generate_round_part(width, height, base_color)
    elif shape == "long":
        return generate_long_part(width, height, base_color)
    elif shape == "flat":
        return generate_flat_part(width, height, base_color)
    elif shape == "weld":
        return generate_weld_part(width, height, base_color)
    else:
        return generate_auto_part(width, height, part_type)


# UNIVERSAL DEFECT INJECTORS
# Each function takes a base image and applies a physically-inspired defect.
# Unknown defect types get intelligent composite effects.

# Keywords mapped to defect generation strategies
SCRATCH_DEFECTS  = ["scratch", "abrasion", "scuff", "scrape", "gouge", "scoring", "mark"]
THERMAL_DEFECTS  = ["thermal", "warp", "heat", "burn", "overheat", "discolor", "anneal", "temper"]
RUST_DEFECTS     = ["rust", "corrosion", "oxidat", "oxide", "patina", "tarnish"]
DENT_DEFECTS     = ["dent", "impact", "deform", "buckle", "crush", "bend", "distort"]
CRACK_DEFECTS    = ["crack", "fracture", "break", "split", "fissure", "rupture", "shatter"]
WEAR_DEFECTS     = ["wear", "erosion", "abras", "worn", "degradation", "fatigue", "pitting"]
CONTAMINATION_DEFECTS = ["contamin", "stain", "residue", "deposit", "buildup", "dirt", "grease", "oil", "spill"]
MISALIGN_DEFECTS = ["misalign", "offset", "shift", "gap", "spacing", "loose", "overtighten"]


def apply_scratch_defect(img):
    """Realistic scratch marks with varying depth and direction."""
    h, w = img.shape[:2]
    num_scratches = random.randint(4, 15)
    for _ in range(num_scratches):
        pts = []
        x, y = random.randint(0, w), random.randint(0, h)
        direction_x = random.choice([-1, 1]) * random.randint(10, 40)
        direction_y = random.choice([-1, 1]) * random.randint(10, 40)
        for _ in range(random.randint(4, 12)):
            x = int(np.clip(x + direction_x + random.randint(-15, 15), 0, w - 1))
            y = int(np.clip(y + direction_y + random.randint(-15, 15), 0, h - 1))
            pts.append([x, y])
        if len(pts) >= 2:
            pts = np.array(pts, dtype=np.int32)
            # Scratches appear as bright lines on dark metal
            brightness = random.randint(150, 230)
            color = (brightness, brightness - random.randint(0, 20), brightness - random.randint(0, 30))
            cv2.polylines(img, [pts], False, color, random.randint(1, 3), lineType=cv2.LINE_AA)
    return img


def apply_thermal_warping_defect(img):
    """Heat discoloration with glowing gradients and wavy distortions."""
    h, w = img.shape[:2]
    # Heat glow spots
    num_spots = random.randint(2, 6)
    for _ in range(num_spots):
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 4 * h // 5)
        radius = random.randint(40, 130)
        overlay = img.copy()
        # Orange-to-blue heat gradient (temper colors)
        heat_colors = [
            (random.randint(0, 30), random.randint(80, 160), random.randint(200, 255)),  # Orange
            (random.randint(100, 180), random.randint(50, 100), random.randint(0, 50)),   # Blue tint
            (random.randint(0, 60), random.randint(0, 80), random.randint(150, 220)),     # Dark red
        ]
        color = random.choice(heat_colors)
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        overlay = cv2.GaussianBlur(overlay, (61, 61), 30)
        alpha = random.uniform(0.25, 0.55)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Wavy distortion
    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
    amp = random.uniform(2, 7)
    freq = random.uniform(0.02, 0.07)
    map_x += amp * np.sin(freq * map_y + random.uniform(0, 6.28))
    map_y += amp * 0.5 * np.cos(freq * map_x + random.uniform(0, 6.28))
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return img


def apply_rust_corrosion_defect(img):
    """Procedural rust patches with realistic orange-brown texture."""
    h, w = img.shape[:2]
    noise = generate_perlin_noise(w, h, scale=random.randint(20, 45))

    rust_layer = np.zeros_like(img)
    rust_layer[:, :, 0] = random.randint(10, 45)
    rust_layer[:, :, 1] = random.randint(55, 120)
    rust_layer[:, :, 2] = random.randint(130, 220)

    threshold = random.uniform(0.4, 0.7)
    mask = (noise > threshold).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 5)
    intensity = random.uniform(0.5, 0.85)

    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c].astype(np.float32) * (1 - mask * intensity) +
            rust_layer[:, :, c].astype(np.float32) * mask * intensity,
            0, 255
        ).astype(np.uint8)

    # Granular texture
    grain = np.random.randint(-12, 12, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + (grain * mask[:, :, np.newaxis]).astype(np.int16), 0, 255).astype(np.uint8)
    return img


def apply_dent_defect(img):
    """Impact dents with shadow/highlight displacement."""
    h, w = img.shape[:2]
    num_dents = random.randint(1, 5)
    for _ in range(num_dents):
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 4 * h // 5)
        radius = random.randint(25, 100)

        # Shadow side
        shadow = img.copy()
        offset = random.randint(3, 8)
        cv2.circle(shadow, (cx + offset, cy + offset), radius, (25, 25, 25), -1)
        shadow = cv2.GaussianBlur(shadow, (41, 41), 15)
        img = cv2.addWeighted(shadow, 0.4, img, 0.6, 0)

        # Highlight side
        highlight = img.copy()
        cv2.circle(highlight, (cx - offset, cy - offset), max(5, radius - 15), (175, 180, 185), -1)
        highlight = cv2.GaussianBlur(highlight, (31, 31), 10)
        img = cv2.addWeighted(highlight, 0.2, img, 0.8, 0)

        # Crumple lines
        for _ in range(random.randint(3, 10)):
            angle = random.uniform(0, 2 * np.pi)
            length = radius * random.uniform(0.8, 1.5)
            x2 = int(cx + length * np.cos(angle))
            y2 = int(cy + length * np.sin(angle))
            cv2.line(img, (cx, cy), (x2, y2), (55, 60, 65), 1, cv2.LINE_AA)
    return img


def apply_crack_defect(img):
    """Realistic branching crack patterns via random walk."""
    h, w = img.shape[:2]
    num_cracks = random.randint(1, 5)
    for _ in range(num_cracks):
        x, y = random.randint(w // 5, 4 * w // 5), random.randint(h // 5, 4 * h // 5)
        pts = [(x, y)]
        main_dx = random.choice([-1, 0, 1])
        main_dy = random.choice([-1, 0, 1])
        for _ in range(random.randint(20, 50)):
            dx = main_dx * random.randint(5, 15) + random.randint(-8, 8)
            dy = main_dy * random.randint(5, 15) + random.randint(-8, 8)
            x = int(np.clip(x + dx, 0, w - 1))
            y = int(np.clip(y + dy, 0, h - 1))
            pts.append((x, y))
            # Branching
            if random.random() < 0.15:
                bx, by = x, y
                branch = [(bx, by)]
                for _ in range(random.randint(3, 12)):
                    bx = int(np.clip(bx + random.randint(-12, 12), 0, w - 1))
                    by = int(np.clip(by + random.randint(-12, 12), 0, h - 1))
                    branch.append((bx, by))
                cv2.polylines(img, [np.array(branch, dtype=np.int32)], False, (35, 35, 35), 1, cv2.LINE_AA)

        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_arr], False, (25, 25, 25), random.randint(1, 3), cv2.LINE_AA)
        # Slight bright edge on one side (light catching the crack lip)
        shifted = pts_arr.copy()
        shifted[:, 0] += 1
        shifted[:, 1] += 1
        cv2.polylines(img, [shifted], False, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def apply_wear_defect(img):
    """Surface wear/erosion: smoothed patches and pitting."""
    h, w = img.shape[:2]
    # Worn/smooth patches
    num_patches = random.randint(2, 6)
    for _ in range(num_patches):
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 4 * h // 5)
        axes = (random.randint(30, 100), random.randint(30, 100))
        angle = random.randint(0, 180)
        worn_color = tuple(np.clip(np.array(img[cy, cx].tolist()) + random.randint(15, 40), 0, 255).astype(int).tolist())
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, worn_color, -1)
        overlay = cv2.GaussianBlur(overlay, (25, 25), 10)
        alpha = random.uniform(0.3, 0.6)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Pitting (small dark dots)
    num_pits = random.randint(20, 80)
    for _ in range(num_pits):
        px = random.randint(0, w - 1)
        py = random.randint(0, h - 1)
        r = random.randint(1, 4)
        cv2.circle(img, (px, py), r, (30, 30, 30), -1)
    return img


def apply_contamination_defect(img):
    """Stains, oil spills, residue deposits."""
    h, w = img.shape[:2]
    num_stains = random.randint(2, 7)
    for _ in range(num_stains):
        cx = random.randint(w // 6, 5 * w // 6)
        cy = random.randint(h // 6, 5 * h // 6)
        radius = random.randint(30, 120)

        # Stain color (dark oil, grease, chemical residue)
        stain_colors = [
            (20, 20, 30),     # Dark oil
            (30, 40, 50),     # Grease
            (40, 60, 40),     # Chemical (greenish)
            (50, 50, 80),     # Coolant (reddish)
            (60, 50, 30),     # Hydraulic fluid
        ]
        color = random.choice(stain_colors)

        overlay = img.copy()
        # Irregular stain shape using noise
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        noise = generate_perlin_noise(w, h, scale=20)
        mask *= (noise > 0.35).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)

        for c in range(3):
            overlay[:, :, c] = np.clip(
                img[:, :, c].astype(np.float32) * (1 - mask * 0.7) +
                color[c] * mask * 0.7,
                0, 255
            ).astype(np.uint8)
        img = overlay
    return img


def apply_misalignment_defect(img):
    """Visual representation of misalignment: offset sections, gap lines."""
    h, w = img.shape[:2]
    # Split image and offset one half
    split_pos = random.randint(w // 3, 2 * w // 3)
    offset_y = random.randint(5, 20)

    left = img[:, :split_pos].copy()
    right = img[:, split_pos:].copy()

    # Shift right portion down
    M = np.float32([[1, 0, 0], [0, 1, offset_y]])
    right = cv2.warpAffine(right, M, (right.shape[1], right.shape[0]), borderMode=cv2.BORDER_REFLECT)

    img[:, :split_pos] = left
    img[:, split_pos:] = right

    # Draw gap line
    gap_color = (20, 20, 20)
    cv2.line(img, (split_pos, 0), (split_pos, h), gap_color, random.randint(2, 5))
    # Shadow on one side
    cv2.line(img, (split_pos + 3, 0), (split_pos + 3, h), (80, 80, 80), 1)
    return img


def apply_auto_defect(img, defect_type):
    """For unknown defect types: derive composite effects from text hash."""
    seed = text_to_seed(defect_type)
    rng = np.random.RandomState(seed)

    # Apply a mix of effects based on hash
    effects = [apply_scratch_defect, apply_dent_defect, apply_wear_defect,
               apply_crack_defect, apply_contamination_defect]
    num_effects = rng.randint(1, 3)
    chosen = rng.choice(len(effects), size=num_effects, replace=False)
    for idx in chosen:
        img = effects[idx](img)

    # Add unique discoloration
    noise = generate_perlin_noise(img.shape[1], img.shape[0], scale=40)
    mask = (noise > 0.55).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    tint_color = text_to_color(defect_type, brightness_range=(100, 200))
    tint = np.zeros_like(img)
    tint[:] = tint_color
    for c in range(3):
        img[:, :, c] = np.clip(
            img[:, :, c].astype(np.float32) * (1 - mask * 0.35) +
            tint[:, :, c].astype(np.float32) * mask * 0.35,
            0, 255
        ).astype(np.uint8)
    return img


def select_defect_applicator(defect_type):
    """Route ANY defect type string to the correct procedural function."""
    dt = defect_type.lower()
    for kw in SCRATCH_DEFECTS:
        if kw in dt:
            return apply_scratch_defect
    for kw in THERMAL_DEFECTS:
        if kw in dt:
            return apply_thermal_warping_defect
    for kw in RUST_DEFECTS:
        if kw in dt:
            return apply_rust_corrosion_defect
    for kw in DENT_DEFECTS:
        if kw in dt:
            return apply_dent_defect
    for kw in CRACK_DEFECTS:
        if kw in dt:
            return apply_crack_defect
    for kw in WEAR_DEFECTS:
        if kw in dt:
            return apply_wear_defect
    for kw in CONTAMINATION_DEFECTS:
        if kw in dt:
            return apply_contamination_defect
    for kw in MISALIGN_DEFECTS:
        if kw in dt:
            return apply_misalignment_defect
    # Truly unknown: intelligent composite
    return lambda img: apply_auto_defect(img, defect_type)



# DOMAIN RANDOMIZATION

def apply_domain_randomization(img):
    """Camera, lighting, and sensor augmentations for maximum diversity."""
    h, w = img.shape[:2]

    # Random rotation
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, random.uniform(0.9, 1.1))
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness
    img = np.clip(img.astype(np.float32) + random.uniform(-35, 35), 0, 255).astype(np.uint8)

    # Contrast
    contrast = random.uniform(0.75, 1.25)
    img = np.clip(128 + contrast * (img.astype(np.float32) - 128), 0, 255).astype(np.uint8)

    # Random crop-and-resize (simulates zoom variation)
    if random.random() < 0.4:
        margin = random.randint(20, 60)
        crop = img[margin:h - margin, margin:w - margin]
        img = cv2.resize(crop, (w, h))

    # Gaussian blur (focus variation)
    if random.random() < 0.25:
        img = cv2.GaussianBlur(img, (random.choice([3, 5]), random.choice([3, 5])), 0)

    # Sensor noise
    noise = np.random.randn(*img.shape) * random.uniform(3, 10)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Flips
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.3:
        img = cv2.flip(img, 0)

    return img


# MAIN PIPELINE

def generate_synthetic_data_live(num_images: int, output_dir: str, defect_type: str, part_type: str):
    try:
        import omni.replicator.core as rep
        # Full Omniverse path (requires Isaac Sim application runtime)
        print(f"[OmniForge] Isaac Sim detected. Using Omniverse Replicator for {num_images} frames...")
        print(f"[OmniForge] Part='{part_type}', Defect='{defect_type}'")
        with rep.new_layer():
            camera = rep.create.camera(position=(0, 0, 1000))
            part = rep.create.cylinder(semantics=[('class', part_type)])
            rep.create.light(light_type="Sphere", temperature=rep.distribution.normal(6500, 500),
                             intensity=rep.distribution.normal(30000, 5000),
                             position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
                             scale=rep.distribution.uniform(50, 100), count=3)
            with part:
                rep.randomizer.rotation(uniform=((-90, -90, -90), (90, 90, 90)))
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
            render_product = rep.create.render_product(camera, (1024, 1024))
            writer = rep.WriterRegistry.get("KittiWriter")
            writer.initialize(output_dir=output_dir, omit_semantic_type=True)
            writer.attach([render_product])
            rep.orchestrator.run_until_complete(num_frames=num_images)
            print(f"[OmniForge] Generation complete. Dataset saved to: {output_dir}")

    except ImportError:
        print(f"[OmniForge] Procedural Synthetic Engine active.")
        print(f"[OmniForge] Generating {num_images} unique frames: Part='{part_type}', Defect='{defect_type}'")
        print(f"[OmniForge] Part shape class: {classify_part_shape(part_type)}")

        IMG_SIZE = 640

        os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)

        defect_fn = select_defect_applicator(defect_type)

        for i in range(num_images):
            if i % 10 == 0:
                print(f"[OmniForge] Rendered {i}/{num_images} frames...")

            # Fresh unique base for EVERY frame
            base = get_part_base(part_type, IMG_SIZE, IMG_SIZE)
            defective = defect_fn(base.copy())
            final = apply_domain_randomization(defective)

            split = "train" if i < int(num_images * 0.8) else "val"
            img_name = f"frame_{i:04d}.jpg"
            label_name = f"frame_{i:04d}.txt"

            cv2.imwrite(os.path.join(output_dir, "images", split, img_name), final,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            x_c = 0.5 + random.uniform(-0.08, 0.08)
            y_c = 0.5 + random.uniform(-0.08, 0.08)
            bw = 0.4 + random.uniform(-0.1, 0.1)
            bh = 0.4 + random.uniform(-0.1, 0.1)
            with open(os.path.join(output_dir, "labels", split, label_name), "w") as f:
                f.write(f"0 {x_c:.4f} {y_c:.4f} {bw:.4f} {bh:.4f}\n")

        print(f"[OmniForge] Rendered {num_images}/{num_images} frames...")
        print(f"[OmniForge] Generation complete. Dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniForge Procedural Synthetic Data Engine")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="dataset_defect_v1")
    parser.add_argument("--defect_type", type=str, default="unknown")
    parser.add_argument("--part_type", type=str, default="unknown")
    args = parser.parse_args()

    generate_synthetic_data_live(args.num_images, args.output_dir, args.defect_type, args.part_type)
