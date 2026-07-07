import sys
import argparse
import os

# 1. Initialize Isaac Sim headless app BEFORE importing any omni modules
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep

def generate_isaac_data(num_images: int, output_dir: str, defect_type: str, part_type: str, 
                        fallback_primitive: str, fallback_scale: tuple, fallback_diffuse: tuple, fallback_metallic: float, fallback_roughness: float):
    """
    Enterprise-grade synthetic data generation using NVIDIA Isaac Sim (omni.replicator.core).
    """
    print(f"[Isaac Replicator] Initializing Replicator graph for part: {part_type}", flush=True)
    
    # Enforce absolute path to prevent Replicator from routing to C:/Users/.../omni.replicator_out
    output_dir = os.path.abspath(output_dir)
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    with rep.new_layer():
        # Force Path Traced rendering for physically accurate headless data
        rep.settings.set_render_pathtraced(samples_per_pixel=16)
        
        # Setup basic lighting
        camera = rep.create.camera(position=(0, 200, 400), look_at=(0, 0, 0))
        
        # Check if a custom CAD model exists in the directory (.usd, .usdz, .usda)
        # Intelligently check singular and plural forms to prevent frustrating mismatches
        cad_path = None
        
        base_name = part_type
        base_name_underscore = part_type.replace(" ", "_")
        
        search_names = [base_name, base_name_underscore]
        if base_name.endswith('s'):
            search_names.append(base_name[:-1])
            search_names.append(base_name_underscore[:-1])
        else:
            search_names.append(base_name + 's')
            search_names.append(base_name_underscore + 's')
            
        search_names = list(set(search_names))
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        for name in search_names:
            for ext in [".usd", ".usdz", ".usda"]:
                # Check for exact matches first
                test_path = os.path.join(project_root, f"{name}{ext}")
                if os.path.exists(test_path):
                    cad_path = test_path
                    break
            if cad_path:
                break
                
        # If no exact match, do a robust substring search in the root folder
        if cad_path is None:
            for f in os.listdir(project_root):
                if f.endswith((".usd", ".usdz", ".usda")):
                    f_name = os.path.splitext(f)[0].lower()
                    if base_name.lower() in f_name or base_name_underscore.lower() in f_name:
                        cad_path = os.path.join(project_root, f)
                        print(f"[Isaac Replicator] Robust substring match found CAD model: {f}", flush=True)
                        break
        
        # Create the AI-hallucinated Physical Material (for fallback shapes)
        ai_material = rep.create.material_omnipbr(
            diffuse=fallback_diffuse,
            metallic=fallback_metallic,
            roughness=fallback_roughness
        )
                
        if cad_path is not None:
            print(f"[Isaac Replicator] Loading custom photorealistic CAD model: {cad_path}", flush=True)
            part = rep.create.from_usd(cad_path, semantics=[('class', 'part')])
            
            # CAD models are already physically proportioned, but some raw exports are natively massive.
            # Scale down to 0.3 by default to guarantee perfect framing within the camera's FOV.
            base_scale = (0.3, 0.3, 0.3)
            base_rot = (0, 0, 0)
            z_offset = 0.0
        else:
            print(f"[Isaac Replicator] No CAD model found for {part_type}. Using AI-generated fallback: {fallback_primitive} with scale {fallback_scale}", flush=True)
            prim_str = str(fallback_primitive).lower()
            part_str = str(part_type).lower()
            
            # Intelligent Primitive Normalization:
            # LLMs output arbitrary [X, Y, Z] scales which ruin primitive proportions.
            # We mathematically force the scales to make physical sense for the chosen shape.
            max_s = max(fallback_scale)
            mid_s = sorted(fallback_scale)[1]
            
            # Compute Absolute Base State
            base_scale = fallback_scale
            base_rot = (0, 0, 0)
            z_offset = (fallback_scale[2] / 2.0) + 0.1
            
            # INTELLIGENT SEMANTIC SHAPE CLASSIFIER
            # Overrides the LLM's hallucinated primitive string based on actual industrial taxonomy
            cylindrical_keywords = ["cylinder", "pipe", "bottle", "roller", "can", "tube", "shaft", "rod", "pin", "wire", "cable", "drum", "barrel"]
            toroidal_keywords = ["torus", "gear", "ring", "washer", "tire", "wheel", "gasket", "bearing", "nut"]
            spherical_keywords = ["sphere", "ball", "marble", "knob", "droplet"]
            conical_keywords = ["cone", "funnel", "nozzle", "tip"]
            
            is_cylinder = any(k in part_str or k in prim_str for k in cylindrical_keywords)
            is_torus = any(k in part_str or k in prim_str for k in toroidal_keywords)
            is_sphere = any(k in part_str or k in prim_str for k in spherical_keywords)
            is_cone = any(k in part_str or k in prim_str for k in conical_keywords)
            
            if is_cylinder:
                # Cylinders (pipes/rollers) should be uniform in X/Y (circular base) and long in Z.
                # If (0, 90, 0) stands it upright, Isaac Sim's cylinder might be Y-up or X-up.
                # Let's force it to lie flat by applying an 89.9-degree rotation on X to avoid Gimbal lock.
                base_scale = (mid_s, mid_s, max_s)
                base_rot = (89.9, 0.0, 0.0)
                z_offset = (mid_s / 2.0) + 0.1
                part = rep.create.cylinder(scale=base_scale, position=(0, 0, z_offset), rotation=base_rot, semantics=[('class', 'part')], material=ai_material)
            elif is_torus:
                # Toruses (gears/rings) should be uniform in X/Y (circular) and mathematically thin in Z.
                # If the AI hallucinates a uniform scale like [20, 20, 20], min_s is 20, making a fat donut!
                # We forcefully crush the Z-thickness to 10% of the maximum width to guarantee a flat washer/gear.
                thickness = max_s * 0.1
                base_scale = (max_s, max_s, thickness)
                base_rot = (0, 0, 0)
                z_offset = (thickness / 2.0) + 0.1
                part = rep.create.torus(scale=base_scale, position=(0, 0, z_offset), rotation=base_rot, semantics=[('class', 'part')], material=ai_material)
            elif is_cone:
                base_scale = (mid_s, mid_s, max_s)
                base_rot = (0, 90, 0)
                z_offset = (mid_s / 2.0) + 0.1
                part = rep.create.cone(scale=base_scale, position=(0, 0, z_offset), rotation=base_rot, semantics=[('class', 'part')], material=ai_material)
            elif is_sphere:
                # Spheres must be perfectly uniform in all 3 axes.
                base_scale = (max_s, max_s, max_s)
                base_rot = (0, 0, 0)
                z_offset = (max_s / 2.0) + 0.1
                part = rep.create.sphere(scale=base_scale, position=(0, 0, z_offset), rotation=base_rot, semantics=[('class', 'part')], material=ai_material)
            else:
                # Cubes can safely accept non-uniform XYZ scales, but we must ensure they lie flat!
                # By sorting the dimensions and assigning the smallest to Z, gravity is respected.
                s_sorted = sorted(fallback_scale, reverse=True)
                base_scale = (s_sorted[0], s_sorted[1], s_sorted[2]) # Z is the smallest
                base_rot = (0, 0, 0)
                z_offset = (base_scale[2] / 2.0) + 0.1
                part = rep.create.cube(scale=base_scale, position=(0, 0, z_offset), rotation=base_rot, semantics=[('class', 'part')], material=ai_material)

        # Create a realistic factory floor plane to hide the default grid (Isaac Sim is Z-up)
        # Scale to 1000 (100 meters) and elevate slightly to Z=0.1 to perfectly avoid Z-fighting with the default grid!
        floor = rep.create.plane(scale=1000, position=(0, 0, 0.1))
        
        # Apply industrial matte floor material
        floor_mat = rep.create.material_omnipbr(
            diffuse=(0.08, 0.08, 0.08),
            roughness=0.9,
            metallic=0.2
        )
        with floor:
            rep.modify.material(floor_mat)

        # Create a directional light (Distant) for harsh shadows
        light_dir = rep.create.light(light_type="distant", intensity=2500, rotation=(315, 0, 0))
        
        # Photorealistic Factory Warehouse HDRI Background (loaded directly from the cloud)
        hdri_url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/abandoned_workshop_02_4k.hdr"
        light_amb = rep.create.light(light_type="dome", texture=hdri_url, intensity=1000)

        # Ground plane to catch shadows and provide realistic context
        floor_material = rep.create.material_omnipbr(diffuse=(0.05, 0.05, 0.05), roughness=0.9, metallic=0.1)
        floor = rep.create.plane(scale=50, position=(0, 0, 0), visible=True, material=floor_material)
        with floor:
            rep.modify.semantics([('class', 'background')])

        # Dynamically place the camera: far away for massive CAD models, closer for primitives
        if cad_path is not None:
            cam_min_bounds = (-150, -150, 100)
            cam_max_bounds = (150, 150, 250)
        else:
            cam_min_bounds = (-40, -40, 60)
            cam_max_bounds = (40, 40, 100)

        # Domain Randomization: Scatter, Tumble, Light Jitter, Camera Variations
        with rep.trigger.on_frame(max_execs=num_images):
            with camera:
                rep.modify.pose(
                    # Z is up. Dynamically framed based on CAD vs Primitive sizes.
                    position=rep.distribution.uniform(cam_min_bounds, cam_max_bounds),
                    look_at=(0, 0, 0)
                )
            with part:
                rep.modify.pose(
                    # Keep the object tightly centered to ensure it never clips outside the camera's FOV
                    position=rep.distribution.uniform((-10, -10, z_offset), (10, 10, z_offset)),
                    # Randomize Z-rotation so it faces different directions, with slight XYZ tumbling
                    # CRITICAL: We strictly jitter relative to the computed base_rot to avoid Gimbal lock
                    rotation=rep.distribution.uniform(
                        (float(base_rot[0]-1), float(base_rot[1]-1), 0.0), 
                        (float(base_rot[0]+1), float(base_rot[1]+1), 360.0)
                    ),
                    # Jitter the AI's base scale by +/- 15% to simulate differently sized defects
                    # CRITICAL: We strictly jitter relative to the computed base_scale, completely ignoring raw fallback_scale
                    scale=rep.distribution.uniform(
                        (base_scale[0]*0.85, base_scale[1]*0.85, base_scale[2]*0.85),
                        (base_scale[0]*1.15, base_scale[1]*1.15, base_scale[2]*1.15)
                    )
                )
                
                # Dynamically force the AI-generated physical material every frame
                # This works for primitives AND raw CAD models that lack native materials
                rep.randomizer.materials([ai_material])
                    
            with light_dir:
                # Clamp harsh light rotation so it always acts as a ceiling light pointing down, 
                # but moves enough to cast dynamic shadows in every frame
                rep.modify.pose(rotation=rep.distribution.uniform((230, -30, -30), (310, 30, 30)))

        # Initialize the BasicWriter which outputs images, bounding boxes, segmentation masks
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=output_dir,
            rgb=True,
            bounding_box_2d_tight=True
        )
        
        # Attach the writer to the render product (the camera)
        render_product = rep.create.render_product(camera, (512, 512))
        writer.attach([render_product])

    print(f"[Isaac Replicator] Graph constructed. Orchestrating {num_images} frames synchronously...", flush=True)
    
    # Run the orchestrator synchronously
    for i in range(num_images):
        print(f"[Isaac Replicator] Rendering frame {i+1}/{num_images}...", flush=True)
        rep.orchestrator.step(rt_subframes=1)
        # Force flush to disk
        rep.orchestrator.wait_until_complete()

    print(f"[Isaac Replicator] Successfully wrote {num_images} physical AI frames to {output_dir}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--defect_type", type=str, required=True)
    parser.add_argument("--part_type", type=str, required=True)
    parser.add_argument("--fallback_primitive", type=str, default="cube")
    parser.add_argument("--fallback_scale", type=str, default="20.0,20.0,20.0")
    parser.add_argument("--fallback_diffuse", type=str, default="0.5,0.5,0.5")
    parser.add_argument("--fallback_metallic", type=float, default=0.0)
    parser.add_argument("--fallback_roughness", type=float, default=0.5)
    args = parser.parse_args()

    # Parse comma separated strings into tuples
    try:
        scale_vals = list(map(float, args.fallback_scale.split(",")))[:3]
        while len(scale_vals) < 3:
            scale_vals.append(scale_vals[-1] if scale_vals else 20.0)
            
        # DYNAMIC SCALE NORMALIZATION ENGINE
        # Decouple from LLM absolute scale hallucinations. Treat the LLM's guess as a pure aspect ratio.
        # Mathematically normalize the vector so the maximum scale applied is 60.0.
        # Since base primitives are 1cm, a 60.0 scale yields a 60cm object which frames perfectly.
        max_dim = max(scale_vals)
        if max_dim > 0:
            scale_multiplier = 60.0 / max_dim
            scale_vals = [s * scale_multiplier for s in scale_vals]
            print(f"[Isaac Replicator] Dynamically normalized LLM scale ratio to perfectly fit camera: {scale_vals}", flush=True)
            
        scale_vals = tuple(scale_vals)
    except:
        scale_vals = (20.0, 20.0, 20.0)
        
    try:
        diffuse_vals = list(map(float, args.fallback_diffuse.split(",")))[:3]
        if len(diffuse_vals) < 3:
            diffuse_vals = [0.5, 0.5, 0.5]
            
        # DYNAMIC COLOR NORMALIZATION ENGINE
        if any(v > 1.0 for v in diffuse_vals):
            diffuse_vals = [v / 255.0 for v in diffuse_vals]
            
        diffuse_vals = tuple(max(0.0, min(1.0, v)) for v in diffuse_vals)
    except:
        diffuse_vals = (0.5, 0.5, 0.5)
        
    fallback_metallic = args.fallback_metallic
    fallback_roughness = args.fallback_roughness

    # If the LLM hallucinates generic grey/white for a rust defect, forcefully inject perfect rust physical properties.
    if "rust" in args.defect_type.lower():
        diffuse_vals = (0.6, 0.2, 0.05)
        fallback_metallic = 0.2
        fallback_roughness = 0.9
        print(f"[Isaac Replicator] HACKATHON OVERRIDE: Forced photorealistic rust material for defect: {args.defect_type}", flush=True)

    try:
        generate_isaac_data(
            args.num_images, args.output_dir, args.defect_type, args.part_type,
            args.fallback_primitive, scale_vals, diffuse_vals, fallback_metallic, fallback_roughness
        )
    except Exception as e:
        print(f"[Isaac Replicator Error] {e}", flush=True)
        sys.exit(1)
    finally:
        # Crucial: gracefully shutdown Isaac Sim to free up GPU memory
        simulation_app.close()
