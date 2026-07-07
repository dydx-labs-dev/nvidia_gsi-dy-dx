# OmniForge
**The Agentic Engine for Physical AI Manufacturing**
*(Bridging Sim-to-Real through Autonomous Orchestration and Synthetic Data)*

OmniForge is an end-to-end, decoupled software pipeline that acts as the intelligent orchestration layer for industrial digital twins. Instead of writing manual code to update a factory inspection workflow, you simply prompt an Agentic AI with a natural language command.

## Key Features

- **Natural Language → Structured Execution**: Type any factory floor command and the NVIDIA NIM Cloud LLM (Llama-3.1-8B) parses it into a structured JSON execution plan
- **Universal Procedural Synthetic Data Engine**: Generates hundreds of unique, varied synthetic defect images for ANY part type (gears, pipes, casings, bearings, turbines, PCBs, etc.) and ANY defect type (scratches, rust, thermal warping, dents, cracks, wear, contamination, misalignment, etc.)
- **Edge AI Training**: Automatically fine-tunes a YOLOv8-nano model on the generated dataset using local GPU (CUDA)
- **Sim-to-Real Deployment**: Broadcasts inspection parameters to physical factory robots via ROS 2

## Architecture

```text
User Prompt ──► LangGraph Orchestrator (NVIDIA NIM Llama 3.1) ──► Structured Intent
                                                                       │
                    ┌──────────────────────────────────────────────────┤
                    ▼                                                  ▼
      NVIDIA Isaac Sim & Omniverse Replicator            YOLOv8 Edge Fine-Tuning (CUDA)
      (Ray-Traced Synthetic Data Generation)             (Ultralytics Transfer Learning)
      Generates N unique defect frames                           │
      using matched CAD models or AI primitives                  ▼
                                                  ROS 2 Hardware Bridge (rclpy)
                                                  Deploys physical robot kinematics 
                                                  and YOLO network to hardware
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your NVIDIA API key in .env
echo NVIDIA_API_KEY=nvapi-your_key_here > .env

# 3. Launch the dashboard
streamlit run app.py
```

## NVIDIA Technologies Used

| Technology | Purpose |
|-----------|---------|
| **NVIDIA NIM** | Cloud-hosted Llama-3.1-8B for intent parsing |
| **NVIDIA Omniverse Replicator** | Production-grade synthetic data generation (when Isaac Sim is installed) |
| **NVIDIA CUDA** | GPU-accelerated YOLOv8 training on RTX 3070 Ti |
| **LangChain + LangGraph** | Agentic AI orchestration framework |

## Example Prompts

- *"Generate 50 images of scratched gears and train a model"*
- *"Simulate 100 rusty factory pipes for inspection"*
- *"Create 75 dented drone casings and deploy to robot"*
- *"Detect oil contamination on conveyor rollers"*
- *"Find cracked weld joints on factory seams"*
