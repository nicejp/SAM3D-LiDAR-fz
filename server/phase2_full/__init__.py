# Phase 2 Full Pipeline
from .sam3_segmentation import (
    SAM3Segmentor,
    apply_mask_to_depth,
    masked_depth_to_pointcloud,
    process_session_with_sam3,
    HAS_SAM3
)
from .click_selector import select_clicks, get_session_frame
from .claude_interface import (
    generate_blender_code_claude,
    analyze_image,
    get_session_images,
    check_api_available as check_claude_available
)
from .pipeline import run_phase2_pipeline

__all__ = [
    # SAM 3
    "SAM3Segmentor",
    "apply_mask_to_depth",
    "masked_depth_to_pointcloud",
    "process_session_with_sam3",
    "HAS_SAM3",
    # Click selector
    "select_clicks",
    "get_session_frame",
    # Claude
    "generate_blender_code_claude",
    "analyze_image",
    "get_session_images",
    "check_claude_available",
    # Pipeline
    "run_phase2_pipeline"
]
