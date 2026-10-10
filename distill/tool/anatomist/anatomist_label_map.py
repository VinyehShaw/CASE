"""
Vista3D Label Mapping — Organ name ↔ Vista3D label index.

Vista3D (NVIDIA NV-Segment-CTMR bundle) uses integer label indices
in the segmentation mask (seg_trans.nii.gz). This module maps human-readable
organ names to those indices and provides volume computation helpers.

Reference: https://github.com/Project-MONAI/VISTA
"""

# Vista3D label index → organ name (MRI_BODY relevant subset)
VISTA3D_LABEL_MAP = {
    1: "liver",
    3: "spleen",
    4: "pancreas",
    5: "kidney_right",
    6: "aorta",
    7: "inferior_vena_cava",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "gallbladder",
    11: "esophagus",
    12: "stomach",
    13: "duodenum",
    14: "kidney_left",
    15: "bladder",
    17: "portal_vein_and_splenic_vein",
    18: "rectum",
    19: "small_bowel",
    22: "brain",
    62: "colon",
    115: "heart",
    118: "prostate",
    135: "lung_left",
    136: "lung_right",
    146: "vertebrae",
    134: "intervertebral_discs",
    147: "prostate_transitional_zone",
    148: "prostate_peripheral_zone",
    149: "left_atrium",
    151: "left_ventricle",
    152: "right_ventricle",
    154: "left_ventricle_myocardium",
}

# Reverse map: organ name (lowercase, underscore) → label index
ORGAN_TO_LABEL = {v: k for k, v in VISTA3D_LABEL_MAP.items()}

# Common aliases — maps user-friendly names to canonical organ names
ORGAN_ALIASES = {
    "liver": "liver",
    "hepatic": "liver",
    "spleen": "spleen",
    "splenic": "spleen",
    "pancreas": "pancreas",
    "pancreatic": "pancreas",
    "kidney": "kidney_left",  # ambiguous, default to left
    "kidneys": "kidney_left",
    "kidney_left": "kidney_left",
    "kidney_right": "kidney_right",
    "left_kidney": "kidney_left",
    "right_kidney": "kidney_right",
    "heart": "heart",
    "cardiac": "heart",
    "myocardium": "left_ventricle_myocardium",
    "left_ventricle": "left_ventricle",
    "right_ventricle": "right_ventricle",
    "left_atrium": "left_atrium",
    "lung": "lung_left",  # ambiguous, default to left
    "lungs": "lung_left",
    "lung_left": "lung_left",
    "left_lung": "lung_left",
    "lung_right": "lung_right",
    "right_lung": "lung_right",
    "brain": "brain",
    "bladder": "bladder",
    "urinary_bladder": "bladder",
    "prostate": "prostate",
    "aorta": "aorta",
    "stomach": "stomach",
    "colon": "colon",
    "small_bowel": "small_bowel",
    "small_intestine": "small_bowel",
    "rectum": "rectum",
    "gallbladder": "gallbladder",
    "gall_bladder": "gallbladder",
    "esophagus": "esophagus",
    "duodenum": "duodenum",
    "adrenal_gland": "adrenal_gland_left",
    "left_adrenal": "adrenal_gland_left",
    "right_adrenal": "adrenal_gland_right",
    "vertebrae": "vertebrae",
    "spine": "vertebrae",
    "vertebral": "vertebrae",
}

# Body region → station mapping (approximate)
# Stations go from head (dixon_4_W) to feet (dixon_24_W)
STATION_REGIONS = {
    "dixon_4_W": "head_neck",
    "dixon_8_W": "thorax_upper",
    "dixon_12_W": "thorax_lower_abdomen_upper",
    "dixon_16_W": "abdomen",
    "dixon_20_W": "pelvis",
    "dixon_24_W": "legs",
}


def resolve_organ(name: str) -> str | None:
    """Resolve an organ name (with aliases) to canonical name.

    Returns canonical name or None if not recognized.
    """
    key = name.lower().strip().replace(" ", "_")
    if key in ORGAN_ALIASES:
        return ORGAN_ALIASES[key]
    if key in ORGAN_TO_LABEL:
        return key
    return None


def get_label_index(organ_name: str) -> int | None:
    """Get Vista3D label index for an organ name.

    Returns label index or None if not found.
    """
    canonical = resolve_organ(organ_name)
    if canonical and canonical in ORGAN_TO_LABEL:
        return ORGAN_TO_LABEL[canonical]
    return None


def get_organ_name(label_index: int) -> str | None:
    """Get organ name from Vista3D label index."""
    return VISTA3D_LABEL_MAP.get(label_index)


def compute_organ_volume(seg_mask, label_index: int) -> dict:
    """Compute volume for a specific organ from a segmentation mask.

    Args:
        seg_mask: numpy array of segmentation labels
        label_index: Vista3D label index for the organ

    Returns:
        {"voxel_count": int, "volume_ml": float} or {"voxel_count": 0, "volume_ml": 0.0}
    """
    import numpy as np

    voxel_count = int((seg_mask == label_index).sum())
    return {
        "voxel_count": voxel_count,
        "volume_ml": 0.0,  # filled by caller with voxel spacing
    }


def compute_volumes_with_spacing(seg_mask, label_index: int, spacing: tuple) -> dict:
    """Compute organ volume with physical spacing.

    Args:
        seg_mask: numpy array of segmentation labels
        label_index: Vista3D label index
        spacing: (x, y, z) voxel spacing in mm

    Returns:
        {"voxel_count": int, "volume_ml": float, "volume_cm3": float}
    """
    import numpy as np

    voxel_count = int((seg_mask == label_index).sum())
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0  # 1 ml = 1000 mm³

    return {
        "voxel_count": voxel_count,
        "volume_ml": round(volume_ml, 1),
        "volume_cm3": round(volume_mm3 / 1000.0, 1),
    }


# List of organs available for tool calling (user-facing names)
AVAILABLE_ORGANS = sorted(set(ORGAN_ALIASES.values()))
