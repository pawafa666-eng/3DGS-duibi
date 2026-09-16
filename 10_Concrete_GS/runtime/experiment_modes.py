EXPERIMENT_MODE_SETTINGS = {
    "A": {
        "name": "standard_3dgs",
        "use_2d_gaussian": False,
        "use_normal_alignment": False,
        "use_opacity_regularization": False,
        "freeze_ground_geometry": False,
        "freeze_z_scale": False,
        "freeze_all_xyz": False,
        "disable_densification": False,
    },
    "B": {
        "name": "lidar_3dgs",
        "use_2d_gaussian": False,
        "use_normal_alignment": False,
        "use_opacity_regularization": False,
        "freeze_ground_geometry": False,
        "freeze_z_scale": False,
        "freeze_all_xyz": False,
        "disable_densification": False,
    },
    "C": {
        "name": "lidar_2dgs",
        "use_2d_gaussian": True,
        "use_normal_alignment": False,
        "use_opacity_regularization": False,
        "freeze_ground_geometry": False,
        "freeze_z_scale": True,
        "freeze_all_xyz": False,
        "disable_densification": False,
    },
    "D": {
        "name": "lidar_2dgs_normal",
        "use_2d_gaussian": True,
        "use_normal_alignment": True,
        "use_opacity_regularization": False,
        "freeze_ground_geometry": False,
        "freeze_z_scale": True,
        "freeze_all_xyz": False,
        "disable_densification": False,
    },
    "E": {
        "name": "constrained_2dgs",
        "use_2d_gaussian": True,
        "use_normal_alignment": True,
        "use_opacity_regularization": True,
        "freeze_ground_geometry": False,
        "freeze_z_scale": True,
        "freeze_all_xyz": True,
        "disable_densification": False,
    },
    "F": {
        "name": "concrete_gs",
        "use_2d_gaussian": True,
        "use_normal_alignment": True,
        "use_opacity_regularization": True,
        "freeze_ground_geometry": True,
        "freeze_z_scale": True,
        "freeze_all_xyz": True,
        "disable_densification": True,
    },
}


def get_experiment_settings(mode):
    normalized_mode = str(mode).upper()
    if normalized_mode not in EXPERIMENT_MODE_SETTINGS:
        known = " ".join(EXPERIMENT_MODE_SETTINGS)
        raise ValueError(f"Unknown experiment_mode '{mode}'. Known internal modes: {known}")
    return EXPERIMENT_MODE_SETTINGS[normalized_mode]
