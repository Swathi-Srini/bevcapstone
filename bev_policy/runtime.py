"""Shared MetaDrive configuration for the BEV BC collector and evaluator."""

from __future__ import annotations


def make_env(
    *, use_idm: bool, render: bool, traffic_density: float, horizon: int,
    composite_view: bool = False,
    num_scenarios: int = 10000, start_seed: int = 10,
):
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.policy.idm_policy import IDMPolicy
    from manual_drive_stereo_yolo_weather import CAMERA_RIGS, CAM_H, CAM_W

    sensors = {name: (RGBCamera, CAM_W, CAM_H) for name in CAMERA_RIGS}
    if composite_view:
        sensors["main_camera"] = ()
    config = {
        "manual_control": False, "use_render": render, "image_observation": True, "norm_pixel": False,
        "traffic_density": traffic_density, "num_scenarios": num_scenarios,
        "start_seed": start_seed, "horizon": horizon,
        "decision_repeat": 5,
        "random_agent_model": False, "random_lane_width": True, "random_lane_num": True,
        "out_of_route_done": True, "on_continuous_line_done": False,
        "vehicle_config": {"image_source": "front_left_camera", "show_lidar": False,
                           "show_navi_mark": False, "show_line_to_navi_mark": False,
                           "show_navigation_arrow": False},
        "sensors": sensors,
    }
    if composite_view:
        config.update({"window_size": (640, 360), "show_interface": False})
    if use_idm:
        config["agent_policy"] = IDMPolicy
    return MetaDriveEnv(config)
