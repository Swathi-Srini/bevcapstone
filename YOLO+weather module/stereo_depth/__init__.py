"""Stereo depth estimation module for autonomous driving perception."""

from .camera_params import CameraParameters, CameraConfig
from .stereo_matcher import StereoMatcher
from .depth_processor import DepthProcessor
from .depth_utils import DisparityToDepth, PhysicalSizeEstimator

__all__ = [
    'CameraParameters',
    'CameraConfig',
    'StereoMatcher',
    'DepthProcessor',
    'DisparityToDepth',
    'PhysicalSizeEstimator',
]

__version__ = '1.0.0'
