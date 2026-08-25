"""Stereo depth estimation module for autonomous driving perception."""

from .camera_params import CameraParameters, CameraConfig, CameraPosition
from .stereo_matcher import StereoMatcher
from .depth_processor import DepthProcessor
from .depth_utils import CoordinateTransform, DisparityToDepth, MonocularDepth, PhysicalSizeEstimator

__all__ = [
    'CameraParameters',
    'CameraConfig',
    'CameraPosition',
    'StereoMatcher',
    'DepthProcessor',
    'CoordinateTransform',
    'DisparityToDepth',
    'MonocularDepth',
    'PhysicalSizeEstimator',
]

__version__ = '1.0.0'
