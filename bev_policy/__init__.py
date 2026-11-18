"""CNN + MLP behavioural-cloning policy for the BEV-state contract."""

from pathlib import Path
import sys

# The perception modules predate this package and live in a directory with a
# space in its name. Make their imports available whenever ``bev_policy`` is
# imported, including direct module imports that do not first import bev_state.
_PERCEPTION_ROOT = Path(__file__).resolve().parents[1] / "YOLO+weather module"
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))

from .model import BEVScalarPolicy

__all__ = ["BEVScalarPolicy"]
