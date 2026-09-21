"""poreml — machine-learning benchmark for pore-scale multiphase flow."""

__version__ = "0.1.0"

# Import for side effects: these modules populate the registries, so any entry point
# sees a full set of tasks, models, and metrics regardless of import order.
from . import metrics as metrics  # noqa: F401
from . import models as models  # noqa: F401
from . import tasks as tasks  # noqa: F401
