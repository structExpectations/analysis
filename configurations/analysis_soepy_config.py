"""This module provides some configuration for the package."""
import sys
import os
from pathlib import Path

import numpy as np

# We only support modern Python.
np.testing.assert_equal(sys.version_info[:2] >= (3, 6), True)

# We rely on relative paths throughout the package.
PACKAGE_DIR = Path(os.path.abspath(__file__)).parents[1]
RESOURCES_DIR = PACKAGE_DIR / "estimation" / "basecamp" / "resources"
LOGGING_DIR = PACKAGE_DIR / "estimation" / "basecamp" / "logging"
FIGURES_DIR = PACKAGE_DIR / "estimation" / "basecamp" / "figures"
