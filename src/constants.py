"""
Smart Constants Module - Auto-detects OS and imports correct paths

This module automatically detects whether the code is running on Windows or WSL/Linux
and imports the appropriate constants (file paths) accordingly.

Usage:
    from src.constants import IMG, ANNOTATIONS_PATH
    
    # IMG and ANNOTATIONS_PATH will automatically point to the correct paths
    # based on the detected operating system

Supported Platforms:
    - Windows: Uses paths from win_constants.py
    - WSL/Linux: Uses paths from wsl_constants.py
"""

import platform
import sys

# Detect operating system
def _detect_os():
    """
    Detect the current operating system.
    
    Returns:
        str: 'windows' for Windows, 'wsl' for WSL/Linux
    """
    system = platform.system().lower()
    
    if system == 'windows':
        return 'windows'
    elif system == 'linux':
        # Check if running in WSL
        try:
            with open('/proc/version', 'r') as f:
                if 'microsoft' in f.read().lower():
                    return 'wsl'
        except:
            pass
        return 'wsl'  # Treat all Linux as WSL for now
    else:
        # Default to WSL for macOS or other Unix-like systems
        return 'wsl'


# Auto-import the correct constants based on OS
_current_os = _detect_os()

if _current_os == 'windows':
    from src.win_constants import IMG, ANNOTATIONS_PATH
    print(f"[Chakshu] Loaded Windows constants")
else:
    from src.wsl_constants import IMG, ANNOTATIONS_PATH
    print(f"[Chakshu] Loaded WSL/Linux constants")

# Export for use by other modules
__all__ = ['IMG', 'ANNOTATIONS_PATH']
