#!/usr/bin/env python3
"""
Training script for A2SF RL agent
"""

import sys
import os

# Add parent directory to path to import A2SF modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.main import main

if __name__ == "__main__":
    main()
