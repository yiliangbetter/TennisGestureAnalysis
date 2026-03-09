#!/usr/bin/env python3
"""
Setup script for Tennis Gesture Analysis
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install required Python packages"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)


def main():
    print("Setting up Tennis Gesture Analysis Project...")

    # Install dependencies
    install_dependencies()

    print("\nSetup complete!")
    print("To run the analysis, use: python main.py <path_to_your_video.mp4>")
    print("\nExample: python main.py sample_tennis_video.mp4")


if __name__ == "__main__":
    main()