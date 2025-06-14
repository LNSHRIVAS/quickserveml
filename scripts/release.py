#!/usr/bin/env python3
"""
QuickServeML Release Script

This script automates the release process by:
1. Bumping version numbers
2. Creating git tags
3. Pushing to GitHub
4. Triggering automated release workflow
"""

import os
import sys
import subprocess
import re
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def get_current_version():
    """Get current version from pyproject.toml"""
    with open("pyproject.toml", "r") as f:
        content = f.read()
        match = re.search(r'version = "([^"]+)"', content)
        if match:
            return match.group(1)
    raise ValueError("Could not find version in pyproject.toml")

def update_version(version):
    """Update version in pyproject.toml"""
    with open("pyproject.toml", "r") as f:
        content = f.read()
    
    content = re.sub(r'version = "[^"]+"', f'version = "{version}"', content)
    
    with open("pyproject.toml", "w") as f:
        f.write(content)
    
    print(f"Updated version to {version}")

def update_init_version(version):
    """Update version in __init__.py"""
    init_file = Path("quickserveml/quickserveml/__init__.py")
    if init_file.exists():
        with open(init_file, "r") as f:
            content = f.read()
        
        content = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', content)
        
        with open(init_file, "w") as f:
            f.write(content)
        
        print(f"Updated __init__.py version to {version}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/release.py <version>")
        print("Example: python scripts/release.py 1.0.0")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+$', new_version):
        print("Error: Version must be in format X.Y.Z (e.g., 1.0.0)")
        sys.exit(1)
    
    print(f"Starting release process for version {new_version}")
    
    # Check if we're on main branch
    result = run_command("git branch --show-current", check=False)
    current_branch = result.stdout.strip()
    if current_branch != "main":
        print(f"Warning: You're on branch '{current_branch}', not 'main'")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Check if working directory is clean
    result = run_command("git status --porcelain", check=False)
    if result.stdout.strip():
        print("Error: Working directory is not clean. Please commit or stash changes.")
        print("Uncommitted changes:")
        print(result.stdout)
        sys.exit(1)
    
    # Update version files
    update_version(new_version)
    update_init_version(new_version)
    
    # Commit version changes
    run_command("git add pyproject.toml quickserveml/quickserveml/__init__.py")
    run_command(f'git commit -m "Bump version to {new_version}"')
    
    # Create and push tag
    tag_name = f"v{new_version}"
    run_command(f'git tag -a {tag_name} -m "Release {new_version}"')
    run_command("git push origin main")
    run_command(f"git push origin {tag_name}")
    
    print(f"\n🎉 Release {new_version} created successfully!")
    print(f"Tag: {tag_name}")
    print("\nThe GitHub Actions workflow will automatically:")
    print("1. Build the package")
    print("2. Create a GitHub release")
    print("3. Upload to PyPI (if configured)")
    print(f"\nCheck the release at: https://github.com/LNSHRIVAS/quickserveml/releases/tag/{tag_name}")

if __name__ == "__main__":
    main() 