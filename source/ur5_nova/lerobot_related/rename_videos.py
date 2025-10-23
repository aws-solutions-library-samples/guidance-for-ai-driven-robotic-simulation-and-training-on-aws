#!/usr/bin/env python3
"""
Video file renumbering script
Renumbers files starting from episode_000001.mp4 -> episode_000029.mp4 format
"""

import os
import re
import shutil
from pathlib import Path

def rename_videos(directory_path, start_episode=29):
    """
    Renumbers video files
    
    Args:
        directory_path: Directory containing video files
        start_episode: New starting episode number (default: 29)
    """
    
    # Check directory existence
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return
    
    # Find video files
    video_files = []
    for file in os.listdir(directory_path):
        if file.startswith('episode_') and file.endswith('.mp4'):
            # Extract episode number
            match = re.match(r'episode_(\d+)\.mp4', file)
            if match:
                episode_num = int(match.group(1))
                video_files.append((episode_num, file))
    
    # Sort by episode number
    video_files.sort(key=lambda x: x[0])
    
    print(f"Found video files: {len(video_files)}")
    for episode_num, filename in video_files:
        print(f"  {filename}")
    
    # Show renumbering plan
    print(f"\nRenumbering plan:")
    print(f"Starting episode number: {start_episode}")
    
    # Plan for temporary files
    temp_files = []
    for i, (old_episode_num, filename) in enumerate(video_files):
        new_episode_num = start_episode + i
        new_filename = f"episode_{new_episode_num:06d}.mp4"
        temp_filename = f"temp_episode_{new_episode_num:06d}.mp4"
        
        print(f"  {filename} -> {new_filename}")
        temp_files.append((filename, temp_filename, new_filename))
    
    # Get user confirmation
    confirm = input(f"\nAre you sure you want to renumber {len(video_files)} video files? (y/N): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    try:
        # Step 1: Copy files with temporary names
        print("\nStep 1: Copying video files with temporary names...")
        for old_filename, temp_filename, new_filename in temp_files:
            old_path = os.path.join(directory_path, old_filename)
            temp_path = os.path.join(directory_path, temp_filename)
            shutil.copy2(old_path, temp_path)
            print(f"  Copied: {old_filename} -> {temp_filename}")
        
        # Step 2: Delete old files
        print("\nStep 2: Deleting old video files...")
        for old_episode_num, old_filename in video_files:
            old_path = os.path.join(directory_path, old_filename)
            os.remove(old_path)
            print(f"  Deleted: {old_filename}")
        
        # Step 3: Rename temporary files with new names
        print("\nStep 3: Renaming temporary files with new names...")
        for old_filename, temp_filename, new_filename in temp_files:
            temp_path = os.path.join(directory_path, temp_filename)
            new_path = os.path.join(directory_path, new_filename)
            os.rename(temp_path, new_path)
            print(f"  Renamed: {temp_filename} -> {new_filename}")
        
        print(f"\n✅ Successfully completed! {len(video_files)} video files renumbered.")
        print(f"New episode numbers: {start_episode} - {start_episode + len(video_files) - 1}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check files manually.")
        
        # Clean temporary files in case of error
        for _, temp_filename, _ in temp_files:
            temp_path = os.path.join(directory_path, temp_filename)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"Temporary file deleted: {temp_filename}")
                except:
                    print(f"Could not delete temporary file: {temp_filename}")

def main():
    # Default directory path
    default_directory = "/home/beable/ur5_simulation/src/data_collection/scripts/my_pusht/videos/chunk_000/observation.images.state"
    
    print("Video File Renumbering Script")
    print("=" * 50)
    
    # Get directory path
    directory_path = input(f"Directory path (Enter for default: {default_directory}): ").strip()
    if not directory_path:
        directory_path = default_directory
    
    # Get starting episode number
    start_episode_input = input("Starting episode number (Enter for default: 29): ").strip()
    if start_episode_input:
        try:
            start_episode = int(start_episode_input)
        except ValueError:
            print("Invalid number, using default value (29).")
            start_episode = 29
    else:
        start_episode = 29
    
    # Run script
    rename_videos(directory_path, start_episode)

if __name__ == "__main__":
    main()
