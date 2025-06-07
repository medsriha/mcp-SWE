#!/usr/bin/env python3
"""
Repository Downloader Script

This script downloads repositories from various sources:
- Git repositories (HTTP/HTTPS/SSH)
- Direct HTTP/HTTPS file downloads
- Local filesystem paths
"""

import os
import subprocess
import urllib.request
import urllib.parse
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RepoDownloader:
    """A class to handle downloading repositories from various sources."""
    
    def is_git_url(self, url: str) -> bool:
        """Check if the URL is a Git repository."""
        git_patterns = [
            '.git',
            'github.com',
            'gitlab.com',
            'bitbucket.org',
            'git@'
        ]
        return any(pattern in url.lower() for pattern in git_patterns)
    
    def is_local_path(self, url: str) -> bool:
        """Check if the URL is a local filesystem path."""
        logger.info(f"Checking if the URL is a local filesystem path: {url}")
        return os.path.exists(url) or url.startswith('file://')
    
    def download_git_repo(self, url: str, destination: str) -> bool:
        """Download a Git repository."""
        try:
            # Remove destination directory if it exists
            if os.path.exists(destination):
                logger.info(f"Removing existing directory: {destination}")
                shutil.rmtree(destination)
            
            logger.info(f"Cloning Git repository: {url}")
            cmd = ['git', 'clone', url, destination]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned repository to: {destination}")
                return True
            else:
                logger.error(f"Error cloning repository: {result.stderr}")
                print(f"Error cloning repository: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("Git is not installed or not in PATH")
            print("Error: Git is not installed or not in PATH")
            return False
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            print(f"Error cloning repository: {e}")
            return False
    
    def download_http_file(self, url: str, destination: str) -> bool:
        """Download a file from HTTP/HTTPS URL."""
        try:
            logger.info(f"Downloading from HTTP/HTTPS: {url}")
            
            # Remove destination directory if it exists, then create it
            if os.path.exists(destination):
                logger.info(f"Removing existing directory: {destination}")
                shutil.rmtree(destination)
            os.makedirs(destination, exist_ok=True)
            
            # Get filename from URL
            parsed_url = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed_url.path) or "downloaded_file"
            file_path = os.path.join(destination, filename)
            
            # Download the file
            urllib.request.urlretrieve(url, file_path)
            
            logger.info(f"Successfully downloaded to: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            print(f"Error downloading file: {e}")
            return False
    
    def copy_local_path(self, source: str, destination: str) -> bool:
        """Copy from local filesystem path."""
        try:
            logger.info(f"Copying from local path: {source}")
            
            # Handle file:// URLs
            if source.startswith('file://'):
                source = source[7:]  # Remove file:// prefix
            
            source_path = Path(source)
            dest_path = Path(destination)
            
            if source_path.is_file():
                # Copy single file
                dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path / source_path.name)
            elif source_path.is_dir():
                # Copy entire directory
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
            else:
                error_msg = f"Error: Source path does not exist: {source}"
                logger.error(error_msg)
                print(error_msg)
                return False
            
            logger.info(f"Successfully copied to: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying local path: {e}")
            print(f"Error copying local path: {e}")
            return False
    
    def download(self, url: str, destination: str) -> bool:
        """Main method to download repository based on URL type."""
        # try:
        # Normalize the destination path
        destination = os.path.abspath(destination)
        
        if self.is_local_path(url):
            return self.copy_local_path(url, destination)
        elif self.is_git_url(url):
            return self.download_git_repo(url, destination)
        else:
            # Assume it's an HTTP/HTTPS URL
            return self.download_http_file(url, destination)
                
        # except Exception as e:
        #     print(f"Error downloading repository: {e}")
        #     return False
