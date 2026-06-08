#!/usr/bin/env python3
"""
Push PaintsUndo-Vector to GitHub

Usage:
    python push_to_github.py --token YOUR_GITHUB_TOKEN [--repo-name PaintsUndo-Vector] [--private]
    
Get your token from: https://github.com/settings/tokens
Required scopes: repo, delete_repo (optional)
"""

import argparse
import os
import subprocess
import sys

try:
    from github import Github
except ImportError:
    print("Installing PyGithub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyGithub"])
    from github import Github


def main():
    parser = argparse.ArgumentParser(description="Push PaintsUndo-Vector to GitHub")
    parser.add_argument("--token", "-t", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--repo-name", "-r", default="PaintsUndo-Vector", help="Repository name")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--description", "-d", default="Vector stroke generation via differentiable rendering optimization", help="Repository description")
    args = parser.parse_args()
    
    # Authenticate
    g = Github(args.token)
    user = g.get_user()
    print(f"Authenticated as: {user.login}")
    
    # Create repository
    repo_name = args.repo_name
    try:
        repo = user.get_repo(repo_name)
        print(f"Repository {repo_name} already exists, using existing repo")
    except Exception:
        repo = user.create_repo(
            repo_name,
            description=args.description,
            private=args.private,
            auto_init=False,
        )
        print(f"Created repository: {repo.full_name}")
    
    # Set remote and push
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Check if remote already exists
    try:
        subprocess.run(["git", "remote", "remove", "origin"], capture_output=True)
    except Exception:
        pass
    
    # Set remote URL with token for authentication
    remote_url = f"https://{args.token}@github.com/{repo.full_name}.git"
    subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
    
    # Push
    print("Pushing to GitHub...")
    result = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"\nSuccessfully pushed to: https://github.com/{repo.full_name}")
    else:
        print(f"Push failed: {result.stderr}")
        # Try with main branch
        subprocess.run(["git", "branch", "-M", "main"], capture_output=True)
        result = subprocess.run(["git", "push", "-u", "origin", "main"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\nSuccessfully pushed to: https://github.com/{repo.full_name}")
        else:
            print(f"Push failed again: {result.stderr}")
    
    # Clean up token from remote URL
    subprocess.run(["git", "remote", "set-url", "origin", f"https://github.com/{repo.full_name}.git"], capture_output=True)


if __name__ == "__main__":
    main()
