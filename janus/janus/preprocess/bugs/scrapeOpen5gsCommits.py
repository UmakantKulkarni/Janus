#!/usr/bin/env python3
from github import Github
import json

# Authenticate with your GitHub token
token = ""
repo_name = "open5gs/open5gs"  # Repository name
g = Github(token)

# Access the repository
repo = g.get_repo(repo_name)

# Fetch all commits
commits = repo.get_commits()

commit_data = []

for commit in commits:
    commit_dict = {
        "sha": commit.sha,
        "author": commit.author.login if commit.author else "Unknown",
        "date": commit.commit.author.date.isoformat(),
        "message": commit.commit.message,
        "url": commit.html_url,
    }

    # Check if the commit message references a ticket number
    referenced_tickets = [
        word for word in commit.commit.message.split() if word.startswith("#")
    ]
    commit_dict["referenced_tickets"] = referenced_tickets

    # Fetch the files changed in the commit
    files_changed = []
    for file in commit.files:
        files_changed.append({
            "filename": file.filename,
            "additions": file.additions,
            "deletions": file.deletions,
            "changes": file.changes,
            "patch": file.patch if hasattr(file, "patch") else "No patch available"
        })
    
    commit_dict["files_changed"] = files_changed
    commit_data.append(commit_dict)

# Save all data to a JSON file
with open("open5gs_commits.json", "w") as f:
    json.dump(commit_data, f, indent=4)

print("Data saved to open5gs_commits.json")