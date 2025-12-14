#!/usr/bin/env python3
from github import Github
import json

# Authenticate with your GitHub token
token = ""
repo_name = "open5gs/open5gs"  # Repository name
g = Github(token)

# Access the repository
repo = g.get_repo(repo_name)

issues = repo.get_issues(state="all")  # Fetch both open and closed issues

issue_data = []

for issue in issues:
    issue_dict = {
        "id": issue.id,
        "number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "labels": [label.name for label in issue.labels],
        "created_at": issue.created_at.isoformat(),
        "updated_at": issue.updated_at.isoformat(),
        "state": issue.state,
        "comments_count": issue.comments,
        "author": issue.user.login,
        "url": issue.html_url,
    }

    # Fetch comments for the issue
    comments = []
    if issue.comments > 0:
        for comment in issue.get_comments():
            comments.append({
                "author": comment.user.login,
                "created_at": comment.created_at.isoformat(),
                "body": comment.body,
                "url": comment.html_url,
            })

    issue_dict["comments"] = comments

    # Check for attachments or URLs in the body and comments
    attachments = []
    if issue.body:
        attachments += [word for word in issue.body.split() if word.startswith("http")]

    for comment in comments:
        if comment["body"]:
            attachments += [word for word in comment["body"].split() if word.startswith("http")]

    issue_dict["attachments"] = attachments

    issue_data.append(issue_dict)

# Save all data to a JSON file
with open("open5gs_issues.json", "w") as f:
    json.dump(issue_data, f, indent=4)

print("Data saved to open5gs_issues_with_comments.json")