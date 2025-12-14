#!/usr/bin/env python3
import json

# Load the issues JSON file
with open('open5gs_issues.json', 'r') as issues_file:
    issues_data = json.load(issues_file)

# Load the commits JSON file
with open('open5gs_commits.json', 'r') as commits_file:
    commits_data = json.load(commits_file)

# Create a dictionary to hold the mapping of issue numbers to their comments
issues_mapping = {}

# Iterate through the issues and add a new field "comments" for each issue
for issue in issues_data:
    issue_number = issue['number']
    issue['commits'] = []  # Initialize an empty comments field
    issues_mapping[issue_number] = issue

# Iterate through the commits and map comments to the respective issues
for commit in commits_data:
    commit_message = commit['message']

    # Check if any issue number is mentioned in the commit message
    for issue_number in issues_mapping:
        if f"#{issue_number}" in commit_message:
            # Add the commit details to the comments field of the corresponding issue
            issues_mapping[issue_number]['commits'].append(commit)

# Convert the dictionary back to a list
mapped_issues = list(issues_mapping.values())

# Save the new JSON file with the mapping
with open('mapped_issues.json', 'w') as mapped_file:
    json.dump(mapped_issues, mapped_file, indent=4)

print("Mapped issues JSON file has been created as 'mapped_issues.json'.")
