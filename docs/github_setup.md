# GitHub setup

This directory is already initialized as a local git repository on branch
`main`.

Before the first commit, configure a local git identity if the machine does not
already have one:

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

Then commit:

```bash
git add .
git commit -m "Initial BindGraph algorithm release"
```

Create a GitHub repository, then add the remote and push:

```bash
git remote add origin git@github.com:YOUR_ORG/YOUR_REPO.git
git push -u origin main
```

If using GitHub CLI:

```bash
gh repo create YOUR_ORG/YOUR_REPO --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` only after confirming the training tables,
embedding files, checkpoints, and candidate prediction outputs are not included.
