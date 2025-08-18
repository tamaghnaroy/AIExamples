@echo off
echo Cleaning up secrets from git history...

REM First, let's add the current changes to staging
git add .

REM Commit the current clean version
git commit -m "Secure API keys moved to .env file"

REM Use git filter-branch to remove secrets from history
echo Rewriting git history to remove exposed API keys...
git filter-branch --force --tree-filter "if exist DeepResearcher\deepresearcher.py (powershell -Command \"(Get-Content DeepResearcher\deepresearcher.py) -replace 'sk-proj-[^\"\"]*', '<REDACTED>' -replace 'tvly-dev-[^\"\"]*', '<REDACTED>' -replace 'fc-[^\"\"]*', '<REDACTED>' | Set-Content DeepResearcher\deepresearcher.py\")" HEAD

REM Clean up the backup refs
git for-each-ref --format="delete %%(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now

echo Git history cleaned. You can now push to GitHub:
echo git push origin main --force
pause
