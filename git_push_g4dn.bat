@echo off
echo ========================================
echo CyberBot AI - Git Push for G4dn Deployment
echo ========================================
echo.

:: Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

:: Check if we're in a git repository
if not exist ".git" (
    echo [INFO] Initializing git repository...
    git init
)

echo [INFO] Adding files to git...

:: Add core application files
git add models/cyberbot_real.py
git add requirements.txt

:: Add G4dn.xlarge specific files
git add .env.g4dn-xlarge
git add deploy_g4dn_xlarge.sh
git add test_g4dn_performance.py
git add README_G4DN.md
git add DEPLOYMENT_GUIDE_G4DN.md

:: Add configuration files
git add .gitignore

:: Add documentation
git add README.md

:: Add environment configs for reference
git add .env.example
git add .env.t3-large

echo [SUCCESS] Files added to git

:: Show git status
echo.
echo [INFO] Git status:
git status --short

:: Get commit message
echo.
echo Enter commit message (or press Enter for default):
set /p commit_message="Commit message: "

if "%commit_message%"=="" (
    set "commit_message=🚀 G4dn.xlarge GPU optimization: Enhanced AI with T4 GPU, emotion analysis, 10x performance boost"
)

:: Commit changes
echo.
echo [INFO] Committing changes...
git commit -m "%commit_message%"

if %errorlevel% neq 0 (
    echo [ERROR] Commit failed
    pause
    exit /b 1
)

echo [SUCCESS] Commit successful

:: Check if remote origin exists
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] No remote origin found
    echo Enter your GitHub repository URL:
    echo Example: https://github.com/username/cyberbot-ai.git
    set /p repo_url="Repository URL: "
    
    if not "%repo_url%"=="" (
        git remote add origin "%repo_url%"
        echo [SUCCESS] Remote origin added
    ) else (
        echo [ERROR] No repository URL provided
        pause
        exit /b 1
    )
)

:: Push to remote
echo.
echo [INFO] Pushing to remote repository...

:: Check if main branch exists on remote
git ls-remote --heads origin main | findstr "main" >nul 2>&1
if %errorlevel% equ 0 (
    git push origin main
) else (
    :: First push - set upstream
    git branch -M main
    git push -u origin main
)

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Push successful!
    echo.
    echo 🎉 Code pushed to GitHub successfully!
    echo.
    echo Next steps for G4dn.xlarge deployment:
    echo 1. SSH to your G4dn.xlarge instance
    echo 2. Clone the repository: git clone [your-repo-url]
    echo 3. Run deployment script: sudo ./deploy_g4dn_xlarge.sh
    echo 4. Test performance: python test_g4dn_performance.py
    echo.
    for /f "tokens=*" %%i in ('git remote get-url origin') do set repo_url=%%i
    echo Repository URL: %repo_url%
    echo Deployment Guide: DEPLOYMENT_GUIDE_G4DN.md
) else (
    echo.
    echo [ERROR] Push failed
    echo.
    echo Common solutions:
    echo - Check your GitHub credentials
    echo - Verify repository URL is correct
    echo - Ensure you have push permissions
    pause
    exit /b 1
)

echo.
echo 🚀 Ready for G4dn.xlarge deployment! 🎮
echo.
pause
