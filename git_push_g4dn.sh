#!/bin/bash

# ==========================================
# CyberBot AI - Git Push Script for G4dn Deployment
# ==========================================

echo "🚀 CyberBot AI - Preparing for G4dn.xlarge deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Git repository not initialized${NC}"
    echo -e "${YELLOW}🔧 Initializing git repository...${NC}"
    git init
fi

# Add important files for G4dn deployment
echo -e "${BLUE}📁 Adding files to git...${NC}"

# Core application files
git add models/cyberbot_real.py
git add requirements.txt

# G4dn.xlarge specific files
git add .env.g4dn-xlarge
git add deploy_g4dn_xlarge.sh
git add test_g4dn_performance.py
git add README_G4DN.md
git add DEPLOYMENT_GUIDE_G4DN.md

# Configuration files
git add .gitignore

# Documentation
git add README.md

# Optional: Add other environment configs for reference
git add .env.example
git add .env.t3-large

echo -e "${GREEN}✅ Files added to git${NC}"

# Check git status
echo -e "${BLUE}📊 Git status:${NC}"
git status --short

# Get commit message from user or use default
echo -e "${YELLOW}💬 Enter commit message (or press Enter for default):${NC}"
read -r commit_message

if [ -z "$commit_message" ]; then
    commit_message="🚀 G4dn.xlarge GPU optimization: Enhanced AI with T4 GPU, emotion analysis, 10x performance boost"
fi

# Commit changes
echo -e "${BLUE}💾 Committing changes...${NC}"
git commit -m "$commit_message"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Commit successful${NC}"
else
    echo -e "${RED}❌ Commit failed${NC}"
    exit 1
fi

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo -e "${YELLOW}🔗 No remote origin found${NC}"
    echo -e "${YELLOW}📝 Enter your GitHub repository URL:${NC}"
    echo -e "${BLUE}   Example: https://github.com/username/cyberbot-ai.git${NC}"
    read -r repo_url
    
    if [ ! -z "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo -e "${GREEN}✅ Remote origin added${NC}"
    else
        echo -e "${RED}❌ No repository URL provided${NC}"
        exit 1
    fi
fi

# Push to remote
echo -e "${BLUE}📤 Pushing to remote repository...${NC}"

# Check if main branch exists on remote
if git ls-remote --heads origin main | grep -q main; then
    git push origin main
else
    # First push - set upstream
    git branch -M main
    git push -u origin main
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Push successful!${NC}"
    echo ""
    echo -e "${GREEN}🎉 Code pushed to GitHub successfully!${NC}"
    echo ""
    echo -e "${BLUE}📋 Next steps for G4dn.xlarge deployment:${NC}"
    echo -e "${YELLOW}1.${NC} SSH to your G4dn.xlarge instance"
    echo -e "${YELLOW}2.${NC} Clone the repository: git clone $(git remote get-url origin)"
    echo -e "${YELLOW}3.${NC} Run deployment script: sudo ./deploy_g4dn_xlarge.sh"
    echo -e "${YELLOW}4.${NC} Test performance: python test_g4dn_performance.py"
    echo ""
    echo -e "${BLUE}🔗 Repository URL:${NC} $(git remote get-url origin)"
    echo -e "${BLUE}📖 Deployment Guide:${NC} DEPLOYMENT_GUIDE_G4DN.md"
else
    echo -e "${RED}❌ Push failed${NC}"
    echo -e "${YELLOW}🔧 Common solutions:${NC}"
    echo -e "   - Check your GitHub credentials"
    echo -e "   - Verify repository URL is correct"
    echo -e "   - Ensure you have push permissions"
    exit 1
fi

echo ""
echo -e "${GREEN}🚀 Ready for G4dn.xlarge deployment! 🎮${NC}"
