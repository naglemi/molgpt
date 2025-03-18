#!/bin/bash
# ninjascroll.sh: A simple script to commit and track ninja scrolls

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if commit message is provided
if [ $# -lt 2 ]; then
    echo -e "${RED}Usage: $0 \"commit message\" file1 [file2 ...]${NC}"
    exit 1
fi

COMMIT_MESSAGE=$1
shift # Remove the first argument (commit message)

# Check if all files exist
for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: File '$file' does not exist${NC}"
        exit 1
    fi
done

# Create .scrolls directory if it doesn't exist
mkdir -p .scrolls

# Copy files that are not already in .scrolls
for file in "$@"; do
    if [[ ! "$file" =~ ^\.scrolls/ ]]; then
        filename=$(basename "$file")
        cp "$file" ".scrolls/$filename"
        echo -e "${BLUE}Copied $file to .scrolls/$filename${NC}"
    fi
done

# Git operations
echo -e "${BLUE}Adding files to git...${NC}"
git add "$@"

echo -e "${BLUE}Committing changes...${NC}"
git commit -m "$COMMIT_MESSAGE"

echo -e "${GREEN}Ninja scroll(s) successfully committed!${NC}"
echo -e "${YELLOW}To push changes to remote, run: git push${NC}" 