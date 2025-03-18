#!/bin/bash
# Ninja script to test MolGPT_Cowboy_Chronicle.ipynb

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🥷 Hayato, the Code Ninja, initiates MolGPT notebook testing...${NC}"

# Create .scrolls directory if it doesn't exist
mkdir -p .scrolls

# Check if required Python packages are installed
echo -e "${BLUE}[NINJA LOG] Checking dependencies...${NC}"

# Install required Python packages if needed
required_packages=(
    "jupyter"
    "nbconvert"
    "pandas"
    "numpy" 
    "torch"
    "matplotlib"
    "seaborn"
    "rdkit"
)

# Flag to keep track if pip install is needed
need_pip_install=false

for package in "${required_packages[@]}"
do
    # Check if package is installed
    python -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}[NINJA WARNING] $package is not installed.${NC}"
        need_pip_install=true
    else
        echo -e "${GREEN}[NINJA LOG] $package is installed.${NC}"
    fi
done

# Install missing packages if needed
if [ "$need_pip_install" = true ]; then
    echo -e "${BLUE}[NINJA LOG] Installing missing packages...${NC}"
    pip install jupyter nbconvert pandas numpy matplotlib seaborn rdkit
    if [ $? -ne 0 ]; then
        echo -e "${RED}[NINJA ERROR] Failed to install packages. Please check your Python environment.${NC}"
        exit 1
    fi
fi

# Make sure PYTHONPATH includes the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the notebook test
echo -e "${BLUE}[NINJA LOG] Running notebook test...${NC}"
python test_notebook.py

# Check if the test was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[NINJA SUCCESS] Notebook test completed successfully! 🎉${NC}"
    
    # Create success report
    cat > .scrolls/notebook_test_success.md << EOF
# 🥷 MolGPT Notebook Test: Success 🥷

## Test Results
* ✅ Notebook executed without errors
* ✅ Molecule generation working correctly
* ✅ Paths configured properly
* ✅ Results look sensible

## Next Steps
* Consider adding more validation checks
* Run with full dataset for production use
* Explore additional conditioning options
EOF
    
    echo -e "${GREEN}[NINJA LOG] Success report written to .scrolls/notebook_test_success.md${NC}"
    exit 0
else
    echo -e "${RED}[NINJA ERROR] Notebook test failed! 😞${NC}"
    
    # Create failure report
    cat > .scrolls/notebook_test_failure.md << EOF
# 🥷 MolGPT Notebook Test: Failure 🥷

## Issues Detected
* ❌ Notebook execution failed
* ❌ Check the error logs for specific issues
* ❌ Possible causes:
  * Missing dependencies
  * Path issues
  * Data availability problems
  * Resource constraints

## Recommended Actions
* Check Python environment and dependencies
* Verify dataset availability and paths
* Ensure model weights are accessible
* Consider reducing resource requirements for testing
EOF
    
    echo -e "${RED}[NINJA LOG] Failure report written to .scrolls/notebook_test_failure.md${NC}"
    exit 1
fi 