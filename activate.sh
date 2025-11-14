#!/bin/bash
# Quick activation script for Multi-Agent Data Command Center

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Multi-Agent Data Command Center${NC}"
echo -e "${BLUE}=================================${NC}"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
fi

# Check for .env file
if [ ! -f ".env" ]; then
    if [ ! -f ".env.example" ]; then
        echo ""
        echo "⚠️  No .env file found. Create one with:"
        echo "   echo 'CLAUDE_API_KEY=your-key-here' > .env"
    else
        echo ""
        echo "ℹ️  Copy .env.example to .env and add your API key:"
        echo "   cp .env.example .env"
        echo "   # Then edit .env to add your CLAUDE_API_KEY"
    fi
else
    # Load environment variables from .env
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✓ Environment variables loaded from .env${NC}"
fi

echo ""
echo -e "${BLUE}Ready to use! Try:${NC}"
echo "  python main.py ui              # Start web UI"
echo "  python main.py test_data.csv   # Run pipeline on test file"
echo ""
