#!/bin/bash
# Ollama Setup Script
# Ensures Ollama is installed, running, and required models are pulled

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default model to pull
DEFAULT_MODEL="devstral:24b"

# All supported models for the LLM post-processing feature
SUPPORTED_MODELS=(
    "devstral:24b"
    "qwen2.5:7b"
    "meditron:7b"
    "medgemma:latest"
)

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Ollama is installed
check_ollama_installed() {
    print_status "Checking if Ollama is installed..."
    
    if command -v ollama &> /dev/null; then
        OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
        print_success "Ollama is installed: $OLLAMA_VERSION"
        return 0
    else
        print_warning "Ollama is not installed"
        return 1
    fi
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    # Detect OS
    OS="$(uname -s)"
    case "$OS" in
        Darwin)
            print_status "Detected macOS"
            if command -v brew &> /dev/null; then
                print_status "Installing via Homebrew..."
                brew install ollama
            else
                print_status "Installing via curl..."
                curl -fsSL https://ollama.com/install.sh | sh
            fi
            ;;
        Linux)
            print_status "Detected Linux"
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        *)
            print_error "Unsupported OS: $OS"
            print_status "Please install Ollama manually from https://ollama.com"
            exit 1
            ;;
    esac
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully"
    else
        print_error "Failed to install Ollama"
        exit 1
    fi
}

# Check if Ollama server is running
check_ollama_running() {
    print_status "Checking if Ollama server is running..."
    
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_success "Ollama server is running on localhost:11434"
        return 0
    else
        print_warning "Ollama server is not running"
        return 1
    fi
}

# Start Ollama server
start_ollama() {
    print_status "Starting Ollama server..."
    
    # Check if already running
    if check_ollama_running 2>/dev/null; then
        return 0
    fi
    
    # Start in background
    ollama serve &> /tmp/ollama.log &
    OLLAMA_PID=$!
    
    # Wait for server to start (max 30 seconds)
    print_status "Waiting for Ollama server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            print_success "Ollama server started (PID: $OLLAMA_PID)"
            return 0
        fi
        sleep 1
    done
    
    print_error "Failed to start Ollama server. Check /tmp/ollama.log for details"
    exit 1
}

# List available models
list_models() {
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    for m in models:
        print(m['name'])
except:
    pass
" 2>/dev/null)
    
    if [ -n "$MODELS" ]; then
        echo "$MODELS"
    fi
}

# Check if a specific model is available
is_model_available() {
    local model=$1
    local models=$(list_models)
    
    # Check exact match or partial match (model name without tag)
    if echo "$models" | grep -q "^${model}$"; then
        return 0
    fi
    
    # Check partial match (e.g., "devstral" matches "devstral:24b")
    local model_name="${model%%:*}"
    if echo "$models" | grep -q "^${model_name}"; then
        return 0
    fi
    
    return 1
}

# Pull a model
pull_model() {
    local model=$1
    
    print_status "Pulling model: $model"
    print_status "This may take a while depending on your internet connection..."
    
    if ollama pull "$model"; then
        print_success "Model $model pulled successfully"
        return 0
    else
        print_error "Failed to pull model $model"
        return 1
    fi
}

# Ensure a model is available, pull if not
ensure_model() {
    local model=$1
    
    print_status "Checking model: $model"
    
    if is_model_available "$model"; then
        print_success "Model $model is available"
        return 0
    else
        print_warning "Model $model not found, pulling..."
        pull_model "$model"
        return $?
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS] [MODEL]"
    echo ""
    echo "Ensures Ollama is installed, running, and specified model is available."
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -l, --list      List available models"
    echo "  -a, --all       Pull all supported models"
    echo "  -s, --status    Show Ollama status only"
    echo ""
    echo "Arguments:"
    echo "  MODEL           Model to pull (default: $DEFAULT_MODEL)"
    echo ""
    echo "Supported models:"
    for model in "${SUPPORTED_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "Examples:"
    echo "  $0                    # Setup with default model ($DEFAULT_MODEL)"
    echo "  $0 qwen2.5:7b         # Setup with specific model"
    echo "  $0 --all              # Setup and pull all supported models"
    echo "  $0 --list             # List currently available models"
}

# Main function
main() {
    echo "=============================================="
    echo "       Ollama Setup for Document Recognition"
    echo "=============================================="
    echo ""
    
    # Parse arguments
    MODEL="$DEFAULT_MODEL"
    PULL_ALL=false
    LIST_ONLY=false
    STATUS_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -l|--list)
                LIST_ONLY=true
                shift
                ;;
            -a|--all)
                PULL_ALL=true
                shift
                ;;
            -s|--status)
                STATUS_ONLY=true
                shift
                ;;
            *)
                MODEL="$1"
                shift
                ;;
        esac
    done
    
    # Step 1: Check/Install Ollama
    if ! check_ollama_installed; then
        read -p "Would you like to install Ollama? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_ollama
        else
            print_error "Ollama is required. Please install it manually from https://ollama.com"
            exit 1
        fi
    fi
    
    # Step 2: Check/Start Ollama server
    if ! check_ollama_running; then
        start_ollama
    fi
    
    # Status only mode
    if [ "$STATUS_ONLY" = true ]; then
        echo ""
        echo "Available models:"
        list_models | while read -r m; do
            echo "  - $m"
        done
        exit 0
    fi
    
    # List only mode
    if [ "$LIST_ONLY" = true ]; then
        echo ""
        echo "Available models:"
        list_models | while read -r m; do
            echo "  - $m"
        done
        exit 0
    fi
    
    echo ""
    
    # Step 3: Ensure model(s) are available
    if [ "$PULL_ALL" = true ]; then
        print_status "Pulling all supported models..."
        for model in "${SUPPORTED_MODELS[@]}"; do
            ensure_model "$model"
        done
    else
        ensure_model "$MODEL"
    fi
    
    echo ""
    echo "=============================================="
    print_success "Ollama setup complete!"
    echo ""
    echo "Available models:"
    list_models | while read -r m; do
        echo "  - $m"
    done
    echo ""
    print_status "You can now start the application with:"
    echo "  ./venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
    echo "=============================================="
}

# Run main
main "$@"
