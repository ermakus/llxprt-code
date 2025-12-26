#!/usr/bin/env bash
#
# Build and run the A2A server Docker container
#
# Usage:
#   ./scripts/build-a2a.sh [--build-only] [--run-only] [--workspace PATH]
#
# Options:
#   --build         Only build the image, don't run
#   --run           Only run the container (assumes image exists)
#   --workspace     Path to workspace directory (default: ./workspace)
#   --port          Port to expose (default: 41242)
#   --detach, -d    Run container in background
#   --help, -h      Show this help message
#

set -euo pipefail

source .env
export "$(grep -v '^#' .env | xargs)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
IMAGE_NAME="llxprt-a2a-server"
CONTAINER_NAME="llxprt-a2a"
PORT="${CODER_AGENT_PORT:-41242}"
WORKSPACE_PATH="${PROJECT_ROOT}/workspace"
BUILD_ONLY=false
RUN_ONLY=false
DETACH=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

show_help() {
    head -20 "$0" | tail -16
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_ONLY=true
            shift
            ;;
        --run)
            RUN_ONLY=true
            shift
            ;;
        --workspace)
            WORKSPACE_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

cd "$PROJECT_ROOT"

build_packages() {
    log_info "Building packages..."

    # Build only core and a2a-server packages (not the full project)
    log_info "Building @vybestack/llxprt-code-core..."
    npm run build --workspace @vybestack/llxprt-code-core

    log_info "Building @vybestack/llxprt-code-a2a-server..."
    npm run build --workspace @vybestack/llxprt-code-a2a-server

    # Create tarballs
    log_info "Creating package tarballs..."

    cd "$PROJECT_ROOT/packages/core"
    npm pack
    mkdir -p dist
    mv ./*.tgz dist/

    cd "$PROJECT_ROOT/packages/a2a-server"
    npm pack
    mkdir -p dist
    mv ./*.tgz dist/

    cd "$PROJECT_ROOT"

    log_success "Packages built successfully"
}

build_image() {
    log_info "Building Docker image: $IMAGE_NAME"

    docker build \
        -f packages/a2a-server/Dockerfile \
        -t "$IMAGE_NAME" \
        .

    log_success "Docker image built: $IMAGE_NAME"
}

run_container() {
    # Create workspace directory if it doesn't exist
    mkdir -p "$WORKSPACE_PATH"

    # Stop existing container if running
    if docker ps -q -f "name=$CONTAINER_NAME" | grep -q .; then
        log_warn "Stopping existing container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi

    # Remove existing container if exists
    if docker ps -aq -f "name=$CONTAINER_NAME" | grep -q .; then
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi

    log_info "Starting container: $CONTAINER_NAME"
    log_info "  Port: $PORT"
    log_info "  Workspace: $WORKSPACE_PATH"

    # Build docker run command
    DOCKER_ARGS=(
        "--name" "$CONTAINER_NAME"
        "-p" "${PORT}:41242"
        "-v" "${WORKSPACE_PATH}:/workspace"
        "-e" "CODER_AGENT_PORT=41242"
    )

    # Add API key if set (check OpenAI first, then Gemini)
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        DOCKER_ARGS+=("-e" "OPENAI_API_KEY=${OPENAI_API_KEY}")
        log_info "  OPENAI_API_KEY: (set)"

        if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
            DOCKER_ARGS+=("-e" "OPENAI_BASE_URL=${OPENAI_BASE_URL}")
            log_info "  OPENAI_BASE_URL: ${OPENAI_BASE_URL}"
        fi

        if [[ -n "${OPENAI_MODEL:-}" ]]; then
            DOCKER_ARGS+=("-e" "OPENAI_MODEL=${OPENAI_MODEL}")
            log_info "  OPENAI_MODEL: ${OPENAI_MODEL}"
        fi
    elif [[ -n "${GEMINI_API_KEY:-}" ]]; then
        DOCKER_ARGS+=("-e" "GEMINI_API_KEY=${GEMINI_API_KEY}")
        log_info "  GEMINI_API_KEY: (set)"
    else
        log_warn "No API key set - provide OPENAI_API_KEY or GEMINI_API_KEY"
    fi

    # Add other optional env vars
    if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
        DOCKER_ARGS+=("-e" "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}")
    fi

    if [[ -n "${GCS_BUCKET_NAME:-}" ]]; then
        DOCKER_ARGS+=("-e" "GCS_BUCKET_NAME=${GCS_BUCKET_NAME}")
    fi

    if [[ "$DETACH" == "true" ]]; then
        DOCKER_ARGS+=("-d")
        docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"
        log_success "Container started in background"
        echo ""
        log_info "View logs: docker logs -f $CONTAINER_NAME"
        log_info "Stop: docker stop $CONTAINER_NAME"
        log_info "Agent card: http://localhost:${PORT}/.well-known/agent-card.json"
    else
        DOCKER_ARGS+=("-it" "--rm")
        log_info "Starting in foreground (Ctrl+C to stop)..."
        echo ""
        docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"
    fi
}

# Main execution
if [[ "$RUN_ONLY" == "true" ]]; then
    # Check if image exists
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        log_error "Image $IMAGE_NAME not found. Run without --run-only first."
        exit 1
    fi
    run_container
elif [[ "$BUILD_ONLY" == "true" ]]; then
    build_packages
    build_image
else
    build_packages
    build_image
    run_container
fi
