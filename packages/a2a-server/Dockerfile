FROM docker.io/library/node:20-slim

ARG A2A_SERVER_NAME="llxprt-a2a-server"
ENV A2A_SERVER="$A2A_SERVER_NAME"

# Install minimal set of packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  make \
  g++ \
  curl \
  git \
  ca-certificates \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Set up npm global package folder
RUN mkdir -p /usr/local/share/npm-global \
  && chown -R node:node /usr/local/share/npm-global
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Create workspace directory
RUN mkdir -p /workspace && chown -R node:node /workspace

# Switch to non-root user
USER node

# Copy packages with proper ownership
COPY --chown=node:node packages/core/dist/vybestack-llxprt-code-core-*.tgz /tmp/
COPY --chown=node:node packages/a2a-server/dist/vybestack-llxprt-code-a2a-server-*.tgz /tmp/

# Install packages globally
RUN npm install -g /tmp/vybestack-llxprt-code-core-*.tgz && \
    npm install -g /tmp/vybestack-llxprt-code-a2a-server-*.tgz && \
    npm cache clean --force && \
    rm -f /tmp/*.tgz

# Set environment variables
ENV CODER_AGENT_PORT=41242
ENV CODER_AGENT_WORKSPACE_PATH=/workspace

# Provider configuration (set via docker run -e or docker-compose)
# For Gemini: GEMINI_API_KEY
# For OpenAI-compatible (DeepSeek, etc.):
#   OPENAI_API_KEY - Required API key
#   OPENAI_BASE_URL - Optional base URL (e.g., https://api.deepseek.com/v1)
#   OPENAI_MODEL - Optional model name (default: gpt-4)

# Expose the A2A server port
EXPOSE 41242

# Set workspace as working directory
WORKDIR /workspace

# Default entrypoint runs the A2A server
CMD ["llxprt-code-a2a-server"]
