# Dockerfile for the Express backend.
# Build:  docker build -t mern-tasks-backend ./backend

# Small official Node.js image (Alpine Linux).
FROM node:22-alpine

# Work inside /app.
WORKDIR /app

# Tell Express and npm that this is a production install.
ENV NODE_ENV=production

# Copy the dependency manifests first so `npm ci` is cached between code changes.
COPY package.json package-lock.json ./

# Install the exact versions from package-lock.json, skipping devDependencies (test tools).
RUN npm ci --omit=dev

# Copy the application code.
COPY src ./src

# Run as the unprivileged "node" user that the image provides, not as root.
USER node

# The server listens on port 4000 (see PORT in config.js).
EXPOSE 4000

# Start the server directly with node (signals like SIGTERM reach it without npm in between).
CMD ["node", "src/index.js"]
