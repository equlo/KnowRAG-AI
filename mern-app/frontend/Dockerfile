# Dockerfile for the React frontend - a two-stage build.
# Build:  docker build -t mern-tasks-frontend ./frontend

# ---------- Stage 1: compile the React app into static files ----------
# Node.js image used only for building; it is thrown away afterwards.
FROM node:22-alpine AS build

# Work inside /app.
WORKDIR /app

# Copy the dependency manifests first so `npm ci` is cached between code changes.
COPY package.json package-lock.json ./

# Install the exact versions recorded in package-lock.json.
RUN npm ci

# Copy the rest of the source code.
COPY . .

# Produce optimised HTML/CSS/JS in /app/dist.
RUN npm run build

# ---------- Stage 2: serve the static files with nginx ----------
# A tiny web server image; the final image contains no Node.js at all.
FROM nginx:1.27-alpine

# Our nginx config: serves the app and proxies /api to the backend.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built files from stage 1 into nginx's web root.
COPY --from=build /app/dist /usr/share/nginx/html

# nginx listens on port 80 inside the container.
EXPOSE 80
