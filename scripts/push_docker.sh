#!/bin/bash
# Usage: ./scripts/push_docker.sh <dockerhub-username> <tag>
USERNAME=${1:-yourusername}
TAG=${2:-latest}
docker build -t $USERNAME/parlay:$TAG .
docker push $USERNAME/parlay:$TAG
echo "Pushed $USERNAME/parlay:$TAG"
echo "For HF Spaces: set Dockerfile app_port to 7860 and push repo to huggingface.co/spaces/$USERNAME/parlay"
