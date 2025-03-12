#!/bin/bash

echo "Sourcing environment variables"
source .env

# Variables
DOCKER_USERNAME="${DOCKER_USERNAME:-your_default_username}"
IMAGE_NAME="tft_vod_analysis"
TAG="latest"
EC2_USER="ubuntu"
EC2_HOST="${EC2_HOST:-your_default_host}"
SSH_KEY_PATH="~/Desktop/tft-pipeline-key.pem"  # Path to your SSH key file
CLOUDWATCH_LOG_GROUP="tft-pipeline-logs"
CLOUDWATCH_REGION="us-west-1"

# Prompt for Docker Hub credentials
read -p "Enter Docker Hub username: " DOCKER_HUB_USERNAME
read -sp "Enter Docker Hub password: " DOCKER_HUB_PASSWORD
echo

# Build the Docker image specifically for x86_64
echo "Building Docker image for x86_64..."
docker build --no-cache --platform linux/amd64 -t $DOCKER_USERNAME/$IMAGE_NAME:$TAG .

# Push to Docker Hub
echo "Pushing to Docker Hub..."
echo "$DOCKER_HUB_PASSWORD" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
docker push $DOCKER_USERNAME/$IMAGE_NAME:$TAG

# Prepare .env file for secure transfer
echo "Transferring .env file..."
scp -i $SSH_KEY_PATH .env $EC2_USER@$EC2_HOST:~/.env

# Connect to EC2 and update
echo "Deploying to EC2..."
ssh -i $SSH_KEY_PATH $EC2_USER@$EC2_HOST "
  # Update package lists and install AWS CLI if not already installed
  sudo apt-get update
  sudo apt-get install -y awscli

  # Create CloudWatch log group (will not error if it already exists)
  aws logs create-log-group --log-group-name $CLOUDWATCH_LOG_GROUP --region $CLOUDWATCH_REGION || true

  # Configure Docker daemon to use CloudWatch logs
  sudo mkdir -p /etc/docker
  echo '{
    \"log-driver\": \"awslogs\",
    \"log-opts\": {
      \"awslogs-region\": \"$CLOUDWATCH_REGION\",
      \"awslogs-group\": \"$CLOUDWATCH_LOG_GROUP\",
      \"awslogs-create-group\": \"true\",
      \"tag\": \"{{.Name}}\"
    }
  }' | sudo tee /etc/docker/daemon.json

  # Restart Docker to apply logging configuration
  sudo systemctl restart docker

  # Login to Docker Hub
  docker login -u "$DOCKER_HUB_USERNAME" -p '$DOCKER_HUB_PASSWORD'

  # Pull the latest image
  docker pull $DOCKER_USERNAME/$IMAGE_NAME:$TAG
  
  # Stop and remove existing container
  docker stop $IMAGE_NAME || true
  docker rm $IMAGE_NAME || true
  
  # Create volume for logs if it doesn't exist
  docker volume create pipeline-logs
  
  # Run the TFT pipeline container with env file and CloudWatch logging
  docker run -d --name $IMAGE_NAME \\
    --log-driver awslogs \\
    --log-opt awslogs-region=$CLOUDWATCH_REGION \\
    --log-opt awslogs-group=$CLOUDWATCH_LOG_GROUP \\
    --log-opt tag={{.Name}} \\
    --env-file ~/.env \\
    -v pipeline-logs:/app/logs \\
    --restart unless-stopped \\
    $DOCKER_USERNAME/$IMAGE_NAME:$TAG
  
  # Show running containers
  docker ps
  
  # Clean up unused images
  docker system prune -f
"

echo "Deployment completed!"