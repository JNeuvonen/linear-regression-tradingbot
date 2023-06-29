#!/bin/bash
yum update -y
yum install -y docker
service docker start
usermod -a -G docker ec2-user
chkconfig docker on
source .env
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
sudo docker pull jneuv/live_env:latest
sudo docker run -d -p 80:8080 jneuv/live_env:latest