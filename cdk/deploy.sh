#!/bin/bash
# AWS CDK Deployment Script for Mexican Revolution RAG System

set -e

echo "🚀 Starting AWS CDK deployment for Mexican Revolution RAG System..."

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if CDK is installed
if ! command -v cdk &> /dev/null; then
    echo "❌ AWS CDK not installed. Installing..."
    npm install -g aws-cdk
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Bootstrap CDK (if not already bootstrapped)
echo "🔧 Bootstrapping CDK..."
cdk bootstrap

# Set up OpenAI API key in AWS Secrets Manager
echo "🔐 Setting up OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY environment variable not set."
    echo "Please set it: export OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Create or update the secret
aws secretsmanager create-secret \
    --name "rag-system/openai-api-key" \
    --description "OpenAI API Key for RAG System" \
    --secret-string "{\"OPENAI_API_KEY\":\"$OPENAI_API_KEY\"}" \
    --region us-east-1 2>/dev/null || \
aws secretsmanager update-secret \
    --secret-id "rag-system/openai-api-key" \
    --secret-string "{\"OPENAI_API_KEY\":\"$OPENAI_API_KEY\"}" \
    --region us-east-1

echo "✅ OpenAI API key configured in AWS Secrets Manager"

# Deploy the stack
echo "🏗️ Deploying CDK stack..."
cdk deploy --require-approval never

echo "✅ Deployment completed successfully!"
echo ""
echo "🌐 Your RAG system is now deployed on AWS!"
echo "📊 Check the outputs above for your application URLs"
echo "📈 Monitor your application in the AWS Console"
echo ""
echo "🔗 Useful links:"
echo "   - ECS Console: https://console.aws.amazon.com/ecs/"
echo "   - CloudWatch: https://console.aws.amazon.com/cloudwatch/"
echo "   - Load Balancer: https://console.aws.amazon.com/ec2/v2/home#LoadBalancers"
