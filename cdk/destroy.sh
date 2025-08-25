#!/bin/bash
# AWS CDK Destruction Script for Mexican Revolution RAG System

set -e

echo "🗑️ Starting AWS CDK destruction for Mexican Revolution RAG System..."

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

# Destroy the stack
echo "🏗️ Destroying CDK stack..."
cdk destroy --force

echo "✅ Destruction completed successfully!"
echo ""
echo "🗑️ All AWS resources have been removed."
echo "💡 Note: Some resources may take a few minutes to be fully cleaned up."
