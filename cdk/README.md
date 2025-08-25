# AWS CDK Deployment for Mexican Revolution RAG System

This directory contains the AWS CDK infrastructure code to deploy the Mexican Revolution RAG system to AWS with production-ready architecture.

## 🏗️ Infrastructure Overview

The CDK stack creates a complete production environment with:

### **Core Infrastructure**
- **VPC**: Multi-AZ VPC with public and private subnets
- **ECS Fargate**: Serverless container orchestration with Blue/Green deployment
- **Application Load Balancer**: High-availability load balancing
- **Auto Scaling**: CPU and memory-based scaling (2-10 instances)
- **WAF v2**: Rate limiting and security protection

### **Security & Monitoring**
- **IAM Roles**: Least-privilege access for ECS tasks
- **Secrets Manager**: Secure storage of OpenAI API key
- **CloudWatch**: Comprehensive monitoring and logging
- **Health Checks**: Application-level health monitoring
- **WAF v2**: Web application firewall with rate limiting

### **CI/CD Pipeline**
- **CodePipeline**: Automated build and deployment pipeline
- **CodeBuild**: Docker image building and testing
- **CodeDeploy**: Blue/Green deployment strategy
- **GitHub Actions**: Alternative CI/CD workflow
- **S3 Artifacts**: Secure artifact storage

### **Networking**
- **HTTPS Support**: SSL/TLS termination for secure communication
- **Multi-AZ Deployment**: High availability across availability zones
- **Security Groups**: Network-level security controls

## 📋 Prerequisites

### **AWS Setup**
1. **AWS CLI**: Install and configure AWS CLI
   ```bash
   aws configure
   ```

2. **AWS CDK**: Install CDK globally
   ```bash
   npm install -g aws-cdk
   ```

3. **Python Dependencies**: Install CDK Python dependencies
   ```bash
   pip install -r requirements.txt
   ```

### **Environment Variables**
Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 Deployment

### **Quick Deploy**
```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy to AWS
./deploy.sh
```

### **Manual Deployment**
```bash
# Install dependencies
pip install -r requirements.txt

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy the stack
cdk deploy
```

## 🗑️ Cleanup

### **Destroy Infrastructure**
```bash
# Make destruction script executable
chmod +x destroy.sh

# Destroy all resources
./destroy.sh
```

### **Manual Cleanup**
```bash
cdk destroy
```

## 📊 Architecture Details

### **ECS Fargate Configuration (Blue/Green)**
- **CPU**: 1 vCPU (1024 CPU units)
- **Memory**: 2 GB RAM
- **Blue Service**: 2 instances (auto-scaling 2-10)
- **Green Service**: 0 instances initially (scaled during deployment)
- **Health Check**: HTTP health check on `/health` endpoint
- **Deployment Strategy**: Blue/Green with automatic rollback

### **Load Balancer Configuration**
- **FastAPI**: HTTP on port 80
- **Streamlit**: HTTPS on port 443
- **Health Checks**: 30-second intervals
- **Target Groups**: Separate groups for FastAPI and Streamlit

### **Auto Scaling Rules**
- **CPU Scaling**: Scale out at 70% CPU utilization
- **Memory Scaling**: Scale out at 80% memory utilization
- **Cooldown**: 60 seconds between scaling actions
- **Blue/Green**: Independent scaling for each environment

### **Rate Limiting & Security**
- **WAF v2**: 2000 requests per 5 minutes per IP
- **API Gateway**: Per-endpoint rate limiting
- **DDoS Protection**: AWS Shield integration
- **Security Rules**: Common attack protection

### **Monitoring Dashboard**
CloudWatch dashboard includes:
- Blue/Green service CPU and memory utilization
- Request count and response time
- WAF rate limiting metrics
- Error rates and health check status
- Deployment status and rollback metrics

## 🔧 Configuration

### **Environment Variables**
The application uses these environment variables:
- `ENVIRONMENT`: Set to "production"
- `LOG_LEVEL`: Set to "INFO"
- `OPENAI_API_KEY`: Retrieved from AWS Secrets Manager

## 📈 Monitoring & Logging

### **CloudWatch Logs**
- **Log Group**: `/aws/ecs/rag-system`
- **Retention**: 1 month
- **Stream Prefix**: `rag-system`

### **Metrics Available**
- ECS service metrics (CPU, memory, network)
- Load balancer metrics (requests, response time)
- Application metrics (health check status)
## 🔒 Security Features

### **Network Security**
- VPC with private subnets for ECS tasks
- Security groups limiting access
- HTTPS termination at load balancer

### **Access Control**
- IAM roles with least privilege
- Secrets Manager for sensitive data
- No direct SSH access to containers

### **Data Protection**
- Encryption in transit (HTTPS)
- Encryption at rest (EBS volumes)
- Secure secret management


## 🚨 Troubleshooting

### **Common Issues**

1. **Deployment Fails**
   ```bash
   # Check CDK bootstrap
   cdk bootstrap
   
   # Verify AWS credentials
   aws sts get-caller-identity
   ```

2. **Container Health Check Fails**
   - Verify application is listening on correct ports
   - Check application logs in CloudWatch
   - Ensure health check endpoint returns 200

3. **High Costs**
   - Review auto-scaling configuration
   - Check for resource leaks
   - Monitor CloudWatch metrics

```

