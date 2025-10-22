# Robotics Training CDK Deployment

## Overview

This CDK application deploys a complete robotics training environment with Isaac Sim, NVIDIA DCV, and ROS 2 on AWS using nested CloudFormation stacks.

## Architecture

The deployment creates four nested stacks:

### 1. VPC Stack
- **VPC** with configurable CIDR block (default: 10.0.0.0/16)
- **2 Public Subnets** across 2 Availability Zones
- **2 Private Subnets** across 2 Availability Zones
- **1 NAT Gateway** in first public subnet for private subnet internet access
- **Internet Gateway** for public internet access
- **Route Tables** for public and private subnet routing

### 2. IAM & Secrets Stack
- **IAM Role** with permissions for:
  - Amazon Bedrock Full Access
  - SSM Managed Instance Core
  - Secrets Manager read access
  - S3 Full Access to project bucket
- **Instance Profile** for EC2 instance
- **Secrets Manager Secret** with auto-generated password for ubuntu user
- **S3 Bucket** with naming: `{projectName}-{environment}-{accountId}-{region}`

### 3. EC2 Stack
- **Security Group** allowing:
  - SSH (port 22) from anywhere
  - DCV (port 8443) from anywhere
- **EC2 Instance** with:
  - GPU-enabled instance type (default: g4dn.xlarge)
  - Deep Learning AMI with NVIDIA drivers
  - Configurable root volume size (default: 100GB)
  - User data script for Isaac Sim and DCV installation

### 4. EKS Stack
- **EKS Cluster** (Kubernetes v1.30) with:
  - Public and private endpoint access
  - CloudWatch logging enabled
  - Deployed in private subnets
- **Managed Node Group** with:
  - Trainium instances (trn1.2xlarge)
  - Amazon Linux 2023 Neuron AMI
  - 2 nodes (min/max/desired)
  - Deployed in private subnets

## Prerequisites

- AWS CLI configured with appropriate credentials
- Node.js 18+ installed
- AWS CDK CLI installed: `npm install -g aws-cdk`

## Quick Start

```bash
# Basic deployment
./deploy-nested-stackset.sh robotics-training dev "us-east-1" "g4dn.xlarge" "MyKeyPair"

# Custom parameters
./deploy-nested-stackset.sh myproject prod "us-west-2" "g4dn.2xlarge" "MySSHKey"
```

## Parameters

### Script Parameters (Positional)
```bash
./deploy-nested-stackset.sh [PROJECT_NAME] [ENVIRONMENT] [REGION] [INSTANCE_TYPE] [SSH_KEY] [ALLOWED_CIDRS]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| PROJECT_NAME | robotics-training | Project identifier for resource naming |
| ENVIRONMENT | dev | Environment name (dev/staging/prod) |
| REGION | us-east-1 | AWS region for deployment |
| INSTANCE_TYPE | g4dn.xlarge | EC2 instance type |
| SSH_KEY | "" | SSH key pair name (optional) |
| ALLOWED_CIDRS | 0.0.0.0/0 | Comma-separated CIDR blocks for SSH/DCV access |

### CDK Context Parameters

You can also pass parameters directly via CDK context:

```bash
cdk deploy \
  --context projectName="my-robotics" \
  --context environment="prod" \
  --context instanceType="g4dn.4xlarge" \
  --context keyName="MyKeyPair" \
  --context rootVolumeSize=200 \
  --context vpcCidr="172.16.0.0/16"
```

| Context Parameter | Type | Default | Description |
|-------------------|------|---------|-------------|
| projectName | string | robotics-training | Project name for resource naming |
| environment | string | dev | Environment identifier |
| instanceType | string | g4dn.xlarge | EC2 instance type |
| keyName | string | undefined | SSH key pair name |
| rootVolumeSize | number | 100 | Root volume size in GB |
| vpcCidr | string | 10.0.0.0/16 | VPC CIDR block |
| allowedCidrBlocks | array | ["0.0.0.0/0"] | CIDR blocks allowed for SSH/DCV access |

## Instance Types & Costs

### Recommended GPU Instance Types

| Instance Type | vCPU | RAM | GPU | GPU Memory | Cost/Hour (us-east-1) |
|---------------|------|-----|-----|------------|----------------------|
| g4dn.xlarge | 4 | 16GB | 1x T4 | 16GB | ~$0.526 |
| g4dn.2xlarge | 8 | 32GB | 1x T4 | 16GB | ~$0.752 |
| g4dn.4xlarge | 16 | 64GB | 1x T4 | 16GB | ~$1.505 |
| g4dn.8xlarge | 32 | 128GB | 1x T4 | 16GB | ~$2.176 |
| g4dn.12xlarge | 48 | 192GB | 4x T4 | 64GB | ~$3.912 |

### Storage Options

| Volume Size | Use Case | Monthly Cost |
|-------------|----------|--------------|
| 150GB | Basic Isaac Sim (default) | ~$15 |
| 250GB | Multiple projects | ~$25 |
| 500GB | Large datasets | ~$50 |

## Deployment Examples

### Development Environment
```bash
# Open access (default)
./deploy-nested-stackset.sh robotics-dev dev "us-east-1" "g4dn.xlarge" "dev-keypair"

# Restrict to your IP only
MY_IP=$(curl -s https://checkip.amazonaws.com)
./deploy-nested-stackset.sh robotics-dev dev "us-east-1" "g4dn.xlarge" "dev-keypair" "$MY_IP/32"

# Multiple CIDR blocks
./deploy-nested-stackset.sh robotics-dev dev "us-east-1" "g4dn.xlarge" "dev-keypair" "203.0.113.0/24,198.51.100.0/24"

# Custom volume size (250GB)
./deploy-nested-stackset.sh robotics-dev dev "us-east-1" "g4dn.xlarge" "dev-keypair" "0.0.0.0/0" "250"
```

### Production Environment with Custom Settings
```bash
cdk deploy \
  --context projectName="robotics-prod" \
  --context environment="prod" \
  --context instanceType="g4dn.4xlarge" \
  --context keyName="prod-keypair" \
  --context rootVolumeSize=500 \
  --context vpcCidr="172.16.0.0/16" \
  --context allowedCidrBlocks='["203.0.113.0/24","198.51.100.0/24"]'
```

### Multi-Region Deployment
```bash
# Deploy to us-west-2
./deploy-nested-stackset.sh robotics-west dev "us-west-2" "g4dn.2xlarge" "west-keypair"

# Deploy to eu-west-1
./deploy-nested-stackset.sh robotics-eu dev "eu-west-1" "g4dn.xlarge" "eu-keypair"
```

## Post-Deployment

### Get Instance Information
```bash
# Get all stack outputs
aws cloudformation describe-stacks \
  --stack-name robotics-training-dev-stack \
  --query 'Stacks[0].Outputs'

# Get instance public IP
aws cloudformation describe-stacks \
  --stack-name robotics-training-dev-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`InstancePublicIp`].OutputValue' \
  --output text
```

### Access the Instance

#### SSH Access
```bash
ssh -i MyKeyPair.pem ubuntu@<instance-public-ip>
```

#### DCV Access
1. Get the password:
```bash
aws secretsmanager get-secret-value \
  --secret-id robotics-training-dev-password \
  --query SecretString --output text | jq -r '.password'
```

2. Open browser to: `https://<instance-public-ip>:8443`
3. Login with username: `ubuntu` and the retrieved password

### Isaac Sim Usage
```bash
# SSH into instance
ssh -i MyKeyPair.pem ubuntu@<instance-public-ip>

# Navigate to Isaac Sim directory
cd /home/ubuntu/isaacsim

# Run Isaac Sim
./isaac-sim.sh
```

### S3 Bucket Usage
```bash
# Get bucket name from stack outputs
BUCKET_NAME=$(aws cloudformation describe-stacks \
  --stack-name robotics-training-dev-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
  --output text)

# Upload files to S3
aws s3 cp myfile.txt s3://$BUCKET_NAME/

# List bucket contents
aws s3 ls s3://$BUCKET_NAME/

# Download files from S3
aws s3 cp s3://$BUCKET_NAME/myfile.txt ./
```

### EKS Cluster Usage
```bash
# Get cluster name from stack outputs
CLUSTER_NAME=$(aws cloudformation describe-stacks \
  --stack-name robotics-training-dev-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`EksClusterName`].OutputValue' \
  --output text)

# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name $CLUSTER_NAME

# Verify cluster access
kubectl get nodes

# Check node group status
kubectl get nodes -o wide

# Deploy a sample workload
kubectl create deployment nginx --image=nginx
kubectl expose deployment nginx --port=80 --type=LoadBalancer
```

## Resource Naming Convention

All resources follow the pattern: `{projectName}-{environment}-{resourceType}`

S3 buckets include account ID and region: `{projectName}-{environment}-{accountId}-{region}`

Examples:
- Stack: `robotics-training-dev-stack`
- VPC: `robotics-training-dev-vpc`
- Instance: `robotics-training-dev-instance`
- Secret: `robotics-training-dev-password`
- Role: `robotics-training-dev-ec2-role`
- S3 Bucket: `robotics-training-dev-123456789012-us-east-1`
- EKS Cluster: `robotics-training-dev-cluster`
- EKS Node Group: `robotics-training-dev-nodegroup`

## Cleanup

### Delete Stack
```bash
cdk destroy robotics-training-dev-stack
```

### Manual Cleanup (if needed)
```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name robotics-training-dev-stack

# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name robotics-training-dev-stack
```

## Troubleshooting

### Common Issues

#### 1. Instance Launch Failures
- Check if the specified instance type is available in the region
- Verify you have sufficient EC2 limits for GPU instances
- Ensure the SSH key exists in the target region

#### 2. Permission Errors
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify CDK bootstrap
cdk bootstrap aws://ACCOUNT-ID/REGION
```

#### 3. DCV Connection Issues
- Verify security group allows port 8443
- Check instance is running and DCV service is started
- Ensure you're using HTTPS (not HTTP)

#### 4. Isaac Sim Installation Issues
- SSH into instance and check logs: `tail -f /var/log/user-data.log`
- Verify GPU drivers: `nvidia-smi`
- Check disk space: `df -h`

### Monitoring

#### CloudWatch Logs
- Instance logs: `/aws/ec2/user-data`
- Application logs: Custom log groups created by applications

#### Instance Health
```bash
# Check instance status
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0

# Check system logs
aws ec2 get-console-output --instance-id i-1234567890abcdef0
```

## Security Considerations

### Network Security
- Security group allows SSH and DCV from anywhere (0.0.0.0/0)
- Consider restricting to your IP range for production:
  ```bash
  # Get your IP
  MY_IP=$(curl -s https://checkip.amazonaws.com)
  
  # Update security group
  aws ec2 authorize-security-group-ingress \
    --group-id sg-12345678 \
    --protocol tcp \
    --port 22 \
    --cidr $MY_IP/32
  ```

### Access Control
- Use IAM roles instead of access keys
- Enable CloudTrail for API logging
- Consider using Session Manager instead of SSH

### Data Protection
- Root volume is not encrypted by default
- Consider enabling EBS encryption for sensitive data
- Secrets Manager automatically encrypts passwords

## Cost Optimization

### Instance Management
```bash
# Stop instance when not in use
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance when needed
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

### Scheduled Operations
Consider using AWS Instance Scheduler to automatically start/stop instances based on schedule.

## Support

### Getting Help
1. Check CloudFormation events in AWS Console
2. Review instance user-data logs
3. Verify IAM permissions and service limits
4. Check AWS service health dashboard

### Useful Commands
```bash
# List all stacks
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE

# Get stack events
aws cloudformation describe-stack-events --stack-name robotics-training-dev-stack

# Check CDK diff
cdk diff robotics-training-dev-stack
```