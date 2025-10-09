# Robotics Training StackSet Deployment Guide

## Quick Start

```bash
# Deploy to single region with your IP
cd deployment/cdk-nodejs
./deploy-nested-stackset.sh robotics-training dev "us-east-1" "g4dn.xlarge" "MyAWSKeyPair"

# Deploy to multiple regions
./deploy-nested-stackset.sh robotics-training dev "us-east-1,us-west-2,eu-west-1"
```

## Prerequisites

### Required Tools
```bash
# Install Node.js (>=18.x)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install AWS CDK
npm install -g aws-cdk

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
```

### AWS Configuration
```bash
# Configure AWS credentials
aws configure

# Create SSH key pair (if needed)
aws ec2 create-key-pair --key-name MyKeyPair --query 'KeyMaterial' --output text > MyKeyPair.pem
chmod 400 MyKeyPair.pem
```

### Get Your IP Address
```bash
# Get your public IP for security group
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "Your IP: $MY_IP/32"
```

## Configuration

### Basic Configuration
```bash
# Edit cdk.context.json
{
  "projectName": "robotics-training",
  "environment": "dev",
  "targetRegions": ["us-east-1", "us-west-2"],
  "allowedCidrBlocks": ["YOUR.IP.ADDRESS/32"]
}
```

### Advanced Configuration
```bash
# Multi-account deployment
{
  "projectName": "robotics-training",
  "environment": "prod",
  "targetRegions": ["us-east-1", "us-west-2", "eu-west-1"],
  "targetAccounts": ["123456789012", "234567890123"],
  "allowedCidrBlocks": ["203.0.113.0/24", "198.51.100.0/24"]
}
```

## Deployment Options

### 1. Single Region Deployment
```bash
./deploy-nested-stackset.sh robotics-training dev "us-east-1" "g4dn.xlarge" "MyKeyPair"
```

### 2. Multi-Region Deployment
```bash
./deploy-nested-stackset.sh robotics-training dev "us-east-1,us-west-2" "g4dn.2xlarge" "MyKeyPair"
```

### 3. Custom Parameters
```bash
# Deploy with custom settings
cdk deploy \
  --context projectName=robotics-training \
  --context environment=prod \
  --context targetRegions='["us-east-1","us-west-2"]' \
  --context allowedCidrBlocks='["203.0.113.0/24"]' \
  --parameters InstanceType=g4dn.4xlarge \
  --parameters KeyName=MyKeyPair \
  --parameters RootVolumeSize=200
```

## Parameters

### Required Parameters
| Parameter | Description | Example |
|-----------|-------------|---------|
| `projectName` | Project identifier | `robotics-training` |
| `environment` | Environment name | `dev`, `staging`, `prod` |
| `targetRegions` | Deployment regions | `["us-east-1", "us-west-2"]` |

### Optional Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `instanceType` | `g4dn.xlarge` | EC2 instance type |
| `keyName` | `""` | SSH key pair name |
| `rootVolumeSize` | `100` | Root volume size (GB) |
| `vpcCidr` | `10.0.0.0/16` | VPC CIDR block |
| `allowedCidrBlocks` | `0.0.0.0/0` | Allowed IP ranges |

### Security Configuration
```bash
# Restrict access to your IP only
--context allowedCidrBlocks='["YOUR.IP.ADDRESS/32"]'

# Allow multiple IP ranges
--context allowedCidrBlocks='["203.0.113.0/24","198.51.100.0/24"]'

# Corporate network access
--context allowedCidrBlocks='["10.0.0.0/8","172.16.0.0/12"]'
```

## Architecture

### StackSet Structure
```
Parent StackSet (Multi-Region)
├── VPC Child Stack
│   ├── VPC (10.0.0.0/16)
│   ├── 2 Public Subnets (auto-calculated)
│   ├── 2 Private Subnets (auto-calculated)
│   ├── Internet Gateway
│   ├── NAT Gateway (single)
│   └── Route Tables
├── IAM/Secrets Child Stack
│   ├── Secrets Manager Secret
│   ├── IAM Role (Bedrock + SSM access)
│   └── Instance Profile
└── EC2 Child Stack
    ├── Security Group (SSH + DCV)
    ├── EC2 Instance (GPU-enabled)
    └── User Data (Isaac Sim setup)
```

### Resources Created Per Region
- **1 VPC** with 4 subnets across 2 AZs
- **1 NAT Gateway** (cost-optimized)
- **1 EC2 Instance** with GPU support
- **1 Secrets Manager Secret** with auto-generated password
- **1 IAM Role** with required permissions
- **1 Security Group** with restricted access

## Management Commands

### Check Deployment Status
```bash
# List all StackSets
aws cloudformation list-stack-sets

# Check specific StackSet
aws cloudformation describe-stack-set \
  --stack-set-name robotics-training-dev-stackset

# List stack instances
aws cloudformation list-stack-instances \
  --stack-set-name robotics-training-dev-stackset
```

### Monitor Operations
```bash
# List operations
aws cloudformation list-stack-set-operations \
  --stack-set-name robotics-training-dev-stackset

# Check operation status
aws cloudformation describe-stack-set-operation \
  --stack-set-name robotics-training-dev-stackset \
  --operation-id <operation-id>
```

### Update StackSet
```bash
# Update with new parameters
cdk deploy \
  --context projectName=robotics-training \
  --context environment=dev \
  --parameters InstanceType=g4dn.2xlarge
```

### Access Instances
```bash
# Get instance IPs
aws cloudformation describe-stacks \
  --stack-name robotics-training-dev-stackset-<region> \
  --query 'Stacks[0].Outputs[?OutputKey==`InstancePublicIp`].OutputValue' \
  --output text

# SSH to instance
ssh -i MyKeyPair.pem ubuntu@<instance-ip>

# Get DCV password
aws secretsmanager get-secret-value \
  --secret-id robotics-training-dev-password \
  --query SecretString --output text | jq -r '.password'
```

## Cost Optimization

### Instance Types & Costs
| Instance Type | vCPU | RAM | GPU | Cost/Hour |
|---------------|------|-----|-----|-----------|
| g4dn.xlarge | 4 | 16GB | T4 | $0.526 |
| g4dn.2xlarge | 8 | 32GB | T4 | $0.752 |
| g4dn.4xlarge | 16 | 64GB | T4 | $1.505 |

### Multi-Region Costs
```bash
# 3 regions × g4dn.xlarge = ~$1.58/hour
# Additional costs per region:
# - NAT Gateway: ~$45/month
# - Data transfer: Variable
```

### Cost Management
```bash
# Stop instances when not in use
aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=robotics-training" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

# Start instances when needed
aws ec2 start-instances --instance-ids <instance-ids>
```

## Troubleshooting

### Common Issues

#### Bootstrap Required
```bash
# Bootstrap all target regions
aws sts get-caller-identity --query Account --output text | \
  xargs -I {} cdk bootstrap aws://{}/<region>
```

#### Permission Errors
```bash
# Check IAM permissions
aws iam get-role --role-name robotics-training-StackSetAdministrationRole
aws iam get-role --role-name robotics-training-StackSetExecutionRole
```

#### Deployment Failures
```bash
# Check failed operations
aws cloudformation list-stack-set-operations \
  --stack-set-name robotics-training-dev-stackset \
  --query 'Summaries[?Status==`FAILED`]'
```

#### Instance Access Issues
```bash
# Check security group rules
aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=robotics-training-*-ec2-sg"

# Verify your IP
curl -s https://checkip.amazonaws.com
```

## Cleanup

### Delete StackSet
```bash
# Delete all stack instances first
aws cloudformation delete-stack-instances \
  --stack-set-name robotics-training-dev-stackset \
  --regions us-east-1 us-west-2 \
  --accounts $(aws sts get-caller-identity --query Account --output text) \
  --retain-stacks false

# Wait for deletion to complete, then delete StackSet
aws cloudformation delete-stack-set \
  --stack-set-name robotics-training-dev-stackset
```

### CDK Cleanup
```bash
# Destroy via CDK
cdk destroy --all
```

## Security Best Practices

### Network Security
- **Restrict CIDR blocks** to your IP address only
- **Use VPC endpoints** for AWS services (optional)
- **Enable VPC Flow Logs** for monitoring

### Access Control
- **SSH key-based authentication** only
- **Secrets Manager** for password storage
- **IAM roles** with least privilege
- **Session Manager** for secure access

### Monitoring
```bash
# Enable CloudTrail
aws cloudtrail create-trail --name robotics-training-trail \
  --s3-bucket-name your-cloudtrail-bucket

# Set up CloudWatch alarms
aws cloudwatch put-metric-alarm \
  --alarm-name "High-CPU-Usage" \
  --alarm-description "Alert when CPU exceeds 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold
```

## Support

### Getting Help
1. Check deployment logs in CloudFormation console
2. Verify IAM permissions and service limits
3. Review security group and network configuration
4. Check AWS service health dashboard

### Useful Resources
- [AWS StackSets Documentation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/what-is-cfnstacksets.html)
- [CDK StackSets Guide](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_cloudformation.CfnStackSet.html)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)