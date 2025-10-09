# Robotics Training Infrastructure - CDK StackSet Deployment

## Prerequisites
- AWS CLI configured with appropriate permissions
- Node.js 18+ and npm installed
- CDK CLI installed: `npm install -g aws-cdk`

## Quick Deploy

1. **Install dependencies**
```bash
npm install
```

2. **Deploy StackSet** (includes bootstrapping)
```bash
chmod +x deploy-nested-stackset.sh
./deploy-nested-stackset.sh [PROJECT_NAME] [ENVIRONMENT] [REGIONS] [INSTANCE_TYPE] [KEY_NAME]
```

**Examples:**
```bash
# Default deployment (dev environment, us-east-1,us-west-2)
./deploy-nested-stackset.sh

# Custom deployment
./deploy-nested-stackset.sh robotics-training prod us-east-1,eu-west-1 g4dn.2xlarge my-key-pair
```

## Configuration

**Script Parameters:**
- `PROJECT_NAME`: Project identifier (default: robotics-training)
- `ENVIRONMENT`: Environment name (default: dev)
- `TARGET_REGIONS`: Comma-separated regions (default: us-east-1,us-west-2)
- `INSTANCE_TYPE`: EC2 instance type (default: g4dn.xlarge)
- `KEY_NAME`: SSH key pair name (optional)

**Additional settings** in `cdk.json`:
```json
{
  "context": {
    "vpcCidr": "10.0.0.0/16",
    "rootVolumeSize": 100
  }
}
```

## Validation

1. **Check StackSet status**
```bash
aws cloudformation describe-stack-set --stack-set-name robotics-training-dev-stackset
```

2. **Verify EC2 instance**
```bash
aws ec2 describe-instances --filters "Name=tag:Name,Values=robotics-training-dev-instance"
```

3. **Test DCV connection**
- Get instance public IP from EC2 console
- Connect to `https://<public-ip>:8443`
- Login with username `ubuntu` and password from Secrets Manager

4. **Verify Isaac Sim**
```bash
# SSH to instance
ssh -i your-key.pem ubuntu@<public-ip>
cd ~/isaacsim/
./isaac-sim.selector.sh
```

## Cleanup

1. **Delete stack instances first**
```bash
aws cloudformation delete-stack-instances --stack-set-name robotics-training-dev-stackset --regions us-east-1 us-west-2 --accounts $(aws sts get-caller-identity --query Account --output text)
```

2. **Delete StackSet**
```bash
aws cloudformation delete-stack-set --stack-set-name robotics-training-dev-stackset
```

## Architecture

**Nested StackSet Structure:**
```
Parent StackSet
├── VPC Child Stack (VPC, Subnets, NAT Gateway)
├── IAM/Secrets Child Stack (IAM Role, Secrets Manager)
└── EC2 Child Stack (Instance, Security Group)
```

## Troubleshooting

- **StackSet creation fails**: Check IAM permissions for CloudFormation StackSets
- **Bootstrap fails**: Ensure AWS CLI has admin permissions in target regions
- **EC2 launch fails**: Verify key pair exists and AMI is available in region
- **DCV not accessible**: Check security group allows port 8443 from your IP
- **Isaac Sim issues**: Check userdata script logs in `/var/log/cloud-init-output.log`
- **Multi-region issues**: Verify CDK bootstrap completed in all target regions