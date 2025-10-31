#!/bin/bash

# Deploy Robotics Training Infrastructure with Named Arguments

set -e

# Default values
PROJECT_NAME="robotics-training"
ENVIRONMENT="dev"
TARGET_REGIONS="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"
KEY_NAME=""
ALLOWED_CIDRS="$(curl -s https://checkip.amazonaws.com)/32"
ROOT_VOLUME_SIZE="150"
VPC_CIDR="10.0.0.0/16"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--project)
      PROJECT_NAME="$2"
      shift 2
      ;;
    -e|--environment)
      ENVIRONMENT="$2"
      shift 2
      ;;
    -r|--region)
      TARGET_REGIONS="$2"
      shift 2
      ;;
    -i|--instance-type)
      INSTANCE_TYPE="$2"
      shift 2
      ;;
    -k|--key-name)
      KEY_NAME="$2"
      shift 2
      ;;
    -c|--allowed-cidrs)
      ALLOWED_CIDRS="$2"
      shift 2
      ;;
    -v|--volume-size)
      ROOT_VOLUME_SIZE="$2"
      shift 2
      ;;
    --vpc-cidr)
      VPC_CIDR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -p, --project NAME          Project name (default: robotics-training)"
      echo "  -e, --environment ENV       Environment (default: dev)"
      echo "  -r, --region REGION         AWS region (default: us-east-1)"
      echo "  -i, --instance-type TYPE    EC2 instance type (default: g4dn.xlarge)"
      echo "  -k, --key-name KEY          SSH key pair name (optional)"
      echo "  -c, --allowed-cidrs CIDRS   Comma-separated CIDR blocks (default: your IP/32)"
      echo "  -v, --volume-size SIZE      Root volume size in GB (default: 150)"
      echo "  --vpc-cidr CIDR             VPC CIDR block (default: 10.0.0.0/16)"
      echo "  -h, --help                  Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --project my-robotics --environment prod --region us-west-2"
      echo "  $0 -p robotics-dev -e dev -i g4dn.2xlarge -k my-keypair"
      echo "  $0 --allowed-cidrs 203.0.113.0/24,198.51.100.0/24 --volume-size 250"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "=== Deploying Robotics Training Infrastructure ==="
echo "Project: $PROJECT_NAME"
echo "Environment: $ENVIRONMENT"
echo "Region: $TARGET_REGIONS"
echo "Instance Type: $INSTANCE_TYPE"
echo "SSH Key: ${KEY_NAME:-None}"
echo "Allowed CIDRs: $ALLOWED_CIDRS"
echo "Root Volume Size: ${ROOT_VOLUME_SIZE}GB"
echo "VPC CIDR: $VPC_CIDR"
echo ""

# Install dependencies and build
echo "Installing dependencies..."
npm install
npm run build

# Bootstrap CDK in target regions
echo "Bootstrapping CDK in target regions..."
IFS=',' read -ra REGION_ARRAY <<< "$TARGET_REGIONS"
for region in "${REGION_ARRAY[@]}"; do
    echo "Bootstrapping in $region..."
    cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/$region || true
done

# Convert comma-separated CIDRs to JSON array
CIDRS_ARRAY=$(echo "$ALLOWED_CIDRS" | sed 's/,/","/g' | sed 's/^/["/' | sed 's/$/"]/')

# Deploy nested stacks
echo "Deploying nested stacks..."
cdk deploy \
  --context projectName="$PROJECT_NAME" \
  --context environment="$ENVIRONMENT" \
  --context instanceType="$INSTANCE_TYPE" \
  --context keyName="$KEY_NAME" \
  --context rootVolumeSize=$ROOT_VOLUME_SIZE \
  --context vpcCidr="$VPC_CIDR" \
  --context allowedCidrBlocks="$CIDRS_ARRAY" \
  --require-approval never

# Upload source files to S3
echo ""
echo "Uploading source files to S3..."
S3_BUCKET_NAME=$(aws cloudformation describe-stacks \
  --stack-name $PROJECT_NAME-$ENVIRONMENT-stack \
  --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" \
  --output text)

if [ -d "../../source/ur5_nova" ]; then
  aws s3 sync ../../source/ur5_nova s3://$S3_BUCKET_NAME/source/ur5_nova
  echo "Source files uploaded to s3://$S3_BUCKET_NAME/source/ur5_nova"
else
  echo "Source directory not found, skipping upload"
fi

echo ""
echo "=== Deployment Complete ==="
echo "Stack Name: $PROJECT_NAME-$ENVIRONMENT-stack"
echo "S3 Bucket: $S3_BUCKET_NAME"
echo ""
echo "Get instance details:"
echo "aws cloudformation describe-stacks --stack-name $PROJECT_NAME-$ENVIRONMENT-stack --query 'Stacks[0].Outputs'"
echo ""
echo "Get DCV password:"
echo "aws secretsmanager get-secret-value --secret-id $PROJECT_NAME-$ENVIRONMENT-password --query SecretString --output text | jq -r '.password'"
