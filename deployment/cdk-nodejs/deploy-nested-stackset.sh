#!/bin/bash

# Deploy Single StackSet with Nested Child Stacks

set -e

MY_IP=$(curl -s https://checkip.amazonaws.com)
PROJECT_NAME=${1:-"robotics-training"}
ENVIRONMENT=${2:-"dev"}
TARGET_REGIONS=${3:-"us-east-1"}
INSTANCE_TYPE=${4:-"g4dn.xlarge"}
KEY_NAME=${5:-""}
ALLOWED_CIDRS=${6:-"$MY_IP/32"}
ROOT_VOLUME_SIZE=${7:-"150"}

echo "=== Deploying Nested StackSet ==="
echo "Project: $PROJECT_NAME"
echo "Environment: $ENVIRONMENT"
echo "Target Regions: $TARGET_REGIONS"
echo "Instance Type: $INSTANCE_TYPE"
echo "SSH Key: ${KEY_NAME:-"None"}"
echo "Allowed CIDRs: $ALLOWED_CIDRS"
echo "Root Volume Size: ${ROOT_VOLUME_SIZE}GB"

# Convert comma-separated regions to JSON array
REGIONS_ARRAY=$(echo "$TARGET_REGIONS" | sed 's/,/","/g' | sed 's/^/["/' | sed 's/$/"]/')

# Build context parameters
CONTEXT_PARAMS="--context projectName=$PROJECT_NAME --context environment=$ENVIRONMENT --context targetRegions=$REGIONS_ARRAY"

if [ -n "$KEY_NAME" ]; then
    CONTEXT_PARAMS="$CONTEXT_PARAMS --context keyName=$KEY_NAME"
fi

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

# Get account ID for S3 bucket naming
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
CONTEXT_PARAMS="$CONTEXT_PARAMS --context accountId=$ACCOUNT_ID"

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
  --context allowedCidrBlocks="$CIDRS_ARRAY" \
  --require-approval never

echo ""
echo "=== Deployment Complete ==="
echo "Stack Name: $PROJECT_NAME-$ENVIRONMENT-stack"
echo "Instance Type: $INSTANCE_TYPE"
echo "SSH Key: $KEY_NAME"
echo ""
echo "Get instance details:"
echo "aws cloudformation describe-stacks --stack-name $PROJECT_NAME-$ENVIRONMENT-stack --query 'Stacks[0].Outputs'"
echo ""
echo "Get DCV password:"
echo "aws secretsmanager get-secret-value --secret-id $PROJECT_NAME-$ENVIRONMENT-password --query SecretString --output text | jq -r '.password'"