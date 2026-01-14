# NVIDIA Isaac GR00T VLA Training on AWS

This guidance showcases NVIDIA Isaac GR00T Vision-Language-Action (VLA) model fine-tuning that combines the intelligence of foundation models with robotic learning capabilities, all accelerated by AWS GPU infrastructure and managed through modern cloud-native technologies. This guidance enables developers to fine-tune GR00T VLA models using teleoperation/simulation datasets on [Amazon EKS](https://aws.amazon.com/eks/) with [AWS Batch](https://aws.amazon.com/batch/), then automatically deploy trained models to physical robots through [AWS IoT services](https://aws.amazon.com/iot/).

## Table of Contents

### Required

1. [Overview](#overview)
   - [Cost](#cost)
   - [Sample Cost Table](#sample-cost-table)
2. [Prerequisites](#prerequisites)
   - [Operating System](#operating-system)
   - [Third-party tools](#third-party-tools)
   - [AWS account requirements](#aws-account-requirements)
3. [Deployment Steps](#deployment-steps)
   - [Architecture Overview](#architecture-overview)
   - [Deployment Path 1: Fully Automated Deployment (Recommended)](#deployment-path-1-fully-automated-deployment-recommended)
   - [Deployment Path 2: Using Existing AWS Resources](#deployment-path-2-using-existing-aws-resources)
   - [Deployment Path 3: Using Pre-built Container Image](#deployment-path-3-using-pre-built-container-image)
   - [Configuration Options](#configuration-options)
   - [Triggering Manual Container Rebuilds](#triggering-manual-container-rebuilds)
   - [Deployment Checklist](#deployment-checklist)
4. [Deployment Validation](#deployment-validation)
5. [Running the Guidance](#running-the-guidance)
6. [Next Steps](#next-steps)
7. [Cleanup](#cleanup)
8. [FAQ, known issues, additional considerations, and limitations](#faq-known-issues-additional-considerations-and-limitations-optional)
   - [Known issues](#known-issues)
   - [Service Limit](#service-limits)
9. [Revisions](#revisions-optional)
10. [Authors](#authors-optional)
11. [Notices](#notices)

## Overview

This guidance demonstrates how to build an AI-assisted robotic learning system using NVIDIA Isaac GR00T VLA models with AWS infrastructure. The system enables fine-tuning of pre-trained GR00T models using teleoperation and simulation datasets, with automatic deployment to physical robots via [AWS IoT services](https://aws.amazon.com/iot/).

**Why did you build this Guidance?**
Traditional robotic training requires extensive manual programming and domain expertise. This guidance solves the challenge of fine-tuning advanced VLA models like GR00T that can understand natural language instructions and perform complex manipulation tasks, leveraging AWS's AI/ML services for scalable robot training.

**What problem does this Guidance solve?**

- Reduces the complexity of fine-tuning NVIDIA Isaac GR00T VLA models
- Enables scalable training infrastructure using [AWS Batch](https://aws.amazon.com/batch/) and [GPU instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- Provides monitoring and evaluation capabilities through [Amazon DCV](https://aws.amazon.com/dcv/)
- Integrates with existing robotics datasets in LeRobot format
- Automates the deployment pipeline from training to physical robots

**Architecture Flow:**

1. **Dataset Preparation**: LeRobot datasets from teleoperation or simulation (sample dataset included in [sample_dataset/](sample_dataset/))
2. **Model Fine-tuning**: GR00T VLA models fine-tuned using [AWS Batch](https://aws.amazon.com/batch/) with GPU acceleration
3. **Infrastructure**: [AWS Batch](https://aws.amazon.com/batch/) manages training jobs, [Amazon DCV](https://aws.amazon.com/dcv/) provides monitoring interface
4. **Model Storage**: Trained models stored in [Amazon S3](https://aws.amazon.com/s3/) or Hugging Face Hub
5. **Deployment**: Fine-tuned models deployed to physical robots via [AWS IoT services](https://aws.amazon.com/iot/)

### Cost

_You are responsible for the cost of the AWS services used while running this Guidance. As of December 2024, the cost for running this Guidance with the default settings in the US East (N. Virginia) Region is approximately $456.80 per month for processing 100 training episodes with continuous model fine-tuning._

_We recommend creating a [Budget](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) through [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/) to help manage costs. Prices are subject to change. For full details, refer to the pricing webpage for each AWS service used in this Guidance._

### Sample Cost Table

The following table provides a sample cost breakdown for deploying this Guidance with the default parameters in the [US East (N. Virginia)](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/) Region for one month.

| AWS service                                                                                               | Dimensions                                   | Cost [USD]   |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------ |
| [Amazon EC2](https://aws.amazon.com/ec2/) ([g6e.4xlarge](https://aws.amazon.com/ec2/instance-types/g6e/)) | 2 instances for GR00T training (8 hours/day) | $ 285.60     |
| [Amazon EKS](https://aws.amazon.com/eks/) Cluster                                                         | 1 cluster for container orchestration        | $ 73.00      |
| [AWS Batch](https://aws.amazon.com/batch/)                                                                | Compute environment and job queue management | $ 15.00      |
| [Amazon DCV](https://aws.amazon.com/dcv/) ([g4dn.xlarge](https://aws.amazon.com/ec2/instance-types/g4/))  | 1 instance for monitoring and evaluation     | $ 45.00      |
| [Amazon S3](https://aws.amazon.com/s3/)                                                                   | 1TB storage for datasets and trained models  | $ 23.00      |
| [Amazon EFS](https://aws.amazon.com/efs/)                                                                 | 500 GB shared storage for checkpoints        | $ 12.50      |
| [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)                                            | 2 secrets for credentials management         | $ 0.80       |
| [Amazon VPC](https://aws.amazon.com/vpc/)                                                                 | NAT Gateway and data transfer                | $ 1.90       |
| **Total**                                                                                                 |                                              | **$ 456.80** |

## Prerequisites

### Operating System

These deployment instructions are optimized to best work on **Ubuntu 24.04 LTS with NVIDIA GPU support**. Deployment on other OS may require additional steps.

**Required packages:**

- Python 3.10+
- [AWS CLI](https://aws.amazon.com/cli/) v2
- Docker and Docker Compose
- [AWS CDK](https://aws.amazon.com/cdk/) v2
- Node.js 18+ and npm (for CDK deployment)

**Installation commands:**

```bash
# Install Python and pip
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-venv

# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install Docker
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER

# Install AWS CDK
npm install -g aws-cdk
```

### Third-party tools

**Required third-party tools:**

- **NVIDIA Isaac GR00T**: Vision-Language-Action foundation model
- **LeRobot**: Robotics learning framework for dataset handling
- **PyTorch 2.3.1**: Deep learning framework with CUDA support
- **Transformers**: Hugging Face transformers library
- **Amazon DCV**: Remote desktop for monitoring training progress

### AWS account requirements

**Required AWS account setup:**

- **EC2 instance limits**: Ensure sufficient quota for [g6e.4xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) and [g4dn.xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instances
- **EKS service**: Enable [Amazon EKS](https://aws.amazon.com/eks/) service in your target region
- **AWS Batch**: Enable [AWS Batch](https://aws.amazon.com/batch/) service for job management
- **IAM permissions**: Administrator access or specific permissions for:
  - [EC2](https://aws.amazon.com/ec2/) instance management
  - [EKS](https://aws.amazon.com/eks/) cluster creation
  - [AWS Batch](https://aws.amazon.com/batch/) job submission
  - [S3](https://aws.amazon.com/s3/) bucket operations
  - [EFS](https://aws.amazon.com/efs/) file system access
  - [Secrets Manager](https://aws.amazon.com/secrets-manager/) access
  - [DCV](https://aws.amazon.com/dcv/) session management
- **VPC**: Default [VPC](https://aws.amazon.com/vpc/) or custom VPC with internet gateway
- **Key Pair**: [EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html) for SSH access to instances

**Note**: Replace `ACCOUNT-NUMBER` with your AWS account ID and `REGION` with your target AWS region.

## Deployment Steps

This guidance provides multiple deployment paths to accommodate different environments and requirements. Choose the path that best fits your needs.

### Architecture Overview

The infrastructure consists of two independent but complementary CDK stacks:

1. **BatchStack** - Creates [AWS Batch](https://aws.amazon.com/batch/) resources for scalable fine-tuning jobs

   - [Amazon VPC](https://aws.amazon.com/vpc/) with public and private subnets (optional, can import existing)
   - [Amazon EFS](https://aws.amazon.com/efs/) file system for shared storage (optional, can import existing)
   - [Security group](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html) for EFS access
   - (Optional) [Amazon S3](https://aws.amazon.com/s3/) bucket for storing model checkpoints
   - (Optional) [AWS CodeBuild](https://aws.amazon.com/codebuild/) project for building the container image
   - [Amazon ECR](https://aws.amazon.com/ecr/) repository and container image
   - [EC2 Launch Template](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-templates.html) with increased root volume
   - [AWS Batch](https://aws.amazon.com/batch/) Compute Environment ([EC2](https://aws.amazon.com/ec2/) with [GPU instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing))
   - [AWS Batch](https://aws.amazon.com/batch/) Job Queue and Job Definition
   - [IAM roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) for Batch instances and job execution

2. **DcvStack** - Deploys an [Amazon EC2](https://aws.amazon.com/ec2/) instance with [Amazon DCV](https://aws.amazon.com/dcv/) for visualization and evaluation
   - [Amazon EC2](https://aws.amazon.com/ec2/) instance ([g6.4xlarge](https://aws.amazon.com/ec2/instance-types/g6/) with GPU and [Amazon DCV](https://aws.amazon.com/dcv/))
   - [Security group](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html) for DCV and TensorBoard access
   - [Elastic IP](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html) for stable connectivity
   - [IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) for EC2 instance
   - Mounts shared [EFS](https://aws.amazon.com/efs/) file system (if provided)

When both stacks are deployed together, they share [VPC](https://aws.amazon.com/vpc/), [EFS](https://aws.amazon.com/efs/), and [Security Groups](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html) to enable seamless data flow between training and evaluation workflows.

### Deployment Path 1: Fully Automated Deployment (Recommended)

Deploy both stacks automatically with [CDK](https://aws.amazon.com/cdk/). Works on **any architecture** (x86 or ARM) - the infrastructure automatically handles container building in the cloud when needed.

1. Clone the repository using command:

   ```bash
   # Make sure git-lfs is installed (https://git-lfs.com)
   git lfs install
   git clone https://github.com/aws-solutions-library-samples/guidance-for-ai-driven-robotic-simulation-and-training-on-aws.git
   ```

2. Navigate to the repository folder:

   ```bash
   cd guidance-for-ai-driven-robotic-simulation-and-training-on-aws
   ```

3. Navigate to the GR00T infrastructure directory:

   ```bash
   cd reinforcement_learning_vla/nvidia-gr00t-vla/gr00t/infra
   ```

4. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure AWS credentials:

   ```bash
   aws configure
   ```

6. Set AWS region for deployment:

   ```bash
   export AWS_REGION=us-west-2  # or your preferred region
   ```

7. Bootstrap CDK (if first time using CDK in this account/region):

   ```bash
   cdk bootstrap
   ```

8. Deploy the infrastructure stacks:

   ```bash
   # Deploy both Batch and DCV stacks (creates VPC, EFS, and Batch resources)
   cdk deploy IsaacGr00tBatchStack IsaacLabDcvStack
   ```

   **What happens automatically:**

   - If you don't provide an `ecr_image_uri`, the stack will:
     1. Create a CodeBuild project with x86 compute
     2. Build the container image in the cloud from the Dockerfile
     3. Push the image to ECR
     4. Use the built image for Batch jobs
   - **Build time**: First deployment takes 10-20 minutes for the container build

9. Monitor the container build (optional):

   ```bash
   # Get the CodeBuild project name from stack outputs
   export PROJECT_NAME=$(aws cloudformation describe-stacks \
     --stack-name IsaacGr00tBatchStack \
     --query 'Stacks[0].Outputs[?OutputKey==`CodeBuildProjectName`].OutputValue' \
     --output text)

   # Monitor build logs in real-time
   aws logs tail /aws/codebuild/$PROJECT_NAME --follow
   ```

10. Capture the deployed resources:

    ```bash
    # Get Batch job queue name
    aws cloudformation describe-stacks --stack-name IsaacGr00tBatchStack --query 'Stacks[0].Outputs[?OutputKey==`JobQueueName`].OutputValue' --output text

    # Get DCV instance public IP
    aws cloudformation describe-stacks --stack-name IsaacLabDcvStack --query 'Stacks[0].Outputs[?OutputKey==`DcvInstancePublicIp`].OutputValue' --output text

    # Get EFS file system ID
    aws cloudformation describe-stacks --stack-name IsaacGr00tBatchStack --query 'Stacks[0].Outputs[?OutputKey==`EfsFileSystemId`].OutputValue' --output text
    ```

### Deployment Path 2: Using Existing AWS Resources

If you already have [VPC](https://aws.amazon.com/vpc/), [EFS](https://aws.amazon.com/efs/), or other AWS resources, you can import them instead of creating new ones:

```bash
# Navigate to the infrastructure directory
cd reinforcement_learning_vla/nvidia-gr00t-vla/gr00t/infra

# Install dependencies and configure AWS
pip install -r requirements.txt
aws configure
export AWS_REGION=us-west-2

# Bootstrap CDK
cdk bootstrap

# Deploy with existing resources
cdk deploy IsaacGr00tBatchStack IsaacLabDcvStack \
  --context vpc_id=vpc-12345 \
  --context efs_id=fs-12345 \
  --context efs_sg_id=sg-12345 \
  --context dataset_bucket=my-dataset-bucket \
  --context s3_upload_uri=s3://my-checkpoint-bucket/gr00t/checkpoints
```

### Deployment Path 3: Using Pre-built Container Image

If you've already built and pushed a container image to [ECR](https://aws.amazon.com/ecr/), skip the automatic build:

```bash
# Navigate to the infrastructure directory
cd reinforcement_learning_vla/nvidia-gr00t-vla/gr00t/infra

# Install dependencies and configure AWS
pip install -r requirements.txt
aws configure
export AWS_REGION=us-west-2

# Bootstrap CDK
cdk bootstrap

# Deploy with pre-built image
cdk deploy IsaacGr00tBatchStack IsaacLabDcvStack \
  --context ecr_image_uri=123456789012.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest
```

### Configuration Options

You can customize the deployment using [CDK context parameters](https://docs.aws.amazon.com/cdk/v2/guide/context.html) or environment variables:

| Context Parameter | Env Variable     | Description                                         | Default                            |
| ----------------- | ---------------- | --------------------------------------------------- | ---------------------------------- |
| `vpc_id`          | `VPC_ID`         | Existing VPC ID to reuse                            | Creates new VPC                    |
| `efs_id`          | `EFS_ID`         | Existing EFS file system ID                         | Creates new EFS                    |
| `efs_sg_id`       | `EFS_SG_ID`      | EFS security group ID (required if `efs_id` is set) | Creates new SG                     |
| `ecr_image_uri`   | `ECR_IMAGE_URI`  | Pre-built ECR image URI (same region as deployment) | Automatically builds via CodeBuild |
| `dataset_bucket`  | `DATASET_BUCKET` | S3 bucket name for dataset read-only access         | No dataset bucket access           |
| `s3_upload_uri`   | `S3_UPLOAD_URI`  | S3 URI for checkpoint uploads                       | Creates new checkpoint bucket      |

**Using cdk.json for configuration:**

You can also provide configuration in `reinforcement_learning_vla/nvidia-gr00t-vla/gr00t/infra/cdk.json`:

```json
{
  "app": "python app.py",
  "context": {
    "vpc_id": "vpc-xxxxxxxx",
    "efs_id": "fs-xxxxxxxx",
    "efs_sg_id": "sg-xxxxxxxx",
    "ecr_image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/gr00t-finetune:latest",
    "dataset_bucket": "my-dataset-bucket",
    "s3_upload_uri": "s3://my-checkpoint-bucket/gr00t/checkpoints"
  }
}
```

### Triggering Manual Container Rebuilds

If you need to rebuild the container after making changes:

```bash
# Get the CodeBuild project name from stack outputs
export PROJECT_NAME=$(aws cloudformation describe-stacks \
  --stack-name IsaacGr00tBatchStack \
  --query 'Stacks[0].Outputs[?OutputKey==`CodeBuildProjectName`].OutputValue' \
  --output text)

# Trigger a new build
aws codebuild start-build --project-name $PROJECT_NAME

# Monitor the build
aws logs tail /aws/codebuild/$PROJECT_NAME --follow
```

### Deployment Checklist

Before deploying, ensure you have:

- [ ] [AWS CLI](https://aws.amazon.com/cli/) configured with credentials (`aws configure`)
- [ ] Node.js installed for [AWS CDK](https://aws.amazon.com/cdk/) CLI (`npm install -g aws-cdk`)
- [ ] Python 3.8+ installed
- [ ] Git LFS installed for large file support
- [ ] Requested [GPU instance quota](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) ([g6e.2xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) or larger) in your target region
- [ ] At least 8 vCPUs for "Running On-Demand G and VT instances" [service quota](https://docs.aws.amazon.com/servicequotas/latest/userguide/intro.html)

## Deployment Validation

**Validate successful deployment:**

1. **CloudFormation Stack Status:**

   ```bash
   aws cloudformation describe-stacks --stack-name IsaacGr00tBatchStack --query 'Stacks[0].StackStatus'
   aws cloudformation describe-stacks --stack-name IsaacLabDcvStack --query 'Stacks[0].StackStatus'
   ```

   Expected output: `"CREATE_COMPLETE"` for both stacks

2. **AWS Batch Resources:**

   ```bash
   aws batch describe-compute-environments --compute-environments IsaacGr00tComputeEnvironment
   aws batch describe-job-queues --job-queues IsaacGr00tJobQueue
   aws batch describe-job-definitions --job-definitions IsaacGr00tJobDefinition
   ```

   Expected: All resources should be in VALID/ENABLED state

3. **DCV Instance Status:**

   ```bash
   aws ec2 describe-instances --filters "Name=tag:Name,Values=*dcv*" --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' --output table
   ```

   Expected: Instance should be in `running` state

4. **EFS File System:**

   ```bash
   aws efs describe-file-systems --query 'FileSystems[?Name==`IsaacGr00tEFS`].[FileSystemId,LifeCycleState]' --output table
   ```

   Expected: File system should be in `available` state

5. **Container Registry:**

   ```bash
   aws ecr describe-repositories --repository-names isaac-gr00t-training
   aws ecr describe-images --repository-name isaac-gr00t-training
   ```

   Expected: Repository should exist with pushed container image

6. **Resource Sharing Validation:**

   When both stacks are deployed together, verify they share resources correctly:

   - **[VPC](https://aws.amazon.com/vpc/)**: DCV instance and Batch compute nodes run in the same network
   - **[EFS](https://aws.amazon.com/efs/)**: Mounted at `/mnt/efs` on both DCV instance and Batch containers
   - **[Security Group](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html)**: Allows both DCV and Batch instances to access EFS on port 2049

   This enables:

   - Real-time monitoring of training jobs via TensorBoard on the DCV instance
   - Direct access to model checkpoints for evaluation
   - Seamless data flow between training and evaluation workflows

## Running the Guidance

This section covers how to submit training jobs, monitor progress, and evaluate trained models. For detailed configuration parameters, see [gr00t/README.md](gr00t/README.md) and [gr00t/env.example](gr00t/env.example).

**Step 1: Access the DCV Monitoring Environment**

1. Connect to the DCV instance at `https://<DCV_INSTANCE_PUBLIC_IP>:8443`
2. Login with username `ubuntu` and retrieve password from [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/):
   ```bash
   aws secretsmanager get-secret-value --secret-id DcvInstancePassword --query SecretString --output text
   ```

**Step 2: Submit Training Jobs**

Submit GR00T fine-tuning jobs to [AWS Batch](https://aws.amazon.com/batch/) using one of the following methods:

**Basic training job (AWS CLI):**

```bash
aws batch submit-job \
  --job-name "IsaacGr00tFinetuning" \
  --job-queue "IsaacGr00tJobQueue" \
  --job-definition "IsaacGr00tJobDefinition"
```

**With custom dataset and parameters:**

```bash
aws batch submit-job \
  --job-name "IsaacGr00tFinetuning" \
  --job-queue "IsaacGr00tJobQueue" \
  --job-definition "IsaacGr00tJobDefinition" \
  --container-overrides 'environment=[
    {name=HF_DATASET_ID,value=lerobot/your-dataset},
    {name=MAX_STEPS,value=6000},
    {name=SAVE_STEPS,value=2000},
    {name=LEARNING_RATE,value=1e-5}
  ]'
```

**Multi-GPU training (4 GPUs with [g6e.12xlarge](https://aws.amazon.com/ec2/instance-types/g6e/)):**

```bash
aws batch submit-job \
  --job-name "IsaacGr00tMultiGPU" \
  --job-queue "IsaacGr00tJobQueue" \
  --job-definition "IsaacGr00tJobDefinition" \
  --container-overrides '{
    "environment": [
      {"name":"NUM_GPUS","value":"4"},
      {"name":"BATCH_SIZE","value":"8"},
      {"name":"DATALOADER_NUM_WORKERS","value":"2"}
    ],
    "resourceRequirements": [
      {"type":"GPU","value":"4"},
      {"type":"VCPU","value":"48"},
      {"type":"MEMORY","value":"393216"}
    ]
  }'
```

**Important Note on Multi-GPU Training:**
When using multiple GPUs, you may need to reduce `DATALOADER_NUM_WORKERS` (from default of 8) to avoid shared memory exhaustion. The provided Batch stack sets shared memory to 64GB, which is sufficient with reduced workers. Alternatively, you can increase the shared memory size in the job definition to match your instance capabilities.

**Using AWS Console:**

1. Go to AWS Batch → Jobs → Submit new job
2. Select `IsaacGr00tJobDefinition` and `IsaacGr00tJobQueue`
3. Add environment variables and select the number of GPUs as needed
4. Submit the job

**Dataset Format Note:**
If you use a custom dataset in [LerobotDataset:v3.0 format](https://huggingface.co/blog/lerobot-datasets-v3), you need to first convert it back to v2.1. LerobotDataset:v3.0 support is coming soon.

**Step 3: Monitor Training Progress**

```bash
# Check job status
JOB_ID=$(aws batch list-jobs --job-queue IsaacGr00tJobQueue --job-status RUNNING --query 'jobList[0].jobId' --output text)
aws batch describe-jobs --jobs $JOB_ID

# Stream training logs
aws logs tail /aws/batch/job --follow \
  --log-stream-names "$(aws batch describe-jobs --jobs $JOB_ID \
  --query 'jobs[0].container.logStreamName' --output text)"
```

**Training Configuration:**

- Default: 6000 steps (~2 hours on [g6e.4xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) using the provided sample dataset)
- Checkpoints saved every 2000 steps at `/mnt/efs/gr00t/checkpoints`
- Model artifacts uploaded to [S3](https://aws.amazon.com/s3/) or Hugging Face Hub upon completion

**Configuration Environment Variables:**

Key environment variables you can customize (see [gr00t/env.example](gr00t/env.example) for complete list):

- **Dataset sources**: `DATASET_LOCAL_DIR`, `DATASET_S3_URI`, `HF_DATASET_ID`
- **Uploads**: `UPLOAD_TARGET` (hf|s3|none), `HF_TOKEN`, `HF_MODEL_REPO_ID`, `S3_UPLOAD_URI`
- **Training**: `MAX_STEPS`, `SAVE_STEPS`, `NUM_GPUS`, `BATCH_SIZE`, `LEARNING_RATE`
- **Model/data**: `BASE_MODEL_PATH`, `DATA_CONFIG`, `VIDEO_BACKEND`, `EMBODIMENT_TAG`
- **Tuning**: `TUNE_LLM`, `TUNE_VISUAL`, `TUNE_PROJECTOR`, `TUNE_DIFFUSION_MODEL`, LoRA parameters

**Step 4: Evaluate Trained Models**

Access the DCV instance to evaluate trained models:

1. Navigate to `/mnt/efs/gr00t/checkpoints` to view saved checkpoints
2. Use the provided evaluation scripts to test model performance
3. Visualize training metrics and model outputs

## Next Steps

**Customization and Enhancement Options:**

1. **Modify Training Parameters:**

   - Adjust learning rate, batch size, and training steps in environment variables
   - Customize LoRA parameters for efficient fine-tuning
   - Modify model components to tune (LLM, visual encoder, projector, diffusion model)

2. **Extend to Different Robot Tasks:**

   - Use different LeRobot datasets for various manipulation tasks
   - Modify embodiment tags for different robot configurations
   - Update data configuration for custom observation/action spaces

3. **Scale Training Infrastructure:**

   - Deploy multiple compute environments for parallel training
   - Use Amazon SageMaker for managed training workflows
   - Implement distributed training across multiple nodes

4. **Integrate Additional Models:**

   - Fine-tune different GR00T model variants
   - Combine with other foundation models
   - Implement multi-modal learning capabilities

5. **Deploy to Physical Robots:**

   - Configure AWS IoT Core for robot fleet management
   - Implement over-the-air model updates
   - Add real-world sensor integration and calibration

6. **Production Optimization:**
   - Implement model quantization for edge deployment
   - Add monitoring and alerting with Amazon CloudWatch
   - Set up automated retraining pipelines with Amazon EventBridge

## Cleanup

**To completely remove all resources created by this Guidance:**

1. **Stop running Batch jobs:**

   ```bash
   # Cancel any running jobs
   aws batch cancel-job --job-id <JOB_ID> --reason "Cleanup"

   # Wait for jobs to terminate
   aws batch list-jobs --job-queue IsaacGr00tJobQueue --job-status RUNNING
   ```

2. **Empty S3 bucket and EFS:**

   ```bash
   # Get bucket name from CloudFormation output
   BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name IsaacGr00tBatchStack --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' --output text)

   # Empty the bucket
   aws s3 rm s3://$BUCKET_NAME --recursive

   # Clean EFS (from DCV instance)
   sudo rm -rf /mnt/efs/gr00t/*
   ```

3. **Delete container images:**

   ```bash
   # Delete ECR repository
   aws ecr delete-repository --repository-name isaac-gr00t-training --force
   ```

4. **Delete CloudFormation stacks:**

   ```bash
   # Navigate to infra directory
   cd reinforcement_learning_vla/nvidia-gr00t-vla/gr00t/infra

   # Set AWS region
   export AWS_REGION=us-west-2  # or your preferred region

   # Delete all CDK stacks
   cdk destroy IsaacLabDcvStack --force
   cdk destroy IsaacGr00tBatchStack --force

   # Confirm deletion when prompted
   ```

5. **Verify resource deletion:**

   ```bash
   # Check that stacks are deleted
   aws cloudformation list-stacks --stack-status-filter DELETE_COMPLETE

   # Verify Batch resources are deleted
   aws batch describe-compute-environments --compute-environments IsaacGr00tComputeEnvironment
   ```

**Note**: The cleanup process may take 10-15 minutes to complete. Monitor the CloudFormation console to ensure all stacks are successfully deleted.

## FAQ, known issues, additional considerations, and limitations (optional)

### Known issues

1. **Multi-GPU Shared Memory Issues:**

   - **Issue**: Training fails with shared memory exhaustion on multi-GPU setups
   - **Resolution**: Reduce `DATALOADER_NUM_WORKERS` (from default of 8 to 2) or increase shared memory size in the Batch job definition. For example, with a g6e.12xlarge instance, you can set shared memory to 384GB in `reinforcement_learning_vla/nvidia-gr00t-vla/gr00t/infra/batch_stack.py`

2. **Container Build Fails:**

   - **Issue**: Building flash-attn takes too long or fails on ARM machines
   - **Resolution**: The stack automatically uses CodeBuild for cloud-based x86 builds. Check CodeBuild logs for specific errors:
     ```bash
     aws logs tail /aws/codebuild/$PROJECT_NAME --follow
     ```
   - **Docker rate limit exceeded**: The buildspec.yml includes a Docker registry mirror configuration to avoid rate limits. If issues persist, authenticate with Docker Hub or use AWS ECR Public
   - **Build timeout**: Default timeout is 2 hours, sufficient for most builds. Check CloudWatch Logs for specific error messages
   - **Permission errors**: Verify the CodeBuild role has ECR push permissions and the ECR repository exists in the same region

3. **EFS Mount Issues:**

   - **Issue**: Training container cannot access EFS file system
   - **Resolution**: Check security group rules allow NFS (port 2049) from itself. Verify EFS mount targets are in the correct subnets

4. **DCV Connection Issues:**

   - **Issue**: "No session found" error when connecting to DCV or cannot connect to DCV web interface
   - **Resolution**:
     - Wait 10-15 minutes for user data script to complete
     - Check `/var/log/dcv-bootstrap.summary` for status on the DCV instance
     - Verify security group allows port 8443
     - Restart DCV service: `sudo systemctl restart dcvserver`

5. **Batch Job Fails to Start:**

   - **Issue**: Job stuck in RUNNABLE state
   - **Resolution**: Check compute environment has available capacity. Verify launch template and instance types are correct. Ensure GPU instance quotas are sufficient

6. **Permission Denied Errors:**

   - **Issue**: Job cannot access S3 or ECR
   - **Resolution**: Verify IAM roles have appropriate policies. For S3 dataset access, ensure `DATASET_BUCKET` environment variable is set and the Batch job role has read permissions

7. **Dataset Format Compatibility:**
   - **Issue**: LeRobot dataset v3.0 format not supported
   - **Resolution**: Convert datasets to v2.1 format before training. LerobotDataset:v3.0 support is coming soon

**Additional considerations**

- **AWS Batch jobs are billed per second while running, including queue time**
- **GPU instances (g6e family) are premium instances with higher hourly costs**
- **EFS storage incurs charges based on provisioned throughput and storage used**
- **DCV instances create remote desktop sessions accessible over the internet - ensure strong passwords**
- **Training datasets may be large and increase S3 storage costs over time**
- **Multi-GPU training requires careful memory management and may need instance type adjustments**

### Service Limits:

Critical service limits that may require increases:

- **[EC2 Instance Limits](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html)**:

  - [g6e.4xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) instances: Default limit may be 0-5 per region
  - [g4dn.xlarge](https://aws.amazon.com/ec2/instance-types/g4/) instances: Default limit may be 0-5 per region
  - [Request limit increase](https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas)
  - At least 8 vCPUs for "Running On-Demand G and VT instances" required

- **[EKS Cluster Limit](https://docs.aws.amazon.com/eks/latest/userguide/service-quotas.html)**: Default 100 clusters per region (usually sufficient)

- **[AWS Batch Compute Environment](https://docs.aws.amazon.com/batch/latest/userguide/service_limits.html)**: Default limits are typically sufficient

- **[S3 Storage](https://docs.aws.amazon.com/AmazonS3/latest/userguide/BucketRestrictions.html)**: Default limits are typically sufficient for this guidance

**Supported Regions:**

- **[us-east-1](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/)** (N. Virginia)
- **[us-west-2](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/)** (Oregon)
- **[eu-west-1](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/)** (Ireland)

Service availability considerations:

- **[GPU instances (g6e family)](https://aws.amazon.com/ec2/instance-types/g6e/)**: Available in most major regions
- **[AWS Batch](https://aws.amazon.com/batch/)**: Available in all major regions
- **[Amazon DCV](https://aws.amazon.com/dcv/)**: Available in select regions
- **[Amazon EFS](https://aws.amazon.com/efs/)**: Available in all major regions

**Note**: Verify GPU instance availability in your target region before deployment.

**Security Considerations:**

- [DCV](https://aws.amazon.com/dcv/) web interface is exposed to the internet on port 8443
- Ensure strong passwords are set via [Secrets Manager](https://aws.amazon.com/secrets-manager/)
- Consider restricting access to specific IP ranges in [security groups](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html)
- Training data may contain sensitive information - ensure proper [S3 bucket policies](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucket-policies.html)
- Container images may contain proprietary model weights - secure [ECR repositories](https://aws.amazon.com/ecr/) appropriately

_For any feedback, questions, or suggestions, please use the issues tab under this repo._

## Revisions

Document all notable changes to this project.

Consider formatting this section based on Keep a Changelog, and adhering to Semantic Versioning.

## Authors

NVIDIA Isaac GR00T VLA Training Component developed by the AWS Solutions Library team.

## Notices

_Customers are responsible for making their own independent assessment of the information in this Guidance. This Guidance: (a) is for informational purposes only, (b) represents AWS current product offerings and practices, which are subject to change without notice, and (c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided "as is" without warranties, representations, or conditions of any kind, whether express or implied. AWS responsibilities and liabilities to its customers are controlled by AWS agreements, and this Guidance is not part of, nor does it modify, any agreement between AWS and its customers._
