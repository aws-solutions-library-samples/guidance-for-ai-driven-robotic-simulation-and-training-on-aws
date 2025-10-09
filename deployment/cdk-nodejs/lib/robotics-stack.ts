import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as fs from 'fs';
import * as path from 'path';
import { Construct } from 'constructs';

export interface RoboticsStackProps extends cdk.StackProps {
  projectName: string;
  environment: string;
  instanceType?: string;
  keyName?: string;
  rootVolumeSize?: number;
  vpcCidr?: string;
}

export class VpcStack extends cdk.NestedStack {
  public readonly vpc: ec2.Vpc;
  public readonly publicSubnet: ec2.ISubnet;

  constructor(scope: Construct, id: string, props: RoboticsStackProps) {
    super(scope, id, props);

    this.vpc = new ec2.Vpc(this, 'VPC', {
      ipAddresses: ec2.IpAddresses.cidr(props.vpcCidr || '10.0.0.0/16'),
      maxAzs: 1,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
        }
      ]
    });

    this.publicSubnet = this.vpc.publicSubnets[0];
  }
}

export class IamSecretsStack extends cdk.NestedStack {
  public readonly secret: secretsmanager.Secret;
  public readonly role: iam.Role;
  public readonly instanceProfile: iam.CfnInstanceProfile;
  public readonly bucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: RoboticsStackProps) {
    super(scope, id, props);

    this.secret = new secretsmanager.Secret(this, 'Secret', {
      secretName: `${props.projectName}-${props.environment}-password`,
      generateSecretString: {
        secretStringTemplate: '{}',
        generateStringKey: 'password',
        passwordLength: 16,
        excludeCharacters: '"@/\\\'\''
      }
    });

    this.bucket = new s3.Bucket(this, 'S3Bucket', {
      bucketName: `${props.projectName}-${props.environment}-${cdk.Aws.ACCOUNT_ID}-${cdk.Aws.REGION}`,
      versioned: false,
      publicReadAccess: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL
    });

    this.role = new iam.Role(this, 'EC2Role', {
      roleName: `${props.projectName}-${props.environment}-ec2-role`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonBedrockFullAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore')
      ]
    });

    this.secret.grantRead(this.role);
    this.bucket.grantReadWrite(this.role);

    this.instanceProfile = new iam.CfnInstanceProfile(this, 'InstanceProfile', {
      instanceProfileName: `${props.projectName}-${props.environment}-ec2-profile`,
      roles: [this.role.roleName]
    });
  }
}

export class EC2Stack extends cdk.NestedStack {
  public readonly instance: ec2.Instance;
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: RoboticsStackProps & {
    vpc: ec2.Vpc;
    subnet: ec2.ISubnet;
    secret: secretsmanager.Secret;
    role: iam.Role;
  }) {
    super(scope, id, props);

    const scriptPath = path.join(__dirname, '..', 'scripts', 'userdata.sh');
    const userDataScript = fs.readFileSync(scriptPath, 'utf8');
    const templatedUserData = userDataScript.replace(/\$\{secret_name\}/g, props.secret.secretName);

    this.securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', {
      vpc: props.vpc,
      description: 'Security group for robotics training instance',
      allowAllOutbound: true
    });

    this.securityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(22), 'SSH access');
    this.securityGroup.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(8443), 'DCV access');

    this.instance = new ec2.Instance(this, 'Instance', {
      instanceType: new ec2.InstanceType(props.instanceType || 'g4dn.xlarge'),
      machineImage: ec2.MachineImage.fromSsmParameter(
        '/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id'
      ),
      vpc: props.vpc,
      vpcSubnets: { subnets: [props.subnet] },
      securityGroup: this.securityGroup,
      role: props.role,
      keyName: props.keyName,
      blockDevices: [{
        deviceName: '/dev/sda1',
        volume: ec2.BlockDeviceVolume.ebs(props.rootVolumeSize || 100, {
          volumeType: ec2.EbsDeviceVolumeType.GP3
        })
      }],
      userData: ec2.UserData.custom(templatedUserData)
    });
  }
}

export class RoboticsStack extends cdk.Stack {
  public readonly vpcStack: VpcStack;
  public readonly iamSecretsStack: IamSecretsStack;
  public readonly ec2Stack: EC2Stack;

  constructor(scope: Construct, id: string, props: RoboticsStackProps) {
    super(scope, id, props);

    this.vpcStack = new VpcStack(this, 'VpcStack', props);
    this.iamSecretsStack = new IamSecretsStack(this, 'IamSecretsStack', props);
    
    this.ec2Stack = new EC2Stack(this, 'EC2Stack', {
      ...props,
      vpc: this.vpcStack.vpc,
      subnet: this.vpcStack.publicSubnet,
      secret: this.iamSecretsStack.secret,
      role: this.iamSecretsStack.role
    });

    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpcStack.vpc.vpcId,
      description: 'VPC ID'
    });

    new cdk.CfnOutput(this, 'InstanceId', {
      value: this.ec2Stack.instance.instanceId,
      description: 'EC2 Instance ID'
    });

    new cdk.CfnOutput(this, 'InstancePublicIp', {
      value: this.ec2Stack.instance.instancePublicIp,
      description: 'EC2 Instance Public IP'
    });

    new cdk.CfnOutput(this, 'SecretArn', {
      value: this.iamSecretsStack.secret.secretArn,
      description: 'Secrets Manager Secret ARN'
    });

    new cdk.CfnOutput(this, 'S3BucketName', {
      value: this.iamSecretsStack.bucket.bucketName,
      description: 'S3 Bucket Name'
    });

    new cdk.CfnOutput(this, 'S3BucketArn', {
      value: this.iamSecretsStack.bucket.bucketArn,
      description: 'S3 Bucket ARN'
    });
  }
}