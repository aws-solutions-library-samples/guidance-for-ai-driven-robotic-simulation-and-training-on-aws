import * as cdk from 'aws-cdk-lib';
import * as cloudformation from 'aws-cdk-lib/aws-cloudformation';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as fs from 'fs';
import * as path from 'path';
import { Construct } from 'constructs';

export interface RoboticsStackSetProps extends cdk.StackProps {
  projectName: string;
  environment: string;
  targetRegions: string[];
  targetAccounts?: string[];
  organizationalUnitIds?: string[];
}

export class RoboticsStackSet extends cdk.Stack {
  public readonly stackSet: cloudformation.CfnStackSet;
  public readonly stackSetRole: iam.Role;
  public readonly executionRole: iam.Role;

  constructor(scope: Construct, id: string, props: RoboticsStackSetProps) {
    super(scope, id, props);

    // Load and template the user data script
    const scriptPath = path.join(__dirname, '..', 'scripts', 'userdata.sh');
    const userDataScript = fs.readFileSync(scriptPath, 'utf8');
    const templatedUserData = userDataScript.replace(/\$\{secret_name\}/g, '${SecretArn}');

    // Create StackSet Administration Role
    this.stackSetRole = new iam.Role(this, 'StackSetAdministrationRole', {
      roleName: `${props.projectName}-StackSetAdministrationRole`,
      assumedBy: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
      inlinePolicies: {
        StackSetAdministrationPolicy: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              effect: iam.Effect.ALLOW,
              actions: ['sts:AssumeRole'],
              resources: [`arn:aws:iam::*:role/${props.projectName}-StackSetExecutionRole`]
            })
          ]
        })
      }
    });

    // Create StackSet Execution Role Template
    this.executionRole = new iam.Role(this, 'StackSetExecutionRole', {
      roleName: `${props.projectName}-StackSetExecutionRole`,
      assumedBy: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('PowerUserAccess')
      ]
    });

    // Parent StackSet Template with nested stacks
    const parentTemplate = {
      AWSTemplateFormatVersion: '2010-09-09',
      Description: 'Robotics Training Infrastructure Parent StackSet',
      Parameters: {
        ProjectName: { Type: 'String', Default: props.projectName },
        Environment: { Type: 'String', Default: props.environment },
        VpcCidr: { Type: 'String', Default: '10.0.0.0/16' },
        InstanceType: { Type: 'String', Default: 'g4dn.xlarge' },
        KeyName: { Type: 'String', Default: '' },
        RootVolumeSize: { Type: 'Number', Default: 100 }
      },
      Conditions: {
        HasKeyName: { 'Fn::Not': [{ 'Fn::Equals': [{ Ref: 'KeyName' }, ''] }] }
      },
      Resources: {
        // All resources in a single template instead of nested stacks
        VPC: {
          Type: 'AWS::EC2::VPC',
          Properties: {
            CidrBlock: { Ref: 'VpcCidr' },
            EnableDnsHostnames: true,
            EnableDnsSupport: true,
            Tags: [{ Key: 'Name', Value: { 'Fn::Sub': '${ProjectName}-${Environment}-vpc' } }]
          }
        },
        InternetGateway: {
          Type: 'AWS::EC2::InternetGateway',
          Properties: {
            Tags: [{ Key: 'Name', Value: { 'Fn::Sub': '${ProjectName}-${Environment}-igw' } }]
          }
        },
        InternetGatewayAttachment: {
          Type: 'AWS::EC2::VPCGatewayAttachment',
          Properties: {
            InternetGatewayId: { Ref: 'InternetGateway' },
            VpcId: { Ref: 'VPC' }
          }
        },
        PublicSubnet1: {
          Type: 'AWS::EC2::Subnet',
          Properties: {
            VpcId: { Ref: 'VPC' },
            AvailabilityZone: { 'Fn::Select': [0, { 'Fn::GetAZs': { Ref: 'AWS::Region' } }] },
            CidrBlock: { 'Fn::Select': [0, { 'Fn::Cidr': [{ Ref: 'VpcCidr' }, 4, 8] }] },
            MapPublicIpOnLaunch: true,
            Tags: [{ Key: 'Name', Value: { 'Fn::Sub': '${ProjectName}-${Environment}-public-1' } }]
          }
        },
        Secret: {
          Type: 'AWS::SecretsManager::Secret',
          Properties: {
            Name: { 'Fn::Sub': '${ProjectName}-${Environment}-password' },
            GenerateSecretString: {
              SecretStringTemplate: '{}',
              GenerateStringKey: 'password',
              PasswordLength: 16,
              ExcludeCharacters: '"@/\\\'\''
            }
          }
        },
        EC2Role: {
          Type: 'AWS::IAM::Role',
          Properties: {
            RoleName: { 'Fn::Sub': '${ProjectName}-${Environment}-ec2-role' },
            AssumeRolePolicyDocument: {
              Version: '2012-10-17',
              Statement: [{ Effect: 'Allow', Principal: { Service: 'ec2.amazonaws.com' }, Action: 'sts:AssumeRole' }]
            },
            ManagedPolicyArns: ['arn:aws:iam::aws:policy/AmazonBedrockFullAccess', 'arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore'],
            Policies: [{
              PolicyName: 'SecretsManagerAccess',
              PolicyDocument: {
                Version: '2012-10-17',
                Statement: [{ Effect: 'Allow', Action: 'secretsmanager:GetSecretValue', Resource: { Ref: 'Secret' } }]
              }
            }]
          }
        },
        InstanceProfile: {
          Type: 'AWS::IAM::InstanceProfile',
          Properties: {
            InstanceProfileName: { 'Fn::Sub': '${ProjectName}-${Environment}-ec2-profile' },
            Roles: [{ Ref: 'EC2Role' }]
          }
        },
        SecurityGroup: {
          Type: 'AWS::EC2::SecurityGroup',
          Properties: {
            GroupDescription: 'Security group for EC2 instance',
            VpcId: { Ref: 'VPC' },
            SecurityGroupIngress: [
              { IpProtocol: 'tcp', FromPort: 22, ToPort: 22, CidrIp: '0.0.0.0/0' },
              { IpProtocol: 'tcp', FromPort: 8443, ToPort: 8443, CidrIp: '0.0.0.0/0' }
            ]
          }
        },
        Instance: {
          Type: 'AWS::EC2::Instance',
          Properties: {
            InstanceType: { Ref: 'InstanceType' },
            ImageId: { 'Fn::Sub': '{{resolve:ssm:/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}}' },
            SubnetId: { Ref: 'PublicSubnet1' },
            SecurityGroupIds: [{ Ref: 'SecurityGroup' }],
            IamInstanceProfile: { Ref: 'InstanceProfile' },
            KeyName: { 'Fn::If': ['HasKeyName', { Ref: 'KeyName' }, { Ref: 'AWS::NoValue' }] },
            BlockDeviceMappings: [{ DeviceName: '/dev/sda1', Ebs: { VolumeSize: { Ref: 'RootVolumeSize' }, VolumeType: 'gp3' } }],
            UserData: {
              'Fn::Base64': {
                'Fn::Sub': [
                  templatedUserData,
                  { SecretArn: { Ref: 'Secret' } }
                ]
              }
            }
          }
        }
      },
      Outputs: {
        VpcId: { Value: { Ref: 'VPC' } },
        InstanceId: { Value: { Ref: 'Instance' } },
        InstancePublicIp: { Value: { 'Fn::GetAtt': ['Instance', 'PublicIp'] } },
        SecretArn: { Value: { Ref: 'Secret' } }
      }
    };



    // Create StackSet with single template
    this.stackSet = new cloudformation.CfnStackSet(this, 'RoboticsStackSet', {
      stackSetName: `${props.projectName}-${props.environment}-stackset`,
      description: 'Robotics Training Infrastructure StackSet',
      permissionModel: 'SELF_MANAGED',
      templateBody: JSON.stringify(parentTemplate),
      capabilities: ['CAPABILITY_NAMED_IAM'],
      administrationRoleArn: this.stackSetRole.roleArn,
      executionRoleName: this.executionRole.roleName,
      parameters: [
        {
          parameterKey: 'ProjectName',
          parameterValue: props.projectName
        },
        {
          parameterKey: 'Environment',
          parameterValue: props.environment
        },
        {
          parameterKey: 'InstanceType',
          parameterValue: 'g4dn.xlarge'
        },
        {
          parameterKey: 'RootVolumeSize',
          parameterValue: '100'
        }
      ],
      operationPreferences: {
        regionConcurrencyType: 'PARALLEL',
        maxConcurrentPercentage: 100,
        failureTolerancePercentage: 10
      }
    });

    // Note: StackSet instances need to be created via AWS CLI after StackSet deployment
    // Use: aws cloudformation create-stack-instances --stack-set-name <stackset-name> --regions <regions> --accounts <accounts>

    // Outputs
    new cdk.CfnOutput(this, 'StackSetId', {
      value: this.stackSet.ref,
      description: 'StackSet ID'
    });

    new cdk.CfnOutput(this, 'StackSetArn', {
      value: this.stackSet.attrStackSetId,
      description: 'StackSet ARN'
    });
  }
}