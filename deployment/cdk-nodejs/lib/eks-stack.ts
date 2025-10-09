import * as cdk from 'aws-cdk-lib';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';
import { RoboticsStackProps } from './robotics-stack';
import { KubectlV33Layer } from '@aws-cdk/lambda-layer-kubectl-v33';

export class EksStack extends cdk.NestedStack {
  public readonly cluster: eks.Cluster;
  public readonly nodeGroup: eks.Nodegroup;

  constructor(scope: Construct, id: string, props: RoboticsStackProps & {
    vpc: ec2.Vpc;
    privateSubnets: ec2.ISubnet[];
  }) {
    super(scope, id, props);

    // EKS Cluster Service Role
    const clusterRole = new iam.Role(this, 'EksClusterRole', {
      roleName: `${props.projectName}-${props.environment}-eks-cluster-role`,
      assumedBy: new iam.ServicePrincipal('eks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSClusterPolicy')
      ]
    });

    // EKS Node Group Service Role
    const nodeRole = new iam.Role(this, 'EksNodeRole', {
      roleName: `${props.projectName}-${props.environment}-eks-node-role`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSWorkerNodePolicy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKS_CNI_Policy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly')
      ]
    });

    // EKS Cluster
    this.cluster = new eks.Cluster(this, 'EksCluster', {
      clusterName: `${props.projectName}-${props.environment}-cluster`,
      version: eks.KubernetesVersion.V1_33,
      role: clusterRole,
      vpc: props.vpc,
      vpcSubnets: [{ subnets: props.privateSubnets }],
      endpointAccess: eks.EndpointAccess.PUBLIC_AND_PRIVATE,
      clusterLogging: [
        eks.ClusterLoggingTypes.API,
        eks.ClusterLoggingTypes.AUDIT,
        eks.ClusterLoggingTypes.AUTHENTICATOR,
        eks.ClusterLoggingTypes.CONTROLLER_MANAGER,
        eks.ClusterLoggingTypes.SCHEDULER
      ],
      defaultCapacity: 0, // We'll add managed node group separately
      kubectlLayer: new KubectlV33Layer(this, 'kubectl-v33-layer'), // Add the required kubectlLayer here
    });

    // Managed Node Group with Trainium instances
    this.nodeGroup = this.cluster.addNodegroupCapacity('TrainiumNodeGroup', {
      nodegroupName: `${props.projectName}-${props.environment}-nodegroup`,
      instanceTypes: [new ec2.InstanceType('trn1.2xlarge')],
      amiType: eks.NodegroupAmiType.AL2_X86_64,
      nodeRole: nodeRole,
      subnets: { subnets: props.privateSubnets },
      minSize: 2,
      maxSize: 2,
      desiredSize: 2,
      diskSize: 100,
      capacityType: eks.CapacityType.ON_DEMAND
    });
  }
}