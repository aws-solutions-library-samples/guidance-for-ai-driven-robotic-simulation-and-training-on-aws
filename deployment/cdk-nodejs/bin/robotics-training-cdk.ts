#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { RoboticsStack } from '../lib/robotics-stack';

const app = new cdk.App();

// Get context values or use defaults
const projectName = app.node.tryGetContext('projectName') || 'robotics-training';
const environment = app.node.tryGetContext('environment') || 'dev';
const region = app.node.tryGetContext('region') || 'us-east-1';
const instanceType = app.node.tryGetContext('instanceType') || 'g4dn.xlarge';
const keyName = app.node.tryGetContext('keyName');
const rootVolumeSize = Number(app.node.tryGetContext('rootVolumeSize')) || 150;
const vpcCidr = app.node.tryGetContext('vpcCidr') || '10.0.0.0/16';
const allowedCidrBlocksContext = app.node.tryGetContext('allowedCidrBlocks');
const allowedCidrBlocks = allowedCidrBlocksContext ? JSON.parse(allowedCidrBlocksContext) : undefined;

// Create robotics stack with nested stacks
const roboticsStack = new RoboticsStack(app, `${projectName}-${environment}-stack`, {
  env: {
    region: region,
  },
  projectName,
  environment,
  instanceType,
  keyName,
  rootVolumeSize,
  vpcCidr,
  allowedCidrBlocks
});