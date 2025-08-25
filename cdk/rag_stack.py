#!/usr/bin/env python3
"""
CDK Stack for Mexican Revolution RAG System
Defines all AWS infrastructure components for production deployment
"""

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_applicationloadbalancer as alb,
)
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)
from aws_cdk import (
    aws_codedeploy as codedeploy,
)
from aws_cdk import (
    aws_codepipeline as codepipeline,
)
from aws_cdk import (
    aws_codepipeline_actions as codepipeline_actions,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecs as ecs,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_secretsmanager as secretsmanager,
)
from aws_cdk import (
    aws_wafv2 as wafv2,
)
from constructs import Construct


class RAGStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # VPC for the RAG system
        self.vpc = ec2.Vpc(
            self,
            "RAGVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # ECS Cluster
        self.cluster = ecs.Cluster(
            self,
            "RAGCluster",
            vpc=self.vpc,
            container_insights=True,
            enable_fargate_capacity_providers=True,
        )

        # S3 Bucket for artifacts
        self.artifact_bucket = s3.Bucket(
            self,
            "RAGArtifactBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Task Definition for the RAG application (Blue)
        self.blue_task_definition = ecs.FargateTaskDefinition(
            self,
            "RAGBlueTaskDefinition",
            memory_limit_mib=2048,
            cpu=1024,
            execution_role=self._create_execution_role(),
            task_role=self._create_task_role(),
        )

        # Task Definition for the RAG application (Green)
        self.green_task_definition = ecs.FargateTaskDefinition(
            self,
            "RAGGreenTaskDefinition",
            memory_limit_mib=2048,
            cpu=1024,
            execution_role=self._create_execution_role(),
            task_role=self._create_task_role(),
        )

        # Container definition for Blue
        self.blue_container = self.blue_task_definition.add_container(
            "RAGBlueContainer",
            image=ecs.ContainerImage.from_asset("../"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="rag-system-blue",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"},
            secrets={
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(
                    self._create_openai_secret()
                )
            },
            port_mappings=[
                ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP),
                ecs.PortMapping(container_port=8501, protocol=ecs.Protocol.TCP),
            ],
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60),
            ),
        )

        # Container definition for Green
        self.green_container = self.green_task_definition.add_container(
            "RAGGreenContainer",
            image=ecs.ContainerImage.from_asset("../"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="rag-system-green",
                log_retention=logs.RetentionDays.ONE_MONTH,
            ),
            environment={"ENVIRONMENT": "production", "LOG_LEVEL": "INFO"},
            secrets={
                "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(
                    self._create_openai_secret()
                )
            },
            port_mappings=[
                ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP),
                ecs.PortMapping(container_port=8501, protocol=ecs.Protocol.TCP),
            ],
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60),
            ),
        )

        # Application Load Balancer
        self.alb = alb.ApplicationLoadBalancer(
            self,
            "RAGALB",
            vpc=self.vpc,
            internet_facing=True,
            load_balancer_name="rag-system-alb",
        )

        # WAF for rate limiting and security
        self.waf = self._create_waf()

        # Target Group for Blue (FastAPI)
        self.blue_fastapi_target_group = alb.ApplicationTargetGroup(
            self,
            "BlueFastAPITargetGroup",
            vpc=self.vpc,
            port=8000,
            protocol=alb.ApplicationProtocol.HTTP,
            target_type=alb.TargetType.IP,
            health_check=alb.HealthCheck(
                path="/health",
                port="8000",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        # Target Group for Blue (Streamlit)
        self.blue_streamlit_target_group = alb.ApplicationTargetGroup(
            self,
            "BlueStreamlitTargetGroup",
            vpc=self.vpc,
            port=8501,
            protocol=alb.ApplicationProtocol.HTTP,
            target_type=alb.TargetType.IP,
            health_check=alb.HealthCheck(
                path="/",
                port="8501",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        # Target Group for Green (FastAPI)
        self.green_fastapi_target_group = alb.ApplicationTargetGroup(
            self,
            "GreenFastAPITargetGroup",
            vpc=self.vpc,
            port=8000,
            protocol=alb.ApplicationProtocol.HTTP,
            target_type=alb.TargetType.IP,
            health_check=alb.HealthCheck(
                path="/health",
                port="8000",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        # Target Group for Green (Streamlit)
        self.green_streamlit_target_group = alb.ApplicationTargetGroup(
            self,
            "GreenStreamlitTargetGroup",
            vpc=self.vpc,
            port=8501,
            protocol=alb.ApplicationProtocol.HTTP,
            target_type=alb.TargetType.IP,
            health_check=alb.HealthCheck(
                path="/",
                port="8501",
                healthy_http_codes="200",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )

        # Listener for FastAPI (Blue/Green)
        self.fastapi_listener = self.alb.add_listener(
            "FastAPIListener",
            port=80,
            protocol=alb.ApplicationProtocol.HTTP,
            default_action=alb.ListenerAction.forward([self.blue_fastapi_target_group]),
        )

        # Listener for Streamlit (Blue/Green)
        self.streamlit_listener = self.alb.add_listener(
            "StreamlitListener",
            port=443,
            protocol=alb.ApplicationProtocol.HTTPS,
            certificates=[self._create_certificate()],
            default_action=alb.ListenerAction.forward(
                [self.blue_streamlit_target_group]
            ),
        )

        # ECS Service (Blue)
        self.blue_service = ecs.FargateService(
            self,
            "RAGBlueService",
            cluster=self.cluster,
            task_definition=self.blue_task_definition,
            desired_count=2,
            min_healthy_percent=50,
            max_healthy_percent=200,
            assign_public_ip=True,
            service_name="rag-system-blue-service",
        )

        # ECS Service (Green)
        self.green_service = ecs.FargateService(
            self,
            "RAGGreenService",
            cluster=self.cluster,
            task_definition=self.green_task_definition,
            desired_count=0,  # Start with 0, will be scaled up during deployment
            min_healthy_percent=50,
            max_healthy_percent=200,
            assign_public_ip=True,
            service_name="rag-system-green-service",
        )

        # Attach services to target groups
        self.blue_service.attach_to_application_target_group(
            self.blue_fastapi_target_group
        )
        self.blue_service.attach_to_application_target_group(
            self.blue_streamlit_target_group
        )
        self.green_service.attach_to_application_target_group(
            self.green_fastapi_target_group
        )
        self.green_service.attach_to_application_target_group(
            self.green_streamlit_target_group
        )

        # Auto Scaling for Blue
        blue_scaling = self.blue_service.auto_scale_task_count(
            min_capacity=2, max_capacity=10
        )

        blue_scaling.scale_on_cpu_utilization(
            "BlueCpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        blue_scaling.scale_on_memory_utilization(
            "BlueMemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        # Auto Scaling for Green
        green_scaling = self.green_service.auto_scale_task_count(
            min_capacity=0, max_capacity=10
        )

        green_scaling.scale_on_cpu_utilization(
            "GreenCpuScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        green_scaling.scale_on_memory_utilization(
            "GreenMemoryScaling",
            target_utilization_percent=80,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        # CodeDeploy Application
        self.codedeploy_application = codedeploy.EcsApplication(
            self, "RAGCodeDeployApplication", application_name="rag-system-app"
        )

        # CodeDeploy Deployment Group
        self.deployment_group = codedeploy.EcsDeploymentGroup(
            self,
            "RAGDeploymentGroup",
            application=self.codedeploy_application,
            service=self.blue_service,
            deployment_config=codedeploy.EcsDeploymentConfig.ALL_AT_ONCE,
            auto_rollback=codedeploy.AutoRollbackConfig(
                failed_deployment=True, stopped_deployment=True
            ),
        )

        # CI/CD Pipeline
        self._create_cicd_pipeline()

        # CloudWatch Dashboard
        self._create_dashboard()

        # Outputs
        cdk.CfnOutput(
            self,
            "LoadBalancerDNS",
            value=self.alb.load_balancer_dns_name,
            description="Load Balancer DNS Name",
        )

        cdk.CfnOutput(
            self,
            "FastAPIEndpoint",
            value=f"http://{self.alb.load_balancer_dns_name}",
            description="FastAPI Endpoint",
        )

        cdk.CfnOutput(
            self,
            "StreamlitEndpoint",
            value=f"https://{self.alb.load_balancer_dns_name}",
            description="Streamlit Endpoint",
        )

        cdk.CfnOutput(
            self,
            "ArtifactBucket",
            value=self.artifact_bucket.bucket_name,
            description="S3 Artifact Bucket",
        )

    def _create_execution_role(self) -> iam.Role:
        """Create ECS execution role"""
        return iam.Role(
            self,
            "RAGExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
            inline_policies={
                "SecretsAccess": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["secretsmanager:GetSecretValue"],
                            resources=["*"],
                        )
                    ]
                )
            },
        )

    def _create_task_role(self) -> iam.Role:
        """Create ECS task role"""
        return iam.Role(
            self,
            "RAGTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "CloudWatchAgentServerPolicy"
                )
            ],
        )

    def _create_openai_secret(self) -> secretsmanager.Secret:
        """Create OpenAI API key secret"""
        return secretsmanager.Secret(
            self,
            "OpenAISecret",
            secret_name="rag-system/openai-api-key",
            description="OpenAI API Key for RAG System",
            removal_policy=RemovalPolicy.DESTROY,
        )

    def _create_certificate(self) -> str:
        """Create SSL certificate (placeholder - would need ACM certificate)"""
        # In production, you would create an ACM certificate
        # For now, returning a placeholder
        return "arn:aws:acm:us-east-1:123456789012:certificate/placeholder"

    def _create_waf(self) -> wafv2.CfnWebACL:
        """Create WAF for rate limiting and security"""
        return wafv2.CfnWebACL(
            self,
            "RAGWAF",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            scope="REGIONAL",
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name="RAGWAFMetrics",
                sampled_requests_enabled=True,
            ),
            rules=[
                # Rate limiting rule
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimitRule",
                    priority=1,
                    statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                        limit=2000,  # 2000 requests per 5 minutes
                        aggregate_key_type="IP",
                    ),
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="RateLimitRule",
                        sampled_requests_enabled=True,
                    ),
                ),
                # Common attack protection
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSManagedRulesCommonRuleSet",
                    priority=2,
                    statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                        name="AWSManagedRulesCommonRuleSet",
                        vendor="AWS",
                    ),
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="AWSManagedRulesCommonRuleSet",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )

    def _create_cicd_pipeline(self):
        """Create CI/CD pipeline for automated deployments"""
        # Source action (GitHub)
        source_output = codepipeline.Artifact()
        build_output = codepipeline.Artifact()

        # Build action
        build_action = codepipeline_actions.CodeBuildAction(
            action_name="Build",
            project=self._create_build_project(),
            input=source_output,
            outputs=[build_output],
        )

        # Deploy action
        deploy_action = codepipeline_actions.CodeDeployEcsDeployAction(
            action_name="Deploy",
            service=self.blue_service,
            image_file_inputs=[
                codepipeline_actions.CodeDeployEcsContainerImageInput(
                    input=build_output,
                    task_definition_input=build_output,
                    container_name="RAGBlueContainer",
                )
            ],
            task_definition_template_input=build_output,
            task_definition_template_path="taskdef.json",
            app_spec_template_input=build_output,
            app_spec_template_path="appspec.yaml",
            deployment_group=self.deployment_group,
        )

        # Pipeline
        codepipeline.Pipeline(
            self,
            "RAGPipeline",
            pipeline_name="rag-system-pipeline",
            artifact_bucket=self.artifact_bucket,
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[
                        codepipeline_actions.GitHubSourceAction(
                            action_name="GitHub_Source",
                            owner="your-github-username",
                            repo="your-repo-name",
                            branch="main",
                            oauth_token=cdk.SecretValue.secrets_manager("github-token"),
                            output=source_output,
                        )
                    ],
                ),
                codepipeline.StageProps(
                    stage_name="Build",
                    actions=[build_action],
                ),
                codepipeline.StageProps(
                    stage_name="Deploy",
                    actions=[deploy_action],
                ),
            ],
        )

    def _create_build_project(self):
        """Create CodeBuild project for building Docker images"""
        from aws_cdk import aws_codebuild as codebuild

        return codebuild.PipelineProject(
            self,
            "RAGBuildProject",
            project_name="rag-system-build",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_5_0,
                privileged=True,
            ),
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "pre_build": {
                            "commands": [
                                "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com",
                            ]
                        },
                        "build": {
                            "commands": [
                                "docker build -t rag-system .",
                                "docker tag rag-system:latest $IMAGE_REPO_NAME:$IMAGE_TAG",
                            ]
                        },
                        "post_build": {
                            "commands": [
                                "docker push $IMAGE_REPO_NAME:$IMAGE_TAG",
                                'printf \'[{"name":"RAGBlueContainer","imageUri":"%s"}]\' $IMAGE_REPO_NAME:$IMAGE_TAG > imagedefinitions.json',
                            ]
                        },
                    },
                    "artifacts": {
                        "files": [
                            "imagedefinitions.json",
                            "taskdef.json",
                            "appspec.yaml",
                        ]
                    },
                }
            ),
        )

    def _create_dashboard(self):
        """Create CloudWatch dashboard for monitoring"""
        dashboard = cloudwatch.Dashboard(
            self, "RAGDashboard", dashboard_name="RAG-System-Dashboard"
        )

        # CPU Utilization Widget (Blue)
        blue_cpu_widget = cloudwatch.GraphWidget(
            title="Blue Service CPU Utilization",
            left=[self.blue_service.metric_cpu_utilization(period=Duration.minutes(1))],
            width=12,
            height=6,
        )

        # Memory Utilization Widget (Blue)
        blue_memory_widget = cloudwatch.GraphWidget(
            title="Blue Service Memory Utilization",
            left=[
                self.blue_service.metric_memory_utilization(period=Duration.minutes(1))
            ],
            width=12,
            height=6,
        )

        # CPU Utilization Widget (Green)
        green_cpu_widget = cloudwatch.GraphWidget(
            title="Green Service CPU Utilization",
            left=[
                self.green_service.metric_cpu_utilization(period=Duration.minutes(1))
            ],
            width=12,
            height=6,
        )

        # Memory Utilization Widget (Green)
        green_memory_widget = cloudwatch.GraphWidget(
            title="Green Service Memory Utilization",
            left=[
                self.green_service.metric_memory_utilization(period=Duration.minutes(1))
            ],
            width=12,
            height=6,
        )

        # Request Count Widget
        request_widget = cloudwatch.GraphWidget(
            title="Request Count",
            left=[self.alb.metric_request_count(period=Duration.minutes(1))],
            width=12,
            height=6,
        )

        # Target Response Time Widget
        response_time_widget = cloudwatch.GraphWidget(
            title="Target Response Time",
            left=[self.alb.metric_target_response_time(period=Duration.minutes(1))],
            width=12,
            height=6,
        )

        # WAF Rate Limiting Widget
        waf_widget = cloudwatch.GraphWidget(
            title="WAF Rate Limiting",
            left=[self.waf.metric_blocked_requests(period=Duration.minutes(1))],
            width=12,
            height=6,
        )

        dashboard.add_widgets(
            blue_cpu_widget,
            blue_memory_widget,
            green_cpu_widget,
            green_memory_widget,
            request_widget,
            response_time_widget,
            waf_widget,
        )
