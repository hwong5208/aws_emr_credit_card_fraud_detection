import os
from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_emrserverless as emr,
    CfnOutput,
)
from constructs import Construct


class FraudDetectionStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, stage: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        is_dev = stage == "dev"

        # ================================================================
        # 1) S3 BUCKET — data storage (raw, processed, models)
        # ================================================================
        self.bucket = s3.Bucket(
            self, "DataBucket",
            bucket_name=f"fraud-mlops-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY if is_dev else RemovalPolicy.RETAIN,
            auto_delete_objects=is_dev,
        )

        # ================================================================
        # 2) EMR SERVERLESS APPLICATION
        # ================================================================
        emr_app = emr.CfnApplication(
            self, "FraudDetectionApp",
            release_label="emr-7.0.0",
            type="SPARK",
            name=f"fraud-detection-{stage}",
            # FinOps: hard cap prevents runaway spend
            maximum_capacity=emr.CfnApplication.MaximumAllowedResourcesProperty(
                cpu="4 vCPU",
                memory="16 GB",
                disk="200 GB",
            ),
            # FinOps: no pre-warmed workers — cold start ~25s but zero idle cost
            # (omitting initial_capacity is intentional)
            # FinOps: auto-stop after 1 min idle so app doesn't linger in STARTED
            auto_stop_configuration=emr.CfnApplication.AutoStopConfigurationProperty(
                enabled=True,
                idle_timeout_minutes=1,
            ),
        )

        # ================================================================
        # 3) EMR JOB EXECUTION ROLE
        # ================================================================
        emr_job_role = iam.Role(
            self, "EMRJobRole",
            role_name=f"fraud-emr-job-role-{stage}",
            assumed_by=iam.ServicePrincipal("emr-serverless.amazonaws.com"),
        )

        self.bucket.grant_read_write(emr_job_role)

        emr_job_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            resources=["*"],
        ))

        # ================================================================
        # 4) SSM PARAMETERS — consumed by scripts
        # ================================================================
        ssm.StringParameter(
            self, "BucketNameParam",
            parameter_name=f"/fraud-mlops/{stage}/bucket-name",
            string_value=self.bucket.bucket_name,
        )

        ssm.StringParameter(
            self, "EMRAppIdParam",
            parameter_name=f"/fraud-mlops/{stage}/emr-app-id",
            string_value=emr_app.attr_application_id,
        )

        ssm.StringParameter(
            self, "JobRoleArnParam",
            parameter_name=f"/fraud-mlops/{stage}/job-role-arn",
            string_value=emr_job_role.role_arn,
        )

        # ================================================================
        # 5) OUTPUTS
        # ================================================================
        CfnOutput(self, "BucketName", value=self.bucket.bucket_name, description="S3 bucket for data and artifacts")
        CfnOutput(self, "EMRAppId", value=emr_app.attr_application_id, description="EMR Serverless application ID")
        CfnOutput(self, "EMRJobRoleArn", value=emr_job_role.role_arn, description="EMR job execution role")
