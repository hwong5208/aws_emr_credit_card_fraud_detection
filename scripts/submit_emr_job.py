"""
Submit PySpark preprocessing job to AWS EMR Serverless.

Usage:
  python scripts/submit_emr_job.py
"""
import logging
import time
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = "us-west-2"
STAGE = "dev"


def get_ssm(name):
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except ClientError as e:
        logger.error(f"Failed to get SSM parameter {name}: {e}")
        logger.error("Have you run 'make deploy-infra' yet?")
        raise SystemExit(1)


def main():
    bucket = get_ssm(f"/fraud-mlops/{STAGE}/bucket-name")
    app_id = get_ssm(f"/fraud-mlops/{STAGE}/emr-app-id")
    job_role_arn = get_ssm(f"/fraud-mlops/{STAGE}/job-role-arn")

    logger.info(f"Bucket:      {bucket}")
    logger.info(f"EMR App ID:  {app_id}")
    logger.info(f"Job Role:    {job_role_arn}")

    # Upload preprocess.py to S3
    s3 = boto3.client("s3", region_name=REGION)
    script_s3_path = "s3://" + bucket + "/scripts/preprocess.py"
    logger.info(f"Uploading preprocess.py to {script_s3_path}")

    try:
        with open("src/preprocess.py", "rb") as f:
            s3.upload_fileobj(f, bucket, "scripts/preprocess.py")
        logger.info("✓ preprocess.py uploaded")
    except FileNotFoundError:
        raise SystemExit("src/preprocess.py not found. Run from project root.")
    except ClientError as e:
        logger.error(f"Upload failed: {e}")
        raise SystemExit(1)

    # Submit EMR Serverless job
    emr = boto3.client("emr-serverless", region_name=REGION)

    job_name = f"fraud-preprocess-{int(time.time())}"
    logger.info(f"\nSubmitting job: {job_name}")

    try:
        response = emr.start_job_run(
            applicationId=app_id,
            clientToken=job_name,
            executionRoleArn=job_role_arn,
            jobDriver={
                "sparkSubmit": {
                    "entryPoint": script_s3_path,
                    "sparkSubmitParameters": (
                        "--conf spark.driver.cores=1 "
                        "--conf spark.driver.memory=2g "
                        "--conf spark.executor.cores=1 "
                        "--conf spark.executor.memory=2g "
                        "--conf spark.executor.instances=2"
                    ),
                    "entryPointArguments": [
                        "--input", f"s3://{bucket}/raw/creditcard.csv",
                        "--output", f"s3://{bucket}/processed/",
                    ],
                }
            },
        )
        job_run_id = response["jobRunId"]
        logger.info(f"✓ Job submitted. Run ID: {job_run_id}")
    except ClientError as e:
        logger.error(f"Job submission failed: {e}")
        raise SystemExit(1)

    # Poll until completion
    logger.info("\nPolling job status...")
    while True:
        try:
            desc = emr.get_job_run(applicationId=app_id, jobRunId=job_run_id)
            state = desc["jobRun"]["state"]
            state_details = desc["jobRun"].get("stateDetails", "")

            logger.info(f"  State: {state} | {state_details}")

            if state in ("SUCCESS", "FAILED", "CANCELLED"):
                break
            time.sleep(10)
        except ClientError as e:
            logger.error(f"Poll failed: {e}")
            raise SystemExit(1)

    if state == "SUCCESS":
        logger.info("\n✓ Job succeeded!")
        logger.info(f"  Output: s3://{bucket}/processed/")
    else:
        logger.error(f"\n✗ Job {state}: {state_details}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
