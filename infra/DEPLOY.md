# Enterprise Deployment Guide

## Prerequisites
- AWS CLI configured (`aws configure`)
- AWS SAM CLI installed (`brew install aws-sam-cli`)
- Python 3.12

## Quick Deploy

```bash
cd infra/

# Build
sam build

# Deploy (first time — interactive)
sam deploy --guided

# Deploy (subsequent — uses saved config)
sam deploy
```

## What Gets Created

| Resource | Name | Purpose |
|----------|------|---------|
| S3 Bucket | dq-investigator-inbox-{env} | Upload data here |
| S3 Bucket | dq-investigator-output-{env} | Results stored here |
| EventBridge Rule | dq-s3-trigger-{env} | Watches inbox for new files |
| Lambda | dq-trigger-{env} | Receives S3 event, starts pipeline |
| Lambda | dq-engine-{env} | Runs DQ validation (15 min timeout) |
| Lambda | dq-check-threshold-{env} | Evaluates pass/fail |
| Lambda | dq-send-alert-{env} | Sends SNS notification on failure |
| Lambda | dq-archive-{env} | Moves processed files |
| Step Functions | dq-pipeline-{env} | Orchestrates the full flow |
| SNS Topic | dq-investigator-alerts-{env} | Alert notifications |

## How It Works

```
1. Upload file to s3://dq-investigator-inbox-{env}/incoming/data.csv
2. EventBridge detects the upload
3. Trigger Lambda starts Step Functions
4. DQ Engine Lambda runs validation (same code as local)
5. Threshold check evaluates pass/fail
6. If FAIL: alert sent via SNS
7. File archived to processed/ prefix
8. Results written to output bucket
```

## Testing

```bash
# Upload a test file
aws s3 cp ../TEST2_DATA/main_data/customer_transactions.csv \
  s3://dq-investigator-inbox-dev/incoming/

# Watch the execution
aws stepfunctions list-executions \
  --state-machine-arn <ARN from deploy output>

# Check results
aws s3 ls s3://dq-investigator-output-dev/reports/
```

## Local Testing (no AWS needed)

The same engine runs locally:

```bash
# From project root
python -m core.engine --input TEST2_DATA/main_data/customer_transactions.csv --storage all
```
