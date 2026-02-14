# Infrastructure, Monitoring & Deployment

This document records every architectural decision made for deploying the
Acme Dental AI Agent to AWS, including the monitoring strategy, CI/CD
pipelines, and DNS configuration.

---

## 1. High-Level Architecture

```
User Browser
    |
    v
CloudFront (acmeclinic.pauaua.com)
    |
    |--- /* (default) -----> S3 Bucket (React SPA)
    |--- /api/* -----------> ALB ------> ECS Fargate (FastAPI)
                                              |
                                         Task IAM Role (STS)
                                              |
                                    +----+----+----+
                                    |    |         |
                                CloudWatch  Anthropic  Calendly
                                 Metrics      API       API
```

A single CloudFront distribution serves **both** the React frontend
(S3 origin) and the FastAPI backend (ALB origin) using path-based
routing.  This eliminates CORS issues and requires only one ACM
certificate.

---

## 2. Repository Structure

| Repository | Purpose | Deploy trigger |
|------------|---------|----------------|
| `acme-dental-clinic` | FastAPI backend, Dockerfile, unit tests | Push to `main` → Docker build → ECR → ECS rolling update |
| `acme-dental-infra` | Python CDK infrastructure-as-code | Push to `main` → `cdk deploy` |
| `acme-dental-ui` | React + TypeScript + Tailwind chat UI | Push to `main` → `npm build` → S3 sync → CF invalidation |

---

## 3. AWS Services Used

### Compute
- **ECS Fargate** with a single task (0.25 vCPU, 0.5 GB RAM).
- Public subnets (no NAT Gateway) to minimise cost for a demo.
  Production would move to private subnets + NAT Gateway.
- Container health check on `/api/health`.

### Networking
- **VPC** with 2 public subnets across 2 Availability Zones.
- **Application Load Balancer** (internet-facing) forwarding `/api/*`
  to the ECS target group.
- **CloudFront** with two origins:
  - Default: S3 (React SPA, `index.html` fallback for client-side routing).
  - `/api/*`: ALB.

### DNS & TLS
- **Route 53** hosted zone for `acmeclinic.pauaua.com`.
- **ACM certificate** with DNS validation (auto-validated via Route 53).
- NS records delegated from **Namecheap** to Route 53.

### Secrets
- **SSM Parameter Store** SecureString parameters for
  `ANTHROPIC_API_KEY` and `CALENDLY_API_TOKEN`.
- ECS task role has `ssm:GetParameter` permission for these paths.
- No secrets stored in GitHub, Docker images, or environment variables.

### Container Registry
- **ECR** repository for the backend Docker image.

### Frontend Hosting
- **S3 bucket** (private, Origin Access Control) for the React build
  output.

---

## 4. IAM & STS Strategy

All AWS interactions use **IAM roles with STS temporary credentials**.
No static AWS access keys are used anywhere.

| Principal | Role | Permissions |
|-----------|------|-------------|
| ECS Task Execution | `AcmeDentalTaskExecRole` | ECR pull, CloudWatch Logs |
| ECS Task | `AcmeDentalTaskRole` | CloudWatch `PutMetricData`, SSM `GetParameter` |
| GitHub Actions (backend) | `AcmeDentalGHBackendRole` | ECR push, ECS `UpdateService` |
| GitHub Actions (frontend) | `AcmeDentalGHFrontendRole` | S3 sync, CloudFront `CreateInvalidation` |
| GitHub Actions (infra) | `AcmeDentalGHInfraRole` | CDK deploy (CloudFormation, IAM, ECS, S3, etc.) |

GitHub Actions authenticate via **OIDC federation** — the CDK creates
an IAM OIDC Identity Provider trusting `token.actions.githubusercontent.com`,
and each role's trust policy is scoped to the specific repository and
`refs/heads/main`.

---

## 5. Monitoring & Observability

### 5a. CloudWatch Custom Metrics

Emitted from the FastAPI application via `boto3` CloudWatch
`put_metric_data`.  Metrics are batched in a background thread
(60-second flush interval) to avoid blocking request threads.

| Metric Name | Dimensions | Unit | Description |
|-------------|-----------|------|-------------|
| `ExternalAPI/RequestCount` | Service, Status | Count | Incremented per Calendly/Anthropic API call |
| `ExternalAPI/Latency` | Service, Operation | Milliseconds | Round-trip time per external call |
| `ExternalAPI/ErrorCount` | Service, ErrorType | Count | Incremented on failure |

Dimension values:
- **Service**: `calendly`, `anthropic`
- **Status**: `success`, `failure`
- **Operation**: e.g. `GET /event_types`, `POST /invitees`, `llm_invoke`
- **ErrorType**: e.g. `timeout`, `4xx`, `5xx`, `auth_error`

When running locally (no IAM role), metrics are logged to stdout but
not pushed to CloudWatch.  Detection is via the `METRICS_ENABLED`
environment variable (set to `true` in the ECS task definition).

### 5b. Health Check

A **CloudWatch Synthetics Canary** sends `GET https://acmeclinic.pauaua.com/api/health`
every **5 minutes**.  This validates the full request path:
DNS → CloudFront → ALB → ECS container.

The canary auto-publishes `SuccessPercent` and `Duration` metrics.

### 5c. CloudWatch Alarms

| Alarm | Condition | Action |
|-------|-----------|--------|
| HealthCheckFailing | Canary `SuccessPercent < 100` for 2 consecutive periods | SNS notification |
| HighCalendlyErrorRate | `ExternalAPI/ErrorCount{Service=calendly}` > 5 in 5 min | SNS notification |
| HighAnthropicErrorRate | `ExternalAPI/ErrorCount{Service=anthropic}` > 5 in 5 min | SNS notification |
| ECSUnhealthy | ALB target group unhealthy host count > 0 for 5 min | SNS notification |

### 5d. CloudWatch Dashboard

A single dashboard called `AcmeDental-Operations` with widgets for:
- Request count by service (calendly vs anthropic) over time
- P50/P95/P99 latency by service
- Error rate over time
- Health check canary status
- ECS task CPU / memory utilisation

---

## 6. CI/CD Pipelines (GitHub Actions)

### Backend (`acme-dental-clinic`)

```
push to main
    → checkout
    → uv sync --extra dev
    → ruff check
    → pytest
    → docker build + tag with commit SHA
    → OIDC: assume AcmeDentalGHBackendRole
    → docker push to ECR
    → aws ecs update-service --force-new-deployment
```

Tests must pass before any deployment step runs.

### Frontend (`acme-dental-ui`)

```
push to main
    → checkout
    → npm ci
    → npm run lint
    → npm run build
    → OIDC: assume AcmeDentalGHFrontendRole
    → aws s3 sync dist/ s3://<bucket>/
    → aws cloudfront create-invalidation --paths "/*"
```

### Infrastructure (`acme-dental-infra`)

```
push to main
    → checkout
    → pip install -r requirements.txt
    → OIDC: assume AcmeDentalGHInfraRole
    → cdk synth
    → cdk diff (logged for audit)
    → cdk deploy --all --require-approval never
```

---

## 7. DNS Setup (Namecheap → Route 53)

1. CDK deploys the **DnsStack** first, creating a Route 53 hosted zone
   for `acmeclinic.pauaua.com`.
2. After deploy, note the 4 NS records from the hosted zone output.
3. In **Namecheap** → Advanced DNS → add a **NS record** for host
   `acmeclinic` pointing to each of the 4 Route 53 nameservers.
4. Wait for DNS propagation (usually minutes, up to 48h).
5. Deploy remaining stacks (AppStack creates CloudFront + alias records).
6. Verify `https://acmeclinic.pauaua.com` resolves and serves traffic.

---

## 8. Cost Estimate (demo / low-traffic)

| Service | Monthly |
|---------|---------|
| ECS Fargate (0.25 vCPU, 0.5 GB, 24/7) | ~$9 |
| ALB | ~$16 |
| CloudFront | ~$1 |
| S3 | ~$0.03 |
| Route 53 hosted zone | $0.50 |
| Synthetics Canary (5 min interval) | ~$3 |
| CloudWatch Metrics (free tier) | $0 |
| SSM Parameters (free) | $0 |
| ECR | ~$0.10 |
| **Total** | **~$30/month** |

---

## 9. React UI Decisions

- **Vite + React + TypeScript + Tailwind CSS** for a modern, fast build.
- Custom chat components (not a heavy framework) to demonstrate React
  competency as required by the job description.
- Components: `ChatWindow`, `MessageBubble`, `ChatInput`, `TypingIndicator`.
- Custom hooks: `useChat(sessionId)`, `useSession()`.
- In production: relative URL `/api/chat` (CloudFront proxies to ALB).
- In local dev: Vite proxy config forwards `/api` to `http://localhost:8000`.

---

## 10. Security Considerations

- No static AWS keys anywhere — OIDC for CI/CD, task roles for runtime.
- SSM SecureString for API secrets — encrypted at rest with AWS-managed KMS.
- S3 bucket is private; CloudFront accesses it via Origin Access Control.
- ALB security group restricts inbound to CloudFront managed prefix list.
- ECS task security group allows only ALB inbound.
- HTTPS everywhere — ACM certificate on CloudFront, ALB listener on 443.
- GitHub Actions roles are scoped to specific repos + `main` branch only.
