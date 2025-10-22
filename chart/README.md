# Qalby Proxy Helm Chart

A Helm chart for deploying Qalby Proxy - a FastAPI application with audio and agent capabilities.

## Overview

This Helm chart deploys the Qalby Proxy application on a Kubernetes cluster. The application provides audio processing and AI agent capabilities through a FastAPI-based service.

## Features

- **Automatic Configuration**: Host and port are automatically configured based on the service port
- **Environment Variables**: Support for Logfire and OpenTelemetry configuration
- **Health Checks**: Built-in liveness and readiness probes
- **Scalability**: Optional horizontal pod autoscaling
- **Security**: Configurable security contexts and service accounts
- **Ingress Support**: Optional ingress configuration for external access

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+

## Installation

### Add the repository (if hosted)

```bash
helm repo add qalby-proxy https://your-chart-repo-url
helm repo update
```

### Install the chart

```bash
helm install qalby-proxy qalby-proxy/qalby-proxy
```

### Install with custom values

```bash
helm install qalby-proxy qalby-proxy/qalby-proxy -f custom-values.yaml
```

## Configuration

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `kamasalyamov/livellm-proxy` |
| `image.tag` | Container image tag | `""` (uses Chart.appVersion) |
| `service.port` | Service port | `8000` |
| `service.type` | Service type | `ClusterIP` |
| `env.logfire_token` | Logfire write token | `""` |
| `env.otel_exporter_otlp_endpoint` | OpenTelemetry endpoint | `""` |

### Automatic Configuration

The chart automatically configures:
- **HOST**: Always set to `0.0.0.0` for container binding
- **PORT**: Automatically set to `service.port` value

### Example values.yaml

```yaml
replicaCount: 2

image:
  repository: kamasalyamov/livellm-proxy
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

env:
  logfire_token: "your-logfire-token"
  otel_exporter_otlp_endpoint: "https://your-otel-endpoint"

ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: qalby-proxy.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

## Development

### Local Development

1. Clone this repository
2. Make your changes
3. Test the chart locally:

```bash
# Lint the chart
helm lint .

# Test template rendering
helm template qalby-proxy .

# Dry run installation
helm install qalby-proxy . --dry-run --debug
```

### Chart Structure

```
chart/
├── Chart.yaml              # Chart metadata
├── values.yaml             # Default configuration values
├── templates/              # Kubernetes manifest templates
│   ├── deployment.yaml     # Deployment configuration
│   ├── service.yaml        # Service configuration
│   ├── ingress.yaml        # Ingress configuration
│   ├── hpa.yaml           # Horizontal Pod Autoscaler
│   ├── serviceaccount.yaml # Service Account
│   └── tests/             # Test templates
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This chart is licensed under the same license as the Qalby Proxy application.

## Support

For issues and questions:
- Create an issue in the repository
- Contact the maintainers

## Changelog

### 0.1.0
- Initial release
- Automatic host/port configuration
- Support for Logfire and OpenTelemetry
- Health checks and autoscaling support