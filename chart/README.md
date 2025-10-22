# Qalby Proxy Helm Chart

This Helm chart deploys the Qalby Proxy application, a FastAPI-based service with audio and agent capabilities.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+

## Installation

1. Clone or download this chart
2. Copy the example values file and customize it:
   ```bash
   cp values-example.yaml values.yaml
   ```
3. Edit `values.yaml` with your configuration
4. Install the chart:
   ```bash
   helm install qalby-proxy ./chart -f values.yaml
   ```

## Configuration

### Environment Variables

The chart supports configuration through environment variables defined in the `env` section:

```yaml
env:
  # Logfire configuration
  logfire_token: "your-logfire-token"
  otel_exporter_otlp_endpoint: "https://api.logfire.pydantic.dev/otlp"
  
  # Application configuration
  host: "0.0.0.0"
  port: "8000"
  
  # Custom environment variables
  custom:
    YOUR_CUSTOM_VAR: "value"
```

### Node Selection

The chart provides flexible node selection options:

#### Option 1: Simple Node Selector
```yaml
nodeSelector:
  kubernetes.io/os: linux
  node-type: worker
```

#### Option 2: Node Affinity
```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values:
          - worker
```

#### Option 3: Pod Anti-Affinity
```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - qalby-proxy
        topologyKey: kubernetes.io/hostname
```

### Resource Management

Configure resource limits and requests:

```yaml
resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi
```

### Autoscaling

Enable horizontal pod autoscaling:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

### Ingress

Configure ingress for external access:

```yaml
ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: qalby-proxy.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: qalby-proxy-tls
      hosts:
        - qalby-proxy.example.com
```

## Values Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `replicaCount` | int | `1` | Number of replicas |
| `image.repository` | string | `"qalby-proxy"` | Container image repository |
| `image.tag` | string | `""` | Container image tag |
| `env.logfire_token` | string | `""` | Logfire authentication token |
| `env.otel_exporter_otlp_endpoint` | string | `""` | OpenTelemetry exporter endpoint |
| `env.host` | string | `"0.0.0.0"` | Application host |
| `env.port` | string | `"8000"` | Application port |
| `service.type` | string | `"ClusterIP"` | Service type |
| `service.port` | int | `8000` | Service port |
| `ingress.enabled` | bool | `false` | Enable ingress |
| `resources` | object | `{}` | Resource limits and requests |
| `nodeSelector` | object | `{}` | Node selector |
| `affinity` | object | `{}` | Pod affinity rules |

## Examples

### Basic Deployment
```bash
helm install qalby-proxy ./chart \
  --set env.logfire_token="your-token" \
  --set replicaCount=2
```

### Production Deployment with Node Selection
```bash
helm install qalby-proxy ./chart \
  --set env.logfire_token="your-token" \
  --set replicaCount=3 \
  --set nodeSelector.node-type=worker \
  --set resources.limits.cpu=1000m \
  --set resources.limits.memory=1Gi
```

### Development Deployment
```bash
helm install qalby-proxy-dev ./chart \
  --set replicaCount=1 \
  --set resources.requests.cpu=100m \
  --set resources.requests.memory=128Mi
```

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -l app.kubernetes.io/name=qalby-proxy
```

### View Logs
```bash
kubectl logs -l app.kubernetes.io/name=qalby-proxy
```

### Test Health Endpoint
```bash
kubectl port-forward svc/qalby-proxy 8000:8000
curl http://localhost:8000/ping
```

## Upgrading

```bash
helm upgrade qalby-proxy ./chart -f values.yaml
```

## Uninstalling

```bash
helm uninstall qalby-proxy
```
