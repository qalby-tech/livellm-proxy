# Qalby Proxy Helm Chart

This Helm chart deploys the Qalby Proxy application - an LLM proxy server for Google Genai, ElevenLabs, and OpenAI.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- API keys for the providers you want to use (OpenAI, Google, ElevenLabs)

## Installation

### Quick Start

1. **Clone or navigate to the chart directory:**
   ```bash
   cd chart
   ```

2. **Create a values file with your API keys:**
   ```bash
   cp ../config.yaml my-values.yaml
   ```

3. **Edit `my-values.yaml` to configure your deployment:**
   ```yaml
   config:
     masterApiKey: "your-master-api-key"
     providers:
       - kind: openai
         name: openai
         apiKey: "your-openai-api-key"
         baseUrl: "https://api.proxyapi.ru/openai/v1"
       - kind: google
         name: google
         apiKey: "your-google-api-key"
         baseUrl: "https://api.proxyapi.ru/google"
       - kind: elevenlabs
         name: elevenlabs
         apiKey: "your-elevenlabs-api-key"
         baseUrl: "http://your-elevenlabs-endpoint"
   ```

4. **Install the chart:**
   ```bash
   helm install qalby-proxy . -f my-values.yaml
   ```

### Using Kubernetes Secrets (Recommended for Production)

For better security, store API keys in Kubernetes secrets:

1. **Create a secret with your API keys:**
   ```bash
   kubectl create secret generic qalby-proxy-secrets \
     --from-literal=masterApiKey=your-master-api-key \
     --from-literal=openaiApiKey=your-openai-key \
     --from-literal=googleApiKey=your-google-key \
     --from-literal=elevenlabsApiKey=your-elevenlabs-key
   ```

2. **Create a values file that references the secret:**
   ```yaml
   existingSecret: "qalby-proxy-secrets"
   
   config:
     providers:
       - kind: openai
         name: openai
         apiKey: ""  # Will be loaded from secret
         baseUrl: "https://api.proxyapi.ru/openai/v1"
       - kind: google
         name: google
         apiKey: ""  # Will be loaded from secret
         baseUrl: "https://api.proxyapi.ru/google"
       - kind: elevenlabs
         name: elevenlabs
         apiKey: ""  # Will be loaded from secret
         baseUrl: "http://your-elevenlabs-endpoint"
   ```

3. **Install the chart:**
   ```bash
   helm install qalby-proxy . -f values.yaml
   ```

## Configuration

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Container image repository | `kamasalyamov/livellm-proxy` |
| `image.tag` | Container image tag | `dev-0.1.0` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `80` |
| `service.targetPort` | Container target port | `8000` |
| `resources.limits.cpu` | CPU limit | `500m` |
| `resources.limits.memory` | Memory limit | `512Mi` |
| `resources.requests.cpu` | CPU request | `250m` |
| `resources.requests.memory` | Memory request | `256Mi` |
| `config.masterApiKey` | Master API key for authentication | `sk-123` |
| `config.host` | Application host | `0.0.0.0` |
| `config.port` | Application port | `8000` |
| `config.providers` | List of provider configurations | See values.yaml |
| `config.fallback` | Fallback model configurations | See values.yaml |
| `existingSecret` | Name of existing secret for API keys | `""` |

### Enabling Ingress

To expose the service externally via Ingress:

```yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: qalby-proxy.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: qalby-proxy-tls
      hosts:
        - qalby-proxy.yourdomain.com
```

### Enabling Autoscaling

To enable horizontal pod autoscaling:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80
```

## Upgrading

To upgrade an existing release:

```bash
helm upgrade qalby-proxy . -f my-values.yaml
```

## Uninstalling

To uninstall the chart:

```bash
helm uninstall qalby-proxy
```

## Health Checks

The application exposes a health check endpoint at `/healthz` which is used for:
- Liveness probe (checks if the app is running)
- Readiness probe (checks if the app is ready to serve traffic)

## API Endpoints

The proxy exposes the following endpoints:

- `POST /transcribe` - Transcribe audio to text
- `POST /speak` - Convert text to speech
- `POST /chat` - Chat completion

All endpoints require authentication using the master API key in the `Authorization` header:
```
Authorization: Bearer your-master-api-key
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -l app.kubernetes.io/name=qalby-proxy
```

### View pod logs
```bash
kubectl logs -l app.kubernetes.io/name=qalby-proxy
```

### Check configuration
```bash
kubectl get configmap qalby-proxy-config -o yaml
```

### Test the health endpoint
```bash
kubectl port-forward svc/qalby-proxy 8080:80
curl http://localhost:8080/healthz
```

## Support

For issues and questions, please refer to the main repository.

