---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Snapshot
---

> ⚠️ **Experimental Feature**: Dynamo Snapshot is currently in preview and may only be functional in some cluster setups. The `snapshot-agent` DaemonSet runs in privileged mode to perform CRIU operations. See [Limitations](#limitations) for details.

**Dynamo Snapshot** is infrastructure for fast-starting GPU applications in Kubernetes using CRIU (Checkpoint/Restore in Userspace) and NVIDIA's `cuda-checkpoint` utility. The usual flow is:

1. start a worker once and checkpoint its initialized state
2. store that checkpoint on a namespace-local snapshot volume
3. restore later workers from that checkpoint instead of cold-starting again

| Startup Type | Time | What Happens |
|--------------|------|--------------|
| **Cold Start** | ~1 min | Download model, load to GPU, initialize engine |
| **Warm Start** (restore from checkpoint) | ~10 sec | Restore from a ready checkpoint directory |

> ⚠️ Restore time depends on storage bandwidth, GPU model, and whether the restore stays on the same node.

## Prerequisites

- x86_64 (`amd64`) GPU nodes
- NVIDIA driver 580.xx or newer on the target GPU nodes (590.xx or newer if testing multi-GPU snapshots)
- vLLM or SGLang backend today
- `ReadWriteMany` storage for cross-node restore
- **CRI-O / OpenShift:** set `runtime.type=crio` on the snapshot chart (and `openshift.enabled=true` on OpenShift). Defaults are for containerd; see the chart README for sockets and Helm flags.

## Quick Start via `DynamoCheckpoint` CR

1. Build a placeholder image
2. Install the snapshot chart
3. Create a `DynamoCheckpoint` and wait for it to become ready
4. Deploy a `DynamoGraphDeployment` that restores from the corresponding `checkpointRef`

### 1. Build and push a placeholder image

Snapshot-enabled workers must use a placeholder image that wraps the normal runtime image with restore tooling. If you do not already have one, build it and push it to a registry your cluster can pull from:

```bash
export RUNTIME_IMAGE=registry.example.com/dynamo/vllm-runtime:1.0.0
export PLACEHOLDER_IMAGE=registry.example.com/dynamo/vllm-placeholder:1.0.0

cd deploy/snapshot

make docker-build-placeholder \
  PLACEHOLDER_BASE_IMG="${RUNTIME_IMAGE}" \
  PLACEHOLDER_IMG="${PLACEHOLDER_IMAGE}"

make docker-push-placeholder \
  PLACEHOLDER_IMG="${PLACEHOLDER_IMAGE}"
```

The placeholder image preserves the normal runtime entrypoint/command contract and adds the `criu`, `cuda-checkpoint`, and `nsrestore` tooling needed for checkpoint and restore.

To build either snapshot image against a custom CRIU fork or ref, pass
`CRIU_REPO` and `CRIU_REF` through `make`. If they are unset, the Dockerfile
defaults are used.

```bash
make docker-build-agent \
  IMG=registry.example.com/dynamo/snapshot-agent:1.0.0 \
  CRIU_REPO="${YOUR_CRIU_REPO}" \
  CRIU_REF="branch-or-sha"

make docker-build-placeholder \
  PLACEHOLDER_BASE_IMG="${RUNTIME_IMAGE}" \
  PLACEHOLDER_IMG="${PLACEHOLDER_IMAGE}" \
  CRIU_REPO="${YOUR_CRIU_REPO}" \
  CRIU_REF="branch-or-sha"
```

### 2. Enable checkpointing in the platform and verify it

Whether you are installing or upgrading `dynamo-platform`, the operator only needs checkpointing enabled:

```yaml
dynamo-operator:
  checkpoint:
    enabled: true
```

If the platform is already installed, verify that the operator config contains the checkpoint block:

```bash
OPERATOR_CONFIG=$(kubectl get deploy -n "${PLATFORM_NAMESPACE}" \
  -l app.kubernetes.io/name=dynamo-operator,app.kubernetes.io/component=manager \
  -o jsonpath='{.items[0].spec.template.spec.volumes[?(@.name=="operator-config")].configMap.name}')

kubectl get configmap "${OPERATOR_CONFIG}" -n "${PLATFORM_NAMESPACE}" \
  -o jsonpath='{.data.config\.yaml}' | sed -n '/^checkpoint:/,/^[^[:space:]]/p'
```

Verify that the rendered config includes `enabled: true`.

### 3. Install the snapshot chart in the workload namespace

```bash
helm upgrade --install snapshot ./deploy/helm/charts/snapshot \
  --namespace ${NAMESPACE} \
  --create-namespace \
  --set storage.pvc.create=true
```

Cross-node restore requires shared `ReadWriteMany` storage. The chart defaults to that mode. If your cluster does not have a default storage class, also set `storage.pvc.storageClass`.

If you are reusing an existing checkpoint PVC, do not set `storage.pvc.create=true`; install the chart with `storage.pvc.create=false` and set `storage.pvc.name` instead.

CRI-O or OpenShift: append for example `--set runtime.type=crio` and, on OpenShift, `--set openshift.enabled=true` (see `deploy/helm/charts/snapshot/README.md`).

Verify that the PVC and DaemonSet are ready:

```bash
kubectl get pvc snapshot-pvc -n ${NAMESPACE}
kubectl rollout status daemonset/snapshot-agent -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/component=snapshot-agent -o wide
```

### 4. Create a `DynamoCheckpoint`

The checkpoint Job pod template should match the worker container you want to checkpoint. For the snapshot flow, the important parts are the checkpoint identity, a container named `main`, and the placeholder image; the rest of the pod template should mirror your normal worker config.

The Job pod template may contain extra containers (helper sidecars, log shippers, etc.), but the snapshot agent only checkpoints one of them. The operator picks the container named `main` and stamps the `nvidia.com/snapshot-target-containers` annotation on the Job pod template to make that decision explicit to the snapshot agent.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoCheckpoint
metadata:
  name: qwen3-06b-bf16
spec:
  identity:
    model: Qwen/Qwen3-0.6B
    backendFramework: vllm
    tensorParallelSize: 1
    dtype: bfloat16
    maxModelLen: 2048

  job:
    activeDeadlineSeconds: 3600
    podTemplateSpec:
      spec:
        ...
        containers:
          - name: worker
            image: registry.example.com/dynamo/vllm-placeholder:1.0.0
            ...
```

If this checkpoint should capture and restore GPU Memory Service helpers, set:

```yaml
spec:
  gpuMemoryService:
    enabled: true
```

`spec.gpuMemoryService` is outside `spec.identity`, so it does not change the checkpoint identity hash.

For a full working example, see [deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/config/samples/nvidia.com_v1alpha1_dynamocheckpoint.yaml).

Apply it:

```bash
kubectl apply -f qwen3-checkpoint.yaml -n ${NAMESPACE}
```

### 5. Wait for the checkpoint to become ready

```bash
kubectl get dckpt -n ${NAMESPACE} \
  -o custom-columns=NAME:.metadata.name,HASH:.status.identityHash,PHASE:.status.phase

kubectl wait \
  --for=jsonpath='{.status.phase}'=Ready \
  dynamocheckpoint/qwen3-06b-bf16 \
  -n ${NAMESPACE} \
  --timeout=30m
```

The useful status fields are:

- `status.phase`: high-level lifecycle (`Pending`, `Creating`, `Ready`, `Failed`)
- `status.identityHash`: deterministic hash of `spec.identity`
- `status.jobName`: checkpoint Job name
- `status.createdAt`: timestamp recorded when the checkpoint became ready
- `status.message`: progress or failure detail when available

### 6. Deploy a `DynamoGraphDeployment` that restores from `checkpointRef`

Once the checkpoint is `Ready`, restore a worker from it explicitly:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-checkpointref-demo
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-runtime:1.0.0

    VllmDecodeWorker:
      componentType: worker
      replicas: 1
      checkpoint:
        enabled: true
        checkpointRef: qwen3-06b-bf16
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-placeholder:1.0.0
          ...
        ...
```

Apply it:

```bash
kubectl apply -f vllm-checkpointref-demo.yaml -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -w
```

The `VllmDecodeWorker` pod should restore from the ready checkpoint instead of creating a new one.

## DGD Auto Flow

`checkpointRef` is the most explicit path. `mode: Auto` is the higher-level path: the operator computes the checkpoint identity hash, looks for an equivalent `DynamoCheckpoint`, and creates one only when no matching checkpoint exists. If a `DynamoCheckpoint` already exists with the same identity, Auto mode reuses it. If no matching checkpoint exists yet, the first worker cold-starts and the operator creates the checkpoint in the background.

```yaml
checkpoint:
  enabled: true
  mode: Auto
  identity:
    model: Qwen/Qwen3-0.6B
    backendFramework: vllm
    tensorParallelSize: 1
    dtype: bfloat16
    maxModelLen: 2048
```

Inside a `DynamoGraphDeployment`, it looks like this:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-auto-demo
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-runtime:1.0.0

    VllmDecodeWorker:
      componentType: worker
      replicas: 1
      checkpoint:
        enabled: true
        mode: Auto
        identity:
          model: Qwen/Qwen3-0.6B
          backendFramework: vllm
          tensorParallelSize: 1
          dtype: bfloat16
          maxModelLen: 2048
      extraPodSpec:
        mainContainer:
          image: registry.example.com/dynamo/vllm-placeholder:1.0.0
          ...
        ...
```

Auto mode only hashes `checkpoint.identity`. If you need GMS-specific checkpoint behavior, configure it on the `DynamoCheckpoint` object with `spec.gpuMemoryService.enabled`.

Useful inspection commands:

```bash
kubectl get dgd vllm-auto-demo -n ${NAMESPACE} \
  -o jsonpath='{.status.checkpoints.VllmDecodeWorker.checkpointName}{"\n"}{.status.checkpoints.VllmDecodeWorker.identityHash}{"\n"}{.status.checkpoints.VllmDecodeWorker.ready}{"\n"}'

kubectl get dckpt -n ${NAMESPACE}
```

If you want to force a new restore after the checkpoint becomes ready, scale the worker:

```bash
kubectl patch dgd vllm-auto-demo -n ${NAMESPACE} --type=merge \
  -p '{"spec":{"services":{"VllmDecodeWorker":{"replicas":2}}}}'
```

## Target Containers

**Which container gets checkpointed? Which containers get restored into?** That is controlled by a single annotation on the Job pod template (for checkpoints) and on the restore pod (for restores):

```yaml
metadata:
  annotations:
    nvidia.com/snapshot-target-containers: "main"
    # or "engine-0,engine-1" for intra-pod failover restores
```

The annotation is **mandatory**. `snapshot-agent`, `snapshotprotocol`, and `snapshotctl` all refuse to run without it.

- **Checkpoint Jobs** must list **exactly one** target container. Extra containers in the pod template are allowed (helper sidecars, etc.) but only the named one is checkpointed.
- **Restore pods** may list **one or more** target containers. The same checkpoint artifact is replayed into every named container in parallel.

The Dynamo operator stamps this annotation for you:

| Scenario | Annotation set by operator |
|----------|----------------------------|
| `DynamoCheckpoint` Job | `main` |
| Non-failover `DynamoGraphDeployment` restore | `main` |
| Intra-pod failover (`failover.mode: intraPod`) restore | `engine-0,engine-1` |
| Inter-pod failover (`failover.mode: interPod`) restore | `main` on every engine pod (primary and each shadow) |
| Inter-pod GMS weight-server pods | no annotation — GMS pods are not a checkpoint/restore target |

Inter-pod failover restores each engine pod (primary + N shadows) independently from the same checkpoint — one annotation per pod, not one pod with many target containers. Intra-pod failover is the only topology where a single restore pod has multiple target containers, because the operator clones `main` into `engine-0` + `engine-1` in place.

You only need to set the annotation by hand when driving `snapshotctl` directly (see below) or when building a custom pod outside the operator.

Per-container status annotations written by `snapshot-agent`:

| Annotation | Object | Meaning |
|------------|--------|---------|
| `nvidia.com/snapshot-checkpoint-status` | Job | `completed` or `failed`. The checkpoint contract is single-container, so this stays a singleton. |
| `nvidia.com/snapshot-restore-status.<container>` | Pod | Per-container restore status: `in_progress`, `completed`, or `failed`. |
| `nvidia.com/snapshot-restore-container-id.<container>` | Pod | Containerd container ID observed for the last restore attempt (used to dedupe across kubelet restarts). |

There is no stored pod-level restore rollup annotation. Humans and debugging tools can inspect the per-container status annotations directly.

Kubernetes readiness is still the serving-readiness source of truth. For restore targets, the operator/snapshotctl synthesize a `StartupProbe` on each target container by deep-copying that container's existing `StartupProbe`, `LivenessProbe`, or `ReadinessProbe` (in that order of preference) and overriding `FailureThreshold=MaxInt32` + `SuccessThreshold=1`. The kubelet then runs the workload's own probe with effectively-infinite retries during restore, so a probe like `GET /healthz` keeps gating "container is ready" the same way it did before checkpointing was enabled. If the container has no `Startup`/`Liveness`/`Readiness` probe defined at all, a sentinel-file fallback is installed instead — `cat /snapshot-control/restore-complete`, also with `FailureThreshold=MaxInt32` — so kubelet does not mark the placeholder `sleep infinity` Ready before CRIU has run. Cold-started non-target containers keep their normal readiness behavior, so the Pod `Ready` condition only becomes true after every restored and cold-started container is ready.

### Migrating to the annotation-driven contract

If you are upgrading from a Dynamo release before the annotation contract landed, the user-visible changes are:

- **`nvidia.com/snapshot-target-containers` is mandatory.** The protocol helpers (`NewCheckpointJob`, `NewRestorePod`, `PrepareRestorePodSpec`, `ValidateRestorePodSpec`) and `snapshotctl` itself refuse to run without it. The Dynamo operator stamps the annotation for you, so most users only encounter this when driving `snapshotctl` against a hand-rolled pod manifest — supply `--container <name>` (checkpoint) or `--containers <name>[,<name>...]` (restore), or pre-stamp the annotation.
- **`snapshotctl restore` is non-blocking.** It now returns `status=requested` after the restore pod is created/patched and does *not* wait for the workload to become Ready. Scripted callers that relied on the previous blocking behavior should `kubectl wait --for=condition=Ready pod/<name>` (or watch the per-container `nvidia.com/snapshot-restore-status.<container>` annotations) explicitly.
- **No aggregate restore-status annotation.** Pre-merge there was a single `nvidia.com/snapshot-restore-status` rollup written by the controller's informer event loop; that key is gone. Read the per-container `nvidia.com/snapshot-restore-status.<container>` keys directly, or rely on Kubernetes Pod `Ready` for the operational signal.
- **`StartupProbe` is now synthesized, not overwritten.** Earlier drafts of this PR stamped a fixed `Exec: cat /snapshot-control/restore-complete` probe over whatever StartupProbe the workload had. The current behavior preserves your probe handler — only the `FailureThreshold` and `SuccessThreshold` are overridden during restore. If you had a custom HTTP/gRPC/TCP probe on a checkpointed pod template, it is now honored unchanged (with infinite retries while CRIU runs).

## Failover Restore

Dynamo supports two distinct failover topologies. Both are orthogonal to the snapshot-target-containers contract; the operator does the right thing for each, and users only care about the annotation directly when driving `snapshotctl`.

> ⚠️ **Snapshot + GPU Memory Service is currently rejected at admission.** Configurations that set both `spec.checkpoint.enabled=true` and `spec.gpuMemoryService.enabled=true` (which includes every `failover.mode: intraPod` config and every `failover.mode: interPod` config) are rejected by the operator's validating webhook with `checkpointing with gpuMemoryService is temporarily disabled due to known GPU driver issues`. The combination is being stabilized; see the [Internal-only override](#internal-only-override-for-snapshot--gms-not-for-production) below for the escape hatch used in internal testing.

### Intra-pod failover (`failover.mode: intraPod`)

The operator clones the main container into `engine-0` and `engine-1` (primary + shadow) inside a single pod. Checkpoint is still single-container, captured against `main` in a standalone checkpoint Job; restore replays that same checkpoint into both engine containers concurrently. The operator stamps `nvidia.com/snapshot-target-containers = "engine-0,engine-1"` on the restore pod, and the snapshot agent writes per-container status annotations and per-container `restore-complete` sentinels as each engine is restored.

The `snapshot-control` emptyDir is mounted into each engine with `subPath: <containerName>`, so concurrent containers' lifecycle sentinels do not collide on disk. Each container still sees the control directory at `/snapshot-control` (via `$DYN_SNAPSHOT_CONTROL_DIR`) as if it owned the volume exclusively.

### Inter-pod failover (`failover.mode: interPod`)

The operator creates one primary engine pod plus `failover.numShadows` shadow engine pods per rank, each with a single `main` container, alongside a dedicated GMS weight-server pod per rank. Each engine pod is an independent snapshot restore target and gets its own `nvidia.com/snapshot-target-containers = "main"` stamp; the snapshot agent restores each pod independently. GMS weight-server pods are never shaped as restore targets — they run `gpu_memory_service.cli.server` and load weights fresh from disk.

### Internal-only override for snapshot + GMS (NOT FOR PRODUCTION)

> ⚠️ **Internal use only.** This flag exists so the failover wiring above can be exercised in NVIDIA-internal test clusters while the underlying GPU driver path for snapshot + GMS is being stabilized. Do **not** set it on user-facing clusters. The admission rule will be removed entirely once the driver path is fixed; until then, the public contract is "snapshot + GMS is rejected by the webhook."

The operator pod recognizes a single environment variable that disables the
admission rule:

| Variable | Effect |
|----------|--------|
| `DYN_OPERATOR_ALLOW_GMS_CHECKPOINT=1` | Webhook returns `nil` for the snapshot + GMS combination. Every other admission rule in `SharedSpecValidator` (failover topology matching, GMS layout, EPP constraints, …) keeps firing. |
| anything else (incl. unset) | Default. Webhook rejects `Checkpoint.Enabled && GPUMemoryService.Enabled` exactly as today. |

The variable is read **once at operator startup** and threaded into the
validator construction. Changing the value at runtime has no effect; the
operator pod must be restarted for a new value to take effect.

When deploying via the bundled Helm chart (`deploy/helm/charts/platform/components/operator`), set it on the operator deployment via `.Values.env`:

```yaml
dynamo-operator:
  env:
    - name: DYN_OPERATOR_ALLOW_GMS_CHECKPOINT
      value: "1"
```

When deploying via the raw kustomize manifest under
`deploy/operator/config/manager/`, add the variable to the manager container's
`env:` list directly. Either way, restart the operator pod after the change.

When the override is active the operator emits the following log line at
startup so the deviation is visible in operator logs:

```text
INTERNAL OVERRIDE: Checkpoint+GPUMemoryService admission rule disabled via env var; do NOT enable in production
```

## Lower-Level Testing With `snapshotctl`

It is possible to checkpoint and restore pods without the Dynamo operator via the lower-level `snapshotctl` utility. However, the snapshot helm chart must be installed, with a running `snapshot-agent` DaemonSet in the namespace with the checkpoint PVC mounted.

`snapshotctl` is intended for lower-level debugging and validation workflows, not as the primary user-facing checkpoint interface. For command details and manifest requirements, see [deploy/snapshot/cmd/snapshotctl/README.md](../../deploy/snapshot/cmd/snapshotctl/README.md).

### Checkpoint from a worker pod manifest

```bash
snapshotctl checkpoint \
  --manifest ./worker-pod.yaml \
  --container main \
  --namespace ${NAMESPACE}
```

The checkpoint manifest must be for a pod and use a placeholder image. `--container` names the workload container inside the manifest to checkpoint (exactly one). You may omit the flag if the manifest already carries `nvidia.com/snapshot-target-containers` in its annotations; if both are set they must match.

If you do not pass `--checkpoint-id`, `snapshotctl` generates one and prints it:

```text
status=completed
namespace=...
name=...
checkpoint_job=...
checkpoint_id=manual-snapshot-...
checkpoint_location=/checkpoints/...
```

### Restore from a worker pod manifest

```bash
snapshotctl restore \
  --manifest ./worker-pod.yaml \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-... \
  --containers main
```

This creates a new restore pod from the manifest and returns after the request is submitted. For a failover-style restore, pass a comma-separated list, e.g. `--containers engine-0,engine-1`. As with `checkpoint`, the manifest may pre-stamp the annotation and omit the flag. Observe restore progress through Kubernetes readiness, pod events/logs, and the per-container `nvidia.com/snapshot-restore-status.<container>` annotations.

### Restore an existing pod in place

```bash
snapshotctl restore \
  --pod existing-restore-target \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-... \
  --containers main
```

This patches restore metadata (including `nvidia.com/snapshot-target-containers`) onto an existing pod that is already snapshot-compatible and returns after the patch is accepted.

## Checkpoint Identity

Checkpoints are uniquely identified by a **16-character SHA256 hash** (64 bits) of configuration that affects runtime state:

| Field | Required | Affects Hash | Example |
|-------|----------|-------------|---------|
| `model` | ✓ | ✓ | `meta-llama/Llama-3-8B` |
| `backendFramework` | ✓ | ✓ | `sglang`, `vllm` |
| `dynamoVersion` | | ✓ | `0.9.0`, `1.0.0` |
| `tensorParallelSize` | | ✓ | `1`, `2`, `4`, `8` |
| `pipelineParallelSize` | | ✓ | `1`, `2` |
| `dtype` | | ✓ | `float16`, `bfloat16`, `fp8` |
| `maxModelLen` | | ✓ | `4096`, `8192` |
| `extraParameters` | | ✓ | custom key-value pairs |

Fields that do **not** change the checkpoint hash include:

- replica count
- node placement (`nodeSelector`, `affinity`, `tolerations`)
- resource requests/limits
- logging or observability configuration

## `DynamoCheckpoint` CRD

The `DynamoCheckpoint` (shortname: `dckpt`) is the operator-managed resource for checkpoint lifecycle.

Use it when you want:

- pre-warmed checkpoints before any `DynamoGraphDeployment` exists
- explicit lifecycle control independent from a DGD
- a stable human-readable name that services can reference with `checkpointRef`

The operator requires:

- `spec.identity`
- `spec.job.podTemplateSpec`

`spec.job.backoffLimit` is deprecated and ignored. Checkpoint Jobs are always single-attempt.

Check status with:

```bash
kubectl get dckpt -n ${NAMESPACE}
kubectl describe dckpt qwen3-06b-bf16 -n ${NAMESPACE}
kubectl get dckpt qwen3-06b-bf16 -n ${NAMESPACE} -o yaml
```

The `status` block looks like:

```yaml
status:
  phase: Ready
  identityHash: 3bff874d069f0ed5
  jobName: checkpoint-job-3bff874d069f0ed5-1
  createdAt: "2026-01-29T10:05:00Z"
  message: ""
```

## Limitations

- **LLM workers only**: checkpoint/restore supports LLM decode and prefill workers. Specialized workers such as multimodal, embedding, and diffusion are not supported.
- **Multi-GPU remains preview**: tensor-parallel configurations are exercised in internal testing, but they are not yet a broadly supported production path across clusters.
- **Network state is sensitive**: restore is sensitive to live TCP socket state. Loopback bootstrap/control sockets are the most reliable path today.
- **Privileged DaemonSet required**: `snapshot-agent` must run privileged to execute CRIU and `cuda-checkpoint`. Workload pods do not need to be privileged.

## Troubleshooting

### Checkpoint Job finishes but the checkpoint never becomes `Ready`

Snapshot only becomes `Ready` after `snapshot-agent` confirms the checkpoint contents. A completed Job is not enough by itself.

```bash
kubectl get dckpt <checkpoint-name> -n ${NAMESPACE} \
  -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,MESSAGE:.status.message,JOB:.status.jobName

JOB_NAME=$(kubectl get dckpt <checkpoint-name> -n ${NAMESPACE} -o jsonpath='{.status.jobName}')
if [ -n "${JOB_NAME}" ]; then
  kubectl logs job/"${JOB_NAME}" -n ${NAMESPACE}
fi

kubectl logs daemonset/snapshot-agent -n ${NAMESPACE} --all-containers
```

If the worker template is wrong, the most common causes are using the raw runtime image instead of the placeholder image, or leaving out normal mounts and secrets that the worker needs to start.

### Restore cannot find or mount checkpoint storage

Restore discovers checkpoint storage from the `snapshot-agent` DaemonSet in the same namespace. That DaemonSet must be ready and must mount the checkpoint PVC.

```bash
kubectl rollout status daemonset/snapshot-agent -n ${NAMESPACE}
kubectl get daemonset -n ${NAMESPACE} -l app.kubernetes.io/component=snapshot-agent -o wide
kubectl get pvc -n ${NAMESPACE}
```

This is also the path that `snapshotctl` uses when it resolves checkpoint storage.

### `snapshotctl` manifest is rejected or the restore target is wrong

`snapshotctl` requires a `Pod` manifest and a target-container list. Multi-container manifests are supported as long as every name passed via `--container` / `--containers` (or the manifest's `nvidia.com/snapshot-target-containers` annotation) actually exists in the pod spec.

```bash
snapshotctl checkpoint --manifest ./worker-pod.yaml --container main --namespace ${NAMESPACE}
snapshotctl restore  --manifest ./worker-pod.yaml --containers main --namespace ${NAMESPACE} --checkpoint-id <checkpoint-id>
snapshotctl restore  --pod failover-pod --containers engine-0,engine-1 --namespace ${NAMESPACE} --checkpoint-id <checkpoint-id>
```

If the manifest annotation and the CLI flag are both provided they must agree; `snapshotctl` rejects mismatches instead of silently picking one.

## Planned Features

- Stabilize multi-GPU support
- TensorRT-LLM support
- Alternative storage backends

## Related Documentation

- [Installation Guide](installation-guide.md)
- [API Reference](api-reference.md)
