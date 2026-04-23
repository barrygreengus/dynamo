/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package checkpoint

import (
	"context"
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

func ApplyRestorePodMetadata(labels map[string]string, annotations map[string]string, checkpointInfo *CheckpointInfo) {
	enabled := checkpointInfo != nil && checkpointInfo.Enabled && checkpointInfo.Ready
	hash := ""
	artifactVersion := ""
	if enabled {
		hash = checkpointInfo.Hash
		artifactVersion = checkpointInfo.ArtifactVersion
	}
	snapshotprotocol.ApplyRestoreTargetMetadata(labels, annotations, enabled, hash, artifactVersion)
	// Snapshot-agent reads the restore target container list from the
	// nvidia.com/snapshot-target-containers annotation. Default target is
	// the main container; the failover path sets
	// checkpointInfo.RestoreTargetContainers to engine-0/engine-1 so the
	// agent restores the same checkpoint into both engines.
	if !enabled {
		delete(annotations, snapshotprotocol.TargetContainersAnnotation)
		return
	}
	targets := checkpointInfo.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	annotations[snapshotprotocol.TargetContainersAnnotation] = snapshotprotocol.FormatTargetContainers(targets)
}

// restoreTargetsOrDefault returns the restore target container list for a
// given CheckpointInfo, defaulting to the single main container for
// non-failover workloads. Callers are expected to have already confirmed
// that the info is non-nil, enabled, and Ready.
func restoreTargetsOrDefault(info *CheckpointInfo) []string {
	if info != nil && len(info.RestoreTargetContainers) > 0 {
		return info.RestoreTargetContainers
	}
	return []string{commonconsts.MainContainerName}
}

// resolvePodInfoContainer picks the container that should own the pod-info
// downward-API mount. For pods with one restore target, that is the target
// itself. For multi-target (failover) pods, we mount pod-info on every
// target so each engine has the same downward-API view it would have if it
// were the only workload container in the pod.
func resolvePodInfoContainers(podSpec *corev1.PodSpec, targets []string) []*corev1.Container {
	out := make([]*corev1.Container, 0, len(targets))
	for _, name := range targets {
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				out = append(out, &podSpec.Containers[i])
				break
			}
		}
	}
	return out
}

func InjectCheckpointIntoPodSpec(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
) error {
	// Only mutate the worker pod spec once the checkpoint is Ready. Before
	// the checkpoint exists, the worker must cold-start normally without
	// the snapshot-control volume, DYN_SNAPSHOT_CONTROL_DIR, checkpoint PVC
	// mount, or localhost seccomp profile — otherwise the Python worker
	// enters checkpoint mode on env-var presence and sits quiesced waiting
	// for a sentinel that only the checkpoint Job and restore-target path
	// produce. The checkpoint Job itself is built separately through
	// buildCheckpointJob + NewCheckpointJob and does get these.
	if checkpointInfo == nil || !checkpointInfo.Enabled || !checkpointInfo.Ready {
		return nil
	}

	info := checkpointInfo
	if info.Hash == "" {
		if info.Identity == nil {
			return fmt.Errorf("checkpoint enabled but identity is nil and hash is not set")
		}

		hash, err := ComputeIdentityHash(*info.Identity)
		if err != nil {
			return fmt.Errorf("failed to compute identity hash: %w", err)
		}
		info.Hash = hash
	}

	if reader == nil {
		return fmt.Errorf("checkpoint client is required")
	}
	targets := restoreTargetsOrDefault(checkpointInfo)
	// Every named target must exist in the pod spec. Catching this here
	// gives a clearer error than the protocol layer's generic "not found".
	podInfoContainers := resolvePodInfoContainers(podSpec, targets)
	if len(podInfoContainers) != len(targets) {
		return fmt.Errorf("checkpoint restore targets %v do not all exist in pod spec", targets)
	}
	// The target-containers annotation lives on the parent pod metadata
	// (which InjectCheckpointIntoPodSpec does not see directly). For the
	// pod-spec shaping step we synthesize the target list inline so the
	// protocol helper shapes every target; the pod-level annotation is
	// stamped separately by ApplyRestorePodMetadata.
	syntheticAnnotations := map[string]string{
		snapshotprotocol.TargetContainersAnnotation: snapshotprotocol.FormatTargetContainers(targets),
	}
	if err := snapshotprotocol.PrepareRestorePodSpecForCheckpoint(
		ctx,
		reader,
		namespace,
		podSpec,
		syntheticAnnotations,
		info.Hash,
		info.ArtifactVersion,
		snapshotprotocol.DefaultSeccompLocalhostProfile,
		info.Ready,
	); err != nil {
		return err
	}

	EnsurePodInfoVolume(podSpec)
	for _, c := range podInfoContainers {
		EnsurePodInfoMount(c)
	}
	if info.Ready && info.GPUMemoryService != nil && info.GPUMemoryService.Enabled {
		// GMS today is wired to a single main container. Multi-target
		// (failover) support for GMS is tracked separately; stick to
		// the legacy main-container path so single-engine GMS restore
		// continues to work.
		mainContainer := resolveMainContainer(podSpec)
		if mainContainer == nil {
			return fmt.Errorf("gpuMemoryService enabled but no container named %q found in pod spec", commonconsts.MainContainerName)
		}
		storage, err := snapshotprotocol.DiscoverAndResolveStorage(
			ctx,
			reader,
			namespace,
			info.Hash,
			info.ArtifactVersion,
		)
		if err != nil {
			return err
		}
		EnsureGMSRestoreSidecars(podSpec, mainContainer, storage)
	}

	return nil
}

// resolveMainContainer finds the container named "main" in the pod spec.
// ExtraPodSpec.PodSpec.Containers can inject user containers before the main
// container (mergo merge happens before main is appended), so index 0 is
// not guaranteed to be the main container here.
func resolveMainContainer(podSpec *corev1.PodSpec) *corev1.Container {
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			return &podSpec.Containers[i]
		}
	}
	return nil
}
