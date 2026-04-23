// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// reconcileTargetContainers merges a user-supplied container list (from the
// --container / --containers CLI flag) with any pre-existing
// nvidia.com/snapshot-target-containers annotation on the manifest and
// returns the canonical annotation value to stamp back.
//
// Exactly one source of truth wins:
//   - If only the CLI flag is set, the CLI value is authoritative.
//   - If only the manifest annotation is set, it is authoritative.
//   - If both are set, they must match (post-parse normalization); otherwise
//     we error out so users never silently disagree with themselves.
//   - If neither is set, we error out — the annotation is mandatory.
//
// The returned value is always normalized (whitespace trimmed, duplicates
// rejected) and respects the min/max container-count bounds enforced by the
// snapshot contract.
func reconcileTargetContainers(annotations map[string]string, flagValue string, minCount, maxCount int) (string, error) {
	flagNames, flagErr := snapshotprotocol.ParseTargetContainers(flagValue)
	if flagErr != nil {
		return "", fmt.Errorf("--container(s) flag: %w", flagErr)
	}

	manifestRaw := ""
	if annotations != nil {
		manifestRaw = annotations[snapshotprotocol.TargetContainersAnnotation]
	}
	manifestNames, manifestErr := snapshotprotocol.ParseTargetContainers(manifestRaw)
	if manifestErr != nil {
		return "", fmt.Errorf("manifest %s annotation: %w", snapshotprotocol.TargetContainersAnnotation, manifestErr)
	}

	chosen := flagNames
	if len(flagNames) == 0 {
		chosen = manifestNames
	} else if len(manifestNames) > 0 {
		if snapshotprotocol.FormatTargetContainers(flagNames) != snapshotprotocol.FormatTargetContainers(manifestNames) {
			return "", fmt.Errorf(
				"--container(s) flag %q does not match manifest %s %q; pass one or the other",
				snapshotprotocol.FormatTargetContainers(flagNames),
				snapshotprotocol.TargetContainersAnnotation,
				snapshotprotocol.FormatTargetContainers(manifestNames),
			)
		}
	}

	if len(chosen) == 0 {
		return "", fmt.Errorf("target containers are required: pass --container(s) or set %s on the manifest", snapshotprotocol.TargetContainersAnnotation)
	}
	if minCount > 0 && len(chosen) < minCount {
		return "", fmt.Errorf("expected at least %d target container(s), got %d", minCount, len(chosen))
	}
	if maxCount > 0 && len(chosen) > maxCount {
		return "", fmt.Errorf("expected at most %d target container(s), got %d", maxCount, len(chosen))
	}
	return snapshotprotocol.FormatTargetContainers(chosen), nil
}
