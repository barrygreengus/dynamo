# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.utils.multimodal import (
    MmCase,
    MultimodalModelProfile,
    TopologyConfig,
    make_audio_payload,
    make_image_payload,
    make_image_payload_b64,
    make_video_payload,
)
from tests.utils.payload_builder import chat_payload, chat_payload_default

VLLM_TOPOLOGY_SCRIPTS: dict[str, str] = {
    "agg": "agg_multimodal.sh",
    "e_pd": "disagg_multimodal_e_pd.sh",
    "epd": "disagg_multimodal_epd.sh",
    "p_d": "disagg_multimodal_p_d.sh",
}

VLLM_MULTIMODAL_PROFILES: list[MultimodalModelProfile] = [
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.pre_merge],  # default cadence; cases override
                timeout_s=220,
                profiled_vram_gib=9.6,
                requested_vllm_kv_cache_bytes=1_710_490_000,  # 2x safety over min=855_244_800
                tests=[
                    # Vanilla / baseline single-GPU multimodal smoke. Kept on
                    # pre_merge so every PR exercises the simplest happy-path
                    # multimodal config (HTTP-URL image, no frontend decoding).
                    MmCase(payload=make_image_payload(["green"])),
                    # Vanilla inline base64 path (no Rust frontend decode).
                    MmCase(
                        suffix="b64",
                        payload=make_image_payload_b64(["green"]),
                        marks=[pytest.mark.post_merge],
                    ),
                    # --frontend-decoding (HTTP-URL): exercises strip_inline_data_urls
                    # + NIXL RDMA. post_merge-only — local pre-merge builds outside
                    # docker can pick up NIXL stubs that don't support this path;
                    # CI post_merge runs in a container with real NIXL.
                    MmCase(
                        suffix="frontend_decoding",
                        payload=make_image_payload(["green"]),
                        extra_script_args=["--frontend-decoding"],
                        marks=[pytest.mark.post_merge],
                    ),
                    # --frontend-decoding (inline base64). Same NIXL stub
                    # caveat as the HTTP-URL variant above.
                    MmCase(
                        suffix="b64_frontend_decoding",
                        payload=make_image_payload_b64(["green"]),
                        extra_script_args=["--frontend-decoding"],
                        marks=[pytest.mark.post_merge],
                    ),
                ],
            ),
            "e_pd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=340,
                single_gpu=True,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=300,
                single_gpu=True,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
            "p_d": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=300,
                single_gpu=True,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen3.5-0.8B",
        short_name="qwen3.5-0.8b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=600,
                profiled_vram_gib=4.0,
                tests=[
                    # HTTP-URL color test on hybrid Mamba/full-attention VL.
                    # post_merge — qwen3-vl-2b carries the pre_merge baseline.
                    MmCase(payload=make_image_payload(["green"])),
                    # Inline-base64 + --frontend-decoding (NIXL RDMA path) on
                    # the hybrid Mamba/full-attention VL. post_merge for the
                    # same NIXL-stub reason as qwen3-vl-2b's frontend_decoding
                    # cases — see that topology for the rationale.
                    MmCase(
                        suffix="b64_frontend_decoding",
                        payload=make_image_payload_b64(["green"]),
                        extra_script_args=["--frontend-decoding"],
                        marks=[pytest.mark.post_merge],
                    ),
                ],
            ),
        },
    ),
    MultimodalModelProfile(
        name="Qwen/Qwen3-VL-2B-Instruct",
        short_name="qwen3-vl-2b-video",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                delayed_start=60,
                tests=[MmCase(payload=make_video_payload(["red", "static", "still"]))],
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=600,
                delayed_start=60,
                single_gpu=True,
                tests=[MmCase(payload=make_video_payload(["red", "static", "still"]))],
            ),
        },
    ),
    # Audio: uses agg topology with DYN_CHAT_PROCESSOR=vllm because the Rust
    # Jinja engine cannot render multimodal content arrays (audio_url).
    MultimodalModelProfile(
        name="Qwen/Qwen2-Audio-7B-Instruct",
        short_name="qwen2-audio-7b",
        topologies={
            "agg": TopologyConfig(
                marks=[
                    pytest.mark.skip(
                        reason="vLLM engine core init fails on amd64 post-merge. "
                        "OPS-4445"
                    ),
                    pytest.mark.post_merge,
                ],
                timeout_s=600,
                env={"DYN_CHAT_PROCESSOR": "vllm"},
                tests=[MmCase(payload=make_audio_payload(["Hester", "Pynne"]))],
            ),
        },
        extra_vllm_args=["--max-model-len", "7232"],
    ),
    MultimodalModelProfile(
        name="google/gemma-3-4b-it",
        short_name="gemma3-4b",
        topologies={
            "agg": TopologyConfig(
                marks=[pytest.mark.post_merge],
                timeout_s=300,
                profiled_vram_gib=12.0,
                tests=[MmCase(payload=make_image_payload(["green"]))],
            ),
        },
        extra_vllm_args=["--dtype", "bfloat16"],
        gated=True,
    ),
    # [gluo NOTE] LLaVA 1.5 7B is big model and require at least 3 GPUs to run.
    # We may use less GPUs by squeezing the model onto 2 GPUs.
    # LLaVA 1.5 color naming varies across CUDA backends under vLLM 0.20;
    # keep this as a multimodal serving smoke check, not a color oracle.
    MultimodalModelProfile(
        name="llava-hf/llava-1.5-7b-hf",
        short_name="llava-1.5-7b",
        topologies={
            "agg": TopologyConfig(
                # nightly-only: 7B 1-GPU footprint is tight (vram=19.2 GiB).
                # Exercises a different image (coco bus) + a string-content
                # smoke check that the multimodal templating handles.
                marks=[pytest.mark.nightly],
                timeout_s=360,
                gpu_marker="gpu_1",
                profiled_vram_gib=19.2,
                requested_vllm_kv_cache_bytes=4_318_854_000,  # 2x safety over min=2_159_426_560
                tests=[
                    MmCase(
                        payload=chat_payload(
                            [
                                {"type": "text", "text": "What is in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
                                    },
                                },
                            ],
                            repeat_count=1,
                            expected_response=["bus"],
                            temperature=0.0,
                        ),
                    ),
                    # String content (not array) — verifies string → array
                    # conversion for multimodal templates. Just validate no error.
                    MmCase(
                        suffix="default",
                        payload=chat_payload_default(
                            repeat_count=1,
                            expected_response=[],
                        ),
                    ),
                ],
            ),
            "e_pd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=340,
                gpu_marker="gpu_4",
                tests=[
                    MmCase(
                        payload=make_image_payload(
                            [
                                "green",
                                "white",
                                "black",
                                "purple",
                                "red",
                                "pink",
                                "yellow",
                                "blue",
                                "orange",
                            ]
                        )
                    )
                ],
            ),
            "epd": TopologyConfig(
                marks=[pytest.mark.pre_merge],
                timeout_s=300,
                gpu_marker="gpu_4",
                tests=[
                    MmCase(
                        payload=make_image_payload(
                            [
                                "green",
                                "white",
                                "black",
                                "purple",
                                "red",
                                "pink",
                                "yellow",
                                "blue",
                                "orange",
                            ]
                        )
                    )
                ],
            ),
        },
    ),
]
