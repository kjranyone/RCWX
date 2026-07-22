"""Tests for device-resident IVF retrieval."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rcwx.models.accelerator_index import AcceleratorIVFIndex


def test_accelerator_ivf_retrieves_weighted_neighbors_on_cpu() -> None:
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    features = np.array(
        [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]],
        dtype=np.float32,
    )
    index = AcceleratorIVFIndex(
        centroids=centroids,
        features_by_id=features,
        packed_ids=np.array([0, 1, 2, 3], dtype=np.int64),
        list_starts=np.array([0, 2], dtype=np.int64),
        list_sizes=np.array([2, 2], dtype=np.int64),
        max_list_size=2,
        device="cpu",
        dtype=torch.float32,
    )
    query = torch.tensor([[[0.1, 0.0], [10.1, 10.0]]])

    output = index.retrieve(query, k=1)

    torch.testing.assert_close(
        output,
        torch.tensor([[[0.0, 0.0], [10.0, 10.0]]]),
    )


def test_accelerator_ivf_masks_padding_candidates() -> None:
    index = AcceleratorIVFIndex(
        centroids=np.array([[0.0, 0.0]], dtype=np.float32),
        features_by_id=np.array([[3.0, 4.0]], dtype=np.float32),
        packed_ids=np.array([0], dtype=np.int64),
        list_starts=np.array([0], dtype=np.int64),
        list_sizes=np.array([1], dtype=np.int64),
        max_list_size=4,
        device="cpu",
        dtype=torch.float32,
    )

    output = index.retrieve(torch.tensor([[[2.9, 4.1]]]), k=1)

    torch.testing.assert_close(output, torch.tensor([[[3.0, 4.0]]]))


def test_accelerator_ivf_weighted_distance_matches_direct_formula() -> None:
    features = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=np.float32,
    )
    index = AcceleratorIVFIndex(
        centroids=np.array([[0.0, 0.0]], dtype=np.float32),
        features_by_id=features,
        packed_ids=np.arange(4, dtype=np.int64),
        list_starts=np.array([0], dtype=np.int64),
        list_sizes=np.array([4], dtype=np.int64),
        max_list_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    query = torch.tensor([[[1.25, 0.0]]])

    output = index.retrieve(query, k=2)

    distances = np.square(features[:, 0] - 1.25)
    nearest = np.argsort(distances)[:2]
    weights = np.square(1.0 / (distances[nearest] + 1e-6))
    weights /= weights.sum()
    expected = (features[nearest] * weights[:, None]).sum(axis=0)
    torch.testing.assert_close(output[0, 0], torch.from_numpy(expected))


def test_from_faiss_rejects_lists_above_fixed_candidate_cap() -> None:
    faiss = pytest.importorskip("faiss")
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(2), 2, 1, faiss.METRIC_L2)
    index.train(features)
    index.add(features)

    with pytest.raises(ValueError, match="exceeds cap"):
        AcceleratorIVFIndex.from_faiss(
            index,
            features,
            device="cpu",
            dtype=torch.float32,
            max_list_size=4,
        )
