"""Device-resident IVF retrieval for the fixed-shape streaming path."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from rcwx.accelerator_graph import (
    clear_accelerator_graph_cache,
    get_accelerator_graph_stats,
    run_accelerator_graph,
)

logger = logging.getLogger(__name__)


class AcceleratorIVFIndex(nn.Module):
    """Search one IVF list entirely on an accelerator.

    RVC indexes normally use ``IndexIVFFlat`` with ``nprobe=1``. Packing the
    inverted lists once lets each hop select a centroid and search only its
    fixed-size candidate list without synchronizing through CPU FAISS.
    """

    def __init__(
        self,
        *,
        centroids: np.ndarray,
        features_by_id: np.ndarray,
        packed_ids: np.ndarray,
        list_starts: np.ndarray,
        list_sizes: np.ndarray,
        max_list_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        storage_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float32
        device = torch.device(device)

        centroids_tensor = torch.from_numpy(centroids).to(device, dtype=storage_dtype)
        features_tensor = torch.from_numpy(features_by_id).to(device, dtype=storage_dtype)
        packed_ids_tensor = torch.from_numpy(packed_ids).to(device=device, dtype=torch.long)
        packed_features = features_tensor[packed_ids_tensor]
        del features_tensor, packed_ids_tensor

        self.max_list_size = int(max_list_size)
        # Distance accumulation stays in fp32. Keep the relatively small
        # centroid table ready in that dtype instead of converting it every
        # hop, while the much larger packed feature table remains compact.
        centroids_float = centroids_tensor.float()
        self.register_buffer("centroids", centroids_float)
        self.register_buffer("centroid_norms", centroids_float.square().sum(dim=1))
        self.register_buffer("packed_features", packed_features)
        self.register_buffer(
            "packed_feature_norms",
            packed_features.float().square().sum(dim=1),
        )
        self.register_buffer(
            "list_starts",
            torch.from_numpy(list_starts).to(device=device, dtype=torch.long),
        )
        self.register_buffer(
            "list_sizes",
            torch.from_numpy(list_sizes).to(device=device, dtype=torch.long),
        )
        self.register_buffer(
            "candidate_offsets",
            torch.arange(self.max_list_size, device=device, dtype=torch.long),
        )
        self.eval()

    @classmethod
    def from_faiss(
        cls,
        index: Any,
        features_by_id: np.ndarray,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        max_list_size: int = 256,
    ) -> "AcceleratorIVFIndex":
        """Pack a FAISS ``IndexIVFFlat`` while preserving its vector IDs."""
        import faiss

        if int(getattr(index, "metric_type", -1)) != int(faiss.METRIC_L2):
            raise ValueError("accelerator IVF requires an L2 index")
        if int(getattr(index, "nprobe", 1)) != 1:
            raise ValueError("accelerator IVF currently requires nprobe=1")
        if not all(hasattr(index, name) for name in ("nlist", "quantizer", "invlists")):
            raise ValueError("accelerator IVF requires an inverted-list index")
        if features_by_id.ndim != 2 or features_by_id.shape[1] != int(index.d):
            raise ValueError("reconstructed index features have an invalid shape")

        max_list_size = max(4, int(max_list_size))
        source_sizes = np.array(
            [index.invlists.list_size(i) for i in range(index.nlist)],
            dtype=np.int64,
        )
        if np.any(source_sizes <= 0):
            raise ValueError("accelerator IVF requires non-empty inverted lists")
        largest_list = int(source_sizes.max())
        if largest_list > max_list_size:
            raise ValueError(
                f"accelerator IVF list size {largest_list} exceeds cap {max_list_size}"
            )
        candidate_size = max(4, largest_list)
        centroids = index.quantizer.reconstruct_n(0, index.nlist).astype(
            np.float32,
            copy=False,
        )
        starts = np.empty(index.nlist, dtype=np.int64)
        sizes = np.empty(index.nlist, dtype=np.int64)
        id_parts: list[np.ndarray] = []
        packed_count = 0
        source_count = 0
        for list_id in range(index.nlist):
            source_size = int(source_sizes[list_id])
            kept_size = source_size
            starts[list_id] = packed_count
            sizes[list_id] = kept_size
            source_count += source_size
            if kept_size:
                ids = faiss.rev_swig_ptr(
                    index.invlists.get_ids(list_id),
                    source_size,
                )
                part = np.asarray(ids[:kept_size], dtype=np.int64).copy()
                if part.min() < 0 or part.max() >= len(features_by_id):
                    raise ValueError("inverted-list vector ID is out of range")
                id_parts.append(part)
                packed_count += kept_size

        if not id_parts:
            raise ValueError("accelerator IVF index is empty")
        packed_ids = np.concatenate(id_parts)
        coverage = packed_count / max(1, source_count)
        logger.info(
            "Preparing accelerator IVF: lists=%d vectors=%d/%d coverage=%.3f%% cap=%d",
            index.nlist,
            packed_count,
            source_count,
            coverage * 100.0,
            candidate_size,
        )
        return cls(
            centroids=centroids,
            features_by_id=features_by_id,
            packed_ids=packed_ids,
            list_starts=starts,
            list_sizes=sizes,
            max_list_size=candidate_size,
            device=device,
            dtype=dtype,
        )

    def _retrieve(self, features: torch.Tensor, k: int) -> torch.Tensor:
        query = features[0].float()
        query_norms = query.square().sum(dim=1, keepdim=True)
        centroid_distances = (
            query_norms
            + self.centroid_norms.unsqueeze(0)
            - 2.0 * (query @ self.centroids.transpose(0, 1))
        )
        coarse_ids = centroid_distances.argmin(dim=1)
        starts = self.list_starts[coarse_ids]
        sizes = self.list_sizes[coarse_ids]
        locations = starts.unsqueeze(1) + self.candidate_offsets.unsqueeze(0)
        valid = self.candidate_offsets.unsqueeze(0) < sizes.unsqueeze(1)
        locations = locations.clamp_max(self.packed_features.shape[0] - 1)

        candidates = self.packed_features[locations].float()
        # ||c-q||^2 = ||c||^2 + ||q||^2 - 2<c,q>. Candidate norms are
        # invariant and precomputed once, avoiding two [T, list, dim]
        # temporaries for subtraction and squaring on every 20ms hop.
        candidate_dots = torch.bmm(
            candidates,
            query.unsqueeze(2),
        ).squeeze(2)
        distances = (
            self.packed_feature_norms[locations]
            + query_norms
            - 2.0 * candidate_dots
        ).clamp_min_(0.0)
        distances = distances.masked_fill(~valid, float("inf"))
        scores, local_ids = torch.topk(distances, k, dim=1, largest=False)
        selected = locations.gather(1, local_ids)
        selected_features = self.packed_features[selected].float()

        weights = (1.0 / (scores + 1e-6)).square()
        weights = weights / weights.sum(dim=1, keepdim=True)
        retrieved = (selected_features * weights.unsqueeze(2)).sum(dim=1)
        return retrieved.to(features.dtype).unsqueeze(0)

    @torch.no_grad()
    def retrieve(self, features: torch.Tensor, k: int = 4) -> torch.Tensor:
        k = max(1, min(int(k), self.max_list_size))
        return run_accelerator_graph(
            self,
            f"accelerator-ivf-k-{k}",
            lambda value: self._retrieve(value, k),
            features,
        )

    def clear_graph_cache(self) -> None:
        clear_accelerator_graph_cache(self)

    def graph_stats(self) -> dict[str, float | int]:
        return get_accelerator_graph_stats(self)
