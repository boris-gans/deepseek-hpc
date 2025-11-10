"""Strategy abstractions covering each parallelism technique from the README."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from ..data.prompts import PromptRepository


class ParallelismStrategy(ABC):
    """Base class for any TP/PP configuration we want to evaluate."""

    def __init__(self, prompt_repository: PromptRepository) -> None:
        """Store the prompt repository used to build job input shards."""
        self.prompt_repository = prompt_repository

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return a short identifier that tags artifacts produced by this strategy."""

    @abstractmethod
    def describe_topology(self) -> str:
        """Explain the TP/PP/DP layout for documentation and logging."""

    @abstractmethod
    def slurm_constraints(self) -> Dict[str, str]:
        """Return Slurm resource hints (nodes, partition, constraint flags)."""

    @abstractmethod
    def deepspeed_args(self) -> Dict[str, str]:
        """Expose the DeepSpeed configuration overrides for this run."""

    @abstractmethod
    def expected_jobs(self) -> int:
        """Return how many sequential jobs make up this strategy."""

    @abstractmethod
    def postprocess(self, dataframe) -> None:
        """Apply any strategy-specific cleanup to the aggregated DataFrame."""


class TensorParallelAcrossNodes(ParallelismStrategy):
    """Tensor parallelism spread across multiple nodes."""

    @property
    def strategy_name(self) -> str:
        """Return the canonical identifier for TP-across-nodes."""
        return "tensor_parallel_across_nodes"

    def describe_topology(self) -> str:
        """Describe the TP>PP configuration for cross-node sharding."""
        raise NotImplementedError("Topology description is not implemented yet.")

    def slurm_constraints(self) -> Dict[str, str]:
        """Return Slurm parameters for cross-node tensor parallelism."""
        raise NotImplementedError("Slurm constraints are not implemented yet.")

    def deepspeed_args(self) -> Dict[str, str]:
        """Return DeepSpeed overrides for cross-node tensor parallelism."""
        raise NotImplementedError("DeepSpeed args are not implemented yet.")

    def expected_jobs(self) -> int:
        """Tensor-parallel strategy typically runs as a single job."""
        raise NotImplementedError("Expected jobs count is not implemented yet.")

    def postprocess(self, dataframe) -> None:
        """Attach metadata to results produced by cross-node TP runs."""
        raise NotImplementedError("Postprocessing is not implemented yet.")


class PipelineParallelAcrossNodes(ParallelismStrategy):
    """Pipeline parallelism across nodes without tensor parallelism."""

    @property
    def strategy_name(self) -> str:
        """Return the canonical identifier for PP-across-nodes."""
        return "pipeline_parallel_across_nodes"

    def describe_topology(self) -> str:
        """Describe the pipeline segmentation used by this strategy."""
        raise NotImplementedError("Topology description is not implemented yet.")

    def slurm_constraints(self) -> Dict[str, str]:
        """Return Slurm parameters for pipeline parallel runs."""
        raise NotImplementedError("Slurm constraints are not implemented yet.")

    def deepspeed_args(self) -> Dict[str, str]:
        """Return DeepSpeed overrides for pipeline parallelism."""
        raise NotImplementedError("DeepSpeed args are not implemented yet.")

    def expected_jobs(self) -> int:
        """Return how many jobs make up the pipeline-only campaign."""
        raise NotImplementedError("Expected jobs count is not implemented yet.")

    def postprocess(self, dataframe) -> None:
        """Attach metadata to results produced by PP-only runs."""
        raise NotImplementedError("Postprocessing is not implemented yet.")


class TensorParallelSingleNode(ParallelismStrategy):
    """Tensor parallelism contained within a single multi-GPU node."""

    @property
    def strategy_name(self) -> str:
        """Return the canonical identifier for single-node TP."""
        return "tensor_parallel_single_node"

    def describe_topology(self) -> str:
        """Describe how GPUs within a node share work."""
        raise NotImplementedError("Topology description is not implemented yet.")

    def slurm_constraints(self) -> Dict[str, str]:
        """Return Slurm parameters for single-node tensor runs."""
        raise NotImplementedError("Slurm constraints are not implemented yet.")

    def deepspeed_args(self) -> Dict[str, str]:
        """Return DeepSpeed overrides for single-node tensor runs."""
        raise NotImplementedError("DeepSpeed args are not implemented yet.")

    def expected_jobs(self) -> int:
        """Return how many jobs make up the single-node tensor campaign."""
        raise NotImplementedError("Expected jobs count is not implemented yet.")

    def postprocess(self, dataframe) -> None:
        """Attach metadata to results produced by single-node TP runs."""
        raise NotImplementedError("Postprocessing is not implemented yet.")


class HybridTensorPipeline(ParallelismStrategy):
    """Hybrid tensor-parallel within nodes plus pipeline across nodes."""

    @property
    def strategy_name(self) -> str:
        """Return the canonical identifier for the hybrid configuration."""
        return "hybrid_tensor_pipeline"

    def describe_topology(self) -> str:
        """Describe the hybrid TP+PP layout used for this strategy."""
        raise NotImplementedError("Topology description is not implemented yet.")

    def slurm_constraints(self) -> Dict[str, str]:
        """Return Slurm parameters for the hybrid campaign."""
        raise NotImplementedError("Slurm constraints are not implemented yet.")

    def deepspeed_args(self) -> Dict[str, str]:
        """Return DeepSpeed overrides for the hybrid campaign."""
        raise NotImplementedError("DeepSpeed args are not implemented yet.")

    def expected_jobs(self) -> int:
        """Return how many jobs make up the hybrid campaign."""
        raise NotImplementedError("Expected jobs count is not implemented yet.")

    def postprocess(self, dataframe) -> None:
        """Attach metadata to results produced by hybrid TP+PP runs."""
        raise NotImplementedError("Postprocessing is not implemented yet.")
