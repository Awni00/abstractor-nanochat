import os
import socket

def get_slurm_args(params: dict, defaults: dict) -> dict:
    """
    Customize SLURM arguments based on job parameters.

    This is a simple demo showing how different parameter values
    can trigger different resource allocations.

    Args:
        params: Job parameters from the job spec
            - algorithm: Algorithm type (algo_a, algo_b, etc.)
            - dataset: Dataset size (small, large)
            - config: Configuration type (default, optimized)
            - duration: Job runtime in seconds

        defaults: Default SLURM arguments from job spec
            - partition, time, mem, cpus_per_task, etc.

    Returns:
        Dictionary of SLURM arguments to use for this job
    """

    cluster = detect_cluster()

    slurm_args = defaults.copy()
    slurm_args["cluster"] = cluster

    if cluster == "misha":
        slurm_args["qos"] = "qos_nmi"
        slurm_args["partition"] = "gpu"
        slurm_args["gpu_type"] = "h100"
    elif cluster == "bouchet":
        if 'qos' in slurm_args:
            del slurm_args["qos"]  # not used on bouchet
        slurm_args["partition"] = "gpu_h200"
        # slurm_args["gpu_type"] = "h200"
        if 'gpu_type' in slurm_args:
            del slurm_args["gpu_type"]  # use default GPU type on bouchet

    return slurm_args

def detect_cluster() -> str:
    # Prefer explicit SLURM variable, fall back to hostname pattern
    slurm_cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if slurm_cluster:
        return slurm_cluster.lower()
    host = socket.gethostname().lower()
    if "misha" in host:
        return "misha"
    if "bouchet" in host:
        return "bouchet"
    return "misha"  # safe default

cluster = detect_cluster()

print(f"Detected cluster: {cluster}")