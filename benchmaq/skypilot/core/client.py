"""SkyPilot API client wrapper.

Provides functions to interact with SkyPilot for cluster management.

Prerequisites:
    Users must authenticate with SkyPilot before using these functions:
    - Run `sky auth` or
    - Set SKYPILOT_API_SERVER_URL and SKYPILOT_API_KEY environment variables
"""

from typing import Dict, Any, Optional, List

import sky


def launch_cluster(
    task_yaml: str,
    cluster_name: str,
    down: bool = True,
    idle_minutes_to_autostop: Optional[int] = None,
) -> Dict[str, Any]:
    """Launch a SkyPilot cluster from task YAML string.
    
    Args:
        task_yaml: YAML string defining the SkyPilot task configuration.
        cluster_name: Name for the cluster.
        down: If True, tear down cluster after job completes.
        idle_minutes_to_autostop: Auto-stop after this many idle minutes.
    
    Returns:
        Dict with job_id and handle information.
    
    Raises:
        Various SkyPilot exceptions if launch fails.
    """
    task = sky.Task.from_yaml_str(task_yaml)
    
    request_id = sky.launch(
        task,
        cluster_name=cluster_name,
        down=down,
        idle_minutes_to_autostop=idle_minutes_to_autostop,
    )
    
    # Stream launch/provisioning logs and wait for completion
    job_id, handle = sky.stream_and_get(request_id)
    
    # Tail the job logs to show setup and run output
    if job_id is not None:
        print()
        print("=" * 64)
        print("STREAMING JOB LOGS")
        print("=" * 64)
        print()
        sky.tail_logs(cluster_name, job_id, follow=True)
    
    return {
        "job_id": job_id,
        "handle": handle,
    }


def teardown_cluster(cluster_name: str, purge: bool = False) -> None:
    """Tear down a SkyPilot cluster.
    
    Args:
        cluster_name: Name of the cluster to tear down.
        purge: If True, forcefully remove from SkyPilot's cluster table
               even if actual termination fails.
    """
    request_id = sky.down(cluster_name, purge=purge)
    sky.get(request_id)


def get_cluster_status(cluster_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get status of SkyPilot clusters.
    
    Args:
        cluster_names: List of cluster names to query. If None, returns all clusters.
    
    Returns:
        List of cluster status dictionaries.
    """
    request_id = sky.status(cluster_names)
    return sky.get(request_id)


def stop_cluster(cluster_name: str) -> None:
    """Stop a SkyPilot cluster (preserves disk).
    
    Args:
        cluster_name: Name of the cluster to stop.
    """
    request_id = sky.stop(cluster_name)
    sky.get(request_id)


def start_cluster(cluster_name: str) -> None:
    """Start a stopped SkyPilot cluster.
    
    Args:
        cluster_name: Name of the cluster to start.
    """
    request_id = sky.start(cluster_name)
    sky.get(request_id)
