"""Machine profiles.

- ``cpu``: a many-core CPU pod for lstmtraining rounds, 4-8 runs side by side. 32 vCPUs of
  compute-optimised ``cpu3c`` (2 GB RAM per vCPU, so 64 GB), about $0.96/h at the 2026-11-18 price.
  Eight lstmtraining runs at their usual four OpenMP threads fill it.
- ``gpu``: one RTX 4090 (24 GB) on RunPod's secure cloud, for Surya, kraken and VLM runs, about
  $0.89/h. ``--community`` takes the community cloud instead (about $0.34/h, less reliable hosts).
- ``smoke``: the smallest CPU pod (2 vCPUs, about $0.06/h), for proving the loop end to end.

Prices are never taken from here: every run reads the live catalogue price first.
Images are RunPod's own, with sshd and runpodctl inside.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

CPU_IMAGE = "runpod/base:1.0.2-ubuntu2404"
GPU_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


@dataclass(frozen=True)
class Profile:
    name: str
    kind: str                 # "cpu" or "gpu"
    image: str
    default_hours: float      # wall-time limit unless --hours says otherwise
    disk_gb: int
    cpu_flavor: str | None = None
    vcpus: int | None = None
    gpu_id: str | None = None
    gpu_count: int = 1
    cloud: str = "SECURE"

    @property
    def pod_type(self) -> str:
        if self.kind == "cpu":
            return f"{self.cpu_flavor}x{self.vcpus}"
        return f"{self.gpu_id}x{self.gpu_count}/{self.cloud.lower()}"

    def body(self, name: str, env: dict) -> dict:
        b = {"name": name, "image": self.image, "disk": self.disk_gb, "ports": ["22/tcp"],
             "startSsh": True, "cloud": self.cloud, "env": env}
        if self.kind == "cpu":
            b["cpu"] = {"id": self.cpu_flavor, "vcpuCount": self.vcpus}
        else:
            b["gpu"] = {"id": self.gpu_id, "count": self.gpu_count}
        return b

    def community(self) -> "Profile":
        if self.kind != "gpu":
            raise ValueError("CPU pods run on the secure cloud only")
        return replace(self, cloud="COMMUNITY")


PROFILES = {
    "cpu": Profile("cpu", "cpu", CPU_IMAGE, default_hours=12, disk_gb=40, cpu_flavor="cpu3c", vcpus=32),
    "gpu": Profile("gpu", "gpu", GPU_IMAGE, default_hours=4, disk_gb=60, gpu_id="NVIDIA GeForce RTX 4090"),
    "smoke": Profile("smoke", "cpu", CPU_IMAGE, default_hours=0.5, disk_gb=10, cpu_flavor="cpu3c", vcpus=2),
}


def live_compute_price(api, p: Profile) -> float:
    if p.kind == "cpu":
        lo, hi = api.cpu_vcpu_range(p.cpu_flavor)
        if not (lo <= p.vcpus <= hi) or p.vcpus & (p.vcpus - 1):
            raise ValueError(f"{p.cpu_flavor} takes a power of two from {lo} to {hi} vCPUs, not {p.vcpus}")
        return api.cpu_price_per_vcpu(p.cpu_flavor) * p.vcpus
    return api.gpu_price(p.gpu_id, p.cloud) * p.gpu_count
