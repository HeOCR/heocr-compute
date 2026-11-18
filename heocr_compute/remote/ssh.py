"""SSH to a pod: commands, and tar streams for files in and out.

Uses the key runpodctl made (``~/.runpod/ssh/runpodctl-ssh-key``); its public half is also passed to the
pod as ``PUBLIC_KEY``, which RunPod's images add to ``authorized_keys``. Host keys of short-lived pods
are not recorded.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tarfile
from pathlib import Path

KEY = "~/.runpod/ssh/runpodctl-ssh-key"


def public_key(key: str = KEY) -> str:
    return Path(os.path.expanduser(key + ".pub")).read_text(encoding="utf-8").strip()


class SSHError(Exception):
    pass


class Remote:
    def __init__(self, host: str, port: int, user: str = "root", key: str = KEY):
        self.host, self.port, self.user, self.key = host, int(port), user, os.path.expanduser(key)

    @classmethod
    def from_pod(cls, pod: dict) -> "Remote | None":
        direct = ((pod or {}).get("ssh") or {}).get("direct")
        if not direct:
            return None
        return cls(direct["host"], direct["port"], direct.get("username") or "root")

    def argv(self, command: str) -> list[str]:
        return ["ssh", "-i", self.key, "-p", str(self.port),
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
                f"{self.user}@{self.host}", command]

    def run(self, command: str, input: str | None = None, timeout: float = 600, check: bool = True):
        try:
            p = subprocess.run(self.argv(command), input=input, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise SSHError(f"ssh timed out after {timeout:.0f}s") from e
        if check and p.returncode != 0:
            raise SSHError(f"ssh exited {p.returncode}: {p.stderr.strip()[-300:]}")
        return p

    def upload(self, root, rels, dest: str, timeout: float = 6 * 3600) -> None:
        """Send ``rels`` (relative to ``root``) as one tar stream, unpacked under ``dest`` on the pod."""
        root = Path(root).expanduser()
        cmd = f"mkdir -p {shlex.quote(dest)} && tar -xf - --no-same-owner -C {shlex.quote(dest)}"
        p = subprocess.Popen(self.argv(cmd), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=p.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tf:
                for rel in rels:
                    info = tf.gettarinfo(str(root / rel), arcname=rel)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with open(root / rel, "rb") as f:
                        tf.addfile(info, f)
            p.stdin.close()
            err = p.stderr.read().decode(errors="replace")
            if p.wait(timeout=timeout) != 0:
                raise SSHError(f"upload failed: {err.strip()[-300:]}")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            for stream in (p.stdin, p.stdout, p.stderr):
                if stream:
                    stream.close()

    def download(self, src: str, rels, dest, timeout: float = 6 * 3600) -> list[str]:
        """Fetch ``rels`` (relative to ``src`` on the pod) into ``dest``. Only regular files in ``rels``."""
        wanted = set(rels)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        cmd = f"cd {shlex.quote(src)} && tar -cf - -- " + " ".join(shlex.quote(r) for r in sorted(wanted))
        p = subprocess.Popen(self.argv(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        got = []
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
                for m in tf:
                    name = m.name[2:] if m.name.startswith("./") else m.name
                    if not m.isfile() or name not in wanted:
                        continue
                    m.name = name
                    tf.extract(m, dest, filter="data")
                    got.append(name)
            err = p.stderr.read().decode(errors="replace")
            if p.wait(timeout=timeout) != 0:
                raise SSHError(f"download failed: {err.strip()[-300:]}")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            for stream in (p.stdin, p.stdout, p.stderr):
                if stream:
                    stream.close()
        return got
