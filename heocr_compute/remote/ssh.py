"""SSH to a pod: commands, and tar streams for files in and out.

Uses HeOCR's own key pair, ``~/.config/heocr/remote/ssh/id_ed25519``, made on first use. Its public half
is passed to the pod as ``PUBLIC_KEY``, which RunPod's images add to ``authorized_keys``, so the key needs
no registration with any account. Host keys of short-lived pods are not recorded.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Timer

KEY = "~/.config/heocr/remote/ssh/id_ed25519"


def ensure_key(key: str = KEY) -> Path:
    path = Path(os.path.expanduser(key))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "heocr-remote", "-f", str(path)],
                       check=True)
    return path


def public_key(key: str = KEY) -> str:
    return Path(str(ensure_key(key)) + ".pub").read_text(encoding="utf-8").strip()


class SSHError(Exception):
    pass


@contextmanager
def _tar_process(argv, timeout, *, stdin=None, stdout=None):
    """Bound the whole transfer, including blocked reads, writes and tar processing."""
    with tempfile.TemporaryFile() as stderr, subprocess.Popen(
        argv, stdin=stdin, stdout=stdout, stderr=stderr,
    ) as process:
        expired = False

        def abort():
            nonlocal expired
            expired = True
            process.kill()

        timer = Timer(timeout, abort)
        timer.start()
        try:
            yield process
            if process.wait() != 0 and not expired:
                stderr.seek(max(0, stderr.seek(0, os.SEEK_END) - 300))
                raise SSHError(f"tar transfer failed: {stderr.read().decode(errors='replace').strip()}")
        finally:
            timer.cancel()
            timer.join()
            if process.poll() is None:
                process.kill()
            process.wait()
            if expired:
                raise SSHError(f"tar transfer timed out after {timeout:.0f}s") from None


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
        with _tar_process(self.argv(cmd), timeout, stdin=subprocess.PIPE) as p:
            with tarfile.open(fileobj=p.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tf:
                for rel in rels:
                    info = tf.gettarinfo(str(root / rel), arcname=rel)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with open(root / rel, "rb") as f:
                        tf.addfile(info, f)
            p.stdin.close()

    def download(self, src: str, rels, dest, timeout: float = 6 * 3600, max_bytes: int | None = None) -> list[str]:
        """Fetch ``rels`` (relative to ``src`` on the pod) into ``dest``. Only regular files in ``rels``."""
        wanted = set(rels)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        cmd = f"cd {shlex.quote(src)} && tar -cf - -- " + " ".join(shlex.quote(r) for r in sorted(wanted))
        got = []
        with _tar_process(self.argv(cmd), timeout, stdout=subprocess.PIPE) as p:
            with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
                for m in tf:
                    name = m.name[2:] if m.name.startswith("./") else m.name
                    if not m.isfile() or name not in wanted:
                        continue
                    if max_bytes is not None:
                        max_bytes -= m.size
                        if max_bytes < 0 or name in got:
                            raise SSHError("tar transfer exceeded the verified size or repeated a file")
                    m.name = name
                    tf.extract(m, dest, filter="data")
                    got.append(name)
        return got
