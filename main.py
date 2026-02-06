import os
import sys
import socket
import subprocess


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    import torch

    ngpus = torch.cuda.device_count()

    # ensure project root is in PYTHONPATH
    root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if root not in pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = root + (os.pathsep + pythonpath if pythonpath else "")

    script = os.path.join(root, "dino.py")
    args = sys.argv[1:]  # forward all CLI args

    if ngpus > 1:
        port = find_free_port()
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=gpu",
            "--master_port",
            str(port),
            script,
            *args,
        ]
    else:
        cmd = [sys.executable, script, *args]

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
