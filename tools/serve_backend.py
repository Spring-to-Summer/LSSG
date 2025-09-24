import ray
import subprocess
import os
import sys
import time
import requests

ray.init(configure_logging=False)
MAX_SERVER_START_WAIT_S = 120


@ray.remote(num_gpus=1)
class VllmWorker:

    def __init__(self, args, port):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            ["python3", "-m", "vllm.entrypoints.openai.api_server"]
            + args
            + ["--port", str(port)],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self.port = port
        self._wait_for_server()

    def ready(self):
        return True

    def _wait_for_server(self):
        # run health check
        start = time.time()
        while True:
            try:
                if (
                    requests.get(f"http://localhost:{self.port}/health").status_code
                    == 200
                ):
                    break
            except Exception as err:
                if self.proc.poll() is not None:
                    raise RuntimeError("Server exited unexpectedly.") from err

                time.sleep(0.5)
                if time.time() - start > MAX_SERVER_START_WAIT_S:
                    raise RuntimeError("Server failed to start in time.") from err

    def __del__(self):
        if hasattr(self, "proc"):
            self.proc.terminate()


def spin_up_vllm_workers(num_workers, gpus_per_worker, init_port, args):
    vllm_workers = [
        VllmWorker.options(num_gpus=gpus_per_worker).remote(args, init_port + i)
        for i in range(num_workers)
    ]
    ray.get([worker.ready.remote() for worker in vllm_workers])
    return [
        {"worker": vllm_worker, "port": init_port + i}
        for i, vllm_worker in enumerate(vllm_workers)
    ]


if __name__ == "__main__":
    model_name = "./ckpt"
    args = ["--model", model_name, "--dtype", "bfloat16"]
    vllm_workers = spin_up_vllm_workers(
        num_workers=8, gpus_per_worker=1, init_port=8000, args=args
    )

    time.sleep(1000)
    print(vllm_workers)

    # # test sending requests
    # for i in range(8):
    #     response = requests.post(
    #         f"http://localhost:{8000+i}/complete",
    #         json={"prompt": "hello", "max_tokens": 10},
    #     )
    #     print(response.json())
    #     time.sleep(1)
