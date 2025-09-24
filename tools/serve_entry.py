from openai import AsyncOpenAI
import asyncio
import time
import math
import uuid
from dataclasses import dataclass
from tools.serve_backend import spin_up_vllm_workers

BATCH_SIZE = 1250
BATCH_TIMEOUT = 0.5
TOTAL_GPUS = 8


@dataclass
class request:
    id: str
    prompt: str


@dataclass
class response:
    id: str
    text: str


class Client:
    def __init__(self, backend, base_url, api_key, model_name):
        self.backend = backend
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self._is_free = True

    async def get_batch_completion(self, reqs, **gen_kwargs):
        prompts = [req.prompt for req in reqs]
        ids = [req.id for req in reqs]
        print(f"Processing {len(reqs)} requests")
        if len(prompts) > 0:
            self._is_free = False
            completion = await self.client.completions.create(
                model=self.model_name, prompt=prompts, **gen_kwargs
            )
            responses = [
                response(id=id, text=item.text)
                for id, item in zip(ids, completion.choices)
            ]
            self._is_free = True
            return responses
        else:
            return []

    async def is_free(self):
        return self._is_free


class ClientGroup:
    def __init__(self, workers, model_name):
        self.workers = workers
        self.model_name = model_name
        self.clients = [
            Client(
                base_url=f"http://localhost:{worker['port']}/v1",
                api_key="token-abc123",
                backend=worker["worker"],
                model_name=model_name,
            )
            for worker in self.workers
        ]
        self.num_clients = len(self.clients)
        self.request_queue = asyncio.Queue()
        self.results = []
        self.total_requests = 0

    async def get_free_clients(self):
        return [client for client in self.clients if await client.is_free()]

    async def add_request(self, prompt, id):
        self.total_requests += 1
        await self.request_queue.put(request(prompt=prompt, id=id))

    async def process_requests(self, **gen_kwargs):
        current_batch = []
        batch_start_time = None

        while True:
            try:
                # Wait for the next request or timeout
                req = await asyncio.wait_for(
                    self.request_queue.get(), timeout=BATCH_TIMEOUT
                )
                current_batch.append(req)
                self.request_queue.task_done()
                # Start the timer when the first request is received
                if batch_start_time is None:
                    batch_start_time = time.time()

            except asyncio.TimeoutError:
                # Timeout reached with a partially filled batch
                if current_batch:
                    await self._process_batch(current_batch, **gen_kwargs)
                    current_batch = []
                    batch_start_time = None

            # Check if the batch is full or the timeout has been reached
            free_clients = len(await self.get_free_clients())
            if len(current_batch) >= BATCH_SIZE * free_clients or (
                batch_start_time is not None
                and time.time() - batch_start_time >= BATCH_TIMEOUT
            ):
                await self._process_batch(current_batch, **gen_kwargs)
                current_batch = []
                batch_start_time = None

            # Check if all requests have been processed
            if len(self.results) == self.total_requests:
                break

    async def _process_batch_single_client(self, client, batch, **gen_kwargs):
        # Process the batch of requests concurrently
        results = await client.get_batch_completion(
            batch,
            **gen_kwargs,
        )
        self.results.extend(results)

    async def _process_batch(self, batch, **gen_kwargs):
        # Process the batch of requests concurrently
        free_clients = await self.get_free_clients()
        num_requests = len(batch)
        requests_per_client = math.ceil(num_requests / len(free_clients))

        # Split the batch into sub-batches for each client
        sub_batches = [
            batch[i : i + requests_per_client]
            for i in range(0, num_requests, requests_per_client)
        ]

        await asyncio.gather(
            *[
                self._process_batch_single_client(client, sub_batch, **gen_kwargs)
                for client, sub_batch in zip(free_clients, sub_batches)
            ]
        )

        # Process each sub-batch concurrently
        # tasks = [
        #     self._process_batch_single(client, sub_batch, **gen_kwargs)
        #     for client, sub_batch in zip(free_clients, sub_batches)
        # ]

        # for task in asyncio.as_completed(tasks):
        #     await task

    async def generate(self, prompts, **gen_kwargs):
        start = time.time()
        uuids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        for prompt, id in zip(prompts, uuids):
            await self.add_request(prompt=prompt, id=id)
        end = time.time()
        print(f"Added {len(prompts)} requests in {end - start} seconds")

        # Process requests and collect results
        await self.process_requests(**gen_kwargs)
        results = self.results
        results_dict = {item.id: item.text for item in results}
        results = [results_dict[id] for id in uuids]

        self.results = []
        self.total_requests = 0
        return results


def get_entry(model_name, num_gpus=TOTAL_GPUS, gpus_per_worker=1, verbose=False):
    args = [
        "--model",
        model_name,
        "--dtype",
        "bfloat16",
        "--disable-log-requests",
    ]
    if verbose:
        args.pop(-1)

    workers = spin_up_vllm_workers(
        num_workers=num_gpus // gpus_per_worker,
        gpus_per_worker=gpus_per_worker,
        init_port=8000,
        args=args,
    )
    return ClientGroup(
        workers=workers,
        model_name=model_name,
    )


if __name__ == "__main__":
    num_clients = 8
    model_name = "./ckpt"

    test_num = 100000

    async def main(questions, **gen_kwargs):
        group = ClientGroup(num_clients, model_name)
        start = time.time()
        completions = await group.run(questions, **gen_kwargs)
        end = time.time()
        print(f"{(end - start)/len(questions)} seconds per completion")
        print("total time: ", end - start)
        print(completions[0])
        assert len(completions) == len(questions)

    questions = []
    with open("/scratch/tianhao/spag/data/all_target_words.txt", "r") as f:
        for item in f:
            questions.append(item.strip())

    questions = questions[:test_num]

    gen_kwargs = {
        "max_tokens": 100,
        "temperature": 1.0,
    }

    asyncio.run(main(questions, **gen_kwargs))
