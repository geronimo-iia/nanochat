"""Worker and WorkerPool: per-GPU model replicas for parallel inference."""

import asyncio
from dataclasses import dataclass

import torch

from nanochat.config.checkpoint import CheckpointConfig
from nanochat.evaluation.engine import Engine
from nanochat.model_factory import load_model_from_dir
from nanochat.tokenizer import RustBPETokenizer


@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""

    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: RustBPETokenizer


class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, device_type: str, num_gpus: int | None = None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if device_type == "cuda" else 1
        self.device_type = device_type
        self.num_gpus = num_gpus
        self.workers: list[Worker] = []
        self.available_workers: asyncio.Queue[Worker] = asyncio.Queue()

    async def initialize(self, source: str, config: CheckpointConfig, model_tag: str | None = None, step: int | None = None) -> None:
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert self.device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):
            if self.device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(self.device_type)
                print(f"Loading model on {self.device_type}...")

            model, tokenizer, _ = load_model_from_dir(source, device, config=config, model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id=gpu_id, device=device, engine=engine, tokenizer=tokenizer)
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker) -> None:
        """Return a worker to the pool."""
        await self.available_workers.put(worker)
