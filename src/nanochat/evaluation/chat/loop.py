from dataclasses import asdict
from functools import partial
from typing import cast

import torch
import torch.distributed as dist

from nanochat.common import autodetect_device_type, compute_init, get_dist_info, print0
from nanochat.config import Config
from nanochat.evaluation.chat.state import ChatEvalResult
from nanochat.evaluation.engine import Engine
from nanochat.report import get_report
from nanochat.tasks.arc import ARC
from nanochat.tasks.base import Task
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tasks.humaneval import HumanEval
from nanochat.tasks.mmlu import MMLU
from nanochat.tasks.spellingbee import SpellingBee
from nanochat.model_factory import load_model_from_dir


def run_generative_eval(
    task_object: Task,
    tokenizer: object,
    model: torch.nn.Module,
    engine: Engine,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_problems: int | None = None,
) -> float:
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)
        total += 1
        num_passed += int(passed)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)", end="", flush=True)

    print()

    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")
    return num_passed / total


def run_categorical_eval(
    task_object: Task,
    tokenizer: object,
    model: object,
    batch_size: int,
    max_problems: int | None = None,
) -> float:
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache = {}
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(prompt_ids)  # (B, T, V)

        for idx, conversation in enumerate(conversations):
            letters = cast(list[str], conversation["letters"])
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


def run_chat_eval(
    task_name: str,
    model: object,
    tokenizer: object,
    engine: object,
    batch_size: int = 1,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    max_problems: int | None = None,
) -> float:
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    if task_object.eval_type == "generative":
        return run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
        )
    elif task_object.eval_type == "categorical":
        return run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")


def evaluate_chat_model(
    model: object,
    tokenizer: object,
    engine: object,
    task_names: list[str],
    device: object,
    batch_size: int = 8,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    max_problems: int | None = None,
) -> dict[str, float]:
    """Evaluate chat model on all or selected tasks, return per-task accuracies."""
    result = ChatEvalResult.fresh()
    for name in task_names:
        acc = run_chat_eval(
            name,
            model,
            tokenizer,
            engine,
            batch_size=batch_size,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            max_problems=max_problems,
        )
        result.results[name] = acc
        print0(f"{name} accuracy: {100 * acc:.2f}%")
    return result.results


def chat_eval(
    config: Config,
    source: str,
    task_name: str | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    num_samples: int = 1,
    top_k: int = 50,
    batch_size: int = 8,
    model_tag: str | None = None,
    step: int | None = None,
    max_problems: int | None = None,
) -> None:
    """Run chat model evaluation and log results to report."""
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    _, _, _, _, device = compute_init(device_type)

    model, tokenizer, _ = load_model_from_dir(phase=source, device=device, model_tag=model_tag, step=step)
    engine = Engine(model, tokenizer)

    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    baseline_accuracies = {
        "ARC-Easy": 0.25,
        "ARC-Challenge": 0.25,
        "MMLU": 0.25,
        "GSM8K": 0.0,
        "HumanEval": 0.0,
        "SpellingBee": 0.0,
    }
    task_names = all_tasks if task_name is None else task_name.split("|")

    results = evaluate_chat_model(
        model=model,
        tokenizer=tokenizer,
        engine=engine,
        task_names=task_names,
        device=device,
        batch_size=batch_size,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        max_problems=max_problems,
    )

    all_tasks_were_evaluated = all(t in results for t in all_tasks)
    chatcore_metric_dict: dict[str, object] = {}
    if all_tasks_were_evaluated:
        centered_mean = 0.0
        for t, acc in results.items():
            baseline_acc = baseline_accuracies.get(t, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}

    get_report().log(
        section="Chat evaluation " + source,
        data=[
            asdict(config),
            {
                "source": source,
                "task_name": task_name,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "num_samples": num_samples,
                "top_k": top_k,
                "batch_size": batch_size,
                "model_tag": model_tag,
                "step": step,
                "max_problems": max_problems,
            },
            results,
            chatcore_metric_dict,
        ],
    )
