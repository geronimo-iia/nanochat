import os
from typing import cast

from nanochat import workspace
from nanochat.common import autodetect_device_type, compute_init, print0
from nanochat.config import Config
from nanochat.evaluation.base.state import BaseEvalResult
from nanochat.evaluation.core_benchmark import evaluate_core
from nanochat.evaluation.engine import Engine
from nanochat.evaluation.hf_model import get_hf_token_bytes, load_hf_model
from nanochat.evaluation.loss_eval import evaluate_bpb
from nanochat.model_factory import load_model_from_dir
from nanochat.report import get_report
from nanochat.tokenizer import get_token_bytes
from nanochat.training.dataloader import tokenizing_distributed_data_loader_bos_bestfit


def run_base_eval(config: Config) -> None:
    eval_modes = {mode.strip() for mode in config.evaluation.modes.split(",")}

    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)

    is_hf_model = config.evaluation.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(config.evaluation.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = config.evaluation.hf_path
        model_slug = config.evaluation.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model_from_dir(
            phase="base",
            device=device,
            config=config.checkpoint,
            model_tag=config.common.model_tag,
            step=config.evaluation.step,
        )
        sequence_len = cast(int, cast(dict[str, object], meta["model_config"])["sequence_len"])
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {cast(int, meta['step'])})"
        model_slug = f"base_model_{cast(int, meta['step']):06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    result = BaseEvalResult.fresh()

    # Sampling
    if "sample" in eval_modes and not is_hf_model:
        print0("\n" + "=" * 80)
        print0("Model Samples")
        print0("=" * 80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                result.samples.append(sample_str)
            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                result.unconditioned_samples.append(sample_str)
    elif "sample" in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # BPB evaluation
    if "bpb" in eval_modes:
        print0("\n" + "=" * 80)
        print0("BPB Evaluation")
        print0("=" * 80)
        tokens_per_step = config.evaluation.device_batch_size * sequence_len * ddp_world_size
        if config.evaluation.split_tokens % tokens_per_step != 0:
            config.evaluation.split_tokens = (config.evaluation.split_tokens // tokens_per_step) * tokens_per_step
            print0(
                f"Adjusted split_tokens to {config.evaluation.split_tokens} (must be divisible by {tokens_per_step})"
            )
        steps = config.evaluation.split_tokens // tokens_per_step
        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer, config.evaluation.device_batch_size, sequence_len, split_name, device=device
            )
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            result.bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # CORE evaluation
    if "core" in eval_modes:
        print0("\n" + "=" * 80)
        print0("CORE Evaluation")
        print0("=" * 80)
        core_results = evaluate_core(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_per_task=config.evaluation.max_per_task,
        )
        result.core_results = core_results
        if ddp_rank == 0:
            output_csv_path = os.path.join(workspace.eval_results_dir(), f"{model_slug}.csv")
            results = cast(dict[str, float], core_results["results"])
            centered_results = cast(dict[str, float], core_results["centered_results"])
            with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in results:
                    f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # Report
    report_data: list[dict[str, object]] = [{"model": model_name}]
    if result.core_results:
        report_data[0]["CORE metric"] = result.core_results["core_metric"]
        report_data.append(cast(dict[str, object], result.core_results["centered_results"]))
    if result.bpb_results:
        report_data[0]["train bpb"] = result.bpb_results.get("train")
        report_data[0]["val bpb"] = result.bpb_results.get("val")
    if result.samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(result.samples)})
    if result.unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(result.unconditioned_samples)})

    get_report().log(section="Base model evaluation", data=report_data)
