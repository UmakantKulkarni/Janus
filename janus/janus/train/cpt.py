#!/usr/bin/env python3
import logging
import os
import math
import argparse
import yaml
import csv
from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import transformers
from accelerate.utils import set_seed
from torch.utils.data import random_split
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from janus.utils.paths import load_repo_config

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def get_config():
    """Parses command line arguments to get the config file path."""
    parser = argparse.ArgumentParser(description="Run Continual Pre-Training with FSDP and LoRA.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    return OmegaConf.create(config_dict)

def evaluate(model, eval_dataloader, accelerator):
    """
    Token-average validation loss across the whole eval set.
    """
    model.eval()
    total_num = torch.tensor(0.0, device=accelerator.device)
    total_den = torch.tensor(0.0, device=accelerator.device)

    with torch.no_grad():
        for _, batch in enumerate(eval_dataloader):
            batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}
            with accelerator.autocast():
                outputs = model(**batch)
            # mean over non-ignored tokens in this batch
            loss = outputs.loss.detach()

            # count valid tokens (labels != -100 is the HF convention)
            valid_tokens_local = (batch["labels"] != -100).sum().to(torch.float32)

            # turn batch mean back into sum of token losses for correct weighting
            num_local = (loss.to(torch.float32) * valid_tokens_local)

            # gather across processes
            num_g = accelerator.gather_for_metrics(num_local.unsqueeze(0)).sum()
            den_g = accelerator.gather_for_metrics(valid_tokens_local.unsqueeze(0)).sum()

            total_num += num_g
            total_den += den_g

    if total_den.item() == 0:
        eval_loss = float("nan")
        perplexity = float("inf")
    else:
        eval_loss = (total_num / total_den).item()
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

    return eval_loss, perplexity

def main():
    """
    Main function to run the Continual Pre-Training (CPT) process.
    """
    # --- 1. Initialization and Configuration ---
    cfg = get_config()
    repo_cfg = load_repo_config()
    set_seed(int(getattr(cfg.training, "seed", 1592)))
    base_model_name = repo_cfg.get("model", {}).get("base_model", "")

    # fsdp_plugin = FullyShardedDataParallelPlugin(
    #     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    #     optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    # )

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision")
        or repo_cfg.get("train", {}).get("mixed_precision", "no"),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
    )

    # --- Manual CSV Logger Setup ---
    csv_writer = None
    if accelerator.is_main_process:
        logger.info(f"Starting CPT script on rank {accelerator.process_index}")
        logger.info(f"Full Configuration:\n{OmegaConf.to_yaml(cfg)}")
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        log_file_path = os.path.join(cfg.training.output_dir, "training_log.csv")
        # Open in append mode to continue logging for resumed runs
        log_file = open(log_file_path, 'a', newline='')
        csv_writer = csv.writer(log_file)
        # Write header only if the file is new/empty
        if log_file.tell() == 0:
            csv_writer.writerow(["step", "training_loss_token_avg", "validation_loss", "perplexity", "learning_rate"])
        logger.info(f"CSV log is active at {log_file_path}")

    # --- 2. Load Tokenizer and Dataset ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set tokenizer pad_token to eos_token")

    logger.info(f"Loading dataset from {cfg.data.corpus_path}...")
    raw_dataset = load_dataset('text', data_files={'train': cfg.data.corpus_path})['train']

    def tokenize_function(examples):
        return tokenizer(examples['text'])

    logger.info("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=['text'])

    raw_block_size = int(cfg.data.block_size)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is not None and 0 < tok_max < 10**9:  # HF sometimes uses a huge sentinel
        block_size = min(raw_block_size, int(tok_max))
    else:
        block_size = raw_block_size
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Grouping texts into blocks of size {block_size}")
    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=os.cpu_count())

    split_percentage = cfg.training.get('validation_split_percentage', 5) / 100.0
    num_eval_samples = int(len(lm_dataset) * split_percentage)
    num_train_samples = len(lm_dataset) - num_eval_samples
    train_dataset, eval_dataset = random_split(
        lm_dataset, [num_train_samples, num_eval_samples], generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Dataset split into {len(train_dataset)} training samples and {len(eval_dataset)} validation samples.")

    collate_fn = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    pin_mem = (accelerator.device.type == "cuda")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.per_device_train_batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=pin_mem,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=cfg.training.per_device_train_batch_size, collate_fn=collate_fn, pin_memory=pin_mem,
    )

    # --- 3. Load Model and Apply LoRA ---
    logger.info(f"Loading base model '{base_model_name}'")
    if accelerator.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = None
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        use_cache=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    logger.info("Injecting LoRA adapters")
    target_modules_list = list(cfg.lora.target_modules)
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=target_modules_list,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # --- 4. Prepare for Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.get('num_warmup_steps', 0),
        num_training_steps=max_train_steps,
    )

    if accelerator.is_main_process:
        logger.info(
            f"len(train_dl)={len(train_dataloader)} (per rank), "
            f"grad_accum={cfg.training.gradient_accumulation_steps}, "
            f"steps/epoch={num_update_steps_per_epoch}, "
            f"epochs={cfg.training.num_train_epochs}, "
            f"max_train_steps={max_train_steps}, "
            f"world_size={accelerator.num_processes}"
        )
        dist = accelerator.state.distributed_type
        logger.info(f"{dist.name} + LoRA ready for training.")

    # =================================================================
    # == RESUME FROM CHECKPOINT
    # =================================================================
    resume_from_checkpoint = cfg.training.get('resume_from_checkpoint')
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        path = os.path.basename(resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        completed_steps = int(training_difference.replace("step_", ""))
        logger.info(f"Resumed from step {completed_steps}")
    else:
        completed_steps = 0
    # =================================================================

    progress_bar = tqdm(range(max_train_steps), initial=min(completed_steps, max_train_steps), disable=not accelerator.is_main_process)

    # --- 5. Training Loop ---
    model.train()
    best_val_loss = float("inf")
    best_step = None
    patience_counter = 0
    patience = cfg.training.get('early_stopping_patience', 5)

    # these will hold the per-update (after accumulation) token-avg loss for CSV
    step_num_local = torch.tensor(0.0, device=accelerator.device)  # sums of token losses within the current update (this rank)
    step_den_local = torch.tensor(0.0, device=accelerator.device)  # sums of tokens within the current update (this rank)

    for epoch in range(cfg.training.num_train_epochs):
        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(accelerator.device, non_blocking=True) for k, v in batch.items()}
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    # per-microbatch mean over non-ignored tokens
                    loss = outputs.loss
                accelerator.backward(loss)

                # accumulate token-weighted sums across micro-batches in this update
                valid_tokens_local_mb = (batch["labels"] != -100).sum().to(torch.float32)
                step_num_local += (loss.detach().to(torch.float32) * valid_tokens_local_mb)
                step_den_local += valid_tokens_local_mb

                # only when we’re at the end of an accumulation window:
                if accelerator.sync_gradients:
                    if getattr(cfg.training, "clip_grad_norm", 0) and cfg.training.clip_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # compute per-update token-average training loss across ranks
                    num_g = accelerator.gather_for_metrics(step_num_local.unsqueeze(0)).sum()
                    den_g = accelerator.gather_for_metrics(step_den_local.unsqueeze(0)).sum()
                    train_token_avg = (num_g / den_g.clamp(min=1.0)).item()

                    # reset for the next update
                    step_num_local.zero_()
                    step_den_local.zero_()

                    progress_bar.update(1)
                    completed_steps += 1

                    if completed_steps % cfg.checkpointing.eval_steps == 0:
                        val_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
                        logger.info(f"Step {completed_steps}: Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
                        model.train()

                        if accelerator.is_main_process:
                            lr = optimizer.param_groups[0]["lr"]
                            csv_writer.writerow([completed_steps, train_token_avg, val_loss, perplexity, lr])

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_step = completed_steps
                            logger.info("New best validation loss. Saving model...")

                            adapter_save_path = os.path.join(cfg.training.output_dir, "best_adapter")
                            unwrapped_model = accelerator.unwrap_model(model)
                            state_dict = accelerator.get_state_dict(model)
                            if accelerator.is_main_process:
                                unwrapped_model.save_pretrained(
                                    adapter_save_path,
                                    state_dict=state_dict,
                                    safe_serialization=True
                                )
                                tokenizer.save_pretrained(adapter_save_path)
                                logger.info(f"Successfully saved PEFT adapter to {adapter_save_path}")
                            accelerator.wait_for_everyone()
                        else:
                            patience_counter += 1
                            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

                        if patience_counter >= patience:
                            logger.info("Early stopping triggered. Finishing training.")
                            break

            if completed_steps >= max_train_steps or patience_counter >= patience:
                break
        if patience_counter >= patience:
            break

    # Final eval/log so the last partial interval doesn't miss CSV/val logs
    if completed_steps % cfg.checkpointing.eval_steps != 0:
        val_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
        logger.info(f"Final step {completed_steps}: Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
        # also log the last update's training token-avg if we didn't just log it
        if accelerator.is_main_process:
            lr = optimizer.param_groups[0]["lr"]
            # if the very last update didn't reach an eval boundary, we still have
            # step_num_local/step_den_local = 0 here (they’re reset at boundaries),
            # so just put NaN for the trailing training loss.
            csv_writer.writerow([completed_steps, float('nan'), val_loss, perplexity, lr])
        model.train()

    if best_step is None:
        logger.info("best_step not found. Saving final model.")
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            final_dir = os.path.join(cfg.training.output_dir, "best_adapter")
            unwrapped_model.save_pretrained(final_dir, state_dict=state_dict, safe_serialization=True)
            tokenizer.save_pretrained(final_dir)
        accelerator.wait_for_everyone()

    # --- 6. Final Cleanup ---
    logger.info("Training finished.")
    accelerator.end_training()
    if accelerator.is_main_process:
        log_file.close()
    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()