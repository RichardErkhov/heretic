# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, TypeAlias, cast

import torch
import tqdm
from accelerate.utils import named_module_tensors, set_module_tensor_to_device
from peft import LoraConfig, PeftModel, get_peft_model
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)

try:
    from transformers.masking_utils import create_causal_mask
except ImportError:  # Transformers < 5
    create_causal_mask = None

try:
    from transformers.masking_utils import create_recurrent_attention_mask
except ImportError:  # Transformers without recurrent-mask helper
    create_recurrent_attention_mask = None

from .config import QuantizationMethod, Settings
from .system import empty_cache
from .utils import Prompt, batchify, format_exception, print


def get_model_class(
    model: str,
) -> type[AutoModelForImageTextToText] | type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    if any(("vision_config" in config) for config in configs):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


def check_model_resident(
    model: PreTrainedModel | PeftModel, streamed: bool = False
) -> None:
    """
    Ensure that all of the model's weights are resident in memory.

    Streamed runs tolerate meta placeholders: their weights are loaded from
    disk layer by layer when touched (see `_call_layer`), so the check only
    constrains eager runs.
    """
    if streamed:
        return
    offloaded = [name for name, param in model.named_parameters() if param.is_meta]

    if not offloaded:
        return

    raise Exception(
        "The model does not fit into the available device memory: "
        f"{len(offloaded)} weights (such as {', '.join(offloaded[:3])}) had to "
        "be offloaded to CPU or disk. Heretic cannot process offloaded models "
        "because it requires direct, persistent access to all weights, whereas "
        "offloaded weights only exist temporarily during forward passes.\n"
        "Possible remedies:\n"
        "* Free up memory, e.g. by closing other applications.\n"
        "* Change the device map (device_map/max_memory settings) so that all "
        "weights fit into device memory.\n"
        '* Enable 4-bit quantization (quantization = "bnb_4bit").'
    )


def _collapse_flattened_residual_streams(
    hidden_states: tuple[FloatTensor, ...],
    config: Any,
) -> tuple[FloatTensor, ...]:
    """
    Average flattened multi-stream hidden states to hidden_size.

    Some models (e.g. Qwen4-Exp) flatten their residual streams into the
    component dimension instead of exposing a separate stream dimension, so
    their hidden states have shape (prompt, position, streams * hidden_size).
    The final entry is the merged model output (hidden_size) and is unchanged.
    """
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        config = get_text_config()
    hidden_size = getattr(config, "hidden_size", None)
    streams = getattr(config, "hc_count", 1)
    if hidden_size is None or streams <= 1:
        return hidden_states
    width = hidden_size * streams
    return tuple(
        h.unflatten(-1, (streams, hidden_size)).mean(-2) if h.shape[-1] == width else h
        for h in hidden_states
    )


# Some chat templates (e.g. gpt-oss) insert the current date into the prompt,
# which makes the residuals, and therefore the modified model, depend on the day
# Heretic is run. Rendering templates with a fixed date keeps results reproducible.
def strftime_fixed_date(format: str) -> str:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(format)


def chat_template_supports_system_role(tokenizer: PreTrainedTokenizerBase) -> bool:
    # Some chat templates (e.g. Gemma 2 and its fine-tunes) reject the system role
    # outright, raising when one is present; others silently drop it. The only
    # reliable way to tell is to render a probe chat and check whether a sentinel
    # system message survives into the output. This is also how llama.cpp's template
    # engine (minja) detects system-role support.
    sentinel = "HERETIC_SYSTEM_ROLE_PROBE"
    try:
        rendered = cast(
            str,
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sentinel},
                    {"role": "user", "content": "Hi"},
                ],
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
    except Exception:
        return False
    return sentinel in rendered


# The list contains one element per layer.
# Each element maps from the component name to a (possibly sparse) mapping
# from the module index to an (input, output) tuple containing the I/O
# tensors of shape (prompt, component).
ModuleIO: TypeAlias = list[dict[str, dict[int, tuple[Tensor, Tensor]]]]


def _cuda_usable() -> bool:
    # CUDA_VISIBLE_DEVICES="" keeps is_available() True with zero visible devices on some builds; gate on the count so device work falls back to CPU instead of failing with "Invalid device id".
    return torch.cuda.is_available() and torch.cuda.device_count() > 0

# Pinned staging buffers reused per weight shape; fills and hops stay ordered on the single upload stream, so a buffer is never overwritten while its previous hop is in flight.
_STAGING: dict[tuple[Any, Any], Tensor] = {}


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype
    # Whether the model's chat template accepts a system role (see __init__).
    supports_system_role: bool

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self.stream_fallback_reason: str | None = None
        self.passes_bar: tqdm.tqdm | None = None

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        # PreTrainedTokenizerBase is the "base class for all tokenizer backends"
        # according to the documentation.
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            ),
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if get_model_class(settings.model) == AutoModelForImageTextToText:
            self.processor = AutoProcessor.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            )

        # With HF_HUB_OFFLINE set, from_pretrained records its transport flag in init_kwargs,
        # and save_pretrained re-exports it in tokenizer_config.json; pin it so exports are
        # byte-identical regardless of how the hub cache was populated.
        for tokenizer in (self.tokenizer, getattr(self.processor, "tokenizer", None)):
            if tokenizer is not None and "local_files_only" in tokenizer.init_kwargs:
                tokenizer.init_kwargs["local_files_only"] = False

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        # Detect once whether this tokenizer's chat template accepts a system
        # role. Templates that don't (e.g. Gemma 2) need the system prompt folded
        # into the first user message instead (see _apply_system_role).
        self.supports_system_role = chat_template_supports_system_role(self.tokenizer)

        self.model = None
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
                # Disk-first streaming: accelerate only spills to the offload
                # folder when max_memory carries an explicit "disk" bucket;
                # without it every layer past the RAM bucket just stays in CPU
                # RAM. With the disk bucket, each layer is read back from disk
                # when touched, so peak RAM is about one layer, not the model.
                else ({"cpu": "2GB", **({0: "6GB"} if _cuda_usable() else {}),
                       "disk": float("inf")} if settings.stream_layers else None)
        )

        if settings.stream_layers and os.name == "nt":
            # Windows counts warm file pages in process RSS, so a disk-streamed
            # load looks RAM-backed even though only the touched layer is ever
            # needed. Cap the working set: the OS then trims the disk cache
            # towards the layer working set instead of the whole checkpoint.
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(
                handle, ctypes.c_size_t(512 * 2**20), ctypes.c_size_t(8 * 2**30)
            )
            ctypes.WinDLL("psapi").EmptyWorkingSet(handle)

        self.trusted_models = set()

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config
                # Streaming keeps weights on disk: layers feed the compute
                # device as they are touched, and accelerate spills everything
                # beyond the buckets into offload safetensors.
                if settings.stream_layers:
                    extra_kwargs["low_cpu_mem_usage"] = True
                    # Needed when offloaded weights must be re-saved (renamed or
                    # MoE-packed tensors); plain safetensors are indexed in place.
                    extra_kwargs["offload_folder"] = "hf-cache/offload"

                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if settings.model in self.trusted_models
                    else None,
                    **self.revision_kwargs,
                    **extra_kwargs,
                )

                self.dtype = self.model.dtype

                # If we reach this point and the model requires trust_remote_code,
                # the user must have agreed when prompted to execute remote code,
                # because from_pretrained raises an exception otherwise.
                self.trusted_models.add(settings.model)

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                    # The KV cache is deliberately left enabled here so we can
                    # catch dtype errors related to the cache, which we will need
                    # for generating responses.
                    # use_cache=False,
                )
            except Exception as error:
                self.model = None
                empty_cache()

                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        check_model_resident(self.model, settings.stream_layers)

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)

        print("* Modifiable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def apply_lora(self, lora_rank: int):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for modification (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        # PEFT initializes A randomly from the global RNG. B is zero, so this doesn't
        # change the model, but zeroing A too keeps saved adapters reproducible.
        self._zero_lora_weights()

    def _zero_lora_weights(self):
        device = self.compute_device
        for name, module in self.model.named_modules():
            if ("lora_A" in name or "lora_B" in name) and hasattr(module, "weight"):
                if module.weight.is_meta:
                    # PEFT creates the adapter weights while dispatched base
                    # weights are meta placeholders, so they land on meta too;
                    # rebuild them as real tensors the optimizer can update.
                    module.weight = torch.nn.Parameter(
                        torch.zeros(
                            module.weight.shape,
                            dtype=module.weight.dtype,
                            device=device,
                        )
                    )
                else:
                    torch.nn.init.zeros_(module.weight)

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # Hierarchical device_map keeps some layers on CPU; this flag is
                # what makes transformers accept that mix instead of rejecting it.
                llm_int8_enable_fp32_cpu_offload=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Check if we need special handling for quantized models
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self) -> bool:
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.

        Returns True if the fast path was taken.
        """

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            # Zeroing A as well ensures that adapter weights of modules not modified
            # by the next trial don't depend on what previous trials did.
            self._zero_lora_weights()
            return True

        # Purge existing model object from memory to make space.
        self.model = None
        empty_cache()

        quantization_config = self._get_quantization_config(
            str(self.dtype).split(".")[-1]
        )

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config
        # See __init__: streamed runs load lazily from disk.
        if self.settings.stream_layers:
            extra_kwargs["low_cpu_mem_usage"] = True
            extra_kwargs["disable_mmap"] = True
            extra_kwargs["offload_folder"] = "hf-cache/offload"

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **self.revision_kwargs,
            **extra_kwargs,
        )

        check_model_resident(self.model, self.settings.stream_layers)

        self.needs_reload = False

        return False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

        # Text-only models.
        return model.model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Standard self-attention out-projection (most models).
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)  # ty: ignore[unresolved-attribute]

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)  # ty: ignore[unresolved-attribute]

        # DeepSeek V4.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_b_proj)  # ty: ignore[unresolved-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty: ignore[unresolved-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Shared experts of MoE models (e.g. DeepSeek V4 and GLM 5.3).
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.shared_experts.down_proj)  # ty: ignore[unresolved-attribute]

        # Shared expert of MoE models (e.g. Qwen4-Exp).
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.shared_expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # LFM dense operator blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.conv.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            try_add("mlp.down_proj", layer.feed_forward.w2)  # ty: ignore[unresolved-attribute]

        # LFM transformer blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            for expert in layer.feed_forward.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.output_linear)  # ty: ignore[unresolved-attribute]

        # We need at least one module across all components.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No modifiable modules found in layer"

        return modules

    def get_modifiable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        return sorted(components)

    def _apply_system_role(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        # Templates that support the system role get the messages unchanged, so the
        # rendered chat (and therefore the resulting weights) is byte-for-byte
        # identical to before. For templates that reject it (e.g. Gemma 2), fold each
        # system message into the next user message (what Gemma's own later templates
        # do), so a system role anywhere in the chat is removed before rendering, not
        # just a leading one. Empty system prompts are dropped; a system message with
        # no following user message becomes a user turn.
        if self.supports_system_role:
            return messages
        result: list[dict[str, str]] = []
        pending: list[str] = []
        for message in messages:
            if message["role"] == "system":
                if message["content"]:
                    pending.append(message["content"])
                continue
            if pending and message["role"] == "user":
                message = dict(message)
                message["content"] = "\n\n".join([*pending, message["content"]])
                pending = []
            result.append(message)
        if pending:
            result.append({"role": "user", "content": "\n\n".join(pending)})
        return result

    def _render_chat_prompts(self, prompts: list[Prompt]) -> list[str]:
        chats = [
            self._apply_system_role(
                [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ]
            )
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
                # Overrides the function that templates call to get the current date.
                strftime_now=strftime_fixed_date,
            ),
        )

        if self.settings.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]

        return chat_prompts

    @property
    def compute_device(self) -> torch.device:
        # The input embedding is what consumes the token ids, and the streamed
        # path parks it on the compute device; `model.device` can report a
        # stale value for multi-tower checkpoints. A disk-offloaded embedding
        # rests on the meta device, so fall back to where activations run.
        embedding = self.model.get_input_embeddings()
        if _cuda_usable():
            return torch.device("cuda")
        if embedding.weight.is_meta:
            return torch.device("cpu")
        return cast(torch.device, embedding.weight.device)

    def _round_batches(self, prompts: list[Prompt]) -> list[list[Prompt]]:
        # Streamed runs feed the whole round into a single streamed pass: every
        # layer's weights are then loaded once and reused by each chunk of the
        # round, instead of repeating the disk read once per chunk
        # (see _stream_forward_once). Eager runs keep plain batch-sized chunks.
        if self.settings.stream_layers:
            return [prompts]

        return batchify(prompts, self.settings.batch_size)

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        if self.settings.stream_layers:
            return self._generate_streamed(prompts, **kwargs)

        result = self._eager_generate(prompts, **kwargs)
        self._finish_pass()
        return result

    def _eager_generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        # Plain HF forward/generate. Separate from generate() so the streamed
        # path and its fallback can call it without re-entering the dispatch.
        chat_prompts = self._render_chat_prompts(prompts)

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.compute_device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def _generate_streamed(
        self, prompts: list[Prompt], **kwargs: Any
    ) -> tuple[BatchEncoding, LongTensor]:
        # Greedy decoding on top of the streamed forward: every step re-runs
        # the layer-streaming pass over prompt plus continuation, so weights
        # never need to stay resident between steps (no KV cache). Responses
        # are capped at a few dozen tokens, keeping the recompute cheap; the
        # decoded pieces are concatenated as text and re-tokenized together.
        max_new_tokens = int(kwargs.get("max_new_tokens", 20))
        self._begin_passes(max_new_tokens)

        chat_prompts = self._render_chat_prompts(prompts)
        continuations = [""] * len(chat_prompts)
        eos_token_id = self.tokenizer.eos_token_id
        finished = [False] * len(chat_prompts)

        for _ in range(max_new_tokens):
            _, logits = self._stream_forward(prompts, continuations)

            for row, token_id in enumerate(logits.argmax(dim=-1).tolist()):
                if finished[row]:
                    continue
                if token_id == eos_token_id:
                    finished[row] = True
                else:
                    continuations[row] += self.tokenizer.decode([token_id])

            if all(finished):
                break

        inputs = self.tokenizer(
            chat_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.compute_device)

        outputs = self.tokenizer(
            [
                prompt + text
                for prompt, text in zip(chat_prompts, continuations)
            ],
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )["input_ids"].to(self.compute_device)

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens is not None
                else self.settings.max_response_length
            ),
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        responses = []

        for batch in self._round_batches(prompts):
            responses.extend(
                self.get_responses(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                    max_new_tokens=max_new_tokens,
                )
            )

        return responses

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        if self.settings.stream_layers:
            layer_states, _ = self._stream_forward(prompts)
            return self._finish_residuals(layer_states, winsorization_quantile)

        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # KV cache is unnecessary here because we only need the hidden states
            # for the first generated token.
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        return self._finish_residuals(
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            winsorization_quantile,
        )

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        residuals = []

        for batch in self._round_batches(prompts):
            residuals.append(
                self.get_residuals(
                    batch,
                    winsorization_quantile=winsorization_quantile,
                )
            )

        return torch.cat(residuals, dim=0)

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        running_sum = None
        total_count = 0

        for batch in self._round_batches(prompts):
            batch_residuals = self.get_residuals(
                batch,
                winsorization_quantile=winsorization_quantile,
            )

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    def get_module_io(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # The list contains one element per layer.
        # Each element maps from the component name to a (possibly sparse) mapping
        # from the module index to an (input, output) tuple containing the I/O
        # tensors of shape (prompt, component).
        module_io: ModuleIO = []

        def get_hook(
            layer_index: int,
            component: str,
            module_index: int,
        ) -> Callable[[Module, tuple[Tensor, ...], Tensor], None]:
            def hook(
                module: Module,
                inputs: tuple[Tensor, ...],
                outputs: Tensor,
            ) -> None:
                if len(module_io) == layer_index:
                    # First invocation of the hook for this layer.
                    module_io.append({})

                # Layers are invoked in order during inference,
                # so this should always hold.
                assert len(module_io) == layer_index + 1

                if component not in module_io[layer_index]:
                    module_io[layer_index][component] = {}

                # inputs[0] and outputs have shape (prompt, position, component),
                # so this extracts the input/output at the end of each prompt.
                # Move to CPU to decouple from device assignments, which can
                # change between model reloads in multi-GPU configurations.
                input = inputs[0][:, -1, :].detach().clone().cpu()
                output = outputs[:, -1, :].detach().clone().cpu()

                # Streamed passes queue several chunks of prompts through each
                # resident layer, so one module can be seen again within the same
                # pass; append the new rows instead of replacing the captured ones.
                if module_index in module_io[layer_index][component]:
                    previous_input, previous_output = module_io[layer_index][component][module_index]
                    input = torch.cat([previous_input, input], dim=0)
                    output = torch.cat([previous_output, output], dim=0)

                # The modules associated with a component (e.g. expert MLPs)
                # are not necessarily invoked in order, nor are all of them
                # necessarily invoked in each inference step, so we cannot
                # use a list here.
                module_io[layer_index][component][module_index] = (input, output)

            return hook

        hook_handles: list[RemovableHandle] = []

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    hook_handles.append(
                        module.register_forward_hook(
                            get_hook(layer_index, component, module_index)
                        )
                    )

        if self.settings.stream_layers:
            # The forward hooks installed above capture the module I/O regardless
            # of which forward path runs.
            self._stream_forward(prompts)
        else:
            self.generate(
                prompts,
                max_new_tokens=1,
                use_cache=False,
            )

        for hook_handle in hook_handles:
            hook_handle.remove()

        return module_io

    def get_module_io_batched(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # Aggregating batch results is more complicated for module I/O
        # than for other get_*_batched methods, because the structure of the results
        # might differ between batches, as whether individual modules activate
        # can depend on the prompt (in particular for MoE models).
        # In practice, inhomogeneous results should be very rare, but to be fully
        # generic, this logic is required.
        module_io_batches: list[ModuleIO] = [
            self.get_module_io(batch)
            for batch in self._round_batches(prompts)
        ]

        module_io: ModuleIO = []

        for layer_index in range(len(self.get_layers())):
            module_io.append({})

            for module_io_batch in module_io_batches:
                for component, io_map in module_io_batch[layer_index].items():
                    if component not in module_io[layer_index]:
                        module_io[layer_index][component] = {}

                    for module_index in io_map:
                        if module_index not in module_io[layer_index][component]:
                            # This is a placeholder; the actual aggregation happens below.
                            # We need to iterate over the batches twice because we don't
                            # know in advance which components and module indices are present.
                            module_io[layer_index][component][module_index] = (
                                torch.empty(0),
                                torch.empty(0),
                            )

            for component, io_map in module_io[layer_index].items():
                for module_index in io_map:
                    inputs_outputs = [
                        module_io_batch[layer_index][component][module_index]
                        for module_io_batch in module_io_batches
                        if component in module_io_batch[layer_index]
                        and module_index in module_io_batch[layer_index][component]
                    ]
                    input = torch.cat(
                        [input_output[0] for input_output in inputs_outputs],
                        dim=0,
                    )
                    output = torch.cat(
                        [input_output[1] for input_output in inputs_outputs],
                        dim=0,
                    )

                    # The key already exists, and replacing existing values
                    # in a dictionary while iterating over the same dictionary
                    # is safe in Python.
                    module_io[layer_index][component][module_index] = (input, output)

        return module_io

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        if self.settings.stream_layers:
            _, logits = self._stream_forward(prompts)
            if self.settings.offload_outputs_to_cpu:
                logits = logits.cpu()
                empty_cache()
            return logits

        # We only generate one token, and we return the raw logits over the vocabulary
        # at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        # This cast is valid because we passed output_logits=True above.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        if self.settings.offload_outputs_to_cpu:
            del outputs
            logits = logits.cpu()
            empty_cache()

        return logits

    def get_logits_batched(self, prompts: list[Prompt]) -> Tensor:
        rounds = self._round_batches(prompts)
        self._begin_passes(len(rounds))
        logits = []

        for batch in rounds:
            logits.append(self.get_logits(batch))

        return torch.cat(logits, dim=0)

    def _finish_residuals(
        self,
        layer_states: list[Tensor],
        winsorization_quantile: float,
    ) -> Tensor:
        # Models with Hyper-Connections (e.g. DeepSeek V4 and GLM 5.3) have several
        # residual streams, so their hidden states at the last prompt position have
        # shape (prompt, stream, component). We use the mean of the streams.
        # The last entry has no stream dimension, because Transformers replaces it
        # with the model output, where the streams have already been merged.
        layer_states = [
            layer_state.mean(dim=1) if layer_state.dim() == 3 else layer_state
            for layer_state in layer_states
        ]

        layer_states = list(
            _collapse_flattened_residual_streams(
                cast(tuple[FloatTensor, ...], tuple(layer_states)),
                self.model.config,
            )
        )

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(layer_states, dim=1)

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def _progress_bar(self) -> tqdm.tqdm:
        # Outer bar: counts complete streamed layer sweeps ("loops"). The inner
        # "Streaming layers" bar is disappearable (leave=False); this one stays
        # and shows how many loops ran out of how many the current phase plans.
        if self.passes_bar is None:
            self.passes_bar = tqdm.tqdm(desc=self._passes_desc())
        return self.passes_bar

    def _passes_desc(self) -> str:
        return f"Streaming passes (batch size {self.settings.batch_size})"

    def _begin_passes(self, total: int) -> None:
        # Phases with a known round count announce it here; others leave the
        # bar count-only. Batch-size tuning reuses the bar across trials,
        # so refresh the desc to the size actually in effect.
        bar = self._progress_bar()
        bar.set_description_str(self._passes_desc())
        bar.reset(total=total)

    def _finish_pass(self) -> None:
        self._progress_bar().update(1)

    def _stream_forward(
        self, prompts: list[Prompt], suffixes: list[str] | None = None
    ) -> tuple[list[Tensor], Tensor]:
        """
        Layer-by-layer streamed forward pass: each layer's weights are read from disk
        onto the compute device when the layer is touched and evicted afterwards, so
        peak memory stays about one layer plus activations and cached IO on hosts
        where the whole model does not fit into VRAM. Returns the hidden state at the
        last prompt position for each layer boundary that Transformers records
        (embedding output, intermediate layer outputs, final norm output), plus logits.
        """
        with torch.no_grad():
            if self.stream_fallback_reason is None:
                try:
                    result = self._stream_forward_once(prompts, suffixes)
                    self._finish_pass()
                    return result
                except Exception as error:
                    self.stream_fallback_reason = format_exception(error)
                    print(
                        f"* Streamed forward unavailable ({self.stream_fallback_reason}), falling back to eager forward."
                    )

            # Rounds can hold more prompts than one eager batch; keep the
            # fallback's memory profile at one batch at a time and stitch the
            # per-batch results back together in prompt order.
            batches = batchify(prompts, max(1, self.settings.batch_size))
            if len(batches) <= 1:
                states, logits = self._eager_forward_once(prompts)
                self._finish_pass()
                return states, logits

            batch_states: list[list[Tensor]] = []
            batch_logits: list[Tensor] = []

            for batch in batches:
                states, logits = self._eager_forward_once(batch)
                self._finish_pass()
                batch_states.append(states)
                batch_logits.append(logits)

            return (
                [
                    torch.cat([states[boundary] for states in batch_states], dim=0)
                    for boundary in range(len(batch_states[0]))
                ],
                torch.cat(batch_logits, dim=0),
            )

    def _stream_forward_once(
        self, prompts: list[Prompt], suffixes: list[str] | None = None
    ) -> tuple[list[Tensor], Tensor]:
        chat_prompts = self._render_chat_prompts(prompts)
        if suffixes is not None:
            # Continuations follow the rendered chat text directly, exactly
            # like decoded tokens appended in the eager path. The eager
            # fallback ignores them (see _eager_forward_once).
            chat_prompts = [
                prompt + suffix
                for prompt, suffix in zip(chat_prompts, suffixes)
            ]

        model = self.model

        # Unwrap PeftModel (always true after _apply_lora), then locate the module
        # owning the transformer layers (mirrors get_layers()).
        outer = model.base_model.model if isinstance(model, PeftModel) else model
        inner = getattr(outer, "model", outer)

        # Multimodal models keep their text stack in language_model.
        root = getattr(inner, "language_model", inner)

        layers = self.get_layers()
        norm = getattr(root, "norm", None)
        lm_head = (
            getattr(root, "lm_head", None)
            or getattr(outer, "lm_head", None)
            or model.lm_head
        )
        embedding = model.get_input_embeddings()

        # Streamed mode keeps decoder-layer weights outside VRAM: each layer is
        # moved onto the compute device when touched and evicted afterwards
        # (see _call_layer). Embedding, norm and lm_head are hit by every
        # forward pass, so they stay resident there.
        device = self.compute_device
        # accelerate's disk-offload keeps unloaded weights as meta tensors and fills them when the module is called; a manual `.to()` cannot copy out of meta and would drop the weights.
        def to_compute(module: Module) -> None:
            tensor = next(module.parameters(), None)
            if tensor is None:
                # Rotary-style modules hold only buffers (inv_freq etc.).
                tensor = next(module.buffers(), None)
            if tensor is None or not tensor.is_meta:
                module.to(device)

        to_compute(embedding)
        to_compute(lm_head)
        if norm is not None:
            to_compute(norm)

        rotary_embedding = getattr(root, "rotary_emb", None)
        if rotary_embedding is not None:
            to_compute(rotary_embedding)

        root_config = getattr(root, "config", model.config)

        def prepare(chat_texts: list[str]) -> dict[str, Any]:
            # Tokenize and embed one chunk. Each chunk keeps its own padding,
            # masks and rotary states, so chunks can queue through every layer
            # while that layer's weights are resident.
            inputs = self.tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            hidden_states = embedding(input_ids)

            # Mirror the model's own forward pass exactly: Transformers 5 derives
            # padding-aware position ids and attention masks via the helpers in
            # `transformers.masking_utils`; older versions build the ids and the
            # combined causal+padding mask by hand. Using anything else shifts
            # every attention row and changes which token wins the argmax.
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if create_causal_mask is not None:
                mask_kwargs = {
                    "config": root_config,
                    "inputs_embeds": hidden_states,
                    "attention_mask": attention_mask,
                    "past_key_values": None,
                    "position_ids": position_ids,
                }
                full_mask = create_causal_mask(**mask_kwargs)
                layer_types = list(getattr(root_config, "layer_types", None) or [])
                hybrid = (
                    len(set(layer_types)) > 1
                    and len(layer_types) == len(layers)
                    and create_recurrent_attention_mask is not None
                )
                if hybrid:
                    # Hybrid stacks (e.g. Qwen3.5) mix full-attention and recurrent
                    # layers, and the text tower feeds each layer type its own mask.
                    recurrent_mask = create_recurrent_attention_mask(**mask_kwargs)
                    masks_by_type = {
                        layer_type: recurrent_mask
                        if layer_type == "linear_attention"
                        else full_mask
                        for layer_type in set(layer_types)
                    }
                    causal_masks = [masks_by_type[layer_type] for layer_type in layer_types]
                else:
                    causal_masks = [full_mask] * len(layers)
            else:
                seq_len = input_ids.shape[1]
                causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
                allowed = attention_mask[:, None, None, :].bool() & causal
                full_mask = (1.0 - allowed.to(hidden_states.dtype)) * torch.finfo(
                    hidden_states.dtype
                ).min
                causal_masks = [full_mask] * len(layers)

            position_embeddings = None
            if rotary_embedding is not None:
                try:
                    position_embeddings = rotary_embedding(hidden_states, position_ids)
                except IndexError:
                    # Multi-axis rotary (Qwen3.5) indexes one position row per
                    # spatial axis; the text tower feeds it identical text rows.
                    position_embeddings = rotary_embedding(
                        hidden_states, position_ids[None, ...].expand(3, -1, -1)
                    )

            kwargs: dict[str, Any] = {"position_ids": position_ids}
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings

            return {
                "hidden": hidden_states,
                "kwargs": kwargs,
                "causal_masks": causal_masks,
                "rows": [],
            }

        # Layer-major pass: instead of running one full disk pass per batch-sized
        # chunk, queue every chunk of the round through each layer while that
        # layer's weights are resident, so the weights are read once per pass.
        states = [
            prepare(chunk)
            for chunk in batchify(chat_prompts, max(1, self.settings.batch_size))
        ]

        # Each layer's resting device, so layers whose upload happened in the
        # prefetch are still evicted after use. Prefetch window: three layers
        # ahead. The next layer is fully loaded onto the compute device while
        # this layer's chunks drain (side-stream copy on GPU boxes); the two
        # behind it only get their bucket bytes read RAM-side, so their later
        # hop starts warm. On CPU-only boxes both stages are the same hop.
        resting = {id(layer): next(layer.parameters()).device for layer in layers}
        side = torch.cuda.Stream() if _cuda_usable() else None
        pending: torch.cuda.Event | None = None

        for index, layer in enumerate(tqdm.tqdm(layers, desc="Streaming layers", leave=False)):
            for state in states:
                # Mirrors Transformers' hidden-states collection: record the state entering
                # each layer, then finally the normed last-layer output. Only the activation
                # at the last prompt position is needed; cloning (not viewing) that slice
                # releases the full sequence right away. The slices stay on the compute
                # device until the sweep ends, which avoids one blocking device sync per
                # layer; their combined size is tiny next to the layer weights.
                state["rows"].append(state["hidden"][:, -1, :].detach().clone())

            if pending is not None:
                # Order the compute stream behind the side-stream upload enqueued in
                # the previous iteration without stalling the CPU here.
                torch.cuda.current_stream().wait_event(pending)
                pending = None

            items = [
                (state["hidden"], {**state["kwargs"], "attention_mask": state["causal_masks"][index]})
                for state in states
            ]
            hidden_states_list = self._call_layer(
                layer, items, resting[id(layer)], device
            )
            for state, hidden in zip(states, hidden_states_list):
                state["hidden"] = hidden

            ahead = layers[index + 1 : index + 4]
            if ahead:
                pending = self._prefetch_layer(ahead[0], resting, device, side)
                for lookahead in ahead[1:]:
                    self._warm_layer(lookahead)

        logits: list[Tensor] = []

        for state in states:
            # Both consumers (recorded rows, first-token logits) only use the last
            # position: slice before norm/lm_head so the lm_head intermediate is one
            # row per sequence instead of the full padded sequence (same rows out).
            hidden_states = state["hidden"][:, -1, :].detach().clone()

            if norm is not None:
                hidden_states = norm(hidden_states)

            state["hidden"] = hidden_states
            state["rows"].append(hidden_states)
            logits.append(lm_head(hidden_states))

        layer_states = [
            torch.cat([state["rows"][boundary] for state in states], dim=0)
            for boundary in range(len(states[0]["rows"]))
        ]

        return layer_states, torch.cat(logits, dim=0)

    def _prefetch_layer(
        self, layer: Module, resting: dict[int, Any], device: torch.device, side: torch.cuda.Stream | None
    ) -> torch.cuda.Event | None:
        # Upload one layer on a side stream while the GPU digests the current
        # one; the event orders the compute stream behind the copy. Hook-managed
        # layers load onto the compute device through their own pre_forward, so
        # call that early here instead: the weights stay resident and the
        # layer's own window skips the redundant reload (see _call_layer).
        # Peak stays ~two largest layers + activations. The copy is ordered
        # behind the current compute on the compute stream, so no event needed.
        hook = getattr(layer, "_hf_hook", None)
        if hook is not None:
            if next(layer.parameters()).device != device:
                return self._upload_layer(layer, hook, device, side)
            return None
        if side is None or resting[id(layer)] == device:
            return None
        event = torch.cuda.Event()
        with torch.cuda.stream(side):
            layer.to(device)
        event.record(side)
        return event

    def _warm_layer(self, layer: Module) -> None:
        # Two stages behind the full prefetch: read the layer's offloaded
        # weights through the hook's own loader so the later real load hits a
        # warm cache instead of the disk, then drop the values. Weights without
        # a hook already rest in RAM. Tensors are released per weight, so peak
        # stays one weight above the prefetch window.
        hook = getattr(layer, "_hf_hook", None)
        if hook is None or next(layer.parameters()).device != self.compute_device:
            return
        for name, _ in named_module_tensors(
            layer, include_buffers=hook.offload_buffers, recurse=hook.place_submodules, remove_non_persistent=True
        ):
            hook.weights_map[name]

    def _upload_layer(
        self, layer: Module, hook: Any, device: torch.device, side: torch.cuda.Stream | None
    ) -> torch.cuda.Event | None:
        # Mirror of the hook's own weight-load loop with pinned staging: pageable
        # host->device copies crawl on this box, so with CUDA each weight is copied
        # into a pinned buffer first and the hop is enqueued asynchronously on the
        # side stream. Same source values and dtype casts as hook.pre_forward, so
        # the numbers are bit-for-bit unchanged. The loop skips the hook's tied-
        # pointer dedup: shared tensors inside one layer get one identical copy per
        # name instead of one shared copy. shortcut: if a checkpoint ties huge
        # tensors within a layer, route dedup through hook.tied_params_map like
        # pre_forward does.
        hook.tied_pointers_to_remove = set()
        event = torch.cuda.Event() if side is not None else None
        with torch.cuda.stream(side) if side is not None else suppress():
            for name, _ in named_module_tensors(
                layer,
                include_buffers=hook.offload_buffers,
                recurse=hook.place_submodules,
                remove_non_persistent=True,
            ):
                value = hook.weights_map[name]
                if side is not None:
                    key = (tuple(value.shape), value.dtype)
                    staged = _STAGING.get(key)
                    if staged is None:
                        staged = torch.empty(key[0], dtype=key[1], pin_memory=True)
                        _STAGING[key] = staged
                    staged.copy_(value)
                    value = staged
                set_module_tensor_to_device(
                    layer,
                    name,
                    device,
                    value=value,
                    fp16_statistics=hook._maybe_get_fp16_statistics(name, value),
                    tied_params_map=hook.tied_params_map,
                    non_blocking=side is not None,
                    clear_cache=False,
                )
        if event is None:
            return None
        event.record(side)
        return event

    def _call_layer(
        self, layer: Module, items: list[tuple[Tensor, dict[str, Any]]], origin: torch.device, compute: torch.device
    ) -> list[Tensor]:
        # Load layer -> run every queued chunk through it -> unload: the layer's
        # weights stay resident for all chunks of the round, so each layer is
        # read once per streamed pass instead of once per chunk. With a single
        # device load this is a no-op. accelerate's disk-offload hooks load and
        # unload around forward(); drive the hook once around the whole chunk
        # queue and call the wrapped forward directly per chunk, so the hook's
        # own unload does not fire between chunks of the same round.
        hook = getattr(layer, "_hf_hook", None)
        moved = origin != compute and hook is None
        # Layers uploaded by the previous iteration's prefetch only need the
        # eviction in the finally block below.
        if moved and next(layer.parameters()).device != compute:
            layer.to(compute)
        forward = getattr(layer, "_old_forward", None) if hook is not None else None
        outputs: list[Tensor] = []
        try:
            if hook is not None and next(layer.parameters()).device != compute:
                # Already uploaded by the previous window's prefetch.
                hook.pre_forward(layer)

            for hidden_states, call_kwargs in items:
                # (older decoder layers have no `position_embeddings` parameter and compute
                # rotary embeddings themselves), so retry with a reduced set on TypeError.
                for reduced in (False, True):
                    kwargs = dict(call_kwargs)
                    if reduced:
                        kwargs.pop("position_embeddings", None)
                    try:
                        # No autograd graph: otherwise every layer's weight tensors stay
                        # referenced by the graph hanging off the hidden-state chain, and
                        # eviction after each layer frees nothing.
                        with torch.no_grad():
                            out = (forward if forward is not None else layer)(hidden_states, use_cache=False, **kwargs)
                    except TypeError:
                        continue
                    outputs.append(out[0] if isinstance(out, tuple) else out)
                    break
                else:
                    raise TypeError("Decoder layer rejected both full and reduced keyword sets.")

            return outputs
        finally:
            if hook is not None:
                hook.post_forward(layer, outputs[-1] if outputs else None)
            if moved:
                layer.to(origin)
                empty_cache()

    def _eager_forward_once(self, prompts: list[Prompt]) -> tuple[list[Tensor], Tensor]:
        _, outputs = self._eager_generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        return [hs[:, -1, :] for hs in hidden_states], logits

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat = self._apply_system_role(chat)
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.compute_device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
