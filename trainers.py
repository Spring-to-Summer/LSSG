from logging.config import valid_ident

import torch

from transformers import Trainer, AutoConfig

from utils import print_rank_0, IGNORE_INDEX
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import List,Tuple
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def compute_lm_loglikeli(logits, labels):
    batch_size, seq_length, vocab_size = logits.shape

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels).reshape(
        batch_size, -1
    )  # [bs * seq_len]
    ignore_mask = labels != IGNORE_INDEX

    avg_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1)

    return -avg_loss


class SFTWeightedWithKLTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")

        model_outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        with torch.no_grad():
            ref_model_outputs = model.ref_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            ref_logprob = compute_lm_loglikeli(
                ref_model_outputs.logits, inputs["labels"]
            )  # [batch_size]

        if self.args.debug_mode:
            print_rank_0(f"check ref_model output: {ref_logprob}")

        logprob = compute_lm_loglikeli(model_outputs.logits, inputs["labels"])

        # for MC kl
        kl_divergence = logprob.exp() * (logprob - ref_logprob)

        loss = -logprob + self.args.lm_kl_coeff * kl_divergence

        total_loss = (loss * inputs["weights"]).mean()  # [batch_size]

        # 新增 Entropy 正则化项
        logits = model_outputs.logits  # [batch, seq_len, vocab_size]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [batch, seq_len]
        avg_entropy = entropy.mean()

        # 加入 entropy 正则项
        total_loss = total_loss - self.args.entropy_coeff * avg_entropy

        if self.args.debug_mode:
            print_rank_0(f"check loss : {loss}")
            print_rank_0(f"check total_loss : {total_loss}")
            print_rank_0(f"check kl divergence : {kl_divergence}")
            print_rank_0(f"check entropy : {avg_entropy}")

        return (total_loss, model_outputs) if return_outputs else total_loss

class DummySentenceEmbedder:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)

    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

class OfflineWeightedPolicyTrainer(Trainer):
    def __init__(self,sentiment_classifier, semantic_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_model=semantic_model
        self.sentiment_classifier=sentiment_classifier
        if isinstance(self.model, dict):
            self.model = self.model.get("model", self.model)

    def unwrap_and_sync_model(self, model):
        print(f"[Before unwrap] model = {type(model)}, has module = {hasattr(model, 'module')}")
        if hasattr(model, "module"):
            base_model = model.module
        else:
            base_model = model

        base_model.zero_pad_model_inputs = True

        print("⏳ [DeepSpeed] Start manual parameter prefetch (recursive)")
        num_params = 0
        failed = 0

        try:
            for name, module in base_model.named_modules():  # recurse=True
                for param in module.parameters(recurse=False):
                    if hasattr(param, "ds_status"):
                        try:
                            # 强制触发 param.data 加载
                            _ = param.data
                            num_params += 1
                            if hasattr(param, "param_coordinator") and param.param_coordinator is not None:
                                param.param_coordinator.fetch_sub_module(module, forward=True)
                        except Exception as e:
                            failed += 1
                            print(f"[Prefetch Failed] param in module '{name}' -> {e}")
        except Exception as e:
            print(f"[Fatal Error] Manual prefetch failed: {e}")

        print(f"✅ [DeepSpeed] Manual parameter prefetch complete (recursive). Total: {num_params}, Failed: {failed}")
        return base_model

    def encode_texts(self, texts: List[str], fallback_shape: Tuple[int, int], device):
        try:
            cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
            if not cleaned:
                return torch.zeros(fallback_shape, device=device)
            return self.semantic_model.encode(cleaned, convert_to_tensor=True, truncation=True, max_length=512).to(device)
        except Exception as e:
            print_rank_0(f"[Warning] Failed to encode texts: {e}")
            return torch.zeros(fallback_shape, device=device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 💥 确保模型不是 dict
        if isinstance(model, dict):
            print_rank_0("[Fix] 'model' is a dict, extracting actual model via model['model']")
            model = model.get("model", model)
        print_rank_0(f"[Debug] type(model) = {type(model)}")
        print_rank_0(f"[Debug] hasattr(model, 'config') = {hasattr(model, 'config')}")

        if self.args.debug_mode:
            print_rank_0(f"check inputs :{inputs}")
        model_outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        with torch.no_grad():
            ref_model_outputs = model.ref_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )

            ref_logprob = compute_lm_loglikeli(
                ref_model_outputs.logits, inputs["labels"]
            ).detach()  # [batch_size]

        if self.args.debug_mode:
            print_rank_0(f"check ref_model output: {ref_logprob}")

        logprob = compute_lm_loglikeli(model_outputs.logits, inputs["labels"])
        kl_div = logprob - ref_logprob

        print("===============================================================================")
        print("input_ids.shape:", inputs["input_ids"].shape)
        print("attention_mask.shape:", inputs["attention_mask"].shape)

        print(f"[Before unwrap] model = {type(model)}, has module = {hasattr(model, 'module')}")

        # 使用新方法统一处理
        model = self.unwrap_and_sync_model(model)
        max_input_len = model.config.max_position_embeddings - 64  # 放在解包之后再访问！
        #  确保 generate 前所有参数已拉入内存
        with torch.no_grad():
            _ = model(input_ids=inputs["input_ids"][:, -max_input_len:],
                      attention_mask=inputs["attention_mask"][:, -max_input_len:])
            torch.cuda.synchronize()

        was_training = model.training
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        # 这里生成 generated_texts 文本并 decode 出来
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"][:, -max_input_len:],
                attention_mask = inputs["attention_mask"][:, -max_input_len:],
                # input_ids=inputs["input_ids"],
                # attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2,
                temperature=0.95
            )
        if was_training:
            model.train()
        model.gradient_checkpointing_enable()
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("generated_texts =", generated_texts)


        for i, text in enumerate(generated_texts):
            tokenized = self.semantic_model.tokenizer.encode(text, truncation=True)
            print(f"[{i}] len={len(tokenized)} text={repr(text[:100])}")

        importance_ratio = (logprob - ref_logprob).exp()
        importance_ratio_clipped = torch.clip(
            importance_ratio, 1 - self.args.clip_range, 1 + self.args.clip_range
        )

        advantages = inputs["rewards"] - self.args.lm_kl_coeff * kl_div
        ppo_loss = -torch.minimum(
            advantages * importance_ratio, advantages * importance_ratio_clipped
        )

        sample_size, sft_size = (1 - inputs["sft_mask"]).sum(), (
            inputs["sft_mask"]
        ).sum()
        sft_loss = (
            (-logprob * inputs["sft_mask"]).sum() / sft_size
            if sft_size > 0
            else sft_size
        )
        ppo_loss = (
            (ppo_loss * (1 - inputs["sft_mask"])).sum() / sample_size
            if sample_size > 0
            else sample_size
        )

        #  自定义文本：还需要 inputs["ref_texts"] → 用 label decode 得到
        labels = inputs["labels"].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id or 0
        reference_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 清洗 valid_texts
        valid_texts = [t.strip() for t in generated_texts if isinstance(t, str) and t.strip()]
        dim = self.semantic_model.get_sentence_embedding_dimension()
        min_len = min(len(valid_texts), len(reference_texts))

        if min_len == 0:
            print_rank_0("[Warning] No valid texts for semantic comparison.")
            return torch.tensor(0.0, requires_grad=True, device=logprob.device)  # 或直接 return sft_loss + ppo_loss

        # 计算向量
        gen_emb = self.encode_texts(valid_texts[:min_len], (min_len, dim), logprob.device)
        ref_emb = self.encode_texts(reference_texts[:min_len], (min_len, dim), logprob.device)

        print(f"gen_emb.shape = {gen_emb.shape}")
        print(f"ref_emb.shape = {ref_emb.shape}")

        # 计算语义相似度（鼓励与 ref 不太接近，但与 input 保持语义相关性）
        cos_sim = util.pytorch_cos_sim(gen_emb, ref_emb).diagonal()  # 越大越相似
        semantic_diversity_reward = 1 - cos_sim  # 越大越不同
        semantic_penalty = - self.args.semantic_coeff * semantic_diversity_reward.to(logprob.device)

        # 情绪鲁棒性：情绪平衡损失项
        sentiments = self.sentiment_classifier(generated_texts, truncation=True, max_length=512)

        # 得到每句话的正面概率，例如：{'label': 'POSITIVE', 'score': 0.98}
        emotion_stability = torch.tensor(
            [s["score"] if s["label"] == "POSITIVE" else 1 - s["score"] for s in sentiments], device=logprob.device)
        # 假设目标情绪偏中性，惩罚偏离 0.5 的程度：
        # emotion_penalty = (emotion_stability - 0.5).abs() # smooth版本
        emotion_penalty = torch.square(emotion_stability - 0.5) # strong 版本
        emotion_penalty = self.args.emotion_coeff * emotion_penalty.to(logprob.device)

        # 加权加入 PPO 损失，对齐长度
        ppo_loss = ppo_loss.view(-1) if len(ppo_loss.shape) == 0 else ppo_loss
        semantic_penalty = semantic_penalty.view(-1) if len(semantic_penalty.shape) == 0 else semantic_penalty
        emotion_penalty = emotion_penalty.view(-1) if len(emotion_penalty.shape) == 0 else emotion_penalty
        min_len = min(ppo_loss.shape[0], semantic_penalty.shape[0], emotion_penalty.shape[0])

        ppo_loss = ppo_loss[:min_len]
        semantic_penalty = semantic_penalty[:min_len]
        emotion_penalty = emotion_penalty[:min_len]

        # total_loss 本应是 batch 中每个样本的 loss，sft_loss 是标量，不需要切
        total_loss = self.args.lm_sft_coeff * sft_loss + ppo_loss + semantic_penalty + emotion_penalty

        # weighted_loss = (total_loss * inputs["weights"]).mean()  # [batch_size]
        weighted_loss = (total_loss * inputs["weights"][:min_len]).mean()

        if self.args.debug_mode:
            print_rank_0(f"check total_loss : {total_loss}")
            print_rank_0(f"check weighted loss : {weighted_loss}")
            print_rank_0(f"check kl divergence : {kl_div}")

        return (weighted_loss, model_outputs) if return_outputs else weighted_loss
