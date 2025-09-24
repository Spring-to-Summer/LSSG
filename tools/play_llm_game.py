import json
import glob
from tqdm import tqdm


import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from arguments import CustomTrainingArguments

from utils import print_rank_0, read_json_or_jsonl_data
from utils import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from utils import convert_game_history_to_query, set_special_tokens

from dataloaders import batch_padding

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import os
from transformers import BitsAndBytesConfig

TOTAL_EMPTY = 0

class KeywordDataset(Dataset):
    def __init__(self, keyword_list):
        self.data = keyword_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_keyword_list(args, data_path):
    # 用于保存所有 title 的列表
    all_titles = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            data = json.loads(line)
            title = data.get("name")
            price_str = data.get("price", "$0")
            price = float(price_str.replace("$", "").replace(",", "").strip())
            item = {}
            item["name"] = title
            item["price"] = price
            if title:
                all_titles.append(item)
    return all_titles


def query_data_collactor(args, batch, tokenizer):
    input_ids, attention_mask, labels = [], [], []
    text = [item["query"] for item in batch]
    query_ids = [item["query_id"] for item in batch]

    for sent in text:
        input_query_ids = [tokenizer.bos_token_id] + tokenizer.encode(
            sent, add_special_tokens=False
        )
        input_ids.append(input_query_ids)

    outputs = batch_padding(
        input_ids,
        tokenizer,
        max_length=tokenizer.model_max_length - args.max_new_tokens,
    )
    outputs["query_ids"] = query_ids
    outputs["text"] = text
    return outputs


def load_model_and_tokenizer(args, model_name_or_path):
    print_rank_0(f"start loading model from {model_name_or_path}")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 推荐 4-bit（更省显存）；想稳一点也可换成 load_in_8bit=True
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # 若 bfloat16 不支持就改成 torch.float16
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_cache=True,
        quantization_config=bnb_config,  # <- 使用 4bit
        device_map={"": local_rank},     # <- 直接放到当前 local_rank 对应的 GPU
    )
    if hasattr(model, "ref_model"):
        del model.ref_model

    print_rank_0(model)

    # ❌ 不要再 model.to(device) 了（有 device_map 时会冲突/多搬一遍）
    # device = torch.cuda.current_device()
    # model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        truncation_side="left",
        model_max_length=args.max_length,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        print_rank_0("Warning: pad token is None, set default value to 0")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 0

    model, tokenizer = set_special_tokens(model, tokenizer)
    return {"model": model, "tokenizer": tokenizer}


def main():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    eval_dataset = load_keyword_list(args, args.data_path)[:1000]

    # setup model
    # ---------------------------------------------------------------------------------
    players = dict()
    players["seller"] = load_model_and_tokenizer(
        args, args.attacker_model_name_or_path
    )

    if args.attacker_model_name_or_path == args.defender_model_name_or_path:
        players["buyer"] = players["seller"]
    else:
        players["buyer"] = load_model_and_tokenizer(
            args, args.defender_model_name_or_path
        )
    torch_dataset = KeywordDataset(eval_dataset)
    #构造分布式 DataLoader
    sampler = torch.utils.data.distributed.DistributedSampler(torch_dataset, shuffle=True)
    # 保持 batch 是 list[dict]
    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=sampler,
        collate_fn=lambda x: x  # 返回原始 list，不要默认拼装成 tensor
    )



    # sampler = torch.utils.data.distributed.DistributedSampler(
    #     eval_dataset, shuffle=True
    # )
    # dataloader = DataLoader(
    #     eval_dataset,
    #     shuffle=False,
    #     batch_size=args.per_device_eval_batch_size,
    #     sampler=sampler,
    # )

    all_outputs = []
    progress_bar = tqdm(range(len(dataloader)), disable=(dist.get_rank() != 0))
    for step, batch_words in enumerate(dataloader):
        progress_bar.update(1)

        batch_games = [
            {"history": [], "item": keyword["name"], "price": keyword["price"], "max_turns": args.taboo_max_turns}
            for keyword in batch_words
        ]

        for taboo_turn in range(2 * args.taboo_max_turns):
            next_player = "seller" if taboo_turn % 2 == 0 else "buyer"
            model, tokenizer = (
                players[next_player]["model"],
                players[next_player]["tokenizer"],
            )

            if args.task_type == "testing":
                generation_config = GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1,
                )
            elif args.task_type == "sampling":
                generation_config = GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    temperature=1.2,  # default=0.8
                    do_sample=True,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1,
                )

            batch_queries = [
                {
                    "query": convert_game_history_to_query(
                        game["history"],
                        item=game["item"],
                        price=game["price"],
                        max_turns=game["max_turns"],
                    ),
                    "query_id": game["item"],
                }
                for game in batch_games
            ]

            batch = query_data_collactor(args, batch_queries, tokenizer)

            input_ids = torch.Tensor(batch["input_ids"]).long().to(model.device)
            attention_mask = (
                torch.Tensor(batch["attention_mask"]).float().to(model.device)
            )
            query_ids = batch["query_ids"]
            text = batch["text"]
            batch_size = input_ids.shape[0]

            # print_rank_0("text")
            # print_rank_0(text[0])
            # print_rank_0(input_ids[0])
            # print_rank_0("decode input_ids")
            # print_rank_0(tokenizer.decode(input_ids[0], add_special_tokens=False))

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )

            output_seq = generation_output.sequences.reshape(
                batch_size, generation_config.num_return_sequences, -1
            )
            inputs_string = tokenizer.batch_decode(
                input_ids.reshape(batch_size, -1), skip_special_tokens=True
            )

            finished_ids = []
            for idx in range(batch_size):
                output_response = tokenizer.batch_decode(
                    output_seq[idx], skip_special_tokens=True
                )[
                    0
                ]  # only consider one sample
                response_sample = output_response.replace(inputs_string[idx], "").split(
                    tokenizer.eos_token
                )[0]
                batch_games[idx]["history"].append(
                    {"role": next_player, "content": response_sample.strip()}
                )

                if (
                    "{'action': 'accept'}" in response_sample.lower()
                ) or (
                    "{'action': 'quit'}" in response_sample.lower()
                ):
                    # early stop to speed up inference
                    all_outputs.append(batch_games[idx])
                    finished_ids.append(idx)
                if response_sample == "":
                    print(f"Empty response for {batch_queries[idx]}")
                    global TOTAL_EMPTY
                    TOTAL_EMPTY += 1
                    all_outputs.append(batch_games[idx])
                    finished_ids.append(idx)

            batch_games = [
                game for idx, game in enumerate(batch_games) if idx not in finished_ids
            ]
            if len(batch_games) == 0:
                break

        all_outputs.extend(batch_games)
        if dist.get_rank() == 0 and (step % args.logging_steps == 0):
            print_rank_0(f"finished {step} of {len(dataloader)}")
            # print_rank_0(all_outputs[-1])

    output_file_prefix = (
        f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}"
    )
    with open(f"{output_file_prefix}_rank{dist.get_rank()}.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"rank {dist.get_rank()} finishs inference.")

    if "model" in players["seller"]:
        del players["seller"]["model"]
    if "model" in players["buyer"]:
        del players["buyer"]["model"]

    torch.cuda.empty_cache()
    dist.barrier()
    if dist.get_rank() == 0:
        result_paths = glob.glob(f"{output_file_prefix}_rank*.json")
        all_results = []
        for res_path in result_paths:
            new_results = read_json_or_jsonl_data(res_path)
            all_results.extend(new_results)

        print(f"totally loaded {len(all_results)} results")
        with open(f"{output_file_prefix}_results.json", "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"finished inference results merge.")


if __name__ == "__main__":
    main()
    print(f"TOTAL_EMPTY: {TOTAL_EMPTY}")
