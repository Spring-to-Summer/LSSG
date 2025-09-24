import json
from tqdm import tqdm
import asyncio

import transformers

from arguments import CustomTrainingArguments

from utils import convert_game_history_to_query, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN
from tools.serve_entry import get_entry

TOTAL_EMPTY = 0

# tokenizer = AutoTokenizer.from_pretrained(
#     "./ckpt",
#     padding_side="left",  # for batch decode
#     truncation_side="left",
#     model_max_length=2048,
#     trust_remote_code=True,
# )

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = 0


def load_keyword_list(args, data_path):
    with open(data_path, "r") as f:
        keywords = f.read().strip().split("\n")
    return keywords


def get_player(args, model_name_or_path):
    return get_entry(model_name_or_path, gpus_per_worker=1, verbose=False)


async def play_games(args, players, words, **gen_kwargs):
    batch_games = [
        {"history": [], "target_word": keyword, "max_turns": args.taboo_max_turns}
        for keyword in words
    ]
    all_outputs = []

    for taboo_turn in range(2 * args.taboo_max_turns):
        batch_size = len(batch_games)
        next_player = "attacker" if taboo_turn % 2 == 0 else "defender"
        model = players[next_player]

        batch_queries = [
            {
                "query": convert_game_history_to_query(
                    game["history"],
                    target_word=game["target_word"],
                    max_turns=game["max_turns"],
                ),
                "query_id": game["target_word"],
            }
            for game in batch_games
        ]

        batch_text = [item["query"] for item in batch_queries]
        # print("batch_text[0]", batch_text[0])
        # print(
        #     "tokenized batch_text[0]",
        #     tokenizer(batch_text[0], add_special_tokens=False),
        # )

        output_seq = await model.generate(batch_text, **gen_kwargs)

        finished_ids = []
        for idx in range(batch_size):
            response_sample = output_seq[idx]
            batch_games[idx]["history"].append(
                {"role": next_player, "content": response_sample.strip()}
            )

            if (
                "i know the word" in response_sample.lower()
                and next_player == "defender"
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
    return all_outputs


async def generate_game_trajectory(args, **gen_kwargs):
    # check if there's args.max_sample
    if args.max_samples is None:
        eval_dataset = load_keyword_list(args, args.data_path)[:128]
    else:
        eval_dataset = load_keyword_list(args, args.data_path)[: args.max_samples]

    # setup model
    # ---------------------------------------------------------------------------------
    players = dict()
    players["attacker"] = get_player(args, args.attacker_model_name_or_path)

    if args.attacker_model_name_or_path == args.defender_model_name_or_path:
        players["defender"] = players["attacker"]
    else:
        players["defender"] = get_player(args, args.defender_model_name_or_path)

    all_outputs = []

    for i in tqdm(range(0, len(eval_dataset), args.batch_size)):
        batch_words = eval_dataset[i : i + args.batch_size]
        outputs = await play_games(args, players, batch_words, **gen_kwargs)
        all_outputs.extend(outputs)

    output_path = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}.json"

    # sort according to eval_dataset order
    dic = {item["target_word"]: item for item in all_outputs}
    all_outputs = [dic[word] for word in eval_dataset]

    json.dump(all_outputs, open(output_path, "w"), indent=4)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    gen_kwargs = {
        "max_tokens": args.max_new_tokens,
        "temperature": 1.2,
        "top_p": 1.0,
        "extra_body": {
            "top_k": 50,
            "stop_token_ids": [2],
            "min_tokens": 2,
        },
    }
    asyncio.run(generate_game_trajectory(args, **gen_kwargs))
    print("TOTAL_EMPTY", TOTAL_EMPTY)
