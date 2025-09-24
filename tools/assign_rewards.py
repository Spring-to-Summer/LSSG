import re
import json
import argparse
from tqdm import tqdm
from textblob import TextBlob
from copy import deepcopy

from utils import randomly_convert_game_history_to_query
import ast
# import nltk
# nltk.download('all')

# PREDICT_TEMP = r"i know the word! it.{1,8}"
PREDICT_TEMP = """{"action": "accept"}"""

def get_derivative_words(word: str):
    # fuzzy matching for similar words
    word = word.lower()
    blob_word = TextBlob(word)
    word_list = [word, word + 'ing', word + 'ed', blob_word.words.pluralize()[0]]
    quotation_list = ["\"{word}\"", "'{word}'", '`{word}`']
    word_list += [quotation.format(word=word) for quotation in quotation_list for word in word_list]
    
    return word_list


def has_target_word(content: str, target_word: str):
    derivative_words = get_derivative_words(target_word)
    return any([word in content.lower() for word in derivative_words])

def is_prediction(content: str, target_word: str):
    if re.search(PREDICT_TEMP, content.lower()):
        return True
    else:
        return False

def has_accpeted(content: str):
    if re.search(PREDICT_TEMP, content.lower()):
        return True
    else:
        return False

def is_correct_prediction(content: str, target_word: str):
    derivative_words = get_derivative_words(target_word)
    predict_regex = [PREDICT_TEMP + word for word in derivative_words]

    if any([re.search(temp, content.lower()) for temp in predict_regex]):
        return True
    else:
        return False


def get_game_outcome(history):
    history_length = 0
    for i, item in enumerate(history):
        history_length += 1
        content = item["content"].lower()
        if has_accpeted(content):
            return "all win", history_length
        else:
            return "all lose", history_length



def extract_price_from_history(history):
    """
    提取最后成交价格：
    - 优先使用结构化 action（'action': 'offer', 'price': ...）
    - 其次匹配自然语言中的 $xxx
    - 向前回溯最近一次 offer 动作作为成交价格
    """
    offer_price = None

    # 倒序查找是否有 'accept' 出现
    for i in range(len(history) - 1, -1, -1):
        msg = history[i]
        content = msg.get("content", "")

        # 如果当前内容是一个 dict string，尝试解析
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, dict):
                # 找到接受成交
                if parsed.get("action") == "accept":
                    # 向前找离它最近的 offer
                    for j in range(i - 1, -1, -1):
                        prev_msg = history[j]
                        prev_content = prev_msg.get("content", "")
                        try:
                            prev_parsed = ast.literal_eval(prev_content)
                            if isinstance(prev_parsed, dict) and prev_parsed.get("action") == "offer":
                                return float(prev_parsed.get("price", 0))
                        except:
                            continue
        except:
            pass

    # 如果没有结构化 offer+accept，尝试正则匹配
    price_pattern = re.compile(r"\$?(\d+[,.]?\d*)")
    for msg in reversed(history):
        if msg["role"] in ["seller", "buyer"]:
            matches = price_pattern.findall(msg.get("content", ""))
            if matches:
                try:
                    return float(matches[0].replace(",", ""))
                except:
                    continue
    return None


def evaluate_price_reward(price_orig, price_final, history, role):
    if price_final is None or price_orig == 0:
        return -10

    discount = (price_orig - price_final) / price_orig
    ratio = price_final / price_orig
    gift_keywords = [
        "gift", "free", "bonus", "included", "throw in",
        "complimentary", "extra item", "add-on", "perks", "bundle",
        "include for free", "no charge", "limited warranty"
    ]
    has_gift = any(any(gk in msg.get("content", "").lower() for gk in gift_keywords) for msg in history)

    if role == "buyer":
        if discount > 0.2:
            return 20
        elif 0.1 < discount <= 0.2:
            return 15
        elif discount <= 0.1:
            return 10 if has_gift else 5
        else:
            return 0

    elif role == "seller":
        if ratio >= 0.95:
            return 20
        elif 0.8 <= ratio < 0.95:
            return 15 if has_gift else 10
        elif 0.7 <= ratio < 0.8:
            return 10 if has_gift else 5
        elif ratio < 0.7:
            return 10 if has_gift else 5
        else:
            return 0
    return 0


def compute_self_play_sample_rewards(game_episodes, input_data_path=""):
    outputs = []
    judged_games = []
    buyer_game_num, seller_game_num = 0, 0

    for item in tqdm(game_episodes):
        outcome, history_length = get_game_outcome(item["history"])
        new_item = deepcopy(item)
        new_item["outcome"] = outcome
        judged_games.append(new_item)

        if outcome != "all win":
            continue

        original_price = item.get("price", 0)
        deal_price = extract_price_from_history(item["history"])

        for i in range(history_length):
            message = item["history"][i]
            query = randomly_convert_game_history_to_query(
                item["history"][:i],
                item=item["item"],
                price=original_price,
                max_turns=item["max_turns"]
            )
            target = message["content"]

            reward = evaluate_price_reward(original_price, deal_price, item["history"], message["role"])
            reward = max(min(reward, 20), -10) / 10.0  # clip + scale to [-1, 2]

            outputs.append({
                "query": query,
                "target": target.strip(),
                "reward": reward,
                "role": message["role"]
            })

            if message["role"] == "buyer":
                buyer_game_num += 1
            elif message["role"] == "seller":
                seller_game_num += 1

    json.dump(
        judged_games,
        open(input_data_path.replace(".json", "_judged.json"), "w"),
        ensure_ascii=False,
        indent=4,
    )

    all_game_num = buyer_game_num + seller_game_num
    buyer_weight = all_game_num / (2 * buyer_game_num) if buyer_game_num else 0.0
    seller_weight = all_game_num / (2 * seller_game_num) if seller_game_num else 0.0

    print(f"Processed {all_game_num} turns, buyer_weight={buyer_weight}, seller_weight={seller_weight}")

    for item in outputs:
        item["weight"] = buyer_weight if item["role"] == "buyer" else seller_weight

    return outputs


def extract_price_from_history(history):
    price_pattern = re.compile(r"\$?(\d+[,.]?\d*)")
    for msg in reversed(history):
        if msg["role"] == "seller" or msg["role"] == "buyer":
            matches = price_pattern.findall(msg["content"])
            if matches:
                return float(matches[0].replace(",", ""))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for episode processing.")
    parser.add_argument(
        "--input_data_path", type=str, default="", help="the path to input data."
    )
    parser.add_argument(
        "--output_data_path", type=str, default="", help="the path to output data."
    )
    parser.add_argument(
        "--sft_data_path", type=str, default="", help="the path to load sft data."
    )
    parser.add_argument(
        "--decay_weight", type=float, default=0.8, help="the decay weight of reward."
    )

    args = parser.parse_args()

    with open(args.input_data_path, "r") as f:
        game_episodes = json.load(f)

    results = compute_self_play_sample_rewards(
        game_episodes, args.input_data_path
    )

    if args.sft_data_path:
        with open(args.sft_data_path, "r") as f:
            sft_data = json.load(f)
    else:
        sft_data = []

    for item in sft_data:
        item["type"] = "sft"
        item["weight"] = 1.0

    with open(args.output_data_path, "w") as f:
        json.dump(results + sft_data, f, ensure_ascii=False, indent=2)
