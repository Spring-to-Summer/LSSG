import json
import glob
import random

import torch


IGNORE_INDEX = -100

SEP_TOKEN = "<sep>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

GAME_RULE_PROMPTS = [
    """Play the game of Scorable Negotiation Game. In this game, there are two players, an seller and a buyer.

At the beginning of the game, the buyer learns about the item information provided by the seller, including the item's name, category, starting price, and description. 

The buyer's goal is to purchase the item at a price lower than the starting price or negotiate for additional goods or services to be included. The key is to reach an agreement with the seller because failing to do so will result in a penalty.

The seller's goal is to sell the item at a price not too far below the starting price. This can be achieved by offering small additional items or services. Like the buyer, the seller must also focus on reaching an agreement because failing to negotiate successfully will result in a penalty. 

When either party responds with {{"action": "accept"}} to the other party's offer, it indicates that both parties have reached an agreement and jointly win the game. The game lasts for a maximum of {max_turns} turns. If the two players fail to make a deal within {max_turns} turns, they both lose the game.
""",
    """Welcome to the Scorable Negotiation Game! This game involves two players: a buyer and a seller. 
    
At the start, the seller provides item details, including its name, category, starting price, and description, which the buyer reviews. 
    
The buyer’s aim is to secure the item for less than the starting price or negotiate additional benefits. Reaching an agreement is crucial, as failure to do so leads to penalties. 
    
The seller, on the other hand, tries to sell the item close to its original price, possibly by offering minor incentives. Both players must agree, or penalties will apply. 
    
When one side says {{"action": "accept"}}, the game concludes with both players winning. The game can continue for up to {max_turns} turns, but if no deal is made, both players lose.
    """
    ,
    """
    Let’s dive into the Scorable Negotiation Game, featuring a buyer and a seller. 
    
    The game begins with the seller sharing key details about the item—its name, type, starting price, and description. 
    
    The buyer’s task is to negotiate the price below the starting point or secure extras. 
    
    For the seller, the challenge is to sell the item for a price close to its original value, potentially offering additional services or items to close the deal. 
    
    Both parties need to reach an agreement; otherwise, they face penalties. If either player accepts an offer using {{"action": "accept"}}, both win. The game has a limit of {max_turns} turns, and if no agreement is reached, both lose.
    """
    ,
    """
    Step into the intriguing game known as Scorable Negotiation Game. Scorable Negotiation Game involves two roles: buyer and seller. 
    
    Initially, the buyer learns about the item being sold, including its name, category, price, and description. 
    
    The buyer’s goal is to pay less than the starting price or obtain additional goods or services. 
    
    The seller, meanwhile, seeks to sell the item close to the starting price, using small bonuses to secure an agreement if needed. Both players must negotiate successfully to avoid penalties. 
    
    If one responds with {{"action": "accept"}}, they both win. The game runs for up to {max_turns} turns, after which failure to agree results in a loss for both.
    """
    ,
    """
    Embark on the strategic challenge of Scorable Negotiation Game. 
    
    In this Scorable Negotiation Game, two participants—a buyer and a seller—engage in a simulated negotiation. 
    
    The buyer begins with item information provided by the seller, including the name, type, price, and description. The buyer’s aim is to get a better deal by either lowering the price or obtaining extra items or services. 
    
    The seller aims to close the deal near the starting price, potentially adding minor incentives to reach an agreement. 
    
    If both sides fail to agree, they face penalties. Agreement is signaled by {{"action": "accept"}} and results in a win for both. The game is limited to {max_turns} turns, with failure to negotiate resulting in a shared loss.
    """
    ,
    """
    The Scorable Negotiation Game sets up two players: a buyer and a seller. 
    
    The seller presents the buyer with details about the item, such as its name, type, price, and description. 
    
    The buyer tries to purchase it for less than the starting price or obtain additional benefits. 
    
    The seller aims to sell it at a price near the original value, possibly adding small extras. 
    
    Both must reach an agreement to avoid penalties. A successful negotiation ends when either party uses {{"action": "accept"}}, resulting in a win for both. The game lasts for {max_turns} turns, after which failure to agree means both lose.
    """
    ,
    """
    Immerse yourself in the strategic face-off called Scorable Negotiation Game. In the Scorable Negotiation Game, two players—a buyer and a seller—collaborate to make a deal. 
    
    The seller shares details of the item, such as its name, type, initial price, and description, for the buyer to review. 
    
    The buyer’s objective is to negotiate a lower price or obtain extras, while the seller aims to close the deal close to the starting price, offering incentives if necessary. Both must agree to avoid penalties. An agreement, indicated by {{"action": "accept"}}, signifies a shared win. Players have up to {max_turns} turns to agree; otherwise, both lose the game.
    """
    ,
    """
    Step into the challenge of Scorable Negotiation Game. This Scorable Negotiation Game involves two participants: a buyer and a seller. 
    
    At the start, the seller provides the buyer with key item details, such as its name, category, starting price, and description. 
    
    The buyer’s goal is to negotiate a lower price or request additional benefits, while the seller tries to maintain a price close to the initial value. 
    
    Both must agree to avoid penalties. A deal is finalized when one party responds {{"action": "accept"}}. The game lasts up to {max_turns} turns, and if no agreement is reached, both players lose.
    """
]


INSTRUCT_PROMPTS = {
    "seller": """\n\n### Instruction: You are the seller. The name of the item is `{item}`. The price of the item is `{price}`. Provide your response for the next turn.\n\n### Response:""",
    "buyer": """\n\n### Instruction: Your are the buyer. The name of the item is `{item}`. The price of the item is `{price}`. Provide your response for the next turn.\n\n### Response:""",
}

PLAYER_INSTRUCT_PROMPTS = {
    "seller": "You are the seller. The name of the item is `{item}`. The price of the item is `{price}`. Provide your response for the next turn.",
    "buyer": "Your are the buyer. The name of the item is `{item}`. The price of the item is `{price}`. Provide your response for the next turn.",
}


def convert_game_history_to_query(history, item, price, max_turns=5):
    GAME_RULE_PROMPT = GAME_RULE_PROMPTS[0]
    history_str = ""
    for i, message in enumerate(history):
        history_str += "\n  - {}: {}".format(message["role"], message["content"])

    if len(history) == 0:
        query = (
            GAME_RULE_PROMPT.format(max_turns=max_turns)
            + "The game is just initialized."
        )
        next_player = "seller"

    else:
        query = (
            GAME_RULE_PROMPT.format(max_turns=max_turns)
            + "\n### Game History:"
            + history_str
        )
        if history[-1]["role"] == "seller":
            next_player = "buyer"
        else:
            next_player = "seller"

    query += INSTRUCT_PROMPTS[next_player].format(item=item, price=price)
    return query


def randomly_convert_game_history_to_query(history, item, price, max_turns=5):
    if len(history) == 0:
        next_player = "seller"
    else:
        if history[-1]["role"] == "seller":
            next_player = "buyer"
        else:
            next_player = "seller"

    dialog_prefix = "\n" + random.choice(
        ["\n - ", "\n### ", "\n## ", "\n# ", "\n *** ", "\n **", "\n\n"]
    )
    answer_str, question_str = random.choice(
        [
            (next_player, "buyer" if next_player == "seller" else "seller"),
            ("Assistant", "Human"),
            ("Answer", "Question"),
            ("Response", "Query"),
            ("A", "Q"),
        ]
    )

    player_prefix = {
        "seller": answer_str if next_player == "seller" else question_str,
        "buyer": answer_str if next_player == "buyer" else question_str,
    }

    history_str = ""
    for i, message in enumerate(history):
        history_str += "{}{}: {}".format(
            dialog_prefix, player_prefix[message["role"]], message["content"]
        )

    prompt_type = random.choice(["chat", "chat_inverse", "alpaca"])
    system_prefix = random.choice(["Rules", "Game Rule", "System"])

    GAME_RULE_PROMPT = random.choice(GAME_RULE_PROMPTS)
    system_prompt = GAME_RULE_PROMPT.format(max_turns=max_turns)

    if "chat" in prompt_type:
        system_prompt += "\n\n" + PLAYER_INSTRUCT_PROMPTS[next_player].format(
            item=item,
            price=price,
        )

        if len(history) == 0:
            history_str = ""
            system_prompt += "The game is just initialized. "

        system_str = f"{dialog_prefix}{system_prefix}: {system_prompt}"
        if "inverse" in prompt_type:
            query = (
                history_str
                + system_str
                + dialog_prefix
                + player_prefix[next_player]
                + ": "
            )
        else:
            query = (
                system_str
                + history_str
                + dialog_prefix
                + player_prefix[next_player]
                + ": "
            )

    elif prompt_type == "alpaca":
        if random.uniform(0, 1) < 0.2:
            system_prompt = system_prefix + ": " + system_prompt

        if len(history) == 0:
            query = system_prompt + "The game is just initialized. "
        else:
            query = (
                system_prompt + dialog_prefix + "Game History:" + history_str + "\n\n"
            )

        if random.uniform(0, 1) < 0.2:
            query += (
                PLAYER_INSTRUCT_PROMPTS[next_player].format(item=item, price=price)[
                    :-1
                ]
                + ": "
            )
        else:
            query += (
                PLAYER_INSTRUCT_PROMPTS[next_player].format(item=item, price=price)
                + dialog_prefix
                + player_prefix[next_player]
                + ": "
            )

    return query


def set_special_tokens(model, tokenizer):
    
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        print_rank_0(f"====================================================")
        print_rank_0(f"WARNING: the pad token of the tokenizer is None")
        # We do not resize the vocab embedding, since it ruins the KL value with the ref_model
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = tokenizer.decode(0)
        print_rank_0(f"set pad token to {tokenizer.pad_token}")
        print_rank_0(f"set pad token id to {tokenizer.pad_token_id}")
        print_rank_0(f"====================================================")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print_rank_0(tokenizer)
    return model, tokenizer



def read_json_or_jsonl_data(data_path):
    if data_path[-5:] == ".json":
        with open(data_path, "r") as f:
            data_list = json.load(f)
    else:
        with open(data_path, "r") as f:
            lines = f.read().strip().split("\n")
            data_list = [json.loads(l) for l in lines]

    print_rank_0(f">>> totally load {len(data_list)} data from {data_path}")
    return data_list


def merge_json_or_jsonl_data(data_path_pattern):
    file_names = glob.glob(data_path_pattern)
    print_rank_0(f"load {len(file_names)} files from {data_path_pattern}.")
    outputs = []
    for file_name in file_names:
        new_data = read_json_or_jsonl_data(file_name)
        if isinstance(new_data, list):
            outputs.extend(new_data)
        elif isinstance(new_data, dict):
            outputs.append(new_data)
    return outputs


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
