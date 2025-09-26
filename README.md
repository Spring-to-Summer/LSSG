## Implementation for Language Model Self-play via Scorable Negotiation Game

### Usage

1. **Setup the environment:**
   ```
   pip install -r requirements.txt
   ```

2. **Reproduce the Results from the Paper:**

   **Step 1: start generalization-aware behavioral cloning**
   ```
   bash sft.sh
   ```

   **Step 2: start self-play**
   ```
   bash play_game.sh
   ```

   **Step 3: assign rewards**
   ```
   bash assign_rewards.sh
   ```

   **Step 4: start our training**
   ```
   bash lssg.sh
   ```

3. **Evaluate the Results of Our Model:**

   **Reasoning Evaluation:**
   Use [lm-evaluation-harness]([https://openai.com](https://github.com/EleutherAI/lm-evaluation-harness)) for reasoning evaluation

   **Negotiation Evaluation:**
   Use [NegotiationArena]([https://openai.com]([https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/vinid/NegotiationArena))) for negotiation evaluation
