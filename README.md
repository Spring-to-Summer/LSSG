# Language Model Self-play via Scorable Negotiation Game


environment:
pip install -r requirements.txt


steps:
1. run sft.sh to start generalization-aware behavioral cloning
2. run play_game.sh to start self-play
3. run assign_rewards.sh to assign rewards
4. run lssg.sh to start our training
