# 🔐 Privacy project

## Loading models

You can load huggingface models or openAI models from the `agents`. Please use the `ConversationalGPTBaseAgent` for ChatGPT and GPT-4, and use `GPT3BaseAgent` for the `text-davinci-XXX` or `text-curie-XXX` models. All agents are prompted with the `interact()` method.

```
my_gpt4 = ConversationalGPTBaseAgent({'engine': "gpt-4-0613"})
my_gpt4.interact("Hey, are you GPT-4?")

>>> As an AI developed by OpenAI, I'm currently using GPT-3. So, as of the time I am answering this question, no, I am not GPT-4.
```

## Running evaluation

```
# Full sweep evaluation for all tiers 1 to 4
# get number of samples for non-deterministic models, batch-size needs to be 1 for API-based models
python eval.py --n-samples 16 --model gpt-4-0613 --batch-size 1

# evaluation on a specific tier
python eval.py --n-samples 16 --model gpt-4-0613 --batch-size 1 --data-tier 4
```