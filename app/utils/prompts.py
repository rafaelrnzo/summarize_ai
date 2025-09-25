# app/utils/prompts.py

def create_summary_prompt(text: str, max_words: int = 512) -> str:
    return f"""<|im_start|>system
You're helping someone understand this content by creating a natural, conversational summary. Write like you're explaining it to a friend - use everyday language, avoid jargon, and make it engaging. Keep it under {max_words} words and focus on what's actually important and interesting.
<|im_end|>

<|im_start|>user
Can you help me understand what this text is about? Here it is:

{text}
<|im_end|>W

<|im_start|>assistant
Here's what this is about:"""


def create_cluster_summary_prompt(text: str) -> str:
    return f"""<|im_start|>system
Explain the main points from this text in a natural, conversational way. Keep it brief (under 100 words) and focus on what's most interesting or important. Write like you're telling someone about it over coffee.
<|im_end|>

<|im_start|>user
What are the key points here?

{text}
<|im_end|>

<|im_start|>assistant
The main points are:"""


def create_final_summary_prompt(summaries: str) -> str:
    return f"""<|im_start|>system
You've got several key points from different sections. Now bring them together into one clear, engaging summary that tells the complete story. Write naturally - like you're giving someone the highlights of something you just read. Keep it under 512 words and make sure it flows well from start to finish.
<|im_end|>

<|im_start|>user
Here are the main points from different sections. Can you put this all together for me?

{summaries}
<|im_end|>

<|im_start|>assistant
Here's the complete picture:"""