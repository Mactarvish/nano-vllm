import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"模型路径不存在: {path}\n"
            "请先下载模型，运行:\n"
            "huggingface-cli download --resume-download Qwen/Qwen3-0.6B \\\n"
            "  --local-dir ~/huggingface/Qwen3-0.6B/ \\\n"
            "  --local-dir-use-symlinks False"
        )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
