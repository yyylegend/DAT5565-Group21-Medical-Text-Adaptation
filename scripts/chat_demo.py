"""Serve the merged CPT+SFT model as an independent-question Gradio demo."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcpt.evaluate_qa import generation_eos_token_ids, load_model


def build_demo(
    model_dir: Path, max_new_tokens: int, repetition_penalty: float = 1.0,
    temperature: float = 0.7, top_p: float = 0.95,
):
    import gradio as gr
    import torch
    from transformers import (
        AutoProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration,
        RepetitionPenaltyLogitsProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("This demo needs the CUDA-enabled inference environment.")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = load_model(model_dir, Qwen3_5ForConditionalGeneration).to("cuda").eval()
    stop_ids = generation_eos_token_ids(processor.tokenizer)
    print(
        f"Loaded {model_dir}; stop IDs: {stop_ids}; max answer tokens: {max_new_tokens}; "
        f"temperature: {temperature}; top-p: {top_p}; "
        f"answer-only repetition penalty: {repetition_penalty}",
        flush=True,
    )

    def respond(message, history):
        # Training used single question-answer examples, without chat history.
        question = message.strip()
        if not question:
            return "Please enter a question."
        prompt = f"Question: {question}\nAnswer:"
        inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
        prompt_length = inputs["input_ids"].shape[1]
        processors = LogitsProcessorList()
        if repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(
                repetition_penalty, prompt_ignore_length=prompt_length,
            ))
        sampling = {}
        if temperature > 0:
            sampling = {"temperature": temperature, "top_p": top_p, "top_k": 0}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                use_cache=True,
                eos_token_id=stop_ids,
                repetition_penalty=1.0,
                logits_processor=processors,
                **sampling,
            )
        answer_tokens = output[0, prompt_length:]
        answer = processor.tokenizer.decode(
            answer_tokens, skip_special_tokens=True,
        ).strip()
        if not answer:
            print(f"No visible answer; generated token IDs: {answer_tokens.tolist()}", flush=True)
            return "The model ended without a text answer. Try a clearly phrased English medical question."
        return answer

    demo = gr.ChatInterface(
        fn=respond,
        title="Medical QA - CPT+SFT",
        description=(
            "Course research demo. Each question is answered independently; "
            "training used English medical QA and earlier messages are not used as context. "
            f"Answer limit: {max_new_tokens} tokens. Temperature: {temperature:g} "
            f"(0 = greedy). Top-p: {top_p:g}. "
            f"Answer-only repetition penalty: {repetition_penalty:g} (1 = no penalty). "
            "Answers may repeat or contain "
            "unsupported information. Not for clinical decisions."
        ),
        concurrency_limit=1,
        api_name="answer",
    )
    return demo.queue(default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "runs/qwen3_5_2b_sft_full_1epoch/hf_export_multimodal",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature; 0 selects greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling probability mass; 1 disables nucleus filtering.")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="Experimental penalty on generated answer tokens only; 1.0 disables it.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link.")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Use a positive token limit, temperature >= 0, and 0 < top-p <= 1.")
    build_demo(
        args.model_dir.resolve(), args.max_new_tokens, args.repetition_penalty,
        args.temperature, args.top_p,
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )


if __name__ == "__main__":
    main()
