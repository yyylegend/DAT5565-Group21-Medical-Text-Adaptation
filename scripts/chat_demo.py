"""Serve the merged CPT+SFT model as an independent-question Gradio demo."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcpt.evaluate_qa import generation_eos_token_ids, load_model


def build_demo(model_dir: Path, max_new_tokens: int, repetition_penalty: float = 1.0):
    import gradio as gr
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("This demo needs the CUDA-enabled inference environment.")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = load_model(model_dir, Qwen3_5ForConditionalGeneration).to("cuda").eval()
    stop_ids = generation_eos_token_ids(processor.tokenizer)
    print(
        f"Loaded {model_dir}; stop IDs: {stop_ids}; repetition penalty: {repetition_penalty}",
        flush=True,
    )

    def respond(message, history):
        # Training used single question-answer examples, without chat history.
        question = message.strip()
        if not question:
            return "Please enter a question."
        prompt = f"Question: {question}\nAnswer:"
        inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=stop_ids,
                repetition_penalty=repetition_penalty,
            )
        answer = processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
        return answer or "The model returned no answer."

    demo = gr.ChatInterface(
        fn=respond,
        title="Medical QA - CPT+SFT",
        description=(
            "Course research demo. Each question is answered independently; "
            "training used English medical QA and earlier messages are not used as context. "
            f"Repetition penalty: {repetition_penalty:g} (1 = original decoding). "
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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="Experimental repetition penalty; 1.0 retains the original evaluation decoding.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link.")
    args = parser.parse_args()
    build_demo(args.model_dir.resolve(), args.max_new_tokens, args.repetition_penalty).launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )


if __name__ == "__main__":
    main()
