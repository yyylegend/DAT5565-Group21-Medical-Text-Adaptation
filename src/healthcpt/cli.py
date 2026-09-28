"""Connect terminal commands to the project's data and training functions."""

from argparse import ArgumentParser
from pathlib import Path
import json

from .medquad import audit, prepare


def main() -> None:
    # Define each command's inputs in one place so users can run them with --help.
    parser = ArgumentParser(prog="healthcpt")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit-medquad", help="Count complete and missing QA records")
    audit_parser.add_argument("archive", type=Path)
    prepare_parser = commands.add_parser("prepare-medquad", help="Create source-disjoint QA and CPT splits")
    prepare_parser.add_argument("archive", type=Path)
    prepare_parser.add_argument("output_dir", type=Path)
    prepare_parser.add_argument("--seed", type=int, default=5565)
    download_parser = commands.add_parser(
        "download-medical-sources",
        help="Download MedlinePlus XML and a small licensed PMC article sample",
    )
    download_parser.add_argument("qa_train_path", type=Path, help="MedQuAD qa_train.jsonl; its topics guide the PMC sample")
    download_parser.add_argument("output_dir", type=Path, help="Raw download folder, usually data/raw/medical-sources-v3")
    download_parser.add_argument("--topic-limit", type=int, default=20)
    download_parser.add_argument("--articles-per-topic", type=int, default=30)
    corpus_parser = commands.add_parser(
        "prepare-medical-corpus",
        help="Clean and combine CPT text, then create overlap-clean evaluation files",
    )
    corpus_parser.add_argument("medquad_dir", type=Path, help="Prepared MedQuAD folder, usually data/processed/medquad-v1")
    corpus_parser.add_argument("source_dir", type=Path, help="Downloads folder created by download-medical-sources")
    corpus_parser.add_argument("output_dir", type=Path, help="New output folder, usually data/processed/cpt-medical-v3")
    corpus_parser.add_argument("--medquad-cpt-limit", type=int, default=6000)
    corpus_parser.add_argument("--chunk-words", type=int, default=200)
    cpt_parser = commands.add_parser("cpt-pilot", help="Run GPU continued pretraining")
    cpt_parser.add_argument("train_path", type=Path)
    cpt_parser.add_argument("validation_path", type=Path)
    cpt_parser.add_argument("output_dir", type=Path)
    cpt_parser.add_argument("--preset", default="qwen3_5_2b_base")
    cpt_parser.add_argument("--limit-train", type=int, default=16)
    cpt_parser.add_argument("--limit-validation", type=int, default=4)
    cpt_parser.add_argument("--sequence-length", type=int, default=128)
    cpt_parser.add_argument("--batch-size", type=int, default=1)
    cpt_parser.add_argument("--epochs", type=int, default=1)
    cpt_parser.add_argument("--lora-rank", type=int, default=4)
    cpt_parser.add_argument("--learning-rate", type=float, default=1e-4)
    cpt_parser.add_argument("--warmup-ratio", type=float, default=0.05)
    cpt_parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    cpt_parser.add_argument("--checkpoint-steps", type=int, default=2000)
    export_parser = commands.add_parser(
        "export-hf", help="Merge CPT text updates into a full Qwen3.5 multimodal model"
    )
    export_parser.add_argument(
        "run_dir",
        type=Path,
        help="Completed CPT run folder containing run.json and the LoRA adapter",
    )
    export_parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Original full HF Base snapshot used for training (local directory)",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Empty export folder (default: <run_dir>/hf_export_multimodal)",
    )
    args = parser.parse_args()

    # Run the function selected by the user, then print its result as JSON.
    if args.command == "audit-medquad":
        result = audit(args.archive)
    elif args.command == "prepare-medquad":
        result = prepare(args.archive, args.output_dir, args.seed)
    elif args.command == "download-medical-sources":
        from .medical_data import download_medical_sources

        result = download_medical_sources(
            qa_train_path=args.qa_train_path,
            output_dir=args.output_dir,
            topic_limit=args.topic_limit,
            articles_per_topic=args.articles_per_topic,
        )
    elif args.command == "prepare-medical-corpus":
        from .medical_data import prepare_medical_corpus

        result = prepare_medical_corpus(
            medquad_dir=args.medquad_dir,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            medquad_cpt_limit=args.medquad_cpt_limit,
            chunk_words=args.chunk_words,
        )
    elif args.command == "cpt-pilot":
        from .cpt import train

        result = train(
            train_path=args.train_path,
            validation_path=args.validation_path,
            output_dir=args.output_dir,
            preset=args.preset,
            limit_train=args.limit_train,
            limit_validation=args.limit_validation,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lora_rank=args.lora_rank,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
            checkpoint_steps=args.checkpoint_steps,
        )
    else:
        from .export_hf import export_hf

        result = export_hf(args.run_dir, args.base_dir, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
