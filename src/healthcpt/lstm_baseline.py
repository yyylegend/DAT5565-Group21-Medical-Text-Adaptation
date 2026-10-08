"""Train and evaluate a small TensorFlow/Keras LSTM medical QA baseline."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from healthcpt.qa_metrics import (
    calculate_metrics, file_sha256, read_examples, select_examples, write_jsonl,
)

EOS = "[eos]"


def training_text(example):
    """Use the same QA format as SFT and add a learnable stopping word."""
    return f"Question: {example['question']}\nAnswer: {example['reference_answer']} {EOS}"


def check_cached_answers(examples, cached):
    """Reject mismatched samples before comparing with saved Qwen answers."""
    if len(examples) != len(cached):
        raise ValueError("Cached Qwen predictions have a different sample size.")
    for example, row in zip(examples, cached, strict=True):
        for key in ("source_line", "question", "reference_answer"):
            if example[key] != row.get(key):
                raise ValueError(f"Cached Qwen sample differs at source line {example['source_line']}.")
        if not isinstance(row.get("candidate_prediction"), str):
            raise ValueError("Cached Qwen rows need a candidate_prediction string.")


def runtime(seed):
    """Load TensorFlow only when training or inference is requested."""
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf
    import keras

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.utils.set_random_seed(seed)
    keras.config.set_dtype_policy("float32")
    return tf, keras


def train(args):
    sizes = (args.vocab_size, args.embedding_dim, args.hidden_size,
             args.sequence_length, args.batch_size, args.epochs)
    if min(sizes) < 1 or args.learning_rate <= 0:
        raise ValueError("Training sizes and learning rate must be positive.")
    tf, keras = runtime(args.seed)
    if not tf.config.list_physical_devices("GPU"):
        print("[LSTM] No GPU detected; CPU training may be slow.", flush=True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new, empty training output folder.")
    train_rows = select_examples(read_examples(args.train_file), args.limit_train, args.seed)
    validation_rows = select_examples(read_examples(args.validation_file), args.limit_validation, args.seed)
    train_texts = [training_text(row) for row in train_rows]
    validation_texts = [training_text(row) for row in validation_rows]

    # Learn the word vocabulary from training text only. Preserve punctuation
    # so dosage units and decimal values are not silently changed.
    vectorizer = keras.layers.TextVectorization(
        max_tokens=args.vocab_size, standardize="lower", output_mode="int",
        output_sequence_length=args.sequence_length + 1,
    )
    print("[LSTM] Building a vocabulary from the training split...", flush=True)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(128))
    vocabulary = vectorizer.get_vocabulary()
    if EOS not in vocabulary:
        raise ValueError("The vocabulary must include the end-of-answer word.")

    def dataset(texts, shuffle):
        # Shift each sequence by one word: x predicts the next word in y.
        tokens = vectorizer(tf.constant(texts))
        x, y = tokens[:, :-1], tokens[:, 1:]
        weights = tf.cast(y != 0, tf.float32)
        data = tf.data.Dataset.from_tensor_slices((x, y, weights))
        if shuffle:
            data = data.shuffle(len(texts), seed=args.seed)
        return data.batch(args.batch_size).prefetch(tf.data.AUTOTUNE), tokens

    training, train_tokens = dataset(train_texts, True)
    validation, validation_tokens = dataset(validation_texts, False)
    inputs = keras.Input(shape=(None,), dtype="int64", name="tokens")
    embedded = keras.layers.Embedding(
        len(vocabulary), args.embedding_dim, mask_zero=True, name="embedding",
    )(inputs)
    # A single forward LSTM cannot look at future answer words.
    sequence, _, _ = keras.layers.LSTM(
        args.hidden_size, return_sequences=True, return_state=True, name="lstm",
    )(embedded)
    sequence = keras.layers.Dropout(0.2)(sequence)
    logits = keras.layers.Dense(len(vocabulary), name="word_logits")(sequence)
    model = keras.Model(inputs, logits, name="medical_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="mean_with_sample_weight",
        ),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
        jit_compile=False,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8",
    )
    config = {
        "train_file_sha256": file_sha256(args.train_file),
        "validation_file_sha256": file_sha256(args.validation_file),
        "train_examples": len(train_rows), "validation_examples": len(validation_rows),
        "vocabulary_size": len(vocabulary), "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size, "sequence_length": args.sequence_length,
        "batch_size": args.batch_size, "maximum_epochs": args.epochs,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "dropout": 0.2, "early_stopping_patience": 2,
        "tokenizer": "training-only lowercase whitespace vocabulary",
        "objective": "next word on question and answer; padding excluded",
    }
    (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.summary()
    print("[LSTM] Training starts. Best and latest full models are saved after epochs.", flush=True)
    training_started = time.perf_counter()
    history = model.fit(
        training, validation_data=validation, epochs=args.epochs, shuffle=False, verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(str(args.output_dir / "model.keras"), save_best_only=True),
            keras.callbacks.ModelCheckpoint(str(args.output_dir / "latest.keras")),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
            keras.callbacks.CSVLogger(str(args.output_dir / "history.csv")),
            keras.callbacks.TensorBoard(log_dir=str(args.output_dir / "tensorboard")),
        ],
    )
    training_seconds = time.perf_counter() - training_started
    # Re-evaluate the best checkpoint with a true corpus-level word NLL.
    best = keras.models.load_model(args.output_dir / "model.keras")
    total_loss, total_words = 0.0, 0.0
    for x, y, weights in validation:
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=best(x, training=False))
        total_loss += float(tf.reduce_sum(losses * weights).numpy())
        total_words += float(tf.reduce_sum(weights).numpy())
    nll = total_loss / total_words

    def text_stats(texts, tokens):
        nonpadding = int(tf.reduce_sum(tf.cast(tokens != 0, tf.int64)).numpy())
        unknown = int(tf.reduce_sum(tf.cast(tokens == 1, tf.int64)).numpy())
        return {
            "truncated_examples": sum(len(text.split()) > args.sequence_length + 1 for text in texts),
            "unknown_word_fraction": unknown / nonpadding,
        }

    report = {
        **config, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow": tf.__version__, "keras": keras.__version__,
        "parameters": model.count_params(),
        "model_file_bytes": (args.output_dir / "model.keras").stat().st_size,
        "training_seconds_including_validation_and_saves": training_seconds,
        "completed_epochs": len(history.history["loss"]),
        "selected_epoch": min(range(len(history.history["val_loss"])), key=lambda i: history.history["val_loss"][i]) + 1,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "validation_token_nll": nll, "validation_word_perplexity": math.exp(nll),
        "train_text_stats": text_stats(train_texts, train_tokens),
        "validation_text_stats": text_stats(validation_texts, validation_tokens),
        "note": "Word perplexity is specific to this vocabulary; do not compare directly with Qwen token perplexity.",
    }
    (args.output_dir / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[LSTM] Complete: {args.output_dir}", flush=True)


def evaluate(args):
    if args.max_new_words < 1:
        raise ValueError("--max-new-words must be positive.")
    examples = select_examples(read_examples(args.qa_file), args.limit, args.seed)
    cached = None
    cached_report = None
    if args.cached_qwen_predictions:
        cached = [json.loads(line) for line in args.cached_qwen_predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        check_cached_answers(examples, cached)
        cached_report = json.loads(
            args.cached_qwen_predictions.with_name("metrics.json").read_text(encoding="utf-8")
        )
        if cached_report["evaluation_file_sha256"] != file_sha256(args.qa_file):
            raise ValueError("Cached Qwen results used a different evaluation file.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a new, empty evaluation output folder.")
    tf, keras = runtime(args.seed)
    model = keras.models.load_model(args.model_dir / "model.keras", compile=False)
    vocabulary = json.loads((args.model_dir / "vocabulary.json").read_text(encoding="utf-8"))
    vectorizer = keras.layers.TextVectorization(standardize="lower", vocabulary=vocabulary)
    eos_id = vocabulary.index(EOS)
    lstm = model.get_layer("lstm")

    # Reuse the trained layers and carry the LSTM state between generated words.
    token_input = keras.Input(shape=(None,), dtype="int64")
    h_input = keras.Input(shape=(lstm.units,))
    c_input = keras.Input(shape=(lstm.units,))
    embedded = model.get_layer("embedding")(token_input)
    sequence, h, c = lstm(embedded, initial_state=[h_input, c_input])
    logits = model.get_layer("word_logits")(sequence[:, -1, :])
    inference = keras.Model([token_input, h_input, c_input], [logits, h, c])

    @tf.function(input_signature=[
        tf.TensorSpec([1, None], tf.int64),
        tf.TensorSpec([1, lstm.units], tf.float32),
        tf.TensorSpec([1, lstm.units], tf.float32),
    ])
    def step(tokens, h, c):
        return inference([tokens, h, c], training=False)

    zeros = tf.zeros([1, lstm.units])
    step(tf.constant([[eos_id]], dtype=tf.int64), zeros, zeros)[0].numpy()
    predictions, seconds, rows = [], [], []
    for index, example in enumerate(examples, start=1):
        start = time.perf_counter()
        prompt = f"Question: {example['question']}\nAnswer:"
        tokens = vectorizer(tf.constant([prompt]))
        logits, h, c = step(tokens, zeros, zeros)
        answer_words = []
        for _ in range(args.max_new_words):
            scores = logits.numpy()[0]
            scores[:2] = float("-inf")  # Never generate padding or the unknown-word marker.
            word_id = int(scores.argmax())
            if word_id == eos_id:
                break
            answer_words.append(vocabulary[word_id])
            logits, h, c = step(tf.constant([[word_id]], dtype=tf.int64), h, c)
        seconds.append(time.perf_counter() - start)
        answer = " ".join(answer_words)
        predictions.append(answer)
        rows.append({**example, "candidate_prediction": answer, "seconds": seconds[-1]})
        if index % 25 == 0 or index == len(examples):
            print(f"[LSTM Eval] {index}/{len(examples)} questions", flush=True)
    references = [row["reference_answer"] for row in examples]
    metrics = {"LSTM": calculate_metrics(predictions, references)}
    if cached is not None:
        name = cached_report["models"]["candidate"]["name"]
        metrics[f"{name}_cached"] = calculate_metrics([r["candidate_prediction"] for r in cached], references)
        if all(isinstance(r.get("base_prediction"), str) for r in cached):
            metrics["Base_cached"] = calculate_metrics([r["base_prediction"] for r in cached], references)
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_file_sha256": file_sha256(args.qa_file),
        "evaluation_examples": len(examples), "sampling_seed": args.seed,
        "model_file_sha256": file_sha256(args.model_dir / "model.keras"),
        "generation": {"greedy": True, "max_new_words": args.max_new_words},
        "average_seconds_per_answer": sum(seconds) / len(seconds),
        "average_generated_words": sum(len(p.split()) for p in predictions) / len(predictions),
        "cached_qwen_predictions_sha256": file_sha256(args.cached_qwen_predictions) if cached else None,
        "cached_qwen_generation": cached_report["generation"] if cached_report else None,
        "metrics": metrics,
        "note": "Text overlap is not medical correctness. LSTM word and Qwen subword generation caps are different; their perplexities and token speeds are not directly comparable.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def add_training_arguments(training):
    """Share argument definitions between healthcpt and the module entry point."""
    training.add_argument("--train-file", type=Path, required=True)
    training.add_argument("--validation-file", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--vocab-size", type=int, default=20000)
    training.add_argument("--embedding-dim", type=int, default=128)
    training.add_argument("--hidden-size", type=int, default=256)
    training.add_argument("--sequence-length", type=int, default=512)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--limit-train", type=int)
    training.add_argument("--limit-validation", type=int)
    training.add_argument("--seed", type=int, default=5565)


def add_evaluation_arguments(evaluation):
    """Keep both evaluation entry points on the same options and defaults."""
    evaluation.add_argument("--model-dir", type=Path, required=True)
    evaluation.add_argument("--qa-file", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--limit", type=int, default=200)
    evaluation.add_argument("--max-new-words", type=int, default=128)
    evaluation.add_argument("--cached-qwen-predictions", type=Path)
    evaluation.add_argument("--seed", type=int, default=5565)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_training_arguments(commands.add_parser("train"))
    add_evaluation_arguments(commands.add_parser("evaluate"))
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
