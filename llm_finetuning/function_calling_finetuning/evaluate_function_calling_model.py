

def generate_function_call(model: AutoModelForCausalLM,
                          tokenizer: AutoTokenizer, 
                          prompt: str,
                          max_new_tokens: int = 512,
                          temperature: float = 0.7,
                          do_sample: bool = True) -> str:
    """
    Generate a function call response using the fine-tuned model.
    
    Args:
        model (AutoModelForCausalLM): Fine-tuned model with LoRA adapters
        tokenizer (AutoTokenizer): Model tokenizer
        prompt (str): Input prompt for function calling
        max_new_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature (only used when do_sample=True)
        do_sample (bool): Whether to use sampling
        
    Returns:
        str: Generated response with function calls
        
    Example Prompt Format:
        "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>"
    """
    print(f"🎯 Generating response for prompt...")
    print(f"📝 Input: {prompt}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate response with proper parameter handling
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    
    # Only add sampling parameters if do_sample=True
    if do_sample:
        generation_kwargs["temperature"] = temperature
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )
    
    # Decode result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("✅ Generation completed!")
    print(f"📊 Generated {len(outputs[0]) - len(inputs['input_ids'][0])} new tokens")
    
    return result


# ==========================================================================================================

def test_function_calling_examples(model: AutoModelForCausalLM,
                                   tokenizer: AutoTokenizer,
                                   test_dataset: Dataset) -> list[str]:
    """
    Test the model with various function calling examples.
    
    Args:
        model: Fine-tuned model.
        tokenizer: Model tokenizer.
        test_dataset: Dataset rows containing formatted `text` examples.

    Returns:
        Generated responses aligned positionally with test_dataset.
    """
    print("🧪 Testing function calling capabilities...")
    
    results = []
    for i, test_case in enumerate(test_dataset):
        formatted_text = test_case.get("text")
        if not isinstance(formatted_text, str) or "<calls>" not in formatted_text:
            raise ValueError(f"test_dataset[{i}] has no formatted <calls> block")

        # Exclude the expected answer so the model cannot see the ground truth.
        prompt = formatted_text.split("<calls>", maxsplit=1)[0]

        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case.get('query', 'dataset example')}")
        print(f"{'='*60}")
        
        result = generate_function_call(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=512,  # Adjust as needed
            temperature=0.7,
            do_sample=True  # Fixed: Use sampling with temperature
        )
        results.append(result)
        print(f"\n🔍 Complete Response:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    
    print("\n✅ All test cases completed!")
    return results

# ===============================================================================================================================

import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_CALLS_PATTERN = re.compile(r"<calls\b[^>]*>(.*?)</calls>", re.DOTALL | re.IGNORECASE)
_USER_PATTERN = re.compile(r"<user\b[^>]*>(.*?)</user>", re.DOTALL | re.IGNORECASE)


def _normalize_tool_call(call: Any, *, source: str) -> dict[str, Any]:
    """Normalize a tool call to {'name': str, 'arguments': dict}."""
    if not isinstance(call, Mapping):
        raise ValueError(f"{source} must be a JSON object, got {type(call).__name__}")

    # Also accept the OpenAI-style {'function': {'name': ..., 'arguments': ...}} shape.
    function = call.get("function")
    if isinstance(function, Mapping):
        call = function

    # Some generated examples use 'method'; normalize it to the required 'name'.
    name = call.get("name", call.get("method"))
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source} is missing a non-empty tool name: {call!r}")

    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} has invalid JSON arguments: {arguments!r}") from exc

    if not isinstance(arguments, Mapping):
        raise ValueError(f"{source} arguments must be a JSON object: {arguments!r}")

    return {
        "name": name.strip(),
        "arguments": dict(arguments),
    }


def _parse_tool_calls(value: Any, *, source: str, require_tags: bool = False) -> list[dict[str, Any]]:
    """Parse one or more JSON tool calls from raw JSON or <calls> blocks."""
    payloads: list[Any] = []

    if isinstance(value, str):
        call_blocks = _CALLS_PATTERN.findall(value)
        if require_tags and not call_blocks:
            raise ValueError(f"{source} contains no <calls>...</calls> block")

        serialized_payloads = call_blocks or [value]
        for block_index, serialized in enumerate(serialized_payloads):
            try:
                payloads.append(json.loads(serialized.strip()))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source} call block {block_index} is not valid JSON: {serialized!r}"
                ) from exc
    else:
        payloads.append(value)

    normalized_calls: list[dict[str, Any]] = []
    for payload_index, payload in enumerate(payloads):
        calls = payload if isinstance(payload, list) else [payload]
        for call_index, call in enumerate(calls):
            normalized_calls.append(
                _normalize_tool_call(
                    call,
                    source=f"{source} payload {payload_index}, call {call_index}",
                )
            )

    return normalized_calls


def compare_tool_calls(
    predicted_tool_calls: Sequence[Mapping[str, Any]],
    actual_tool_calls: Sequence[Mapping[str, Any]],
    *,
    prediction_parse_error: str | None = None,
) -> dict[str, bool]:
    """Compare ordered tool names and arguments for one prediction."""
    if prediction_parse_error is not None:
        return {
            "tool_calls_match": False,
            "arguments_match": False,
            "exact_match": False,
        }

    predicted_names = [call["name"] for call in predicted_tool_calls]
    actual_names = [call["name"] for call in actual_tool_calls]
    predicted_arguments = [call["arguments"] for call in predicted_tool_calls]
    actual_arguments = [call["arguments"] for call in actual_tool_calls]

    tool_calls_match = predicted_names == actual_names
    arguments_match = predicted_arguments == actual_arguments
    return {
        "tool_calls_match": tool_calls_match,
        "arguments_match": arguments_match,
        "exact_match": tool_calls_match and arguments_match,
    }


def calculate_function_call_accuracy(
    evaluation_records: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Calculate tool-name, argument, and exact-match accuracy."""
    if not evaluation_records:
        raise ValueError("evaluation_records must contain at least one record")

    total = len(evaluation_records)
    tool_call_matches = sum(bool(record["tool_calls_match"]) for record in evaluation_records)
    argument_matches = sum(bool(record["arguments_match"]) for record in evaluation_records)
    exact_matches = sum(bool(record["exact_match"]) for record in evaluation_records)
    return {
        "total_examples": total,
        "tool_call_matches": tool_call_matches,
        "tool_call_accuracy": tool_call_matches / total,
        "argument_matches": argument_matches,
        "argument_accuracy": argument_matches / total,
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_matches / total,
    }


def evaluate_function_call_predictions(
    test_dataset: Dataset,
    prediction_results: Sequence[str],
) -> list[dict[str, Any]]:
    """Pair prompts and ground-truth calls with calls generated by the model.

    The dataset and prediction list must be positionally aligned. Ground truth
    is read from the `answers` column when present, otherwise from the dataset's
    formatted `text` column. Predictions are parsed from every <calls> block.
    """
    if isinstance(prediction_results, (str, bytes)):
        raise TypeError("prediction_results must be a sequence of response strings")

    if len(test_dataset) != len(prediction_results):
        raise ValueError(
            "test_dataset and prediction_results must have the same length: "
            f"{len(test_dataset)} != {len(prediction_results)}"
        )

    evaluation_records: list[dict[str, Any]] = []

    for index, (example, prediction) in enumerate(zip(test_dataset, prediction_results, strict=True)):
        text = example.get("text", "")

        prompt = example.get("query")
        if not isinstance(prompt, str) or not prompt.strip():
            user_matches = _USER_PATTERN.findall(text)
            if not user_matches:
                raise ValueError(f"test_dataset[{index}] has no query or <user> block")
            prompt = user_matches[0]

        answers = example.get("answers")
        ground_truth_from_text = answers is None
        if answers is None:
            answers = text

        prediction_parse_error = None
        try:
            predicted_tool_calls = _parse_tool_calls(
                prediction,
                source=f"prediction_results[{index}]",
                require_tags=True,
            )
        except ValueError as exc:
            predicted_tool_calls = []
            prediction_parse_error = str(exc)

        actual_tool_calls = _parse_tool_calls(
            answers,
            source=f"test_dataset[{index}] ground truth",
            require_tags=ground_truth_from_text,
        )
        comparison = compare_tool_calls(
            predicted_tool_calls,
            actual_tool_calls,
            prediction_parse_error=prediction_parse_error,
        )

        evaluation_records.append({
            "user_prompt": prompt.strip(),
            "predicted_tool_calls": predicted_tool_calls,
            "actual_tool_calls": actual_tool_calls,
            **comparison,
            "prediction_parse_error": prediction_parse_error,
            "predicted_raw": prediction if prediction_parse_error else None,
        })

    return evaluation_records


def save_accuracy_summary(
    accuracy_metrics: Mapping[str, int | float],
    output_path: str | Path = "tool_call_eval_accuracy_qwen-3-0.6B.json",
) -> Path:
    """Write aggregate function-call accuracy metrics to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as summary_file:
        json.dump(dict(accuracy_metrics), summary_file, ensure_ascii=False, indent=2)
    print(f"Saved accuracy summary to {output_path.resolve()}")
    return output_path


def save_function_call_evaluation(
    evaluation_records: Sequence[Mapping[str, Any]],
    output_stem: str | Path = "tool_call_eval_results_qwen-3-0.6B",
) -> dict[str, Path]:
    """Write evaluation records to CSV and lossless JSON files."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.parent / f"{output_stem.name}.csv"
    json_path = output_stem.parent / f"{output_stem.name}.json"

    normalized_records = [dict(record) for record in evaluation_records]
    required_fields = {
        "user_prompt",
        "predicted_tool_calls",
        "actual_tool_calls",
        "tool_calls_match",
        "arguments_match",
        "exact_match",
    }
    for index, record in enumerate(normalized_records):
        missing_fields = required_fields - record.keys()
        if missing_fields:
            raise ValueError(
                f"evaluation_records[{index}] is missing fields: "
                f"{sorted(missing_fields)}"
            )

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "user_prompt",
                "predicted_tool_calls",
                "actual_tool_calls",
                "tool_calls_match",
                "arguments_match",
                "exact_match",
                "prediction_parse_error",
                "predicted_raw",
            ],
        )
        writer.writeheader()
        for record in normalized_records:
            writer.writerow({
                "user_prompt": record["user_prompt"].replace(
                    "\r", "\\r"
                ).replace("\n", "\\n"),
                "predicted_tool_calls": json.dumps(
                    record["predicted_tool_calls"], ensure_ascii=False
                ),
                "actual_tool_calls": json.dumps(
                    record["actual_tool_calls"], ensure_ascii=False
                ),
                "tool_calls_match": record["tool_calls_match"],
                "arguments_match": record["arguments_match"],
                "exact_match": record["exact_match"],
                "prediction_parse_error": (
                    record.get("prediction_parse_error") or ""
                ).replace("\r", "\\r").replace("\n", "\\n"),
                "predicted_raw": json.dumps(
                    record["predicted_raw"], ensure_ascii=False
                ) if record.get("predicted_raw") is not None else "",
            })

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(normalized_records, json_file, ensure_ascii=False, indent=2)

    print(f"Saved {len(normalized_records):,} records to {csv_path.resolve()}")
    print(f"Saved {len(normalized_records):,} records to {json_path.resolve()}")
    return {"csv": csv_path, "json": json_path}


# Use the same full dataset passed to test_function_calling_examples.
evaluation_results = evaluate_function_call_predictions(test_dataset, results)
accuracy_metrics = calculate_function_call_accuracy(evaluation_results)
print(json.dumps(accuracy_metrics, indent=2))
accuracy_summary_path = save_accuracy_summary(accuracy_metrics)
evaluation_files = save_function_call_evaluation(evaluation_results)
evaluation_results[:2]
