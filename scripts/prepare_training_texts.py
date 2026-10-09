#!/usr/bin/env python3
"""Prepare HumanML3D-format text without changing captions or source files.

The default keeps existing token fields. Only missing / unk/OTHER fields are
tokenized; --retokenize explicitly regenerates all token fields. spaCy is loaded
only when necessary, and never replaced by a tokenizer without POS annotations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import re
from typing import Any, Optional


PROFILES = ("legacy_finemotion", "humanml_reference")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TextFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    clip_id: str
    caption: str
    tokens: Optional[str]
    start: str
    end: str
    origin: str


def check_id(clip_id: str) -> None:
    if not ID_RE.fullmatch(clip_id):
        raise TextFormatError(f"Unsafe clip ID: {clip_id!r}; use a filename stem.")


def validate_caption(caption: str, origin: str) -> None:
    if not caption.strip():
        raise TextFormatError(f"{origin}: empty caption")
    if any(c in caption for c in "#\r\n"):
        raise TextFormatError(f"{origin}: caption contains '#' or a line break")


def normalize_times(record: Record, allow_nan: bool) -> tuple[Record, bool]:
    fields = [record.start, record.end]
    changed = False
    values = []
    for i, field in enumerate(fields):
        if field != field.strip() or not field:
            raise TextFormatError(f"{record.origin}: empty/space-padded time field")
        try:
            value = float(field)
        except ValueError as exc:
            raise TextFormatError(f"{record.origin}: invalid time {field!r}") from exc
        if math.isnan(value) and allow_nan:
            fields[i], value, changed = "0.0", 0.0, True
        if not math.isfinite(value):
            raise TextFormatError(
                f"{record.origin}: non-finite time {field!r}; review it first, "
                "or explicitly use --normalize-nan-times for NaN only"
            )
        values.append(value)
    start, end = values
    if start < 0 or end < 0 or ((start, end) != (0.0, 0.0) and end <= start):
        raise TextFormatError(f"{record.origin}: invalid interval {fields!r}")
    return replace(record, start=fields[0], end=fields[1]), changed


def validate_tokens(tokens: str, origin: str) -> list[str]:
    # The real loader uses split(' '), so doubled spaces are invalid here.
    values = tokens.split(" ")
    if not tokens or tokens != tokens.strip():
        raise TextFormatError(f"{origin}: empty/space-padded tokens")
    for value in values:
        if value.count("/") != 1 or any(c.isspace() for c in value):
            raise TextFormatError(f"{origin}: malformed word/POS token {value!r}")
        word, pos = value.split("/")
        if not word or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", pos):
            raise TextFormatError(f"{origin}: malformed word/POS token {value!r}")
    return values


def tokenize_caption(caption: str, nlp: Any, profile: str) -> str:
    if profile not in PROFILES:
        raise TextFormatError(f"Unknown tokenization profile: {profile}")
    text = (caption.replace("-", "") if profile == "humanml_reference"
            else re.sub(r"\s+", " ", caption).strip())
    doc = nlp(text)
    if not doc.has_annotation("POS") or not doc.has_annotation("LEMMA"):
        raise TextFormatError(
            "The selected spaCy model did not produce both POS and lemma annotations; "
            "use a full English pipeline such as en_core_web_sm"
        )
    if profile == "humanml_reference":
        # Mirrors EricGuo5513/HumanML3D/text_process.py, process_text().
        tokens = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if token.pos_ in ("NOUN", "VERB") and word != "left":
                word = token.lemma_
            tokens.append(f"{word}/{token.pos_}")
    else:
        # Mirrors the FineMotion VLM exporter / placeholder tokenizer.
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            word = token.lemma_.lower().strip()
            if word == "-pron-":
                word = token.text.lower().strip()
            if word:
                tokens.append(f"{word}/{token.pos_ or 'OTHER'}")
    if not tokens:
        raise TextFormatError("Caption yields no usable tokens; supply a real description.")
    result = " ".join(tokens)
    validate_tokens(result, "generated tokens")
    return result


def read_records(input_dir: Optional[Path], input_jsonl: Optional[Path]) -> list[Record]:
    records = []
    if input_dir is not None:
        if not input_dir.is_dir():
            raise TextFormatError(f"Input directory not found: {input_dir}")
        for path in sorted(input_dir.glob("*.txt")):
            check_id(path.stem)
            count = 0
            for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                origin = f"{path.name}:{n}"
                if "#" in line:
                    fields = line.split("#")
                    if len(fields) != 4:
                        raise TextFormatError(f"{origin}: expected exactly four '#' fields")
                    caption, tokens, start, end = fields
                else:
                    caption, tokens, start, end = line, None, "0.0", "0.0"
                records.append(Record(path.stem, caption, tokens, start, end, origin))
                count += 1
            if count == 0:
                raise TextFormatError(f"{path.name}: empty text file")
    else:
        assert input_jsonl is not None
        with input_jsonl.open(encoding="utf-8-sig") as handle:
            for n, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                origin = f"{input_jsonl.name}:{n}"
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TextFormatError(f"{origin}: invalid JSON") from exc
                if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
                    raise TextFormatError(f"{origin}: require a string 'id'")
                if not isinstance(obj.get("caption"), str):
                    raise TextFormatError(f"{origin}: require a string 'caption'")
                check_id(obj["id"])
                tokens = obj.get("tokens")
                if tokens is not None and not isinstance(tokens, str):
                    raise TextFormatError(f"{origin}: 'tokens' must be a string")
                records.append(Record(obj["id"], obj["caption"], tokens,
                                      str(obj.get("from", "0.0")),
                                      str(obj.get("to", "0.0")), origin))
    if not records:
        raise TextFormatError("No captions found")
    return records


def format_record(record: Record) -> str:
    return f"{record.caption}#{record.tokens}#{record.start}#{record.end}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path, help="Non-recursive *.txt: HumanML lines or plain captions")
    inputs.add_argument("--input-jsonl", type=Path, help="One {id, caption, tokens?, from?, to?} per line")
    parser.add_argument("--output-dir", type=Path, help="New or empty directory; existing files are never overwritten")
    parser.add_argument("--profile", choices=PROFILES, default="legacy_finemotion")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--retokenize", action="store_true", help="Explicitly regenerate all token fields")
    parser.add_argument("--normalize-nan-times", action="store_true", help="Explicitly replace NaN tags with 0.0 and record changes")
    parser.add_argument("--validate-only", action="store_true", help="Check already-tokenized input; no spaCy load and no writes")
    parser.add_argument("--max-text-len", type=int, default=20, help="Report loader truncation; does not truncate text files")
    args = parser.parse_args(argv)
    if args.max_text_len <= 0:
        parser.error("--max-text-len must be positive")
    if args.validate_only and (args.output_dir is not None or args.retokenize):
        parser.error("--validate-only cannot use --output-dir or --retokenize")
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only")
    try:
        if args.output_dir is not None:
            output = args.output_dir.resolve()
            if args.input_dir is not None and output == args.input_dir.resolve():
                raise TextFormatError("Output must differ from the input directory")
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise TextFormatError("Output directory must be new or empty")
        records = read_records(args.input_dir, args.input_jsonl)
        report: dict[str, Any] = {
            "profile_for_new_tokens": args.profile, "retokenize": args.retokenize,
            "python": platform.python_version(), "spacy": None, "model": None,
            "normalized_nan_times": [], "max_text_len": args.max_text_len,
            "caption_policy": "preserve verbatim; tokens alone may change",
        }
        nlp = None
        counts: Counter[str] = Counter()
        prepared = []
        for record in records:
            validate_caption(record.caption, record.origin)
            record, changed = normalize_times(record, args.normalize_nan_times)
            if changed:
                report["normalized_nan_times"].append({"id": record.clip_id, "origin": record.origin})
            needs_tokens = args.retokenize or not record.tokens or record.tokens == "unk/OTHER"
            if args.validate_only and needs_tokens:
                raise TextFormatError(f"{record.origin}: missing/placeholder tokens")
            if needs_tokens:
                if nlp is None:
                    try:
                        import spacy
                        nlp = spacy.load(args.model)
                    except (ImportError, OSError) as exc:
                        raise TextFormatError("Install requirements-text.txt; a real spaCy POS model is required") from exc
                    report["spacy"] = spacy.__version__
                    report["model"] = {"name": args.model, "version": nlp.meta.get("version")}
                record = replace(record, tokens=tokenize_caption(record.caption, nlp, args.profile))
                counts["tokenized_lines"] += 1
            else:
                counts["preserved_token_lines"] += 1
            tokens = validate_tokens(record.tokens or "", record.origin)
            counts["lines"] += 1
            counts["lines_above_max_text_len"] += len(tokens) > args.max_text_len
            counts["segmented_lines"] += (float(record.start), float(record.end)) != (0.0, 0.0)
            counts["max_tokens"] = max(counts["max_tokens"], len(tokens))
            prepared.append(record)
        grouped: dict[str, list[Record]] = defaultdict(list)
        for record in prepared:
            grouped[record.clip_id].append(record)
        # Avoid collisions on Windows and other case-insensitive filesystems.
        if len({clip_id.casefold() for clip_id in grouped}) != len(grouped):
            raise TextFormatError("Clip IDs collide when filenames are compared without case")
        counts["files"] = len(grouped)
        report["counts"] = dict(counts)
        report["files"] = []
        for clip_id, lines in sorted(grouped.items()):
            content = "\n".join(format_record(line) for line in lines) + "\n"
            report["files"].append({"id": clip_id, "lines": len(lines),
                                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})
        if not args.validate_only:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for clip_id, lines in sorted(grouped.items()):
                path = args.output_dir / f"{clip_id}.txt"
                # 'x' is a second guard against overwriting, after preflight.
                with path.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write("\n".join(format_record(line) for line in lines) + "\n")
            with (args.output_dir / "_text_manifest.json").open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        print(json.dumps({"status": "valid" if args.validate_only else "prepared",
                          "profile_for_new_tokens": args.profile, "counts": dict(counts),
                          "normalized_nan_time_lines": len(report["normalized_nan_times"]),
                          "output_dir": str(args.output_dir) if args.output_dir else None},
                         ensure_ascii=True, indent=2))
        return 0
    except (TextFormatError, OSError, UnicodeError) as exc:
        parser.exit(2, f"[error] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
