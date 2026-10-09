# Preparing motion text and lemma/POS annotations

The text release is intended to be usable with HumanML3D-style data loaders. A text file and its motion feature share the same ID:

```text
training/
  new_joint_vecs/<clip_id>.npy
  texts/<clip_id>.txt
```

Each nonempty line has four fields:

```text
caption#word/POS word/POS ...#from_time#to_time
```

The sentence is kept verbatim. The tokens contain a word/lemma and its grammatical POS, such as `wave/VERB` or `hand/NOUN`. POS is a language feature; action/style categories are defined separately. Time fields are seconds relative to the paired clip. `0#0` means the entire clip. Do not copy an original long sequence's absolute time range into an already cropped clip.

## From a description to a text file

The historical FineMotion pipeline derived MAIN descriptions from source action/style labels and generated DETAILS from motion videos with [VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3). The two sentences were exported as separate text lines and processed with spaCy. A model-generated detail must be checked against the video before use; POS tagging only analyzes language.

For this example, a short right-hand-wave sentence was reviewed against the motion. The formatter accepts either plain caption files or a simple JSONL format, so other video-captioning models can be used without changing the file format:

```json
{"id":"my_motion_001","caption":"A person waves their right hand while standing still.","from":"0.0","to":"0.0"}
```

An optional `tokens` string can preserve existing annotations. Repeated IDs produce multiple lines in the same text file. This JSONL schema is the formatter's interface; historical VLM logs using `caption_main`/`caption_details` need an explicit field mapping first.

## Install the tokenizer

```bash
python -m pip install -r scripts/requirements-text.txt
```

The pinned environment is spaCy 3.8.11 and `en_core_web_sm` 3.8.0, verified in the preparation environment. Earlier annotation runs do not all have a recoverable environment record; retain their existing tokens when preserving an archived version.

The script requires real POS and lemma annotations. It fails if a selected model lacks them, instead of creating placeholder `OTHER` tags.

## Choose an explicit rule

`legacy_finemotion` reproduces the project's historical VLM exporter and Motion-X tokenizer: skip spaces/punctuation, use lowercase lemmas for all retained tokens.

`humanml_reference` follows [HumanML3D's process_text](https://github.com/EricGuo5513/HumanML3D/blob/main/text_process.py): remove hyphens in the sentence used for tokenization, retain alphabetic tokens, lemmatize NOUN/VERB except the exact lowercase word `left`, and retain the text of other tokens. It does not lowercase everything.

The four-field format is shared, but the two profiles may produce different tokens. Existing HumanML3D tokens should normally be retained.

## Commands

Plain `*.txt` input files can contain one caption per line, or existing four-field annotations. All lines are processed in order. Valid existing tokens are preserved unless `--retokenize` is explicitly used.

```bash
python scripts/prepare_training_texts.py --input-dir my_captions --output-dir prepared_texts --profile legacy_finemotion
python scripts/prepare_training_texts.py --input-jsonl my_captions.jsonl --output-dir prepared_jsonl_texts --profile legacy_finemotion
```

To regenerate the exact bundled example from its reviewed caption input:

```bash
python scripts/prepare_training_texts.py --input-jsonl examples/bandai_wave/caption_input.jsonl --output-dir generated/example_text --profile legacy_finemotion
```

With the pinned tokenizer it produces:

```text
A person waves their right hand while standing still.#a/DET person/NOUN wave/VERB their/PRON right/ADJ hand/NOUN while/SCONJ stand/VERB still/ADV#0.0#0.0
```

This sentence was written and visually reviewed for the example; it is not
presented as a newly generated VLM output. The retained archived motion and
feature files are paired with this single new sentence.

For a new variant following the HumanML reference rule:

```bash
python scripts/prepare_training_texts.py --input-dir my_captions --output-dir reference_texts --profile humanml_reference --retokenize
```

Use a new or empty output directory. The script never overwrites source files or existing outputs. `_text_manifest.json` records the selected rule, package/model versions, hashes, line counts, and long-token statistics. Finite time strings retain their precision.

To check existing annotations without loading spaCy or writing files:

```bash
python scripts/prepare_training_texts.py --input-dir examples/bandai_wave/training/texts --validate-only
```

Invalid fields, tokens, non-finite times, and empty nonzero intervals are rejected. If a reviewed source specifically requires the historical loader's `NaN→0` interpretation, `--normalize-nan-times` records each conversion in the output manifest. It does not silently repair infinity or empty intervals.

## How the current loader uses the fields

The inspected MCM-LDM `Text2MotionDatasetV2` reads all caption lines. Whole-clip descriptions are sampled with `random.choice`; it does not give MAIN a higher priority or automatically merge MAIN and DETAILS. A standalone details sentence must therefore describe enough of the action for its intended use.

The current training features and inspected loader use a 20Hz timeline for nonzero time intervals. If you build features at another rate, update the feature construction and the loader's seconds-to-frame conversion together; text time fields remain in seconds. The loader's default `MAX_TEXT_LEN=20` truncates token lists before adding `sos/OTHER` and `eos/OTHER`; the complete sentence is still returned. Do not insert these special tokens into the text file yourself.

`WordVectorizer` constructs word vectors and POS features. In the inspected motion-transfer implementation, these feed the text-motion evaluation path, while diffusion training conditions are obtained from motion. MotionCLIP text inference separately encodes the original sentence. Use the annotations according to the training/evaluation task; POS tagging itself is not an action-class supervision scheme.

A training run also requires matching motion features, splits, Mean/Std, vocabulary/model files, and the chosen model's dependencies. Downloading annotations alone does not reconstruct motion data.

## Validation and attribution

Both tokenizer profiles were compared against their reference rules on real spaCy inputs, including a complete CLI export. The sample text passes field/time/token validation. New descriptions still require visual review.

The reference profile credits HumanML3D (Chuan Guo); its MIT notice is in `licenses/HUMANML3D_REPOSITORY_LICENSE.txt`. Formatting does not replace the copyright/license of an upstream sentence. Future annotations will be distributed with source-specific provenance and notices.
