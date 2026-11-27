# Datasheet: Hinglish ToolBench

Following the structure of *Datasheets for Datasets* (Gebru et al., 2021). Counts marked † are filled in when the release is frozen.

## Motivation

**Purpose.** To measure and improve how well language models turn requests written the way Indian users type (romanized code-mixed Hinglish, Devanagari Hindi, Indian English) into structured tool calls. Existing tool-calling benchmarks are English-first. The Hindi variants that exist (BFCL-Hi, MLCL) translate English queries into formal Hindi, which is not how most users write.

**Creators.** Rehan Chouhan (IIT Roorkee), as an independent project. No external funding.

## Composition

**Instances.** Each instance is one conversation that ends with a user turn, plus:

- `context`: the current date and time and the user's city, both of which the request may refer to implicitly;
- `expected`: what the assistant should do next. Either `call` with one or more tool calls, `clarify` (a required argument is missing), or `respond` (no tool fits);
- `category`: `single`, `parallel`, `multi_turn`, `clarify` or `irrelevant`;
- `script`: `roman`, `devanagari` or `english`;
- `meta`: how the instance was produced (sampler hints, teacher model, which verifier accepted it).

**Tools.** 17 fictional APIs across banking, travel, food, productivity and utilities (`src/htb/tools.py`). They are deliberately generic and do not correspond to any real bank, airline or delivery service.

**Splits.**

| split | size | how it was checked |
| --- | --- | --- |
| train | † | automatic round-trip verification |
| dev | † | automatic round-trip verification |
| test | † | hand-verified by the author |

The test set comes from a separate scenario seed. Messages that appear in the test set (after normalization) are removed from train and dev.

**Personal information.** None of it is real. Person names are common Indian first names. Phone numbers, PNRs and order ids are random strings in valid formats; a random 10-digit mobile number may coincide with a real one, but it is not associated with any person. Restaurant names are invented.

## Collection process

1. **Label first.** A scenario sampler (`src/htb/gen/sampler.py`) draws a tool call with valid arguments, a date and city context, a category, a script, and *phrasing hints*: the surface forms the message must use, such as `parso` for the day after tomorrow, `saade 7` for 7:30, or `dedh hazaar` for 1,500. Every label is valid by construction and checked against the tool schemas.
2. **Phrasing.** A teacher LLM writes the user's message(s) from the scenario and hints. Teachers: Gemini 3.5 Flash-Lite and Gemini 3.1 Flash-Lite (Google Gemini API, November 2026).
3. **Lint.** Automatic checks for script (Devanagari share), leaked field names or ISO dates, and presence of ids. A message requested in Devanagari but written fully in Latin script is relabelled `roman`.
4. **Round-trip verification.** A different model receives only the conversation and must produce the label. Verifiers are tried in order (Gemma 4 31B, then gpt-oss-120b and Qwen3.8-27B on Groq), and an instance is kept if any of them recovers the label exactly. Instances that no verifier recovers are rejected, but they are kept for analysis and some are offered for test-set review.
5. **Human review (test only).** The author reviewed every test candidate in an offline tool (`htb review`), accepting it, rejecting it, or correcting the label.

## Preprocessing

Normalized-text de-duplication within and across splits; removal of test overlap from train/dev; no other filtering or editing of messages.

## Uses

**Intended.** Evaluating tool-calling models on code-mixed Indian input; training small task-specific models (the companion LoRA models were trained on the train split); studying synthetic-data verification.

**Not intended.** Evaluating real products' safety or compliance; any use that treats the fictional APIs or the generated identifiers as real.

## Known limitations

- **Synthetic text.** Messages were written by an LLM following hints, not by real users. They cover the targeted phenomena reliably but are likely cleaner and more uniform than real chat input. There are no typos from speech-to-text, little slang beyond the hint inventory, and mostly Hindi-belt usage.
- **Verifier selection bias.** Train, dev and test candidates passed a round-trip check by Gemma 4 31B, gpt-oss-120b or Qwen3.8-27B. These models therefore score optimistically on this data and are reported separately from other baselines. Test candidates that every verifier rejected were included in human review to reduce this bias.
- **Single annotator.** The test set was verified by one person, so some label errors and some subjective decisions on ambiguous phrasings remain.
- **Time window.** Contexts fall between November 2026 and March 2027, and relative-date resolution follows the conventions stated in the system prompt.
- **Exact-match scoring** of dates, amounts and enums is strict by design. Free-text fields use fuzzy matching with listed paraphrases, which can still under-credit valid wordings.

## Distribution and maintenance

Released on the Hugging Face Hub under CC BY 4.0, with the generation code (MIT) in this repository. The data was generated through the Gemini and Groq APIs. Anyone planning commercial use should check those providers' terms on using model outputs. Issues and corrections can be reported through the GitHub repository.
