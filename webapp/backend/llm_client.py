import json
import re
import asyncio
from openai import AsyncOpenAI
from typing import Optional

ACTIVITIES = ["Feature Introduction", "Bug Fixing", "Enhancement", "Refactoring"]

# ─── Prompt Templates ─────────────────────────────────────────────────────────

ZERO_SHOT_TEMPLATE = """\
You are an expert software engineer specializing in ML systems and technical debt analysis.

Your task: classify the **developer's primary goal** when making a commit that incidentally introduced an ML-specific code smell. The smell itself is not the activity — focus on what the developer was trying to achieve.

**Smell Type:** {smell_type}{smell_name_suffix}
**Commit Message:** {commit_message}
{issue_context}{pr_context}**Code Diff:**
```
{diff}
```

Choose exactly one primary activity:
- **Feature Introduction** — developer was adding new ML functionality, models, or pipeline components (signals: new files, new classes, new integrations)
- **Bug Fixing** — developer was correcting a defect, incorrect behavior, or failing test (signals: targeted line corrections, error handling, test fixes)
- **Enhancement** — developer was improving performance, efficiency, or usability of existing functionality (signals: algorithmic changes, vectorization, config tuning)
- **Refactoring** — developer was restructuring, renaming, or reorganizing code without changing behavior (signals: renames, moves, interface cleanup, no logic change)

If the commit message conflicts with the diff, prioritize the diff. If ambiguous between two categories, choose the one most directly supported by the diff structure.

Respond in valid JSON only — no markdown, no text outside the JSON object:
{{"primary_activity": "<one of the four>", "sub_activity": "<specific sub-activity, e.g. 'new model integration', 'memory optimization'>", "reasoning": "<1-2 sentences citing specific evidence from the commit message or diff>"}}"""


ONE_SHOT_TEMPLATE = """\
You are an expert software engineer specializing in ML systems and technical debt analysis.

Your task: classify the **developer's primary goal** when making a commit that incidentally introduced an ML-specific code smell. The smell itself is not the activity — focus on what the developer was trying to achieve.

Study this labeled example before classifying:

---
EXAMPLE:
Smell: hyperparameters_not_explicitly_set
Commit Message: "Add LSTM model for time series forecasting"
Diff:
+class LSTMForecaster:
+    def __init__(self):
+        self.model = LSTM(hidden_size=128)
+    def fit(self, X, y):
+        self.model.fit(X, y)
+pipeline.add_step(LSTMForecaster())
Classification: {{"primary_activity": "Feature Introduction", "sub_activity": "new ML model integration", "reasoning": "Commit message explicitly states a new model is being added; the diff confirms new class definition and pipeline wiring with no pre-existing code modified — the missing hyperparameter defaults are incidental to the feature work."}}
---

Now classify:

**Smell Type:** {smell_type}{smell_name_suffix}
**Commit Message:** {commit_message}
{issue_context}{pr_context}**Code Diff:**
```
{diff}
```

Choose exactly one primary activity:
- **Feature Introduction** — developer was adding new ML functionality, models, or pipeline components (signals: new files, new classes, new integrations)
- **Bug Fixing** — developer was correcting a defect, incorrect behavior, or failing test (signals: targeted line corrections, error handling, test fixes)
- **Enhancement** — developer was improving performance, efficiency, or usability of existing functionality (signals: algorithmic changes, vectorization, config tuning)
- **Refactoring** — developer was restructuring, renaming, or reorganizing code without changing behavior (signals: renames, moves, interface cleanup, no logic change)

If the commit message conflicts with the diff, prioritize the diff. If ambiguous between two categories, choose the one most directly supported by the diff structure.

Respond in valid JSON only — no markdown, no text outside the JSON object:
{{"primary_activity": "<one of the four>", "sub_activity": "<specific sub-activity>", "reasoning": "<1-2 sentences citing specific evidence from the commit message or diff>"}}"""


TWO_SHOT_TEMPLATE = """\
You are an expert software engineer specializing in ML systems and technical debt analysis.

Your task: classify the **developer's primary goal** when making a commit that incidentally introduced an ML-specific code smell. The smell itself is not the activity — focus on what the developer was trying to achieve.

Study these two labeled examples before classifying:

---
EXAMPLE 1:
Smell: gradients_not_cleared_before_backward_propagation
Commit Message: "Fix gradient accumulation bug in training loop"
Diff:
 for batch in loader:
-    loss = criterion(output, target)
-    loss.backward()
+    optimizer.zero_grad()
+    loss = criterion(output, target)
+    loss.backward()
Classification: {{"primary_activity": "Bug Fixing", "sub_activity": "gradient management fix", "reasoning": "Commit message explicitly names a bug; the diff shows a targeted single-line insertion to correct a known defect pattern — the smell was introduced incidentally in an adjacent code path, not as the main goal."}}

EXAMPLE 2:
Smell: memory_not_freed
Commit Message: "Refactor model inference module for cleaner architecture"
Diff:
-class OldInference:
-    def predict(self, x):
-        result = self.model(x)
-        return result
+class ModelInference:
+    def run(self, x):
+        result = self.model(x)
+        return result
Classification: {{"primary_activity": "Refactoring", "sub_activity": "code reorganization", "reasoning": "Commit message explicitly states 'refactor'; the diff shows pure rename and restructure with identical logic — the memory management smell pre-existed and was not introduced as part of a functional change."}}
---

Now classify:

**Smell Type:** {smell_type}{smell_name_suffix}
**Commit Message:** {commit_message}
{issue_context}{pr_context}**Code Diff:**
```
{diff}
```

Choose exactly one primary activity:
- **Feature Introduction** — developer was adding new ML functionality, models, or pipeline components (signals: new files, new classes, new integrations)
- **Bug Fixing** — developer was correcting a defect, incorrect behavior, or failing test (signals: targeted line corrections, error handling, test fixes)
- **Enhancement** — developer was improving performance, efficiency, or usability of existing functionality (signals: algorithmic changes, vectorization, config tuning)
- **Refactoring** — developer was restructuring, renaming, or reorganizing code without changing behavior (signals: renames, moves, interface cleanup, no logic change)

If the commit message conflicts with the diff, prioritize the diff. If ambiguous between two categories, choose the one most directly supported by the diff structure.

Respond in valid JSON only — no markdown, no text outside the JSON object:
{{"primary_activity": "<one of the four>", "sub_activity": "<specific sub-activity>", "reasoning": "<1-2 sentences citing specific evidence from the commit message or diff>"}}"""


CHAIN_OF_THOUGHT_TEMPLATE = """\
You are an expert software engineer specializing in ML systems and technical debt analysis.

Your task: classify the **developer's primary goal** when making a commit that incidentally introduced an ML-specific code smell. Work through your reasoning explicitly before committing to a conclusion.

**Smell Type:** {smell_type}{smell_name_suffix}
**Commit Message:** {commit_message}
{issue_context}{pr_context}**Code Diff:**
```
{diff}
```

Reason through the following steps in order:
1. **Intent signal**: What does the commit message explicitly state? Identify trigger words: "add"/"implement" → Feature Introduction; "fix"/"correct"/"resolve" → Bug Fixing; "improve"/"optimize"/"speed up" → Enhancement; "refactor"/"rename"/"reorganize" → Refactoring.
2. **Diff evidence**: Is code being added (new files/classes/functions), corrected (targeted line changes to existing logic), optimized (algorithmic or config changes), or reorganized (renames/moves with equivalent behavior)?
3. **Smell relationship**: Is the smell the direct target of this change, or was it introduced incidentally while pursuing the main goal? This distinction is critical.
4. **Alternative check**: Identify the second-most-likely category and explain why the primary choice is stronger.
5. **Final verdict**: Which category best fits, given all evidence?

Activities:
- **Feature Introduction** — adding new ML functionality, models, or pipeline components
- **Bug Fixing** — correcting a defect, incorrect behavior, or failing test
- **Enhancement** — improving performance, efficiency, or usability of existing functionality
- **Refactoring** — restructuring, renaming, or reorganizing without behavior change

Respond in valid JSON only — embed your full step-by-step reasoning in the "reasoning" field:
{{"primary_activity": "<one of the four>", "sub_activity": "<specific sub-activity>", "reasoning": "<numbered step-by-step reasoning from evidence to conclusion>"}}"""


ROLE_PLAY_TEMPLATE = """\
You are a senior ML engineering lead with 10+ years of experience, conducting a post-mortem code review. A developer on your team introduced an ML-specific code smell during a recent commit. Your goal is to understand their primary intent — not to judge the smell, but to reconstruct the development context that produced it.

**Incident Details:**
- Smell Introduced: {smell_type}{smell_name_suffix}
- Developer's Commit Message: {commit_message}
{issue_context}{pr_context}**Code Changes:**
```
{diff}
```

From your experience: competent developers rarely introduce smells deliberately. Consider what goal, deadline pressure, or task context would lead a developer to make exactly these changes — and what the structure of the diff reveals about their focus at the time.

Classify their primary goal (choose one):
- **Feature Introduction** — developer was building or adding new ML functionality, models, or pipeline components
- **Bug Fixing** — developer was correcting a defect, incorrect behavior, or failing test
- **Enhancement** — developer was improving performance, efficiency, or maintainability of existing code
- **Refactoring** — developer was restructuring architecture or code organization without behavior change

Respond in valid JSON only — no markdown, no text outside the JSON object:
{{"primary_activity": "<one of the four>", "sub_activity": "<specific sub-activity, e.g. 'model architecture update', 'training loop optimization'>", "reasoning": "<your professional assessment citing specific evidence from the commit message and diff>"}}"""


PRESET_TEMPLATES = {
    "Zero-Shot": ZERO_SHOT_TEMPLATE,
    "One-Shot": ONE_SHOT_TEMPLATE,
    "Two-Shot": TWO_SHOT_TEMPLATE,
    "Chain-of-Thought": CHAIN_OF_THOUGHT_TEMPLATE,
    "Role-Play": ROLE_PLAY_TEMPLATE,
}

# ─── Prompt Builder ───────────────────────────────────────────────────────────


def _escape_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


def build_prompt(template: str, task_data: dict) -> str:
    smell_name_suffix = f" ({task_data['smell_type']})" if task_data.get("function_name") else ""
    issue_ctx = f"**Issue Context:** {_escape_braces(task_data['issue_summary'])}\n" if task_data.get("issue_summary") else ""
    pr_ctx = f"**PR Context:** {_escape_braces(task_data['pr_summary'])}\n" if task_data.get("pr_summary") else ""

    return template.format(
        diff=_escape_braces(task_data.get("diff_content") or "[No diff available]"),
        commit_message=_escape_braces(task_data.get("commit_message") or "[No commit message]"),
        smell_type=_escape_braces(task_data.get("smell_type", "Unknown")),
        smell_name_suffix=_escape_braces(smell_name_suffix),
        issue_context=issue_ctx,
        pr_context=pr_ctx,
    )


def build_aggregator_prompt(smell_data: dict, responses: list[dict]) -> str:
    smell_type = smell_data.get("smell_type", "Unknown")
    commit_msg = smell_data.get("commit_message") or "[No commit message]"
    diff_excerpt = (smell_data.get("diff_content") or "[No diff]")[:800]

    model_blocks = ""
    for i, r in enumerate(responses, 1):
        candidates = r.get("sub_activity_candidates") or []
        if len(candidates) > 1:
            sub_line = f"{candidates[0]} (candidates: {', '.join(candidates)})"
        else:
            sub_line = r.get('sub_activity') or '—'
        model_blocks += f"""
Model {i} ({r['model_name']}):
  primary_activity: {r.get('primary_activity') or 'Unknown'}
  sub_activity: {sub_line}
  reasoning: {r.get('reasoning') or '—'}
"""

    n_models = len(responses)
    return f"""\
You are an expert software engineering researcher acting as a meta-classifier.

{n_models} independent AI models have analyzed the following ML code smell commit and produced their classifications. Your task is to synthesize their analyses into a single, well-reasoned final classification.

**Smell Type:** {smell_type}
**Commit Message:** {commit_msg}
**Code Diff (excerpt):**
```
{diff_excerpt}
```

**Model Classifications:**
{model_blocks}

Synthesize these classifications into one authoritative verdict. Follow this procedure:
1. **Evaluate reasoning quality**: prefer reasoning that cites specific evidence from the commit message or diff over reasoning that is vague or generic.
2. **Resolve disagreements by evidence**: if models disagree, identify which reasoning is best supported by the diff structure — not which label appears most often.
3. **Apply tiebreaker**: majority label wins only when reasoning quality is equal across disagreeing models.
4. **Harmonize sub_activity**: review all sub_activity values and candidates across models. Merge synonyms and overlapping descriptions into a single canonical label (2-4 words, lowercase, singular). Example: "gradient fix", "fixing gradient bug", "gradient management fix" → "gradient management fix".

Choose exactly one primary activity:
- **Feature Introduction** — developer was adding new ML functionality, models, or pipeline components (signals: new files, new classes, new integrations)
- **Bug Fixing** — developer was correcting a defect, incorrect behavior, or failing test (signals: targeted line corrections, error handling, test fixes)
- **Enhancement** — developer was improving performance, efficiency, or usability of existing functionality (signals: algorithmic changes, vectorization, config tuning)
- **Refactoring** — developer was restructuring, renaming, or reorganizing code without changing behavior (signals: renames, moves, interface cleanup, no logic change)

Respond in valid JSON only — no markdown, no text outside the JSON object:
{{"primary_activity": "<one of the four>", "sub_activity": "<single canonical label, 2-4 words, lowercase, singular>", "reasoning": "<which model(s) you followed, why their reasoning was strongest, and what evidence supports the final verdict>"}}"""


# ─── Ollama / OpenAI-compatible Client ────────────────────────────────────────


def _make_client(base_url: str, api_key: str = "ollama") -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key=api_key,
    )


async def _sleep_for_ratelimit(e: Exception, attempt: int) -> None:
    """Wait longer on 429 rate-limit errors (Groq / OpenAI), normal backoff otherwise."""
    msg = str(e).lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        await asyncio.sleep(60)
    else:
        await asyncio.sleep(2 ** attempt * 2)


async def run_ollama_query(
    prompt: str,
    model: str,
    temperature: float,
    base_url: str = "http://localhost:11434",
    api_key: str = "ollama",
    max_retries: int = 3,
) -> dict:
    client = _make_client(base_url, api_key)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=900,
            )

            raw = response.choices[0].message.content or ""
            parsed = _parse_json(raw)

            return {
                "primary_activity": _normalize_activity(parsed.get("primary_activity", "")),
                "sub_activity": str(parsed.get("sub_activity", "")),
                "reasoning": str(parsed.get("reasoning", "")),
                "raw_response": raw,
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            }

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await _sleep_for_ratelimit(e, attempt)

    raise RuntimeError(f"LLM query failed after {max_retries} retries")


async def run_aggregator(
    smell_data: dict,
    responses: list[dict],
    model: str,
    temperature: float,
    base_url: str = "http://localhost:11434",
    api_key: str = "ollama",
    max_retries: int = 3,
) -> dict:
    prompt = build_aggregator_prompt(smell_data, responses)
    client = _make_client(base_url, api_key)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=800,
            )

            raw = response.choices[0].message.content or ""
            parsed = _parse_json(raw)

            return {
                "primary_activity": _normalize_activity(parsed.get("primary_activity", "")),
                "sub_activity": str(parsed.get("sub_activity", "")),
                "reasoning": str(parsed.get("reasoning", "")),
                "raw_response": raw,
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            }

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await _sleep_for_ratelimit(e, attempt)

    raise RuntimeError(f"Aggregator query failed after {max_retries} retries")


async def run_normalization_batch(
    labels: list[str],
    model: str,
    base_url: str = "http://localhost:11434",
    api_key: str = "ollama",
    max_retries: int = 3,
) -> dict[str, str]:
    label_list = "\n".join(f"- {l}" for l in labels)
    prompt = f"""You are a research assistant normalizing software engineering activity labels for a scientific study.

Below is a list of sub-activity labels. Assign a single canonical label to each.

Labels:
{label_list}

STRICT RULES (no exceptions):
1. Canonical labels MUST be singular (never plural): "test case" not "test cases", "memory fix" not "memory fixes"
2. Canonical labels MUST be lowercase, 2-4 words max
3. Be VERY aggressive in merging: if two labels describe the same concept even with different wording, they get the same canonical
4. Merge labels that differ only in: plural/singular, capitalization, article (a/the), specific class/file names ("for ReportGenerator" vs "for Inspector" → same canonical if both mean "adding test cases")
5. Prefer the most generic version when merging specifics: "adding test cases for ReportGenerator" → "test case addition"
6. Output every input label exactly once

Respond in valid JSON only:
{{"mappings": [{{"raw": "<original label>", "canonical": "<normalized label>"}}, ...]}}"""

    client = _make_client(base_url, api_key)
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            parsed = _parse_json(raw)
            mappings = parsed.get("mappings", [])
            return {m["raw"]: m["canonical"] for m in mappings if "raw" in m and "canonical" in m}
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await _sleep_for_ratelimit(e, attempt)
    return {}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {}


def _normalize_activity(value: str) -> Optional[str]:
    val = value.lower().strip()
    for act in ACTIVITIES:
        if act.lower() in val or val in act.lower():
            return act
    return None
