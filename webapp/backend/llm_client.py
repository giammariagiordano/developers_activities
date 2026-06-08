import json
import re
import asyncio
from openai import AsyncOpenAI
from typing import Optional
from collections import Counter

ACTIVITIES = ["Feature Introduction", "Bug Fixing", "Enhancement", "Refactoring"]

# ─── Prompt Templates ─────────────────────────────────────────────────────────

ZERO_SHOT_TEMPLATE = """\
You are an expert software engineer specializing in ML systems and technical debt analysis.

Analyze the following commit that introduced an ML-specific code smell into an ML-enabled system.

**Smell Type:** {smell_type}{smell_name_suffix}
**Commit Message:** {commit_message}
{issue_context}{pr_context}
**Code Diff:**
```
{diff}
```

Classify the developer activity that introduced this ML-specific code smell.
Choose exactly one primary activity:
- Feature Introduction: Adding new ML functionality or components
- Bug Fixing: Fixing a defect or incorrect behavior
- Enhancement: Improving or optimizing existing functionality
- Refactoring: Restructuring code without changing external behavior

Respond in valid JSON only:
{{"primary_activity": "<one of the four>", "sub_activity": "<e.g. performance optimization, new model integration>", "reasoning": "<1-2 sentence explanation>"}}"""


FEW_SHOT_TEMPLATE = """\
You are an expert software engineer specializing in ML systems and technical debt analysis.

Here are examples of how to classify developer activities that introduce ML-specific code smells:

EXAMPLE 1:
Smell: hyperparameters_not_explicitly_set
Commit Message: "Add LSTM model for time series forecasting"
Diff: +model = LSTM(hidden_size=128) +model.fit(X_train, y_train)
Classification: {{"primary_activity": "Feature Introduction", "sub_activity": "new ML model integration", "reasoning": "Developer added a new LSTM model without explicitly setting all hyperparameters, typical of initial feature implementation."}}

EXAMPLE 2:
Smell: gradients_not_cleared_before_backward_propagation
Commit Message: "Fix gradient accumulation bug in training loop"
Diff: -for batch in loader: -    loss.backward() +for batch in loader: +    optimizer.zero_grad() +    loss.backward()
Classification: {{"primary_activity": "Bug Fixing", "sub_activity": "gradient management fix", "reasoning": "Developer was fixing incorrect gradient behavior, accidentally left out zero_grad in one path."}}

EXAMPLE 3:
Smell: unnecessary_iteration
Commit Message: "Improve data preprocessing pipeline speed"
Diff: -for i in range(len(df)): -    df.iloc[i] = transform(df.iloc[i]) +df = df.apply(transform)
Classification: {{"primary_activity": "Enhancement", "sub_activity": "performance optimization", "reasoning": "Developer improved processing speed but introduced an iteration smell in an adjacent code path."}}

EXAMPLE 4:
Smell: memory_not_freed
Commit Message: "Refactor model inference module for cleaner architecture"
Diff: -class OldInference: -    def predict(self, x): ... +class ModelInference: +    def run(self, x): ...
Classification: {{"primary_activity": "Refactoring", "sub_activity": "code reorganization", "reasoning": "Developer restructured the inference module without fixing existing memory management issues."}}

Now classify the following:

**Smell Type:** {smell_type}{smell_name_suffix}
**Commit Message:** {commit_message}
{issue_context}{pr_context}
**Code Diff:**
```
{diff}
```

Respond in valid JSON only:
{{"primary_activity": "<one of the four>", "sub_activity": "<specific sub-activity>", "reasoning": "<1-2 sentence explanation>"}}"""


CHAIN_OF_THOUGHT_TEMPLATE = """\
You are an expert software engineer specializing in ML systems and technical debt analysis.

Analyze the following commit that introduced an ML-specific code smell.

**Smell Type:** {smell_type}{smell_name_suffix}
**Commit Message:** {commit_message}
{issue_context}{pr_context}
**Code Diff:**
```
{diff}
```

Think step by step:
1. What does the commit message indicate about the developer's intent?
2. What changes does the diff show? Are new features added, bugs fixed, or existing code restructured?
3. Is the smell incidental to the main change or central to it?
4. Which activity category best fits?

Activities:
- Feature Introduction: Adding new ML functionality
- Bug Fixing: Fixing a defect
- Enhancement: Improving existing functionality
- Refactoring: Restructuring without behavior change

Respond in valid JSON only (include your reasoning steps):
{{"primary_activity": "<one of the four>", "sub_activity": "<specific sub-activity>", "reasoning": "<step-by-step reasoning leading to classification>"}}"""


ROLE_PLAY_TEMPLATE = """\
You are a senior ML engineering lead conducting a post-mortem code review. Your task is to determine what a developer was trying to accomplish when they inadvertently introduced an ML-specific code smell.

**Context:**
- Smell Introduced: {smell_type}{smell_name_suffix}
- Developer's Commit Message: {commit_message}
{issue_context}{pr_context}
**Code Changes Made:**
```
{diff}
```

As a senior engineer, assess: what was this developer's primary goal when they made this change?

Categories (choose one):
- Feature Introduction: Developer was building/adding new ML functionality
- Bug Fixing: Developer was fixing a broken or incorrect behavior
- Enhancement: Developer was improving performance, usability, or maintainability
- Refactoring: Developer was restructuring code architecture

Respond in valid JSON only:
{{"primary_activity": "<one of the four>", "sub_activity": "<e.g. model architecture update, data pipeline improvement>", "reasoning": "<your professional assessment>"}}"""


PRESET_TEMPLATES = {
    "Zero-Shot": ZERO_SHOT_TEMPLATE,
    "Few-Shot": FEW_SHOT_TEMPLATE,
    "Chain-of-Thought": CHAIN_OF_THOUGHT_TEMPLATE,
    "Role-Play": ROLE_PLAY_TEMPLATE,
}

# ─── LLM Client ───────────────────────────────────────────────────────────────


def build_prompt(template: str, task_data: dict) -> str:
    smell_name_suffix = f" ({task_data['smell_type']})" if task_data.get("function_name") else ""
    issue_ctx = f"**Issue Context:** {task_data['issue_summary']}\n" if task_data.get("issue_summary") else ""
    pr_ctx = f"**PR Context:** {task_data['pr_summary']}\n" if task_data.get("pr_summary") else ""

    return template.format(
        diff=task_data.get("diff_content") or "[No diff available]",
        commit_message=task_data.get("commit_message") or "[No commit message]",
        smell_type=task_data.get("smell_type", "Unknown"),
        smell_name_suffix=smell_name_suffix,
        issue_context=issue_ctx,
        pr_context=pr_ctx,
    )


async def run_llm_query(
    prompt: str,
    model: str,
    temperature: float,
    api_key: str,
    max_retries: int = 3,
) -> dict:
    client = AsyncOpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
                max_tokens=600,
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
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = 2 ** attempt * 10
                await asyncio.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            elif attempt == max_retries - 1:
                raise

    raise RuntimeError("LLM query failed after retries")


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


def compute_majority_vote(results: list[dict]) -> dict:
    valid = [r["primary_activity"] for r in results if r.get("primary_activity")]
    if not valid:
        return {
            "primary_activity": None,
            "vote_count": 0,
            "total_votes": len(results),
            "tied": False,
            "tied_activities": [],
            "all_votes": {},
        }

    counts = Counter(valid)
    max_count = max(counts.values())
    winners = [act for act, cnt in counts.items() if cnt == max_count]

    return {
        "primary_activity": winners[0] if len(winners) == 1 else None,
        "vote_count": max_count,
        "total_votes": len(valid),
        "tied": len(winners) > 1,
        "tied_activities": winners if len(winners) > 1 else [],
        "all_votes": dict(counts),
    }
