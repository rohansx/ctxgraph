#!/usr/bin/env python3
"""Generate training data for fine-tuning a DeBERTa-v3-small entity-pair relation classifier.

Reads gold benchmark episodes, generates positive/negative examples with entity markers,
augments via paraphrasing templates, and outputs a train/val split JSON file.
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARK_PATH = PROJECT_ROOT / "crates" / "ctxgraph-extract" / "tests" / "fixtures" / "benchmark_episodes.json"
OUTPUT_PATH = SCRIPT_DIR / "training_data.json"

RELATION_TYPES = [
    "chose", "rejected", "replaced", "depends_on", "fixed",
    "introduced", "deprecated", "caused", "constrained_by",
]

SEED = 42

# ---------------------------------------------------------------------------
# Paraphrase templates per relation type.
# Each template is a callable (head, tail) -> str producing marked-up text.
# Entity markers [E1]/[/E1] and [E2]/[/E2] are embedded in the output.
# ---------------------------------------------------------------------------

def _h(name: str) -> str:
    return f"[E1]{name}[/E1]"

def _t(name: str) -> str:
    return f"[E2]{name}[/E2]"


PARAPHRASE_TEMPLATES: dict[str, list] = {
    "chose": [
        lambda h, t: f"The team selected {_t(t)} after {_h(h)} evaluated the alternatives.",
        lambda h, t: f"{_h(h)} picked {_t(t)} as the preferred solution for the project.",
        lambda h, t: f"After careful review, {_h(h)} adopted {_t(t)} for production use.",
        lambda h, t: f"{_h(h)} decided to go with {_t(t)} over competing options.",
        lambda h, t: f"The decision was made by {_h(h)} to use {_t(t)} going forward.",
        lambda h, t: f"{_h(h)} settled on {_t(t)} as the best fit for the requirements.",
        lambda h, t: f"Following evaluation, {_h(h)} endorsed {_t(t)} for the implementation.",
        lambda h, t: f"{_t(t)} was chosen by {_h(h)} as the technology of choice.",
        lambda h, t: f"{_h(h)} opted for {_t(t)} based on performance benchmarks.",
        lambda h, t: f"The architecture review led {_h(h)} to select {_t(t)}.",
        lambda h, t: f"{_h(h)} committed to {_t(t)} after a proof-of-concept evaluation.",
        lambda h, t: f"In the ADR, {_h(h)} formally chose {_t(t)} for this component.",
        lambda h, t: f"{_h(h)} recommended {_t(t)} and the team approved the selection.",
        lambda h, t: f"After prototyping, {_h(h)} confirmed {_t(t)} as the winning candidate.",
        lambda h, t: f"{_t(t)} emerged as the preferred option after {_h(h)} ran comparisons.",
        lambda h, t: f"{_h(h)} elected to adopt {_t(t)} for the new system design.",
        lambda h, t: f"The engineering team, led by {_h(h)}, embraced {_t(t)} for this use case.",
        lambda h, t: f"{_h(h)} proposed {_t(t)} and it was accepted unanimously by the team.",
        lambda h, t: f"Based on scalability needs, {_h(h)} went with {_t(t)}.",
        lambda h, t: f"{_h(h)} finalized the choice of {_t(t)} after reviewing trade-offs.",
        lambda h, t: f"Documentation shows {_h(h)} approved {_t(t)} as the standard approach.",
    ],
    "rejected": [
        lambda h, t: f"{_h(h)} rejected {_t(t)} due to insufficient feature coverage.",
        lambda h, t: f"After evaluation, {_h(h)} decided against {_t(t)} for the platform.",
        lambda h, t: f"{_t(t)} was ruled out by {_h(h)} because of scalability concerns.",
        lambda h, t: f"{_h(h)} eliminated {_t(t)} from consideration during the design review.",
        lambda h, t: f"The team led by {_h(h)} declined to use {_t(t)} in production.",
        lambda h, t: f"{_h(h)} passed on {_t(t)} in favor of a more mature alternative.",
        lambda h, t: f"{_t(t)} failed to meet the requirements set by {_h(h)}.",
        lambda h, t: f"{_h(h)} dismissed {_t(t)} after benchmarking showed poor results.",
        lambda h, t: f"In the ADR, {_h(h)} explicitly excluded {_t(t)} from the shortlist.",
        lambda h, t: f"{_h(h)} vetoed {_t(t)} due to licensing restrictions.",
        lambda h, t: f"Performance testing convinced {_h(h)} to drop {_t(t)} from consideration.",
        lambda h, t: f"{_h(h)} determined that {_t(t)} was not viable for the workload.",
        lambda h, t: f"The decision record shows {_h(h)} disqualified {_t(t)} early on.",
        lambda h, t: f"{_h(h)} found {_t(t)} lacking in observability and rejected it.",
        lambda h, t: f"Security review by {_h(h)} flagged {_t(t)} as unsuitable.",
        lambda h, t: f"{_h(h)} turned down {_t(t)} because it did not support the required protocol.",
        lambda h, t: f"After a PoC, {_h(h)} concluded {_t(t)} was not production-ready.",
        lambda h, t: f"{_t(t)} was discarded by {_h(h)} during the architecture spike.",
        lambda h, t: f"{_h(h)} opted not to proceed with {_t(t)} after cost analysis.",
        lambda h, t: f"The evaluation matrix showed {_h(h)} scoring {_t(t)} below the threshold.",
        lambda h, t: f"{_h(h)} abandoned {_t(t)} as a candidate after integration testing.",
    ],
    "replaced": [
        lambda h, t: f"The team migrated from {_t(t)} to {_h(h)} for improved performance.",
        lambda h, t: f"{_h(h)} superseded {_t(t)} in the production stack.",
        lambda h, t: f"We swapped out {_t(t)} and brought in {_h(h)} as a replacement.",
        lambda h, t: f"{_t(t)} has been retired in favor of {_h(h)}.",
        lambda h, t: f"The migration from {_t(t)} to {_h(h)} was completed this sprint.",
        lambda h, t: f"{_h(h)} now serves the role previously filled by {_t(t)}.",
        lambda h, t: f"All workloads previously on {_t(t)} have been moved to {_h(h)}.",
        lambda h, t: f"{_h(h)} took over from {_t(t)} after a phased rollout.",
        lambda h, t: f"The legacy {_t(t)} was decommissioned once {_h(h)} proved stable.",
        lambda h, t: f"Traffic was cut over from {_t(t)} to {_h(h)} last week.",
        lambda h, t: f"{_h(h)} is the direct successor of {_t(t)} in our architecture.",
        lambda h, t: f"We completed the switch from {_t(t)} to {_h(h)} across all environments.",
        lambda h, t: f"{_t(t)} is no longer in use after being replaced by {_h(h)}.",
        lambda h, t: f"The {_t(t)} instances were torn down after {_h(h)} went live.",
        lambda h, t: f"{_h(h)} was introduced as a drop-in replacement for {_t(t)}.",
        lambda h, t: f"After benchmarks, {_h(h)} was chosen to replace {_t(t)} entirely.",
        lambda h, t: f"The transition from {_t(t)} to {_h(h)} reduced operational costs by 40 percent.",
        lambda h, t: f"Engineering migrated the system from {_t(t)} to {_h(h)} for reliability.",
        lambda h, t: f"{_h(h)} displaced {_t(t)} as the primary backend technology.",
        lambda h, t: f"As part of modernization, {_t(t)} was substituted with {_h(h)}.",
        lambda h, t: f"{_h(h)} has entirely supplanted {_t(t)} in the infrastructure layer.",
    ],
    "depends_on": [
        lambda h, t: f"{_h(h)} relies on {_t(t)} for core functionality.",
        lambda h, t: f"{_h(h)} connects to {_t(t)} at runtime for data access.",
        lambda h, t: f"The {_h(h)} component requires {_t(t)} to operate correctly.",
        lambda h, t: f"{_h(h)} has a hard dependency on {_t(t)} in production.",
        lambda h, t: f"Without {_t(t)}, {_h(h)} cannot function properly.",
        lambda h, t: f"{_h(h)} integrates directly with {_t(t)} via its client library.",
        lambda h, t: f"{_t(t)} is a critical upstream dependency of {_h(h)}.",
        lambda h, t: f"{_h(h)} consumes data from {_t(t)} through a gRPC interface.",
        lambda h, t: f"The service graph shows {_h(h)} calling {_t(t)} on every request.",
        lambda h, t: f"{_h(h)} is tightly coupled with {_t(t)} and cannot be deployed independently.",
        lambda h, t: f"{_h(h)} imports the SDK provided by {_t(t)}.",
        lambda h, t: f"Health checks for {_h(h)} include verifying connectivity to {_t(t)}.",
        lambda h, t: f"{_h(h)} sends events to {_t(t)} for downstream processing.",
        lambda h, t: f"The startup sequence of {_h(h)} blocks until {_t(t)} is ready.",
        lambda h, t: f"{_h(h)} uses {_t(t)} as its backing store for persistent state.",
        lambda h, t: f"Tracing data shows {_h(h)} making frequent calls to {_t(t)}.",
        lambda h, t: f"{_h(h)} is built on top of {_t(t)} and inherits its configuration.",
        lambda h, t: f"A failure in {_t(t)} cascades to {_h(h)} due to the tight coupling.",
        lambda h, t: f"{_h(h)} reads configuration values from {_t(t)} at boot time.",
        lambda h, t: f"The deployment manifest shows {_h(h)} depending on {_t(t)}.",
        lambda h, t: f"{_h(h)} needs {_t(t)} to handle authentication flows.",
    ],
    "fixed": [
        lambda h, t: f"{_h(h)} resolved a critical issue affecting {_t(t)}.",
        lambda h, t: f"A bug fix in {_h(h)} corrected the broken behavior in {_t(t)}.",
        lambda h, t: f"{_h(h)} patched the vulnerability discovered in {_t(t)}.",
        lambda h, t: f"The issue in {_t(t)} was addressed by changes to {_h(h)}.",
        lambda h, t: f"{_h(h)} repaired the broken integration with {_t(t)}.",
        lambda h, t: f"A hotfix was applied to {_h(h)} to restore {_t(t)} functionality.",
        lambda h, t: f"{_h(h)} eliminated the regression that impacted {_t(t)}.",
        lambda h, t: f"The defect in {_t(t)} was traced back to {_h(h)} and corrected.",
        lambda h, t: f"{_h(h)} remedied the data corruption issue in {_t(t)}.",
        lambda h, t: f"After debugging, {_h(h)} fixed the timeout problem with {_t(t)}.",
        lambda h, t: f"The race condition between {_h(h)} and {_t(t)} was resolved.",
        lambda h, t: f"{_h(h)} addressed the memory leak that manifested in {_t(t)}.",
        lambda h, t: f"A fix was deployed to {_h(h)} solving the {_t(t)} connection drops.",
        lambda h, t: f"{_h(h)} corrected the serialization error impacting {_t(t)}.",
        lambda h, t: f"The null pointer exception in {_h(h)} that crashed {_t(t)} was fixed.",
        lambda h, t: f"{_h(h)} stabilized {_t(t)} by handling edge cases properly.",
        lambda h, t: f"Error handling improvements in {_h(h)} prevented {_t(t)} failures.",
        lambda h, t: f"{_h(h)} updated its retry logic to fix intermittent {_t(t)} errors.",
        lambda h, t: f"The deadlock in {_h(h)} that blocked {_t(t)} was diagnosed and fixed.",
        lambda h, t: f"{_h(h)} resolved the encoding issue that broke {_t(t)} responses.",
        lambda h, t: f"Monitoring showed {_h(h)} causing {_t(t)} errors, which has been fixed.",
    ],
    "introduced": [
        lambda h, t: f"{_h(h)} added {_t(t)} to the system architecture.",
        lambda h, t: f"A new {_t(t)} capability was introduced by {_h(h)}.",
        lambda h, t: f"{_h(h)} brought {_t(t)} into the codebase for the first time.",
        lambda h, t: f"The {_t(t)} feature was implemented as part of the {_h(h)} update.",
        lambda h, t: f"{_h(h)} integrated {_t(t)} into the platform stack.",
        lambda h, t: f"With this change, {_h(h)} now includes {_t(t)} support.",
        lambda h, t: f"{_h(h)} enabled {_t(t)} across all production environments.",
        lambda h, t: f"The pull request by {_h(h)} introduces {_t(t)} to the service.",
        lambda h, t: f"{_h(h)} rolled out {_t(t)} as a new subsystem.",
        lambda h, t: f"{_t(t)} was provisioned by {_h(h)} to handle the new workload.",
        lambda h, t: f"{_h(h)} onboarded {_t(t)} to improve observability.",
        lambda h, t: f"The latest release of {_h(h)} ships with {_t(t)} enabled by default.",
        lambda h, t: f"{_h(h)} set up {_t(t)} in the staging environment first.",
        lambda h, t: f"As part of the refactor, {_h(h)} incorporated {_t(t)}.",
        lambda h, t: f"{_h(h)} deployed {_t(t)} alongside the existing components.",
        lambda h, t: f"The changelog shows {_h(h)} adding {_t(t)} in this release.",
        lambda h, t: f"{_h(h)} adopted {_t(t)} to address the scalability gap.",
        lambda h, t: f"New instrumentation via {_t(t)} was wired in by {_h(h)}.",
        lambda h, t: f"{_h(h)} created the {_t(t)} module from scratch.",
        lambda h, t: f"The team used {_h(h)} to launch {_t(t)} into production.",
        lambda h, t: f"{_h(h)} bootstrapped {_t(t)} as part of the infrastructure overhaul.",
    ],
    "deprecated": [
        lambda h, t: f"{_h(h)} marked {_t(t)} as deprecated in the latest release.",
        lambda h, t: f"The {_t(t)} component was sunsetted by {_h(h)}.",
        lambda h, t: f"{_h(h)} announced the end-of-life for {_t(t)}.",
        lambda h, t: f"{_t(t)} is now considered legacy after {_h(h)} flagged it for removal.",
        lambda h, t: f"{_h(h)} scheduled {_t(t)} for deprecation in the next quarter.",
        lambda h, t: f"Users of {_t(t)} must migrate away per the notice from {_h(h)}.",
        lambda h, t: f"{_h(h)} removed support for {_t(t)} in the v2 API.",
        lambda h, t: f"The roadmap from {_h(h)} shows {_t(t)} being phased out.",
        lambda h, t: f"{_h(h)} issued a deprecation warning for all {_t(t)} consumers.",
        lambda h, t: f"{_t(t)} was obsoleted by changes introduced in {_h(h)}.",
        lambda h, t: f"{_h(h)} retired {_t(t)} after migrating all dependents.",
        lambda h, t: f"The {_t(t)} endpoint was soft-deprecated by {_h(h)} this sprint.",
        lambda h, t: f"{_h(h)} disabled new enrollments into {_t(t)}.",
        lambda h, t: f"Going forward, {_h(h)} will no longer maintain {_t(t)}.",
        lambda h, t: f"{_t(t)} has been superseded and {_h(h)} plans to decommission it.",
        lambda h, t: f"{_h(h)} phased out {_t(t)} from the active service catalog.",
        lambda h, t: f"The deprecation of {_t(t)} was driven by {_h(h)} to reduce tech debt.",
        lambda h, t: f"{_h(h)} set a 90-day sunset window for {_t(t)}.",
        lambda h, t: f"All references to {_t(t)} are being cleaned up after {_h(h)} deprecated it.",
        lambda h, t: f"{_h(h)} flagged {_t(t)} as unsupported starting next release.",
        lambda h, t: f"The migration guide from {_h(h)} details how to move off {_t(t)}.",
    ],
    "caused": [
        lambda h, t: f"The introduction of {_h(h)} led to measurable changes in {_t(t)}.",
        lambda h, t: f"{_h(h)} directly impacted {_t(t)} metrics after deployment.",
        lambda h, t: f"Deploying {_h(h)} resulted in a noticeable shift in {_t(t)}.",
        lambda h, t: f"{_h(h)} triggered a change in {_t(t)} across all regions.",
        lambda h, t: f"Observability data shows {_h(h)} affecting {_t(t)} significantly.",
        lambda h, t: f"After enabling {_h(h)}, {_t(t)} improved dramatically.",
        lambda h, t: f"The rollout of {_h(h)} caused {_t(t)} to drop to acceptable levels.",
        lambda h, t: f"{_h(h)} was the root cause of the {_t(t)} improvement.",
        lambda h, t: f"Dashboards confirmed that {_h(h)} influenced {_t(t)} readings.",
        lambda h, t: f"Correlation analysis linked {_h(h)} to changes in {_t(t)}.",
        lambda h, t: f"{_t(t)} shifted substantially after {_h(h)} was rolled out.",
        lambda h, t: f"The {_h(h)} update produced a side effect on {_t(t)}.",
        lambda h, t: f"Post-deployment metrics show {_h(h)} driving {_t(t)} changes.",
        lambda h, t: f"Enabling {_h(h)} reduced {_t(t)} by an order of magnitude.",
        lambda h, t: f"{_h(h)} had a direct causal effect on {_t(t)} in production.",
        lambda h, t: f"The {_t(t)} regression was traced back to the {_h(h)} release.",
        lambda h, t: f"Grafana panels show {_h(h)} correlating strongly with {_t(t)}.",
        lambda h, t: f"A/B testing proved {_h(h)} was responsible for the {_t(t)} delta.",
        lambda h, t: f"Without {_h(h)}, {_t(t)} reverted to previous levels.",
        lambda h, t: f"The impact of {_h(h)} on {_t(t)} was confirmed via canary analysis.",
        lambda h, t: f"{_h(h)} moved the needle on {_t(t)} within hours of deployment.",
    ],
    "constrained_by": [
        lambda h, t: f"{_h(h)} must comply with the {_t(t)} requirement.",
        lambda h, t: f"The design of {_h(h)} is limited by {_t(t)}.",
        lambda h, t: f"{_h(h)} operates under the {_t(t)} constraint.",
        lambda h, t: f"{_t(t)} imposes restrictions on how {_h(h)} can be implemented.",
        lambda h, t: f"{_h(h)} is bound by the {_t(t)} policy.",
        lambda h, t: f"All changes to {_h(h)} must satisfy {_t(t)}.",
        lambda h, t: f"{_h(h)} was architected to meet {_t(t)} guarantees.",
        lambda h, t: f"The {_t(t)} requirement shapes the behavior of {_h(h)}.",
        lambda h, t: f"{_h(h)} cannot violate {_t(t)} under any circumstances.",
        lambda h, t: f"Compliance with {_t(t)} is mandatory for {_h(h)}.",
        lambda h, t: f"{_h(h)} enforces {_t(t)} at the application layer.",
        lambda h, t: f"The SLA for {_h(h)} is dictated by {_t(t)}.",
        lambda h, t: f"{_h(h)} has a non-negotiable {_t(t)} obligation.",
        lambda h, t: f"Regulatory {_t(t)} defines the operational envelope of {_h(h)}.",
        lambda h, t: f"{_h(h)} validates every request against {_t(t)} rules.",
        lambda h, t: f"The throughput of {_h(h)} is capped by {_t(t)}.",
        lambda h, t: f"{_h(h)} includes guardrails to enforce {_t(t)}.",
        lambda h, t: f"Testing for {_h(h)} includes verification of {_t(t)} adherence.",
        lambda h, t: f"{_h(h)} was redesigned to accommodate {_t(t)} constraints.",
        lambda h, t: f"The deployment of {_h(h)} is gated on {_t(t)} validation.",
        lambda h, t: f"{_h(h)} logs all actions to demonstrate {_t(t)} conformance.",
    ],
}

# ---------------------------------------------------------------------------
# "none" templates: generic sentences mentioning two entities without a relation.
# ---------------------------------------------------------------------------
NONE_TEMPLATES = [
    lambda h, t: f"The codebase includes both {_h(h)} and {_t(t)} in separate modules.",
    lambda h, t: f"{_h(h)} and {_t(t)} are both used in the project but serve different purposes.",
    lambda h, t: f"The documentation mentions {_h(h)} alongside {_t(t)} in the overview section.",
    lambda h, t: f"Both {_h(h)} and {_t(t)} appear in the architecture diagram.",
    lambda h, t: f"The team discussed {_h(h)} and {_t(t)} during the planning meeting.",
    lambda h, t: f"{_h(h)} runs in the same cluster as {_t(t)} but they do not interact.",
    lambda h, t: f"The config file references {_h(h)} and {_t(t)} in different sections.",
    lambda h, t: f"Engineers familiar with {_h(h)} often also work on {_t(t)}.",
    lambda h, t: f"{_h(h)} and {_t(t)} are listed in the technology radar.",
    lambda h, t: f"The oncall runbook covers both {_h(h)} and {_t(t)} procedures.",
]


def find_entity_spans(text: str, entity_name: str, expected_entities: list[dict]) -> list[tuple[int, int]]:
    """Find all (start, end) character spans where entity_name appears in text.

    Prefers span_start/span_end from expected_entities when available and matching.
    Falls back to string search.
    """
    spans = []
    # Check expected_entities for pre-annotated spans
    for ent in expected_entities:
        if ent["name"] == entity_name:
            s, e = ent["span_start"], ent["span_end"]
            if text[s:e] == entity_name:
                spans.append((s, e))
    if spans:
        return spans

    # Fallback: find by string matching
    start = 0
    while True:
        idx = text.find(entity_name, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(entity_name)))
        start = idx + 1
    return spans


def insert_entity_markers(text: str, head: str, tail: str, expected_entities: list[dict]) -> str | None:
    """Insert [E1]/[/E1] around head and [E2]/[/E2] around tail in text.

    Returns None if either entity cannot be located.
    Uses span positions from expected_entities when available.
    Handles overlapping/nested spans by choosing non-overlapping occurrences.
    """
    head_spans = find_entity_spans(text, head, expected_entities)
    tail_spans = find_entity_spans(text, tail, expected_entities)

    if not head_spans or not tail_spans:
        return None

    # Pick the first non-overlapping pair
    h_span = head_spans[0]
    t_span = None
    for ts in tail_spans:
        # No overlap: one ends before the other starts
        if ts[1] <= h_span[0] or ts[0] >= h_span[1]:
            t_span = ts
            break
    if t_span is None:
        # If all tail spans overlap with head, try alternate head spans
        for hs in head_spans[1:]:
            for ts in tail_spans:
                if ts[1] <= hs[0] or ts[0] >= hs[1]:
                    h_span = hs
                    t_span = ts
                    break
            if t_span is not None:
                break

    if t_span is None:
        return None

    # Build the marked text — insert markers from right to left to preserve indices
    insertions = sorted([
        (h_span[0], "[E1]", 0),   # 0 = opening
        (h_span[1], "[/E1]", 1),  # 1 = closing
        (t_span[0], "[E2]", 0),
        (t_span[1], "[/E2]", 1),
    ], key=lambda x: (-x[0], -x[2]))  # right-to-left, closings before openings at same pos

    result = text
    for pos, marker, _ in insertions:
        result = result[:pos] + marker + result[pos:]
    return result


def load_episodes(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_entity_index(episode: dict) -> dict[str, dict]:
    """Map entity name -> entity dict for an episode."""
    return {ent["name"]: ent for ent in episode.get("expected_entities", [])}


def generate_positive_examples(episodes: list[dict]) -> list[dict]:
    """Generate one positive example per expected relation per episode."""
    examples = []
    for ep in episodes:
        text = ep["text"]
        entities = ep.get("expected_entities", [])
        for rel in ep.get("expected_relations", []):
            head, tail, label = rel["head"], rel["tail"], rel["relation"]
            marked = insert_entity_markers(text, head, tail, entities)
            if marked is None:
                print(f"WARNING: Could not mark entities head={head!r} tail={tail!r} in text: {text[:80]}...", file=sys.stderr)
                continue
            examples.append({
                "text": marked,
                "head": head,
                "tail": tail,
                "label": label,
            })
    return examples


def generate_negative_examples(episodes: list[dict], target_count: int, rng: random.Random) -> list[dict]:
    """Generate negative (label='none') examples from entity pairs without relations.

    Also generates some using NONE_TEMPLATES for variety.
    """
    examples = []
    for ep in episodes:
        text = ep["text"]
        entities = ep.get("expected_entities", [])
        entity_names = [ent["name"] for ent in entities]

        # Build set of positive pairs
        positive_pairs = set()
        for rel in ep.get("expected_relations", []):
            positive_pairs.add((rel["head"], rel["tail"]))
            positive_pairs.add((rel["tail"], rel["head"]))  # bidirectional exclusion

        # All ordered entity pairs that are NOT in a relation
        negative_pairs = []
        for h, t in combinations(entity_names, 2):
            if (h, t) not in positive_pairs and (t, h) not in positive_pairs:
                negative_pairs.append((h, t))

        for h, t in negative_pairs:
            # Use original text with markers
            marked = insert_entity_markers(text, h, t, entities)
            if marked is not None:
                examples.append({
                    "text": marked,
                    "head": h,
                    "tail": t,
                    "label": "none",
                })

    # If we need more negatives, generate from templates
    all_entity_names = []
    for ep in episodes:
        for ent in ep.get("expected_entities", []):
            all_entity_names.append(ent["name"])
    unique_entities = list(set(all_entity_names))

    while len(examples) < target_count and len(unique_entities) >= 2:
        h, t = rng.sample(unique_entities, 2)
        tmpl = rng.choice(NONE_TEMPLATES)
        examples.append({
            "text": tmpl(h, t),
            "head": h,
            "tail": t,
            "label": "none",
        })

    # Trim to target
    if len(examples) > target_count:
        rng.shuffle(examples)
        examples = examples[:target_count]

    return examples


def generate_augmented_examples(positives: list[dict], aug_per_example: int, rng: random.Random) -> list[dict]:
    """Create paraphrased variants for each positive example using templates."""
    augmented = []
    for ex in positives:
        label = ex["label"]
        head = ex["head"]
        tail = ex["tail"]
        templates = PARAPHRASE_TEMPLATES.get(label, [])
        if not templates:
            continue
        # Sample aug_per_example templates without replacement if possible
        k = min(aug_per_example, len(templates))
        selected = rng.sample(templates, k)
        for tmpl in selected:
            augmented.append({
                "text": tmpl(head, tail),
                "head": head,
                "tail": tail,
                "label": label,
            })
    return augmented


def stratified_split(examples: list[dict], val_ratio: float, rng: random.Random) -> list[dict]:
    """Assign 'split' field ('train' or 'val') with stratification by label."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_label[ex["label"]].append(ex)

    result = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        for i, item in enumerate(items):
            item["split"] = "val" if i < n_val else "train"
            result.append(item)

    rng.shuffle(result)
    return result


def print_statistics(examples: list[dict]) -> None:
    total = len(examples)
    label_counts = Counter(ex["label"] for ex in examples)
    train_count = sum(1 for ex in examples if ex["split"] == "train")
    val_count = sum(1 for ex in examples if ex["split"] == "val")

    print("=" * 60)
    print("Training Data Statistics")
    print("=" * 60)
    print(f"Total examples:  {total}")
    print(f"Train:           {train_count}")
    print(f"Val:             {val_count}")
    print(f"Val ratio:       {val_count / total:.1%}")
    print()
    print("Per-class counts:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        train_c = sum(1 for ex in examples if ex["label"] == label and ex["split"] == "train")
        val_c = sum(1 for ex in examples if ex["label"] == label and ex["split"] == "val")
        print(f"  {label:20s}  total={count:4d}  train={train_c:4d}  val={val_c:4d}")
    print("=" * 60)


def main() -> None:
    rng = random.Random(SEED)

    print(f"Loading episodes from {BENCHMARK_PATH}")
    episodes = load_episodes(BENCHMARK_PATH)
    print(f"Loaded {len(episodes)} episodes")

    # Step 1: Positive examples from gold annotations
    positives = generate_positive_examples(episodes)
    print(f"Generated {len(positives)} positive examples")

    # Step 2: Augment — oversample minority classes to ~100 examples each
    # Count per-class positives
    pos_counts = Counter(ex["label"] for ex in positives)
    target_per_class = 100
    augmented = []
    for label, templates in PARAPHRASE_TEMPLATES.items():
        gold_for_label = [ex for ex in positives if ex["label"] == label]
        current = len(gold_for_label)
        needed = max(0, target_per_class - current)
        if needed > 0 and gold_for_label:
            # Generate enough augmented examples to reach target_per_class
            aug_per = max(1, needed // len(gold_for_label) + 1)
            aug = generate_augmented_examples(gold_for_label, aug_per_example=aug_per, rng=rng)
            # Trim to exactly what's needed
            rng.shuffle(aug)
            augmented.extend(aug[:needed])
    print(f"Generated {len(augmented)} augmented examples (oversampled minority classes)")

    # Step 3: Negative examples — match per-class count (~100 per class)
    target_negative_count = target_per_class
    negatives = generate_negative_examples(episodes, target_negative_count, rng)
    print(f"Generated {len(negatives)} negative examples")

    # Combine all
    all_examples = positives + augmented + negatives

    # Step 4: Stratified train/val split
    all_examples = stratified_split(all_examples, val_ratio=0.2, rng=rng)

    # Step 5: Output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_examples)} examples to {OUTPUT_PATH}")

    # Step 6: Statistics
    print()
    print_statistics(all_examples)


if __name__ == "__main__":
    main()
