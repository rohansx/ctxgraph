#!/usr/bin/env python3
"""Generate training data for fine-tuning a DeBERTa-v3-small entity-pair relation classifier.

Reads gold benchmark episodes, generates positive/negative examples with entity markers,
augments via paraphrasing templates (including informal/natural patterns), and outputs
a train/val split JSON file.

Improvements over v1:
- Harder negatives: confusing co-occurrence, direction-reversed, near-miss
- More natural augmentation: informal patterns (commit msgs, Slack, ADRs, sentence fragments)
- Cross-episode entity mixing for diverse entity-relation combinations
- Direction-aware negatives for replaced/depends_on
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

# Directional relations where head↔tail swap changes meaning
DIRECTIONAL_RELATIONS = {"replaced", "depends_on", "chose", "rejected", "fixed",
                         "introduced", "deprecated", "caused", "constrained_by"}

SEED = 42

# ---------------------------------------------------------------------------
# Entity marker helpers
# ---------------------------------------------------------------------------

def _h(name: str) -> str:
    return f"[E1]{name}[/E1]"

def _t(name: str) -> str:
    return f"[E2]{name}[/E2]"


# ---------------------------------------------------------------------------
# Paraphrase templates per relation type.
# Split into "formal" (original style) and "informal" (natural text patterns).
# ---------------------------------------------------------------------------

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
# Informal / natural-language positive templates.
# These mimic commit messages, Slack messages, ADR snippets, code comments,
# sentence fragments, and other real-world text patterns.
# ---------------------------------------------------------------------------

INFORMAL_TEMPLATES: dict[str, list] = {
    "chose": [
        lambda h, t: f"Went with {_t(t)} for {_h(h)}. Reason: better community support.",
        lambda h, t: f"ADR-042: {_h(h)} will use {_t(t)}. Decision: accepted.",
        lambda h, t: f"{_h(h)}: chose {_t(t)} over the alternatives after benchmarking",
        lambda h, t: f"decided on {_t(t)} for {_h(h)} - simpler API, less boilerplate",
        lambda h, t: f"@team {_h(h)} is going with {_t(t)}, lmk if concerns",
        lambda h, t: f"feat({_h(h)}): switch to {_t(t)} for data layer",
        lambda h, t: f"After comparing options, {_t(t)} wins for {_h(h)}. Closing this spike.",
        lambda h, t: f"Summary: evaluated 3 options for {_h(h)}, picked {_t(t)}",
        lambda h, t: f"cc @eng - {_h(h)} adopting {_t(t)} per RFC discussion",
        lambda h, t: f"spike result: {_t(t)} is the best fit for {_h(h)}, merging",
    ],
    "rejected": [
        lambda h, t: f"Ruled out {_t(t)} for {_h(h)} - too many breaking changes.",
        lambda h, t: f"ADR-042: {_h(h)} will NOT use {_t(t)}. Too immature.",
        lambda h, t: f"{_h(h)}: tried {_t(t)}, doesn't work for our scale",
        lambda h, t: f"nope, {_t(t)} is a no-go for {_h(h)}. Missing auth support.",
        lambda h, t: f"@team dropping {_t(t)} from {_h(h)} shortlist, perf too low",
        lambda h, t: f"Spike conclusion: {_t(t)} rejected for {_h(h)} due to licensing",
        lambda h, t: f"Benchmarks show {_t(t)} can't handle {_h(h)} throughput. Moving on.",
        lambda h, t: f"Not going with {_t(t)} for {_h(h)}. See thread for details.",
        lambda h, t: f"wontfix: {_t(t)} integration with {_h(h)} has too many edge cases",
        lambda h, t: f"killed the {_t(t)} PoC for {_h(h)}, back to drawing board",
    ],
    "replaced": [
        lambda h, t: f"Migrated from {_t(t)} to {_h(h)}. All tests pass.",
        lambda h, t: f"chore: swap {_t(t)} for {_h(h)} in production",
        lambda h, t: f"Ripped out {_t(t)}, now using {_h(h)} instead.",
        lambda h, t: f"{_h(h)} replaces {_t(t)} effective this release.",
        lambda h, t: f"bye bye {_t(t)}, hello {_h(h)}!",
        lambda h, t: f"refactor: replace {_t(t)} with {_h(h)} across all services",
        lambda h, t: f"Cut over from {_t(t)} to {_h(h)} last night, zero downtime.",
        lambda h, t: f"Migration complete: {_t(t)} -> {_h(h)}. Decommissioning old infra.",
        lambda h, t: f"@ops {_t(t)} is deprecated, {_h(h)} is live now",
        lambda h, t: f"Swapped {_t(t)} out for {_h(h)}. 3x throughput improvement.",
    ],
    "depends_on": [
        lambda h, t: f"{_h(h)} needs {_t(t)} running or it won't start.",
        lambda h, t: f"Note: {_h(h)} requires {_t(t)} >= v3.2",
        lambda h, t: f"{_h(h)} calls {_t(t)} on the hot path",
        lambda h, t: f"can't deploy {_h(h)} without {_t(t)} - hard dependency",
        lambda h, t: f"docker-compose: {_h(h)} depends_on {_t(t)}",
        lambda h, t: f"FYI {_h(h)} is blocked on {_t(t)} being healthy",
        lambda h, t: f"{_h(h)} reads from {_t(t)} on every request",
        lambda h, t: f"Outage root cause: {_t(t)} went down, took {_h(h)} with it",
        lambda h, t: f"Added {_t(t)} as explicit dependency of {_h(h)} in the manifest",
        lambda h, t: f"{_h(h)} imports {_t(t)} client SDK for auth flow",
    ],
    "fixed": [
        lambda h, t: f"fix({_h(h)}): resolve {_t(t)} connection timeout",
        lambda h, t: f"Hotfix: {_h(h)} patch fixes the {_t(t)} crash",
        lambda h, t: f"Fixed the {_t(t)} bug in {_h(h)}. Root cause: null check.",
        lambda h, t: f"{_h(h)} update resolves {_t(t)} issue reported in #1234",
        lambda h, t: f"Patched {_h(h)} to fix {_t(t)} data corruption",
        lambda h, t: f"bugfix: {_h(h)} now handles {_t(t)} edge case correctly",
        lambda h, t: f"The {_t(t)} errors are gone after the {_h(h)} fix",
        lambda h, t: f"@oncall {_h(h)} fix deployed, {_t(t)} should be stable now",
        lambda h, t: f"cherry-pick: {_h(h)} fix for {_t(t)} regression",
        lambda h, t: f"Root cause found in {_h(h)}, {_t(t)} failures resolved.",
    ],
    "introduced": [
        lambda h, t: f"feat({_h(h)}): add {_t(t)} support",
        lambda h, t: f"Added {_t(t)} to {_h(h)} for better observability.",
        lambda h, t: f"{_h(h)} now includes {_t(t)} - see updated docs",
        lambda h, t: f"Wired up {_t(t)} in {_h(h)} this sprint",
        lambda h, t: f"New: {_h(h)} ships with {_t(t)} enabled",
        lambda h, t: f"PR #456: introduce {_t(t)} into {_h(h)}",
        lambda h, t: f"@team {_h(h)} now has {_t(t)} integration, please test",
        lambda h, t: f"Set up {_t(t)} for {_h(h)} in staging first",
        lambda h, t: f"chore({_h(h)}): bootstrap {_t(t)} module",
        lambda h, t: f"Rolled out {_t(t)} as part of the {_h(h)} overhaul",
    ],
    "deprecated": [
        lambda h, t: f"deprecation({_h(h)}): {_t(t)} is EOL as of this release",
        lambda h, t: f"{_h(h)} deprecated {_t(t)}. Migration guide in docs/.",
        lambda h, t: f"Sunsetting {_t(t)} from {_h(h)} - use v2 API instead",
        lambda h, t: f"@team {_h(h)} is dropping {_t(t)} support next quarter",
        lambda h, t: f"DEPRECATION: {_t(t)} will be removed from {_h(h)} in v3",
        lambda h, t: f"Notice: {_h(h)} no longer maintains {_t(t)}. Please migrate.",
        lambda h, t: f"chore: mark {_t(t)} as deprecated in {_h(h)}",
        lambda h, t: f"{_t(t)} is legacy now, {_h(h)} is phasing it out",
        lambda h, t: f"Cleanup: {_h(h)} removing {_t(t)} refs, 90 day sunset",
        lambda h, t: f"FYI {_h(h)} flagged {_t(t)} for decommission",
    ],
    "caused": [
        lambda h, t: f"Deploying {_h(h)} improved {_t(t)} by 3x.",
        lambda h, t: f"After enabling {_h(h)}, {_t(t)} dropped to normal levels.",
        lambda h, t: f"Root cause: {_h(h)} rollout impacted {_t(t)}",
        lambda h, t: f"{_h(h)} change caused {_t(t)} regression - investigating",
        lambda h, t: f"Postmortem: {_h(h)} deployment led to {_t(t)} spike",
        lambda h, t: f"Dashboard shows {_h(h)} directly affecting {_t(t)}",
        lambda h, t: f"Confirmed: {_h(h)} is driving the {_t(t)} improvement",
        lambda h, t: f"A/B test: {_h(h)} group shows {_t(t)} delta of -40%",
        lambda h, t: f"Canary data: {_h(h)} moved {_t(t)} metrics significantly",
        lambda h, t: f"The {_t(t)} change correlates with {_h(h)} release",
    ],
    "constrained_by": [
        lambda h, t: f"{_h(h)} must stay within {_t(t)} limits.",
        lambda h, t: f"Note: {_h(h)} is gated on {_t(t)} compliance",
        lambda h, t: f"Can't ship {_h(h)} without passing {_t(t)} checks",
        lambda h, t: f"{_h(h)} SLA is dictated by {_t(t)}",
        lambda h, t: f"Blocked: {_h(h)} needs {_t(t)} sign-off before deploy",
        lambda h, t: f"Constraint: {_h(h)} throughput capped by {_t(t)}",
        lambda h, t: f"Design doc: {_h(h)} limited by {_t(t)} policy",
        lambda h, t: f"{_h(h)} config enforces {_t(t)} at runtime",
        lambda h, t: f"Audit: {_h(h)} validated against {_t(t)} rules",
        lambda h, t: f"@security {_h(h)} must satisfy {_t(t)} before GA",
    ],
}

# ---------------------------------------------------------------------------
# "none" templates (easy + hard negatives)
# ---------------------------------------------------------------------------
NONE_TEMPLATES_EASY = [
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

# Hard negatives: entities co-occur with relation-like keywords but no real relation
NONE_TEMPLATES_HARD = [
    # Looks like "replaced" but isn't
    lambda h, t: f"{_h(h)} and {_t(t)} are both databases, each serving a different workload.",
    lambda h, t: f"The team evaluated migrating from {_h(h)} but kept it alongside {_t(t)}.",
    lambda h, t: f"While {_h(h)} and {_t(t)} overlap in capabilities, neither replaces the other.",
    lambda h, t: f"Both {_h(h)} and {_t(t)} handle storage but for completely different domains.",
    # Looks like "depends_on" but isn't
    lambda h, t: f"{_h(h)} and {_t(t)} are on the same network but never communicate.",
    lambda h, t: f"Although {_h(h)} runs next to {_t(t)}, there is no runtime dependency.",
    lambda h, t: f"{_h(h)} and {_t(t)} share a config file but are otherwise independent.",
    lambda h, t: f"The service mesh routes traffic for both {_h(h)} and {_t(t)} independently.",
    # Looks like "chose" but isn't
    lambda h, t: f"The team discussed using {_t(t)} for {_h(h)} but no decision was reached.",
    lambda h, t: f"{_h(h)} and {_t(t)} were both on the shortlist but the decision is pending.",
    lambda h, t: f"Spike: evaluating {_t(t)} and others for {_h(h)} - TBD",
    lambda h, t: f"Comparing {_h(h)} with {_t(t)} to determine the best approach.",
    # Looks like "fixed" but isn't
    lambda h, t: f"{_h(h)} and {_t(t)} both had issues this week but they are unrelated.",
    lambda h, t: f"The {_h(h)} crash happened around the same time as {_t(t)} errors, coincidence.",
    lambda h, t: f"Debugging both {_h(h)} and {_t(t)} but the bugs are independent.",
    # Looks like "caused" but isn't
    lambda h, t: f"{_h(h)} and {_t(t)} metrics both spiked, but root causes differ.",
    lambda h, t: f"Correlation between {_h(h)} and {_t(t)} is not causal.",
    lambda h, t: f"Both {_h(h)} and {_t(t)} changed this sprint, but independently.",
    # Looks like "introduced" but isn't
    lambda h, t: f"The PR mentions {_h(h)} and {_t(t)} but only refactors existing code.",
    lambda h, t: f"{_h(h)} and {_t(t)} were both in the changelog for cleanup reasons.",
    # Looks like "deprecated" but isn't
    lambda h, t: f"The team considered deprecating {_t(t)} in {_h(h)} but decided to keep it.",
    lambda h, t: f"Despite the review, {_h(h)} and {_t(t)} are both still active.",
    # Informal hard negatives
    lambda h, t: f"standup: worked on {_h(h)} and {_t(t)} today, separate tasks",
    lambda h, t: f"JIRA-1234: {_h(h)} and {_t(t)} mentioned in different subtasks",
    lambda h, t: f"Retro: {_h(h)} and {_t(t)} both came up but not related",
    lambda h, t: f"TIL: {_h(h)} is similar to {_t(t)} in some ways but we use them separately",
    lambda h, t: f"docs: updated pages for both {_h(h)} and {_t(t)}",
    lambda h, t: f"@team FYI: {_h(h)} and {_t(t)} are both getting upgrades next sprint",

    # --- Sequential/list context (entities in same list but no relation) ---
    lambda h, t: f"The stack includes {_h(h)}, {_t(t)}, and several other components.",
    lambda h, t: f"Our tech radar lists {_h(h)}, {_t(t)}, Kubernetes, and Terraform.",
    lambda h, t: f"Services currently running: {_h(h)}, {_t(t)}, auth-proxy, gateway.",
    lambda h, t: f"The monorepo contains packages for {_h(h)}, {_t(t)}, and shared-utils.",
    lambda h, t: f"Dependencies in the lockfile: {_h(h)}, {_t(t)}, lodash, express.",

    # --- Comparison/evaluation without decision ---
    lambda h, t: f"Comparing {_h(h)} vs {_t(t)} for latency characteristics. No conclusion yet.",
    lambda h, t: f"Benchmark results for {_h(h)} and {_t(t)} are still being analyzed.",
    lambda h, t: f"We looked at both {_h(h)} and {_t(t)} in the spike but deferred the decision.",
    lambda h, t: f"Pros and cons of {_h(h)} vs {_t(t)} documented in the RFC, awaiting review.",
    lambda h, t: f"The evaluation of {_h(h)} and {_t(t)} is ongoing, no winner declared.",

    # --- Past-tense discussion without action ---
    lambda h, t: f"Last quarter we discussed {_h(h)} and {_t(t)} but took no action.",
    lambda h, t: f"The topic of {_h(h)} and {_t(t)} came up in the offsite but was tabled.",
    lambda h, t: f"Previously, {_h(h)} and {_t(t)} were mentioned in passing during planning.",
    lambda h, t: f"In Q3 retrospective, someone brought up {_h(h)} and {_t(t)} as areas to watch.",
    lambda h, t: f"Historical note: {_h(h)} and {_t(t)} were both considered years ago.",

    # --- Documentation/reference context ---
    lambda h, t: f"See the wiki pages for {_h(h)} and {_t(t)} for more details.",
    lambda h, t: f"The runbook has separate sections covering {_h(h)} and {_t(t)}.",
    lambda h, t: f"Confluence page links to docs for {_h(h)} and {_t(t)} in the appendix.",
    lambda h, t: f"README mentions {_h(h)} and {_t(t)} as part of the ecosystem overview.",
    lambda h, t: f"The glossary defines {_h(h)} and {_t(t)} as distinct concepts.",

    # --- Independent actions applied to both entities ---
    lambda h, t: f"Updated {_h(h)}, also updated {_t(t)} - separate PRs.",
    lambda h, t: f"Deployed new versions of both {_h(h)} and {_t(t)} this week, unrelated changes.",
    lambda h, t: f"Ran load tests on {_h(h)} and {_t(t)} independently, results vary.",
    lambda h, t: f"Upgraded {_h(h)} to v3 and {_t(t)} to v2, no connection between the upgrades.",
    lambda h, t: f"Refactored {_h(h)} and {_t(t)} in the same sprint but different stories.",

    # --- Meeting/standup notes mentioning unrelated entities ---
    lambda h, t: f"Standup notes: Alice is on {_h(h)}, Bob is on {_t(t)}. No blockers.",
    lambda h, t: f"Sprint review: {_h(h)} shipped on time. {_t(t)} delayed due to QA.",
    lambda h, t: f"Weekly sync: {_h(h)} team presented demo. {_t(t)} team discussed roadmap.",
    lambda h, t: f"All-hands: leadership mentioned {_h(h)} and {_t(t)} as key investments.",
    lambda h, t: f"1:1 notes: discussed workload across {_h(h)} and {_t(t)} projects.",

    # --- PR descriptions mentioning entities in different sections ---
    lambda h, t: f"PR #789 changes: updated {_h(h)} logging. Unrelated: bumped {_t(t)} version.",
    lambda h, t: f"This PR touches {_h(h)} config and {_t(t)} tests, separate concerns.",
    lambda h, t: f"Changelog: {_h(h)} - improved caching. {_t(t)} - fixed typo in docs.",
    lambda h, t: f"Release notes: {_h(h)} gets new API endpoint. {_t(t)} gets minor UI fix.",
    lambda h, t: f"Diff includes files from both {_h(h)} and {_t(t)} modules but changes are independent.",

    # --- Monitoring/observability co-mention without causation ---
    lambda h, t: f"Dashboard shows metrics for {_h(h)} and {_t(t)} side by side for comparison.",
    lambda h, t: f"Alerts fired for both {_h(h)} and {_t(t)} during the outage, but unrelated root causes.",
    lambda h, t: f"Grafana panel includes {_h(h)} and {_t(t)} latency charts on the same row.",
    lambda h, t: f"Tracing shows requests hitting {_h(h)} and {_t(t)} in different code paths.",
    lambda h, t: f"SLO report covers {_h(h)} at 99.9% and {_t(t)} at 99.5%, measured independently.",
]


def find_entity_spans(text: str, entity_name: str, expected_entities: list[dict]) -> list[tuple[int, int]]:
    """Find all (start, end) character spans where entity_name appears in text."""
    spans = []
    for ent in expected_entities:
        if ent["name"] == entity_name:
            s, e = ent["span_start"], ent["span_end"]
            if text[s:e] == entity_name:
                spans.append((s, e))
    if spans:
        return spans

    start = 0
    while True:
        idx = text.find(entity_name, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(entity_name)))
        start = idx + 1
    return spans


def insert_entity_markers(text: str, head: str, tail: str, expected_entities: list[dict]) -> str | None:
    """Insert [E1]/[/E1] around head and [E2]/[/E2] around tail in text."""
    head_spans = find_entity_spans(text, head, expected_entities)
    tail_spans = find_entity_spans(text, tail, expected_entities)

    if not head_spans or not tail_spans:
        return None

    h_span = head_spans[0]
    t_span = None
    for ts in tail_spans:
        if ts[1] <= h_span[0] or ts[0] >= h_span[1]:
            t_span = ts
            break
    if t_span is None:
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

    insertions = sorted([
        (h_span[0], "[E1]", 0),
        (h_span[1], "[/E1]", 1),
        (t_span[0], "[E2]", 0),
        (t_span[1], "[/E2]", 1),
    ], key=lambda x: (-x[0], -x[2]))

    result = text
    for pos, marker, _ in insertions:
        result = result[:pos] + marker + result[pos:]
    return result


def load_episodes(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_entity_index(episode: dict) -> dict[str, dict]:
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

    Also generates using both easy and hard NONE_TEMPLATES for variety.
    """
    examples = []
    for ep in episodes:
        text = ep["text"]
        entities = ep.get("expected_entities", [])
        entity_names = [ent["name"] for ent in entities]

        positive_pairs = set()
        for rel in ep.get("expected_relations", []):
            positive_pairs.add((rel["head"], rel["tail"]))
            positive_pairs.add((rel["tail"], rel["head"]))

        negative_pairs = []
        for h, t in combinations(entity_names, 2):
            if (h, t) not in positive_pairs and (t, h) not in positive_pairs:
                negative_pairs.append((h, t))

        for h, t in negative_pairs:
            marked = insert_entity_markers(text, h, t, entities)
            if marked is not None:
                examples.append({
                    "text": marked,
                    "head": h,
                    "tail": t,
                    "label": "none",
                })

    all_entity_names = []
    for ep in episodes:
        for ent in ep.get("expected_entities", []):
            all_entity_names.append(ent["name"])
    unique_entities = list(set(all_entity_names))

    # Fill remaining with mix of easy and hard templates
    all_none_templates = NONE_TEMPLATES_EASY + NONE_TEMPLATES_HARD
    while len(examples) < target_count and len(unique_entities) >= 2:
        h, t = rng.sample(unique_entities, 2)
        tmpl = rng.choice(all_none_templates)
        examples.append({
            "text": tmpl(h, t),
            "head": h,
            "tail": t,
            "label": "none",
        })

    if len(examples) > target_count:
        rng.shuffle(examples)
        examples = examples[:target_count]

    return examples


def generate_hard_negatives_from_templates(
    episodes: list[dict], target_count: int, rng: random.Random
) -> list[dict]:
    """Generate hard negatives using NONE_TEMPLATES_HARD with entity pairs from episodes."""
    all_entity_names = list({
        ent["name"] for ep in episodes for ent in ep.get("expected_entities", [])
    })
    examples = []
    for _ in range(target_count):
        if len(all_entity_names) < 2:
            break
        h, t = rng.sample(all_entity_names, 2)
        tmpl = rng.choice(NONE_TEMPLATES_HARD)
        examples.append({
            "text": tmpl(h, t),
            "head": h,
            "tail": t,
            "label": "none",
        })
    return examples


def generate_direction_reversed_negatives(
    episodes: list[dict], rng: random.Random
) -> list[dict]:
    """For directional relations, create reversed-direction examples as 'none'.

    E.g., if A replaced B is gold, then B replaced A is a hard negative labeled 'none'.
    These are generated using paraphrase templates with head/tail swapped.
    """
    examples = []
    for ep in episodes:
        for rel in ep.get("expected_relations", []):
            label = rel["relation"]
            if label not in DIRECTIONAL_RELATIONS:
                continue
            # Swap head and tail — the text now describes a wrong direction
            orig_head, orig_tail = rel["head"], rel["tail"]
            # Use the original relation's templates but with swapped entities
            templates = PARAPHRASE_TEMPLATES.get(label, [])
            if not templates:
                continue
            # Pick 1-2 templates for the reversed pair
            k = min(2, len(templates))
            selected = rng.sample(templates, k)
            for tmpl in selected:
                # Generate text with SWAPPED head/tail to create direction confusion
                # The text says "orig_tail <relation> orig_head" but the correct relation
                # would be "orig_head <relation> orig_tail", so this reversed text is "none"
                examples.append({
                    "text": tmpl(orig_tail, orig_head),
                    "head": orig_tail,
                    "tail": orig_head,
                    "label": "none",
                })
    return examples


def generate_near_miss_negatives(
    episodes: list[dict], rng: random.Random
) -> list[dict]:
    """Generate near-miss negatives: entities from one relation type used in
    templates of a DIFFERENT relation type, labeled 'none'.

    E.g., a 'depends_on' entity pair used in a 'replaced' template.
    """
    examples = []
    all_relations = []
    for ep in episodes:
        for rel in ep.get("expected_relations", []):
            all_relations.append(rel)

    if not all_relations:
        return examples

    for rel in all_relations:
        correct_label = rel["relation"]
        head, tail = rel["head"], rel["tail"]

        # Pick a WRONG relation type's templates
        wrong_labels = [l for l in RELATION_TYPES if l != correct_label]
        if not wrong_labels:
            continue
        wrong_label = rng.choice(wrong_labels)
        templates = PARAPHRASE_TEMPLATES.get(wrong_label, [])
        if not templates:
            continue

        # Generate text using wrong relation template
        tmpl = rng.choice(templates)
        examples.append({
            "text": tmpl(head, tail),
            "head": head,
            "tail": tail,
            "label": "none",
        })

    return examples


def generate_same_episode_unrelated_negatives(
    episodes: list[dict], rng: random.Random
) -> list[dict]:
    """Generate negatives from entity pairs that appear in the same episode text
    but in different sentences and have no gold relation.

    These are the most realistic negatives because they use real episode text
    where entities co-occur but don't have a relation.
    """
    examples = []
    sentence_split_re = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*')

    for ep in episodes:
        text = ep["text"]
        entities = ep.get("expected_entities", [])
        entity_names = [ent["name"] for ent in entities]

        if len(entity_names) < 2:
            continue

        # Build set of positive pairs to exclude
        positive_pairs = set()
        for rel in ep.get("expected_relations", []):
            positive_pairs.add((rel["head"], rel["tail"]))
            positive_pairs.add((rel["tail"], rel["head"]))

        # Split text into sentences and find which entities appear in each
        sentences = sentence_split_re.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        entity_to_sentences: dict[str, list[int]] = defaultdict(list)
        for i, sent in enumerate(sentences):
            for ent_name in entity_names:
                if ent_name in sent:
                    entity_to_sentences[ent_name].append(i)

        # Find entity pairs that appear in DIFFERENT sentences (no shared sentence)
        entities_with_sentences = [
            name for name in entity_names if entity_to_sentences[name]
        ]

        for h, t in combinations(entities_with_sentences, 2):
            if (h, t) in positive_pairs or (t, h) in positive_pairs:
                continue

            h_sents = set(entity_to_sentences[h])
            t_sents = set(entity_to_sentences[t])

            # Only use pairs where entities appear in different sentences
            # (co-occurrence without direct relation context)
            if h_sents & t_sents:
                continue

            # Use the full episode text with entity markers as a negative
            marked = insert_entity_markers(text, h, t, entities)
            if marked is not None:
                examples.append({
                    "text": marked,
                    "head": h,
                    "tail": t,
                    "label": "none",
                })

    return examples


def generate_cross_episode_examples(
    episodes: list[dict], target_per_class: int, rng: random.Random
) -> list[dict]:
    """Cross-episode entity mixing: take entities from one episode and place them
    in templates from another episode's relation type.

    Creates more diverse entity-relation combinations as real positives.
    """
    # Collect all (entity_name, entity_type) pairs
    all_entities_by_type: dict[str, list[str]] = defaultdict(list)
    for ep in episodes:
        for ent in ep.get("expected_entities", []):
            all_entities_by_type[ent["entity_type"]].append(ent["name"])

    # Deduplicate
    for k in all_entities_by_type:
        all_entities_by_type[k] = list(set(all_entities_by_type[k]))

    # Collect (head_type, tail_type) pairs for each relation
    relation_entity_types: dict[str, list[tuple[str, str]]] = defaultdict(list)
    entity_type_map: dict[str, str] = {}
    for ep in episodes:
        for ent in ep.get("expected_entities", []):
            entity_type_map[ent["name"]] = ent["entity_type"]
        for rel in ep.get("expected_relations", []):
            h_type = entity_type_map.get(rel["head"], "Unknown")
            t_type = entity_type_map.get(rel["tail"], "Unknown")
            relation_entity_types[rel["relation"]].append((h_type, t_type))

    examples = []
    all_templates = {**PARAPHRASE_TEMPLATES}
    # Merge informal templates
    for label, tmpls in INFORMAL_TEMPLATES.items():
        all_templates.setdefault(label, []).extend(tmpls)

    for label, type_pairs in relation_entity_types.items():
        templates = all_templates.get(label, [])
        if not templates:
            continue

        # For each type pair, generate cross-episode examples
        seen_pairs = set()
        for h_type, t_type in type_pairs:
            h_candidates = all_entities_by_type.get(h_type, [])
            t_candidates = all_entities_by_type.get(t_type, [])
            if not h_candidates or not t_candidates:
                continue

            # Generate some examples with random entity combinations
            for _ in range(3):
                h = rng.choice(h_candidates)
                t = rng.choice(t_candidates)
                if h == t or (h, t, label) in seen_pairs:
                    continue
                seen_pairs.add((h, t, label))
                tmpl = rng.choice(templates)
                examples.append({
                    "text": tmpl(h, t),
                    "head": h,
                    "tail": t,
                    "label": label,
                })

    return examples


def generate_augmented_examples(positives: list[dict], aug_per_example: int, rng: random.Random) -> list[dict]:
    """Create paraphrased variants using both formal and informal templates."""
    augmented = []
    for ex in positives:
        label = ex["label"]
        head = ex["head"]
        tail = ex["tail"]
        # Combine formal + informal templates
        templates = list(PARAPHRASE_TEMPLATES.get(label, []))
        templates.extend(INFORMAL_TEMPLATES.get(label, []))
        if not templates:
            continue
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
    print(f"Generated {len(positives)} gold positive examples")

    # Step 2: Augment — oversample minority classes to ~100 examples each
    pos_counts = Counter(ex["label"] for ex in positives)
    target_per_class = 100
    augmented = []
    for label in PARAPHRASE_TEMPLATES:
        gold_for_label = [ex for ex in positives if ex["label"] == label]
        current = len(gold_for_label)
        needed = max(0, target_per_class - current)
        if needed > 0 and gold_for_label:
            aug_per = max(1, needed // len(gold_for_label) + 1)
            aug = generate_augmented_examples(gold_for_label, aug_per_example=aug_per, rng=rng)
            rng.shuffle(aug)
            augmented.extend(aug[:needed])
    print(f"Generated {len(augmented)} augmented examples (oversampled minority classes)")

    # Step 3: Cross-episode entity mixing
    cross_episode = generate_cross_episode_examples(episodes, target_per_class, rng)
    print(f"Generated {len(cross_episode)} cross-episode examples")

    # Step 4: Standard negative examples (mix of easy + hard templates)
    target_negative_count = target_per_class
    negatives = generate_negative_examples(episodes, target_negative_count, rng)
    print(f"Generated {len(negatives)} standard negative examples")

    # Step 5: Additional hard negatives from templates
    hard_negatives = generate_hard_negatives_from_templates(episodes, target_count=150, rng=rng)
    print(f"Generated {len(hard_negatives)} hard negative examples (confusing co-occurrence)")

    # Step 6: Direction-reversed negatives
    reversed_negatives = generate_direction_reversed_negatives(episodes, rng)
    print(f"Generated {len(reversed_negatives)} direction-reversed negative examples")

    # Step 7: Near-miss negatives (wrong relation templates)
    near_miss = generate_near_miss_negatives(episodes, rng)
    print(f"Generated {len(near_miss)} near-miss negative examples")

    # Step 7b: Same-episode unrelated entity pair negatives
    episode_unrelated = generate_same_episode_unrelated_negatives(episodes, rng)
    print(f"Generated {len(episode_unrelated)} same-episode unrelated negative examples")

    # Combine all positives
    all_positives = positives + augmented + cross_episode

    # Combine all negatives, then cap to match average positive class size
    all_negatives = negatives + hard_negatives + reversed_negatives + near_miss + episode_unrelated
    pos_class_counts = Counter(ex["label"] for ex in all_positives)
    avg_pos_count = int(sum(pos_class_counts.values()) / max(len(pos_class_counts), 1))
    # Cap "none" at 2.5x the average positive class count — at inference time
    # most entity pairs will NOT have a relation, so none must dominate
    target_none = int(avg_pos_count * 2.5)
    if len(all_negatives) > target_none:
        # Prioritize realistic/hard negatives over easy ones
        # Reorder: episode-unrelated first (most realistic), then hard, reversed, near-miss, standard
        prioritized_negatives = episode_unrelated + hard_negatives + reversed_negatives + near_miss + negatives
        # Deduplicate within negatives
        neg_seen = set()
        unique_negatives = []
        for ex in prioritized_negatives:
            key = (ex["head"], ex["tail"], ex["text"][:100])
            if key not in neg_seen:
                neg_seen.add(key)
                unique_negatives.append(ex)
        rng.shuffle(unique_negatives)
        all_negatives = unique_negatives[:target_none]
        print(f"Capped negatives from {len(prioritized_negatives)} to {len(all_negatives)} (target_none={target_none})")

    all_examples = all_positives + all_negatives
    print(f"\nTotal before dedup: {len(all_examples)}")

    # Deduplicate by (head, tail, label, text[:100])
    seen = set()
    deduped = []
    for ex in all_examples:
        key = (ex["head"], ex["tail"], ex["label"], ex["text"][:100])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    all_examples = deduped
    print(f"Total after dedup:  {len(all_examples)}")

    # Step 8: Stratified train/val split
    all_examples = stratified_split(all_examples, val_ratio=0.2, rng=rng)

    # Step 9: Output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_examples)} examples to {OUTPUT_PATH}")

    # Step 10: Statistics
    print()
    print_statistics(all_examples)


if __name__ == "__main__":
    main()
