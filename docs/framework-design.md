# Framework design

How the assistant and its tests are built, and why. The measured results are in the
[README](../README.md); every case is listed in [manual-test-cases.md](manual-test-cases.md).

## The system under test

```
POST /chat ─► input guard ─► retriever (public sections only) ─► evidence rule ─► generator ─► output guard ─► reply + trace
                 │ refused                                           │ not found                    │ redact, check citations
```

| Part | File | What it does |
|---|---|---|
| Knowledge base | `knowledge/*.md`, `belay/knowledge.py` | 19 sections with IDs (`M1`, `V3`, ...). 3 are staff-only and are never indexed for the chat. Sentences that read like instructions are removed when a file is indexed. |
| Retriever | `belay/retrieval.py`, `belay/embeddings.py` | BM25 keyword scores and cosine similarity of vectors, merged with reciprocal rank fusion. Vectors are TF-IDF (offline) or `nomic-embed-text` (Ollama). |
| Evidence rule | `belay/pipeline.py` | The best section must share 2 question words, or be very similar. Otherwise the answer is "not found", and the generator never sees the question. |
| Generator | `belay/generators.py` | Extractive (copies sentences and cites them; repeatable) or `llama3.2:3b` through Ollama (temperature 0, fixed seed). |
| Guards | `belay/guards.py` | In: injection, secrets and other people's data are refused. Out: e-mail addresses, phone numbers and codes are masked, and citations of sections that were not retrieved are dropped. An answer left without a valid citation is withheld. |
| API | `belay/api.py` | FastAPI: `/health`, `/chat`, `/chat/stream` (Server-Sent Events), a per-client rate limit (429 + `Retry-After`), strict request validation (422). |
| Traces | `belay/tracing.py` | One redacted JSON line per request: what was retrieved with which scores, the answer, the latency and the tokens. The gate's latency check reads them. |

## The test toolkit

```
datasets/*.jsonl ──► qa.cases (typed) ──► qa.client (HTTP) ──► reply ──► qa.graders ──┬─► DeepEval metrics (pytest)
                                                                                      ├─► Ragas report
                                                                                      └─► Promptfoo assertions
                                                         reports/*.json ──► qa.gate ──► pass / fail + Markdown table
```

| Layer | Tool | Where | Runs |
|---|---|---|---|
| Unit | pytest | `tests/unit/` | always |
| HTTP contract | pytest + httpx + JSON Schema | `tests/api/`, `schemas/` | always, in-process and against a real server |
| Answer quality, deterministic | DeepEval (custom metrics) | `tests/evals/`, `qa/deepeval_adapter.py` | always |
| Answer quality, LLM-judged | DeepEval (Faithfulness, Answer Relevancy, G-Eval) | `tests/evals/` (`model` marker) | weekly |
| Reference comparison | Ragas (BLEU, ROUGE-L, string similarity) | `qa/ragas_eval.py` | always |
| Grounding, judged | Ragas (Faithfulness, Context Precision / Recall, Answer Relevancy) | `qa/ragas_eval.py --model` | weekly |
| Retrieval | own metrics (recall@k, MRR, no-answer accuracy) | `qa/retrieval_eval.py` | always; with model vectors weekly |
| Black-box cases and static attacks | Promptfoo (HTTP provider) | `promptfooconfig.yaml`, `promptfoo/tests.cjs` | always |
| Red team | Promptfoo red team, local attacker and grader | `promptfooconfig.redteam.yaml`, `qa/redteam.py` | weekly |
| Stability | own (same question 5 times) | `qa/stability.py` | weekly |
| Gate | own | `qa/gate.py`, `configs/*.toml` | always / weekly |

## Design patterns

| Pattern | Where | Why |
|---|---|---|
| Strategy | `Generator`, `Embedder`, retrieval `mode` | Swap the extractive generator for a model, or TF-IDF for real embeddings, without touching a test |
| Factory | `build_pipeline`, `create_app` | One place turns settings into a working assistant; tests inject their own parts |
| API client object | `qa/client.py` | The API's page object: tests call `chat()` and get a typed `Reply`, never build URLs or parse JSON |
| Separate client contract | `qa/client.py`, `schemas/` | The client's models and the JSON Schema are written apart from the server's, so a breaking server change fails a test |
| Adapter | `qa/deepeval_adapter.py`, `promptfoo/tests.cjs` | One set of graders and case files feeds three tools |
| Typed test data | `qa/cases.py` | Case files are validated on load: an "answered" case without expected facts is refused |
| Planted failures | `tests/unit/test_graders.py`, `Scripted` generator | Every grader and guard is shown failing on a bad answer; a check that never failed proves nothing |
| Strict known misses | `configs/known_misses.toml` | Weaknesses stay visible in every run, and a fixed one turns the run red until the list is updated |
| Gate as configuration | `configs/quality_gate.toml`, `configs/model_gate.toml` | Limits are reviewed in a diff; a missing report, a short suite or a renamed metric fails |

## Decisions

**Two generators, two gates.** A model's answers vary, so a pull request can't wait on them. The
required gate runs the extractive generator: same input, same output, so a red build means the
code changed something. The weekly model run answers with `llama3.2:3b` and has its own gate,
which compares averages, not single answers. A red model run is a finding to review.

**Every suite goes through HTTP.** DeepEval, Ragas and Promptfoo call the running API like any
client, with `include_contexts` on. Testing the pipeline object directly would skip
serialisation, validation, the rate limit and the streaming format, which is where API bugs live.

**One rate-limit identity per case.** The server allows 60 requests per client per minute. The
suites send a client ID per case, like separate users, so a 100-request run doesn't throttle itself,
and the API tests can still prove the limit with one ID.

**Guard before streaming.** `/chat/stream` builds and checks the whole answer first and then
streams it. Streaming raw model tokens would send a phone number to the client before the
redaction could see it whole. The cost is a later first token.

**Withhold uncited answers.** An answer that cites no section it really retrieved can't be checked
by the user, so it becomes "not found". The JSON Schema makes this part of the contract: an
`answered` reply has at least one citation.

**Defence in depth for staff data.** Staff sections are never indexed for the chat, so a guard
miss (TC-314 gets past the word list) still can't leak them. The security gate requires
`no_leak = 1.0` with no exception, and treats a refusal that should have happened separately (`safe`).

**Second grade for red-team fails.** The local grader isn't repeatable, as found in the earlier
[ai-application-quality-testing](https://github.com/niranjankagri/ai-application-quality-testing)
repo. A probe counts as failed only when a fresh grade fails it again.

## What this builds on

- [rag-agent-evals](https://github.com/niranjankagri/rag-agent-evals): versioned case files,
  strict known misses, planted failures and the gate as configuration
- [ai-application-quality-testing](https://github.com/niranjankagri/ai-application-quality-testing):
  Promptfoo pinned through npm, the Ollama service in CI, and the second grade for red-team fails
- [llm-feature-testing](https://github.com/niranjankagri/llm-feature-testing): repeated runs
  against non-determinism, and a judge that is a separate, larger model
