# Testing an LLM chatbot end to end: interview questions, answered with this project

Questions that come up in QA lead and SDET interviews for AI products: testing the API around a model,
evaluating RAG answers, red teaming, and gating releases. Where the project shows the idea, the answer
ends with an **In the project** pointer. Links go to
[github.com/niranjankagri/rag-chatbot-quality-gate](https://github.com/niranjankagri/rag-chatbot-quality-gate).

Versions: Python 3.13, FastAPI 0.165, DeepEval 4.3, Ragas 0.4, Promptfoo 0.125, Ollama 0.40.

## Contents

1. [Testing the API of an LLM feature](#1-testing-the-api-of-an-llm-feature) (Q1-Q12)
2. [Evaluating RAG answers](#2-evaluating-rag-answers) (Q13-Q28)
3. [Models, judges and non-determinism](#3-models-judges-and-non-determinism) (Q29-Q38)
4. [Safety, red teaming and the quality gate](#4-safety-red-teaming-and-the-quality-gate) (Q39-Q48)

## 1. Testing the API of an LLM feature

### Q1. Where do you start testing an LLM feature?

At the API, not the model. Status codes, the response contract, validation, streaming, rate limits and
error bodies behave like any other service and can be tested exactly. Answer quality is a second,
separate layer measured with evals.

**In the project:** `tests/api/test_chat_api.py` (32 tests) never looks at answer quality.

### Q2. What do you check in the response contract?

Required fields, types, allowed values, no unexpected fields, and rules between fields. Here an
`answered` reply must have at least one citation, and a refusal must have none.

**In the project:** `schemas/chat_response.schema.json`, with `if`/`then` rules for the status.

### Q3. Why write the JSON Schema by hand instead of generating it from the server?

A schema generated from the server's models changes when the server changes, so it can never catch a
breaking change. A hand-written one is the client's expectation: if the server renames a field, the test
fails, which is what a real client would experience.

**In the project:** the schema and the client's Pydantic models in `qa/client.py` are separate from `belay/api.py`.

### Q4. How do you test request validation?

One test per kind of bad input, checking the status (422) and that the error names the right field:
missing, empty, blank, too long, wrong type, unknown field. Plus the boundary: the longest allowed value
passes.

**In the project:** `TestInvalidRequests`, including `"questoin"` (a typo) rejected by `extra="forbid"`.

### Q5. How do you test a streaming endpoint?

Check the content type (`text/event-stream`), that the answer arrives in several pieces, that the pieces
join to exactly the final answer, that there is one final event, and that the final event matches the
same schema as the plain call.

**In the project:** `TestStreaming`, and `qa/client.py` parses the Server-Sent Events.

### Q6. Should a chatbot stream tokens straight from the model?

Not if an output guard has to see the whole answer. A phone number split over two tokens can't be
masked by a check that only sees one token. Here the answer is built and checked first, then streamed.
The cost is a later first token; the alternative is a guard that can stream (holding back a few tokens).

**In the project:** the comment in `chat_stream`, `belay/api.py`.

### Q7. How do you test a rate limit?

Send exactly the limit and expect all 200s, then one more and expect 429 with `Retry-After`. Then show
the limit is per client (another client still gets 200) and that every endpoint counts. The limiter
itself is unit-tested with a fake clock, so no test sleeps.

**In the project:** `TestRateLimit` and `tests/api/test_rate_limiter.py`.

### Q8. How do the eval suites avoid hitting the rate limit themselves?

Each case uses its own client ID, like separate users. The limit stays on, and the API test still proves
it with a single ID.

**In the project:** `api.chat(question, client_id=case.id)` in the eval tests.

### Q9. Why run the same tests in-process and against a real server?

In-process is fast and needs no port. A real server also covers what in-process skips: serialisation over
the socket, headers, and SSE framing. The same tests run both ways, switched by one environment variable.

**In the project:** `ChatApiClient.from_env()`: `BELAY_BASE_URL` set means HTTP.

### Q10. What is the API's equivalent of a page object?

An API client object: tests call `chat()` and get a typed reply, never building URLs or parsing JSON.
When the API changes, one class changes.

**In the project:** `qa/client.py`.

### Q11. What do you log for an LLM request?

A trace per request: what was retrieved with its scores, what was answered, latency, token counts, and a
trace ID returned to the client. Personal data is redacted before it's written, so traces can be shared.

**In the project:** `belay/tracing.py`; the gate's latency check reads these traces.

### Q12. What would you add for production observability?

A tracing tool such as Langfuse or LangSmith, or OpenTelemetry spans, so traces are searchable and linked
to user feedback; cost per request; and alerts on the same metrics the gate uses.

## 2. Evaluating RAG answers

### Q13. What can go wrong in a RAG system?

Two halves fail differently. Retrieval can miss the right document or rank it low. Generation can ignore
the document, add facts, or cite the wrong source. You measure them separately to know which half broke.

**In the project:** `qa/retrieval_eval.py` for retrieval, the DeepEval and Ragas suites for answers.

### Q14. Precision, recall and F1?

Precision: of what you returned, how much was right. Recall: of what was right, how much you returned. F1
is their harmonic mean, so it's low if either is low.

### Q15. Recall at k and MRR?

Recall@k: the share of questions with a relevant document in the top k. MRR: the average of 1 / position
of the first relevant document, so rank 1 scores 1 and rank 2 scores 0.5.

**In the project:** reworded questions: MRR 0.857 but recall after the evidence rule only 0.429.

### Q16. What does that gap tell you?

Ranking is fine; the decision to answer is too strict. The right section was usually first, but it shared
only one word with the question, and the rule needs two. That points at the threshold or at better
vectors, not at the ranking.

### Q17. What are embeddings and cosine similarity?

An embedding turns text into a vector, so similar meaning gives nearby vectors. Cosine similarity
measures the angle between two vectors: 1 for the same direction, 0 for nothing in common.

**In the project:** `belay/embeddings.py`: TF-IDF vectors offline, `nomic-embed-text` in the model run.

### Q18. Why combine keyword search and vectors?

Keywords are exact (names, codes, prices); vectors match meaning ("pause" and "freeze"). Hybrid search
gets both. Reciprocal rank fusion merges the two rankings by position, so their scores never need to be
on the same scale.

**In the project:** `Retriever._fuse` in `belay/retrieval.py`.

### Q19. What is "grounded" or "faithful"?

Every claim in the answer is supported by the source it cites. It's the main defence against
hallucination in RAG.

**In the project:** `graders.grounded` checks against the cited section only; DeepEval and Ragas
Faithfulness use an LLM judge.

### Q20. Why check grounding against the cited section and not all retrieved text?

Because the citation is what the user checks. An answer that is true somewhere in the retrieved text but
cites the wrong section misleads the reader.

**In the project:** `test_grounded_only_counts_the_cited_section`.

### Q21. What do BLEU and ROUGE measure?

Word overlap with a reference answer. BLEU counts matching word sequences (precision-like); ROUGE-L uses
the longest common subsequence. They are cheap and repeatable but reward shared words, not shared
meaning, so a correct paraphrase scores low.

**In the project:** Ragas `BleuScore`, `RougeScore`; core ROUGE-L 0.71, reworded 0.31.

### Q22. Then why use them?

As a trend line in the offline gate: if they drop on the same cases, something changed. Meaning is judged
by the LLM metrics in the weekly run.

### Q23. What does Ragas give you?

RAG-specific metrics: Faithfulness, Context Precision and Context Recall (did retrieval bring the right
context, judged against a reference), Answer Relevancy, and non-LLM string metrics.

**In the project:** `qa/ragas_eval.py`, offline by default, judged with `--model`.

### Q24. What does DeepEval give you?

A pytest-style way to assert on LLM outputs, with built-in LLM-judged metrics (Faithfulness, Answer
Relevancy, G-Eval rubrics) and a `BaseMetric` to plug in your own.

**In the project:** our deterministic graders become DeepEval metrics in `qa/deepeval_adapter.py`.

### Q25. What is G-Eval?

An LLM judge given written criteria (a rubric), which scores how well the output meets them. Useful for
"correct compared with the reference" when wording differs.

**In the project:** the `Correctness` rubric in `judge_metrics`.

### Q26. How do you build a golden dataset?

Write each case from the source: the question, the sections that answer it, the facts the answer must
contain, and a reference answer. Version it, review it like code, and validate it on load.

**In the project:** `datasets/*.jsonl`; `qa/cases.py` refuses an "answered" case without expected facts.

### Q27. Why separate core and reworded questions?

Core questions are written from the articles, so they flatter the system. Reworded questions are how
people really ask, and give the honest number.

**In the project:** core 100 %, reworded 50 % right status.

### Q28. How do you test "I don't know"?

Questions the knowledge base can't answer, expected to return "not found", and a measure for it. A wrong
answer is worse than no answer, so they are counted separately.

**In the project:** TC-124 to TC-128 and `no_answer_accuracy`.

## 3. Models, judges and non-determinism

### Q29. How do you deal with a model whose answers vary?

Separate what must be exact from what is measured. The required gate runs a deterministic generator, so
a red build means the code changed. The model runs weekly with limits on averages, and the same question
is asked several times to measure how much answers move.

**In the project:** `ci.yml` and `model-evals.yml`; `qa/stability.py`.

### Q30. Doesn't temperature 0 make a model deterministic?

Mostly, on one machine. Not guaranteed across machines, versions or batch sizes. So you measure it
instead of assuming it.

**In the project:** temperature 0 and a fixed seed in `OllamaGenerator`, plus `same_status`,
`same_citations` and `identical_answer` in the stability report.

### Q31. How does a transformer generate text, in one minute?

It predicts the next token from all previous tokens, using attention to weigh which earlier tokens
matter. Temperature and top-p control how it picks among likely tokens. The context window limits how
much text it can see at once.

### Q32. What is an LLM-as-judge, and what's wrong with it?

A model grading another model's output. It scales, but it varies, can be biased (towards long answers,
or its own style), and can be wrong. Use a different, larger model than the one being graded, calibrate
it against human labels, and don't fail a build on a single judgement.

**In the project:** `qwen2.5:7b` judges `llama3.2:3b`; the gate uses averages.

### Q33. How do you make a judge more reliable?

Clear criteria with what is allowed, not only what is forbidden; low temperature; capped output; a
second grade before a fail counts; and calibration against a labelled set.

**In the project:** the red-team policies say what passes; `qa/redteam.py` keeps only fails both grades agree on.

### Q34. Why is the model run not a required check?

Because a red result there is a finding to review, not proof the code is broken. Making it required would
teach people to re-run until green.

### Q35. Why local models instead of a hosted API?

No cost, no key in CI, and no data leaving the runner. The trade-off is small models on a CPU: an
indication, not a measurement of a production model.

### Q36. How do you choose the number of judged cases?

By time budget. Each judged case costs several judge calls; on a CPU runner that is tens of seconds. The
limit is a parameter, and the same first cases are used each run so runs compare.

**In the project:** `--judge-limit`, default 12.

### Q37. How do you compare retrieval with and without embeddings?

Run the same cases through each mode and compare recall and MRR side by side. Keep the evidence rule the
same, or you're measuring two changes at once.

**In the project:** `qa/retrieval_eval.py` prints bm25, vector and hybrid for both suites.

### Q38. What latency and cost do you track?

Server-side latency per request (p50, p95) from traces, and tokens per request (cost for a paid model).
The gate has a p95 limit to catch an accidental slow path.

## 4. Safety, red teaming and the quality gate

### Q39. What is prompt injection, direct and indirect?

Direct: the user tells the model to ignore its rules. Indirect: the instruction hides in content the
model reads, such as a document or a web page.

**In the project:** TC-301 (direct) and the forum post `B2` (indirect), whose instruction is removed when
the knowledge base is indexed.

### Q40. How do you protect secrets in a RAG system?

Don't index them for that user at all. Guards and prompts are word lists and instructions that can be
talked around; data that was never retrieved can't leak.

**In the project:** staff sections are never indexed for the chat. TC-314 gets past the input guard and still leaks nothing.

### Q41. How do you test for PII leakage?

Plant personal data in the system, attack it directly and indirectly, and assert the data never appears
in any answer, whatever the status. Separately, mask personal data on the way out.

**In the project:** `must_not_contain` in `datasets/attacks.jsonl`; `OutputGuard.redact`.

### Q42. What is red teaming here?

Generated attacks against the running API: a local model writes attacks for each policy, a second model
grades the answers, and failures are graded again before they count.

**In the project:** `promptfooconfig.redteam.yaml`, run weekly.

### Q43. Static attacks or generated ones?

Both. A static set is repeatable and gates every pull request; generated attacks find what you didn't
think of, but vary, so they run weekly.

**In the project:** 16 static attacks in every run; the red team weekly.

### Q44. What is a quality gate for an AI feature?

Limits on measured metrics that fail the build, kept in configuration so changes show in a review. It
must also fail when a report is missing, a suite ran fewer cases, or a metric was renamed.

**In the project:** `qa/gate.py`, `configs/quality_gate.toml`, tested in `tests/gate/`.

### Q45. How do you set the limits?

Must-hold rules at 100 % (no leaks, core questions right). For known weak areas, just below today's
value: it stops things getting worse, and doesn't pretend it's good enough.

### Q46. How do you keep known failures visible?

List them with the measured reason and run them as strict expected failures. They show in every run, and
when one starts passing, the build fails until the list is updated.

**In the project:** `configs/known_misses.toml`.

### Q47. How do you know your checks actually work?

Plant failures: a wrong fact, a wrong citation, an invented detail, a leaked code. A check that has never
failed proves nothing.

**In the project:** `tests/unit/test_graders.py` and the scripted generator in the pipeline tests.

### Q48. Which governance frameworks matter for AI quality?

The EU AI Act (obligations by risk level), the NIST AI Risk Management Framework (govern, map, measure,
manage), and ISO/IEC 42001 (an AI management system standard). For a tester, they turn into documented
evaluations, traceable test data, monitoring and records of known limitations, which is what this
project keeps.
