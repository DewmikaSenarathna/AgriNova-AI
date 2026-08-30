# Phase 14 — Evaluate the System

Phase 14 adds a repeatable evaluation layer for AgriNova AI.

The goal is not only:

> "Does the application return an answer?"

The goal is to measure whether the answer is **retrieved from the right evidence, generated reliably, routed through the right agents/tools, fast enough, and useful to a farmer.**

## Metrics

| Area | Metric | How it is measured |
|---|---|---|
| Retrieval | Recall@K | Relevant gold documents retrieved in top K |
| Retrieval | Precision@K | Relevant gold documents / K |
| Retrieval | Keyword evidence coverage | Temporary proxy before gold labels are available |
| Generation | Groundedness | Evidence-backed answer + valid source citation |
| Generation | Factual accuracy proxy | Required-fact phrase coverage |
| Agent workflow | Task success rate | Expected agent selection + grounded answer + factual proxy |
| Agent workflow | Agent selection accuracy | Actual selected agents vs expected agents |
| Performance | End-to-end latency | Average, median, p95 and maximum |
| Reliability | Error rate | Failed evaluation cases / total cases |
| User experience | Satisfaction | Optional human rating from 1–5 |
| Tools | Tool selection accuracy | Actual inferred tools vs expected tools |

### Important evaluation rule

`factual_accuracy_proxy` is deliberately labelled a **proxy**. Phrase matching cannot prove that an agricultural claim is true. For a serious evaluation, a human/domain expert should verify the answer against the reference answer or source document.

## Folder structure

```text
backend/Agents-Pipeline/
├── evaluation/
│   ├── evaluation_dataset.json
│   ├── ratings.json
│   ├── evaluator.py
│   └── README.md
├── output/
│   └── evaluation/
│       └── latest_report.json
└── run_evaluation.py
```

## Step 1 — Install dependencies

Use the same virtual environment already used for the Agents Pipeline.

```powershell
cd backend\Agents-Pipeline

pip install -r ..\RAG-Pipeline\requirements.txt
pip install -r requirements.txt
```

## Step 2 — Make sure the knowledge base is ready

Phase 14 depends on the existing RAG/agent pipeline.

Run the document-processing and chunking/embedding pipelines first if the ChromaDB collection is empty.

Then make sure your LLM provider is configured in the existing RAG Pipeline `.env`.

## Step 3 — Configure the evaluation dataset

Open:

```text
backend/Agents-Pipeline/evaluation/evaluation_dataset.json
```

For each test case, define:

- `expected_agents`
- `expected_tools`
- `relevant_doc_ids` — manually curated gold document IDs for real Recall@K / Precision@K
- `required_evidence_terms` — optional temporary retrieval proxy
- `required_facts` — optional factual-coverage checks
- `reference_answer` — optional human/domain-expert reference answer

The included cases are starter cases. **Do not treat their empty `relevant_doc_ids` as a completed academic gold set.** After your knowledge base is stable, label the relevant documents/chunks for each question.

## Step 4 — Run the complete evaluation

From:

```text
backend\Agents-Pipeline
```

run:

```powershell
python run_evaluation.py
```

The program will:

1. Load the evaluation cases.
2. Run each case through the real `AgentOrchestrator`.
3. Measure end-to-end latency.
4. Record planner/agent selection.
5. Infer tool usage from the existing Phase 9 evidence.
6. Measure groundedness.
7. Calculate factual-accuracy proxy when facts are supplied.
8. Calculate retrieval metrics when gold labels are supplied.
9. Calculate task success and error rate.
10. Save a JSON report.

The output is:

```text
backend/Agents-Pipeline/output/evaluation/latest_report.json
```

## Step 5 — Run one test case

Example:

```powershell
python run_evaluation.py --case disease_01
```

Multiple cases:

```powershell
python run_evaluation.py --case disease_01 --case weather_01
```

## Step 6 — Record user satisfaction

After a real user/farmer test, enter a 1–5 score in:

```text
evaluation/ratings.json
```

Example:

```json
{
  "disease_01": 5,
  "weather_01": 4,
  "market_01": 4
}
```

Where:

- `1` = very dissatisfied
- `2` = dissatisfied
- `3` = neutral
- `4` = satisfied
- `5` = very satisfied

Then run:

```powershell
python run_evaluation.py --ratings evaluation/ratings.json
```

## Step 7 — Understand the report

Example:

```text
Task success rate       : 85.7%
Error rate              : 0.0%
Groundedness            : 78.6%
Agent selection accuracy: 100.0%
Tool selection accuracy : 85.7%
Satisfaction (1-5)      : 4.33
Latency ms              : avg=..., median=..., p95=..., max=...
```

These numbers answer different questions.

### Task success rate

> Did AgriNova AI complete the intended workflow correctly?

A case is considered successful when:

- the request did not fail,
- expected agents were selected when labels exist,
- the answer has acceptable grounding,
- and factual-accuracy proxy is acceptable when configured.

### Recall@K

Suppose the gold set for a question is:

```text
DOC-A, DOC-B
```

and top-5 retrieval returns:

```text
DOC-A, DOC-X, DOC-Y, DOC-B, DOC-Z
```

Then:

```text
Recall@5 = 2 / 2 = 100%
```

### Precision@K

For the same result:

```text
Precision@5 = 2 / 5 = 40%
```

High recall means the system finds the needed evidence.

High precision means the returned evidence is mostly relevant.

### Groundedness

This checks whether the final answer is actually supported by retrieved/tool evidence and uses source citations.

### Latency

The report provides:

- Average
- Median
- p95
- Maximum

For an agentic system, **p95 is especially useful** because a system can have a good average while some requests are extremely slow.

### Error rate

```text
Error rate = failed cases / total cases
```

This includes orchestration failures.

Individual agent failures are also visible in the individual case output.

### Tool selection accuracy

This compares the expected tools with the tools inferred from the actual agent evidence.

For example:

```text
Expected:
weather_api + vector_database

Actual:
weather_api + vector_database

Result:
100%
```

## Step 8 — Improve the evaluation over time

For a stronger academic/project evaluation:

1. Create at least 30–50 realistic farmer questions.
2. Cover disease, pest, fertilizer, soil, weather, market and government topics.
3. Add multi-intent questions.
4. Add follow-up questions using Phase 11 memory.
5. Add questions where the correct answer is "I don't know".
6. Manually label the relevant documents for each retrieval case.
7. Ask an agriculture-aware reviewer to score factual accuracy.
8. Record 1–5 satisfaction ratings from multiple users.
9. Run the evaluation before and after major architecture changes.
10. Keep each JSON report so improvements can be compared over time.

## Phase 14 architecture

```text
Evaluation Dataset
       |
       v
SystemEvaluator
       |
       +--> AgentOrchestrator
       |       |
       |       +--> Planner
       |       +--> Specialist Agents
       |       +--> External Tools
       |       +--> Report Agent
       |
       +--> Retrieval Metrics
       +--> Generation Metrics
       +--> Workflow Metrics
       +--> Latency Metrics
       +--> Reliability Metrics
       +--> Tool Metrics
       +--> UX Metrics
       |
       v
latest_report.json
```
