"""Frozen, manually labeled evidence cases based on public AI announcements."""

from dataclasses import dataclass


def source(title, url, snippet):
    return {"title": title, "url": url, "snippet": snippet}


GPT_RELEASE = "https://openai.com/index/introducing-gpt-oss/"
GPT_NAMES = source("Introducing gpt-oss", GPT_RELEASE,
                   "On August 5, 2025, OpenAI announced gpt-oss-120b and gpt-oss-20b.")
GPT_LICENSE = source("Introducing gpt-oss", GPT_RELEASE,
                     "OpenAI released gpt-oss-120b and gpt-oss-20b under Apache 2.0.")
GPT_BENCH = source("Introducing gpt-oss", GPT_RELEASE,
                   "OpenAI reported near-parity with o4-mini on core reasoning benchmarks for gpt-oss-120b.")

CLAUDE_RELEASE = "https://www.anthropic.com/news/claude-4"
CLAUDE_NAMES = source("Introducing Claude 4", CLAUDE_RELEASE,
                      "Anthropic announced Claude Opus 4 and Sonnet 4 on May 22, 2025.")
CLAUDE_ACCESS = source("Introducing Claude 4", CLAUDE_RELEASE,
                       "At launch Anthropic said Sonnet 4 was available to free Claude users. Opus 4 and Sonnet 4 were included in Pro, Max, Team, and Enterprise plans.")
CLAUDE_BENCH = source("Introducing Claude 4", CLAUDE_RELEASE,
                      "Anthropic reported that Claude Opus 4 scored 72.5% on SWE-bench and 43.2% on Terminal-bench.")
CONFLICT = source("Simulated contradictory launch note", "https://example.com/claude-launch-fixture",
                  "This launch note says Sonnet 4 was limited to paid Claude plans at launch.")

QWEN_RELEASE = "https://qwenlm.github.io/blog/qwen3/"
QWEN_MOE = source("Qwen3 launch", QWEN_RELEASE,
                  "The Qwen team released two mixture-of-experts models, Qwen3-235B-A22B and Qwen3-30B-A3B.")
QWEN_MODES = source("Qwen3 launch", QWEN_RELEASE,
                    "Qwen3 offers a thinking mode for deeper reasoning and a non-thinking mode for faster responses.")

META_RELEASE = "https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/"
META_ACTION = source("Introducing V-JEPA 2", META_RELEASE,
                     "Meta used about 62 hours of robot data in the action-conditioned stage and demonstrated reaching, picking up, and placing objects.")
META_BENCH = source("Introducing V-JEPA 2", META_RELEASE,
                    "Meta released three benchmarks for evaluating physical reasoning from video.")

R1_RELEASE = "https://github.com/deepseek-ai/DeepSeek-R1"
R1_MODELS = source("DeepSeek-R1 release", R1_RELEASE,
                   "DeepSeek released R1-Zero, R1, and six dense models distilled from R1 using Llama and Qwen bases.")
R1_TRAINING = source("DeepSeek-R1 release", R1_RELEASE,
                     "DeepSeek-R1-Zero used reinforcement learning without a preliminary supervised fine-tuning stage. R1 incorporated cold-start data before reinforcement learning.")


@dataclass(frozen=True)
class ResearchCase:
    id: str
    split: str
    as_of: str
    question: str
    expected: str
    evidence_rubric: str
    results: tuple
    follow_up_results: tuple = ()


CASES = (
    ResearchCase("gpt-names", "dev", "2025-08-05", "What two gpt-oss models did OpenAI announce on August 5?",
                 "supported", "Both model names are present.", (GPT_NAMES,)),
    ResearchCase("gpt-license-gap", "dev", "2025-08-05", "Which gpt-oss models were announced and under what license?",
                 "partial", "Names are supported. License is absent from the initial excerpt.", (GPT_NAMES,), (GPT_LICENSE,)),
    ResearchCase("gpt-benchmark-gap", "dev", "2025-08-05", "What did OpenAI report about gpt-oss-120b on reasoning and on customer support tasks?",
                 "partial", "Reasoning comparison is present. Customer support performance is absent.", (GPT_BENCH,)),
    ResearchCase("gpt-irrelevant", "dev", "2025-08-05", "What license did OpenAI use for gpt-oss?",
                 "insufficient", "A Qwen3 release does not establish the gpt-oss license.", (QWEN_MOE,), (GPT_LICENSE,)),
    ResearchCase("claude-access", "dev", "2025-05-22", "Which Claude 4 model was available to free users at launch?",
                 "supported", "The launch availability excerpt explicitly names Sonnet 4.", (CLAUDE_ACCESS,)),
    ResearchCase("claude-announcement", "dev", "2025-05-22", "Which Claude 4 models were announced and which could free users use at launch?",
                 "partial", "The names are present. Free access is absent from the initial excerpt.", (CLAUDE_NAMES,), (CLAUDE_ACCESS,)),
    ResearchCase("claude-conflict", "dev", "2025-05-22", "Was Sonnet 4 available to free Claude users at launch?",
                 "partial", "The excerpts conflict on free access. The official release is stronger evidence.", (CLAUDE_ACCESS, CONFLICT)),
    ResearchCase("claude-no-results", "dev", "2025-05-22", "What SWE-bench result did Anthropic report for Opus 4?",
                 "insufficient", "No initial excerpts. The follow-up contains the reported score.", (), (CLAUDE_BENCH,)),
    ResearchCase("qwen-moe", "held-out", "2025-04-29", "Which mixture-of-experts models were part of the Qwen3 launch?",
                 "supported", "Both MoE model names are present.", (QWEN_MOE,)),
    ResearchCase("qwen-modes", "held-out", "2025-04-29", "What two response modes did the Qwen3 launch describe?",
                 "supported", "Thinking and non-thinking modes are both described.", (QWEN_MODES,)),
    ResearchCase("qwen-dense-gap", "held-out", "2025-04-29", "How many Qwen3 MoE and dense models were released?",
                 "partial", "Two MoE models are listed. The dense model count is absent.", (QWEN_MOE,)),
    ResearchCase("qwen-unrelated", "held-out", "2025-04-29", "Did the Qwen3 launch describe a non-thinking mode?",
                 "insufficient", "A Claude 4 excerpt supplies no Qwen3 evidence.", (CLAUDE_NAMES,), (QWEN_MODES,)),
    ResearchCase("meta-robot-data", "held-out", "2025-06-11", "How much robot data did Meta report using in the V-JEPA 2 action-conditioned stage?",
                 "supported", "The excerpt states about 62 hours and names the training stage.", (META_ACTION,)),
    ResearchCase("meta-benchmark-count", "held-out", "2025-06-11", "How many physical reasoning benchmarks did Meta release alongside V-JEPA 2?",
                 "supported", "The excerpt gives three and states the benchmark scope.", (META_BENCH,)),
    ResearchCase("meta-benchmark-gap", "held-out", "2025-06-11", "How many benchmarks did Meta release with V-JEPA 2, and what were their names?",
                 "partial", "The count is present. Benchmark names are absent.", (META_BENCH,)),
    ResearchCase("meta-safety-gap", "held-out", "2025-06-11", "Did Meta establish V-JEPA 2 safety for unsupervised factory deployment?",
                 "insufficient", "Short lab demonstrations do not establish factory safety.", (META_ACTION,)),
    ResearchCase("r1-training", "held-out", "2025-01-20", "How did DeepSeek-R1 and R1-Zero differ before reinforcement learning?",
                 "supported", "The excerpt contrasts cold-start data with absence of preliminary supervised fine-tuning.", (R1_TRAINING,)),
    ResearchCase("r1-distill-gap", "held-out", "2025-01-20", "How many R1 distill checkpoints were released, and what were their individual sizes?",
                 "partial", "Six and the model families are present. Individual sizes are absent.", (R1_MODELS,)),
    ResearchCase("r1-irrelevant", "held-out", "2025-01-20", "Which model families did DeepSeek use for its R1 distill checkpoints?",
                 "insufficient", "The Qwen3 modes excerpt does not discuss DeepSeek distillation.", (QWEN_MODES,), (R1_MODELS,)),
    ResearchCase("r1-no-results", "held-out", "2025-01-20", "Did R1 use cold-start data before reinforcement learning?",
                 "insufficient", "No initial excerpts. The follow-up directly addresses cold-start data.", (), (R1_TRAINING,)),
)
