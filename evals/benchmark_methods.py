import sys
import types
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Compatibility shim for Ragas internal LangChain imports
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

import json
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.rag_engine import get_llm
from evals.retrieval_strategies import (
    build_simple_retriever,
    build_semantic_retriever,
    build_hybrid_retriever,
    build_reranked_retriever,
)

warnings.filterwarnings("ignore")
load_dotenv()


def evaluate_retriever_method(method_name: str, builder_func, benchmarks, llm):
    print(f"\n⚡ Running Method Benchmark: [{method_name}]...")
    user_inputs, responses, retrieved_contexts, references = [], [], [], []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert assistant. Answer based ONLY on context:\n\n{context}\nIf missing, say 'I could not find this information in the meeting transcript.'",
            ),
            ("human", "{question}"),
        ]
    )

    for i, item in enumerate(benchmarks):
        q = item.get("user_input") or item.get("question")
        ref = item.get("reference") or item.get("ground_truth")
        ctx_raw = item.get("context")

        retriever = builder_func(ctx_raw, meeting_id=f"test_{i}")
        docs = retriever.invoke(q)
        formatted_ctx = [d.page_content for d in docs]

        chain = prompt | llm | StrOutputParser()
        ans = chain.invoke(
            {"context": "\n\n".join(formatted_ctx), "question": q}
        )

        user_inputs.append(q)
        responses.append(ans)
        retrieved_contexts.append(formatted_ctx)
        references.append(ref)

    eval_dataset = Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
        }
    )

    ragas_llm = LangchainLLMWrapper(llm)
    metrics = [
        Faithfulness(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
        LLMContextPrecisionWithReference(llm=ragas_llm),
    ]

    scores = evaluate(dataset=eval_dataset, metrics=metrics, llm=ragas_llm)
    return scores


def run_full_suite():
    benchmark_path = os.path.join(
        os.path.dirname(__file__), "benchmark_data.json"
    )
    with open(benchmark_path, "r") as f:
        benchmarks = json.load(f)

    llm = get_llm()
    methods = {
        "Simple Chunking": build_simple_retriever,
        "Semantic Chunking": build_semantic_retriever,
        "Hybrid Search": build_hybrid_retriever,
        "Reranking (FlashRank)": build_reranked_retriever,
    }

    summary = []
    for name, func in methods.items():
        eval_result = evaluate_retriever_method(name, func, benchmarks, llm)

        # Convert to dataframe to extract mean column scores
        df_res = eval_result.to_pandas()

        faith_val = (
            float(df_res["faithfulness"].mean())
            if "faithfulness" in df_res
            else 0.0
        )
        recall_val = (
            float(df_res["context_recall"].mean())
            if "context_recall" in df_res
            else 0.0
        )
        prec_val = (
            float(df_res["llm_context_precision_with_reference"].mean())
            if "llm_context_precision_with_reference" in df_res
            else 0.0
        )

        summary.append(
            {
                "Method": name,
                "Faithfulness": round(faith_val, 3),
                "Context Recall": round(recall_val, 3),
                "Context Precision": round(prec_val, 3),
            }
        )

    df = pd.DataFrame(summary)
    print("\n════════════════════════════════════════════════════════════")
    print("🏆 RAG ARCHITECTURE EVALUATION MATRIX")
    print("════════════════════════════════════════════════════════════")
    print(df.to_string(index=False))

    results_path = os.path.join(
        os.path.dirname(__file__), "results", "methods_matrix.csv"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False)

    min_faithfulness = df["Faithfulness"].min()
    if min_faithfulness < 0.85:
        print(
            f"\n❌ Quality Threshold Failed: Minimum faithfulness was {min_faithfulness} (< 0.85)"
        )
        sys.exit(1)
    else:
        print("\n✅ All RAG architectures passed minimum quality threshold.")
        sys.exit(0)


if __name__ == "__main__":
    run_full_suite()
