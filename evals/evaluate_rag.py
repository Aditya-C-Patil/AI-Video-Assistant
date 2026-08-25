import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from langchain_mistralai import ChatMistralAI
from core.vector_store import build_vector_store, get_retriever
from core.rag_engine import get_llm, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def run_evaluation():
    print("🚀 Starting Automated RAG Evaluation Suite...")
    
    with open("evals/benchmark_data.json", "r") as f:
        benchmarks = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting assistant. Answer based ONLY on context:\n\n{context}\nIf not found, state: 'I could not find this information in the meeting transcript.'"),
        ("human", "{question}")
    ])

    for item in benchmarks:
        q = item["user_input"]
        ref = item["reference"]
        ctx_raw = item["context"]

        # 1. Build temporary vector store for this test context
        vstore = build_vector_store(ctx_raw, reset=True)
        retriever = get_retriever(vstore, k=2)
        retrieved_docs = retriever.invoke(q)
        formatted_ctx = [doc.page_content for doc in retrieved_docs]

        # 2. Generate response from LCEL chain
        chain = prompt | llm | StrOutputParser()
        ans = chain.invoke({"context": "\n\n".join(formatted_ctx), "question": q})

        questions.append(q)
        answers.append(ans)
        contexts.append(formatted_ctx)
        ground_truths.append(ref)

    # Prepare dataset for Ragas
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Run Ragas scoring using Mistral
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm
    )

    df_results = results.to_pandas()
    os.makedirs("evals/results", exist_ok=True)
    df_results.to_csv("evals/results/rag_scores.csv", index=False)
    
    print("\n📊 RAG Evaluation Summary Scores:")
    print(results)
    return results

if __name__ == "__main__":
    run_evaluation()
