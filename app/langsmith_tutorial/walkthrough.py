import os
from typing import List

from dotenv import load_dotenv
from langchain import chat_models, prompts
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.evaluation import load_evaluator
from langchain.schema.retriever import BaseRetriever, Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.smith import RunEvalConfig
from langsmith import Client
from langsmith.evaluation import RunEvaluator, EvaluationResult

load_dotenv()


class MyRetriever(BaseRetriever):
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content="Example")]


class FaithfulnessEvaluator(RunEvaluator):

    def __init__(self):
        self.evaluator = load_evaluator(
            "labeled_score_string",
            criteria={"faithful": "How faithful is the submission to the reference context?"},
            normalize_by=10,
        )

    def evaluate_run(self, run, example) -> EvaluationResult:
        res = self.evaluator.evaluate_strings(
            prediction=next(iter(run.outputs.values())),
            input=run.inputs["question"],
            # We are treating the documents as the reference context in this case.
            reference=example.inputs["documents"],
        )
        return EvaluationResult(key="labeled_criteria:faithful", **res)


def main() -> None:
    # unique_id = uuid4().hex[0:8]
    unique_id = "03852968"
    os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"

    # Create a dataset
    examples = [
        {
            "inputs": {
                "question": "What's the company's total revenue for q2 of 2022?",
                "documents": [
                    {
                        "metadata": {},
                        "page_content": "In q1 the lemonade company made $4.95. In q2 revenue increased by a sizeable amount to just over $2T dollars."
                    }
                ],
            },
            "outputs": {
                "label": "2 trillion dollars",
            },
        },
        {
            "inputs": {
                "question": "Who is Lebron?",
                "documents": [
                    {
                        "metadata": {},
                        "page_content": "On Thursday, February 16, Lebron James was nominated as President of the United States."
                    }
                ],
            },
            "outputs": {
                "label": "Lebron James is the President of the USA.",
            },
        }
    ]

    client = Client()
    dataset_name = F"Faithfulness Example - {unique_id}"
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )

    # Define Chain
    response_synthesizer = (
            prompts.ChatPromptTemplate.from_messages(
                [
                    ("system", "Respond using the following documents as context:\n{documents}"),
                    ("user", "{question}")
                ]
            ) | chat_models.ChatAnthropic(model="claude-2", max_tokens=1000)
    )

    chain = (
            {
                "documents": MyRetriever(),
                "question": RunnablePassthrough(),
            }
            | response_synthesizer
    )

    # Evaluate
    eval_config = RunEvalConfig(
        evaluators=["qa"],
        custom_evaluators=[FaithfulnessEvaluator()],
        input_key="question",
    )
    results = client.run_on_dataset(
        llm_or_chain_factory=response_synthesizer,
        dataset_name=dataset_name,
        evaluation=eval_config,
    )
    print(results)


if __name__ == '__main__':
    main()
