from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.langchain import RagasEvaluatorChain


load_dotenv()


def evaluate_qa_chain():
    # load the wikipedia page and create index
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/New_York_City")
    index = VectorstoreIndexCreator().from_loaders([loader])

    # create the QA chain
    llm = ChatOpenAI()
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=index.vectorstore.as_retriever(), return_source_documents=True,
    )

    # testing it out
    question = "How did New York City get its name?"
    result = qa_chain({"query": question})
    print(result["result"])

    # make eval chains
    eval_chains = {
        m.name: RagasEvaluatorChain(metric=m)
        for m in [faithfulness, answer_relevancy, context_recall]
    }

    for name, eval_chain in eval_chains.items():
        score_name = f"{name}_score"
        print(f"{score_name}: {eval_chain(result)[score_name]}")


if __name__ == '__main__':
    evaluate_qa_chain()
