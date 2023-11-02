import os
from functools import partial
from operator import itemgetter

from dotenv import load_dotenv
from langchain.callbacks.manager import trace_as_chain_group
from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.chains import RetrievalQA

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, \
    AIMessagePromptTemplate

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = f"20231101 compare chain_type_rag"


def combine_source_documents(documents: list[Document]) -> str:
    """ソースドキュメントの内容を\n\nで繋げて返す"""
    docs = [doc.page_content for doc in documents]
    docs = "\n\n".join(docs)
    return docs


def get_vectorstore():
    data_path = "../data/jiji.txt"
    loader = TextLoader(data_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0, separator="\n")
    docs = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    return vectorstore


def stuff(text: str):
    model = ChatOpenAI(temperature=0)

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    template = """以下のcontextのみに基づいて質問に答えて下さい。
        {context}

        質問: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    # chain作成
    retrieved_documents = {
        "question": RunnablePassthrough(),
        "docs": itemgetter("question") | retriever,
    }
    inputs = {
        "context": itemgetter("docs"),
        "question": itemgetter("question")
    }
    answer = {
        "answer": inputs | prompt | model | StrOutputParser(),
        "docs": itemgetter("docs") | RunnableLambda(combine_source_documents),
    }
    chain = RunnablePassthrough() | retrieved_documents | answer

    print(chain.invoke({"question": text}))


def refine(text: str):
    model = ChatOpenAI(temperature=0)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    retrieved_documents = {
        "question": RunnablePassthrough(),
        "docs": itemgetter("question") | retriever,
    }

    first_template = """Context information is below.
    ------------
    {context}
    ------------
    Given the context information and not prior knowledge, answer any questions
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(first_template),
        HumanMessagePromptTemplate.from_template(f"{text}"),
    ])

    doc_prompt = PromptTemplate.from_template("{page_content}")
    partial_format_doc = partial(format_document, prompt=doc_prompt)
    first_chain = {"context": partial_format_doc} | prompt | model | StrOutputParser()

    refine_template = """We have the opportunity to refine the existing answer (only if needed) with some more context below.
    ------------
    {context}
    ------------
    Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.
    """
    refine_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(f"{text}"),
        AIMessagePromptTemplate.from_template("{prev_response}"),
        HumanMessagePromptTemplate.from_template(refine_template)
    ])
    _input = {
        "prev_response": itemgetter("prev_response"),
        "context": lambda x: partial_format_doc(x["doc"]),
    }
    refine_chain = (_input | refine_prompt | model | StrOutputParser())

    def refine_loop(docs):
        docs = docs['context']
        summary = first_chain.invoke(docs[0])
        for i, doc in enumerate(docs[1:]):
            summary = refine_chain.invoke({"prev_response": summary, "doc": doc})
        return summary

    inputs = {"context": itemgetter("docs"), "question": itemgetter("question")}
    answer = {
        "answer": inputs | RunnableLambda(refine_loop),
        "docs": itemgetter("docs") | RunnableLambda(combine_source_documents),
    }
    chain = RunnablePassthrough() | retrieved_documents | answer

    print(chain.invoke({"question": text}))


def map_reduce(text: str):
    model = ChatOpenAI(temperature=0)

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    doc_prompt = PromptTemplate.from_template("{page_content}")
    partial_format_doc = partial(format_document, prompt=doc_prompt)
    map_prompt = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
    Return any relevant text verbatim.
    ______________________
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(map_prompt),
        HumanMessagePromptTemplate.from_template(f"{text}"),
    ])

    map_chain = {"context": partial_format_doc} | prompt | model | StrOutputParser()
    map_as_doc_chain = (
            RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
            | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
    )

    def format_docs(docs):
        return "\n\n".join(partial_format_doc(doc) for doc in docs)

    reduce_template = """Given the following extracted parts of a long document and a question, create a final answer. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ______________________
    {context}
    """
    reduce_chain = (
            {"context": format_docs}
            | PromptTemplate.from_template(reduce_template)
            | model
            | StrOutputParser()
    ).with_config(run_name="Reduce")

    retrieved_documents = {
        "question": RunnablePassthrough(),
        "docs": itemgetter("question") | retriever,
    }

    map_reduce_chain = map_as_doc_chain.map() | reduce_chain

    inputs = {"context": itemgetter("docs"), "question": itemgetter("question")}
    answer = {
        "answer": inputs["context"] | map_reduce_chain,
        "docs": itemgetter("docs") | RunnableLambda(combine_source_documents),
    }
    chain = RunnablePassthrough() | retrieved_documents | answer
    print(chain.invoke({"question": text})["answer"])


def map_rerank(text: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    map_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:
    
    Question: [question here]
    Helpful Answer: [answer here]
    Score: [score between 0 and 100]
    
    How to determine the score:
    - Higher is a better answer
    - Better responds fully to the asked question, with sufficient level of detail
    - If you do not know the answer based on the context, that should be a score of 0
    - Don't be overconfident!
    
    Begin!
    
    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Helpful Answer:
    """
    map_prompt = PromptTemplate.from_template(map_template)

    class AnswerAndScore(BaseModel):
        """Return the answer to the question and a relevance score."""

        answer: str = Field(
            description="The answer to the question, which is based ONLY on the provided context."
        )
        score: float = Field(
            decsription="A 0.0-1.0 relevance score, where 1.0 indicates the provided context answers the question completely and 0.0 indicates the provided context does not answer the question at all."
        )

    function = convert_pydantic_to_openai_function(AnswerAndScore)
    model = ChatOpenAI().bind(temperature=0, functions=[function], function_call={"name": "AnswerAndScore"})

    map_chain = (
            map_prompt
            | model
            | PydanticOutputFunctionsParser(pydantic_schema=AnswerAndScore)
    ).with_config(run_name="Map")

    def top_answer(scored_answers):
        return max(scored_answers, key=lambda x: x.score).answer

    document_prompt = PromptTemplate.from_template("{page_content}")
    retrieved_documents = {
        "question": RunnablePassthrough(),
        "docs": itemgetter("question") | retriever,
    }
    doc_chain = RunnablePassthrough() | retrieved_documents
    docs = doc_chain.invoke({"question": text})["docs"]

    map_rerank_chain = (
            (lambda x:
             [{"context": format_document(doc, document_prompt), "question": x["question"]} for doc in x["docs"]])
            | map_chain.map()
            | top_answer
    )

    # print(map_rerank_chain.invoke({"docs": docs, "question": text}))

    inputs = {"docs": itemgetter("docs"), "question": itemgetter("question")}
    answer = {
        "answer": inputs | map_rerank_chain,
        "docs": itemgetter("docs") | RunnableLambda(combine_source_documents),
    }
    chain = RunnablePassthrough() | retrieved_documents | answer
    print(chain.invoke({"question": text}))


if __name__ == '__main__':
    text = "キングオブコント優勝したの誰？"
    # map_reduce(text)
    map_rerank(text)
    # refine(text)
