from operator import itemgetter

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document
from dotenv import load_dotenv


load_dotenv()


def rag():
    vectorstore = FAISS.from_texts(["harriston worked at Kumeta's home"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    template = """Anser the question baed only on the followeing context::
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = chain.invoke("Where did harrison work?")
    print(result)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Answer in the following language: {language}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    } | prompt | model | StrOutputParser()
    
    result = chain.invoke({"question": "where did harrison work?", "language": "japanese"})
    print(result)


def conversational_retrieval_chain():
    vectorstore = FAISS.from_texts(["harriston worked at Kumeta's home"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    _template = """Given the following covnersation and a followi up question, rephrase the follow up question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)

    template = """Answer the question based only the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="page_content")
    
    def _combine_documents(docs, document_prompt = DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
    
    def _format_chat_history(chat_history: list[tuple]) -> str:
        buffer = ""
        for dialogue_turn in chat_history:
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        return buffer

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
        chain_history=lambda x: _format_chat_history(x['chat_history'])
        ) | CONDENSE_QUESTION_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser(),
    )
    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"]
    }
    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
    result = conversational_qa_chain.invoke({
        "question": "where did harrison work?",
        "chat_history": [],
    })
    print(result)

if __name__ == "__main__":
    conversational_retrieval_chain()

