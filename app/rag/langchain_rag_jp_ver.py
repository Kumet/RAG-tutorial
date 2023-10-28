from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import format_document, Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain.vectorstores import FAISS

load_dotenv()


def rag() -> None:
    vectorstore = FAISS.from_texts(["太郎は東京で働いている"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    template = """以下のcontextのみに基づいて質問に答えなさい。
    {context}
    
    質問: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    question = "太郎はどこで働いていますか？"
    answer = chain.invoke(question)
    print(answer)

    template = """以下のcontextのみに基づいて質問に答えなさい。
    {context}

    質問: {question}
    Answer in the following language: {language}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
                "language": itemgetter("language"),
            }
            | prompt
            | model
            | StrOutputParser()
    )
    answer = chain.invoke({"question": '太郎はどこで働いていますか？', "language": "英語"})
    print(answer)


def conversational_retrieval_chain() -> None:
    vectorstore = FAISS.from_texts(["太郎は東京で働いている"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    _template = """次のような会話とフォローアップの質問がある場合、フォローアップの質問を元の言葉で、独立した質問となるように言い換えてください。
    
    会話履歴:
    {chat_history}
    フォローアップの質問: {question}
    独立した質問:"""
    condense_question_prompt = PromptTemplate.from_template(_template)

    template = """以下のcontextのみに基づいて質問に答えなさい。
    {context}

    質問: {question}
    """
    answer_prompt = ChatPromptTemplate.from_template(template)

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x['chat_history'])
        ) | condense_question_prompt | ChatOpenAI(temperature=0) | StrOutputParser(),
    )

    _context = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"]
    }

    conversational_qa_chain = _inputs | _context | answer_prompt | ChatOpenAI()

    answer = conversational_qa_chain.invoke({
        "question": "太郎はどこで働いていますか？",
        "chat_history": [("誰がこの本を書きましたか？", "太郎です")],
    })
    print(answer)


def with_memory_and_returning_source_documents():
    vectorstore = FAISS.from_texts(["太郎は東京で働いている"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    _template = """次のような会話とフォローアップの質問がある場合、フォローアップの質問を元の言葉で、独立した質問となるように言い換えてください。

    会話履歴:
    {chat_history}
    フォローアップの質問: {question}
    独立した質問:"""
    condense_question_prompt = PromptTemplate.from_template(_template)

    template = """以下のcontextのみに基づいて質問に答えなさい。
    {context}

    質問: {question}
    """
    answer_prompt = ChatPromptTemplate.from_template(template)
    memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
    # 最初にメモリをロードするステップを追加
    # input objectに "memory "キーを追加する
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )

    # standalone questionの計算
    standalone_question = {
        "standalone_question": {
                                   "question": lambda x: x["question"],
                                   "chat_history": lambda x: _format_chat_history(x["chat_history"])
                               } | condense_question_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    }

    # retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # 最終プロンプトの入力を作成
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question")
    }

    # returns the answers
    answer = {
        "answer": final_inputs | answer_prompt | ChatOpenAI(),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    inputs = {"question": "太郎はどこで働いていますか？"}
    result = final_chain.invoke(inputs)
    print(result)


def _combine_documents(
        docs: list[Document],
        document_prompt: PromptTemplate = PromptTemplate.from_template(template="{page_content}"),
        document_separator: str = ""
) -> str:
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: list[tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


if __name__ == '__main__':
    # rag()
    # conversational_retrieval_chain()
    with_memory_and_returning_source_documents()
