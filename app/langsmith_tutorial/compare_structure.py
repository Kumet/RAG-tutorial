import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = f"20231019 Compare Structure"


def get_vectorstore():
    data_path = "../data/jiji.txt"
    loader = TextLoader(data_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=10, separator="\n")
    docs = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    return vectorstore


def default_news_role(text: str) -> str:
    llm = ChatOpenAI(temperature=0)
    vectorstore = get_vectorstore()
    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type="map_reduce",
        # chain_type="refine",
    )

    query = f"{text}"
    answer = qa({"query": query})
    return answer['result']


def new_news_role(text: str) -> str:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    template = """以下のcontextのみに基づいて質問に答えて下さい。
    {context}
    
    質問: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    result = chain.invoke(text)
    return result


if __name__ == '__main__':
    # question = "キングオブコントに優勝したのは誰ですか？"
    question = "ジャニーズ事務所の次の名前は？"
    print(default_news_role(question))
    print(new_news_role(question))
