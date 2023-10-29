from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()


def main():
    text_path = "../data/jiji.txt"
    loader = TextLoader(text_path)
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=10, separator="\n")

    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        documents=docs, embedding=embeddings
    )

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
    question = "キングオブコントの優勝者は誰ですか？"
    answer = chain.invoke(question)
    print(answer)


if __name__ == '__main__':
    main()
