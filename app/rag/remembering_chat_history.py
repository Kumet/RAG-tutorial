from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()


def main():
    text_path = "../data/jiji_mini.txt"
    loader = TextLoader(text_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0),
        vectorstore.as_retriever(),
        memory=memory,
    )
    query = "キングオブコントの優勝者は誰ですか？"
    result = qa({"question": query})
    print(result)
    print(result["answer"])

    query = "キングオブコントはいつ行われましたか？"
    result = qa({"question": query})
    print(result)
    print(result["answer"])


def pass_in_chat_history():
    text_path = "../data/jiji_mini.txt"
    loader = TextLoader(text_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever())

    chat_history = []
    query = "キングオブコントの優勝者は誰ですか"
    result = qa({"question": query, "chat_history": chat_history})
    print(result)

    chat_history = [(query, result["answer"])]
    query = "キングオブコントについて知っていることを教えて下さい。"
    result = qa({"question": query, "chat_history": chat_history})
    print(result)


if __name__ == '__main__':
    pass_in_chat_history()
