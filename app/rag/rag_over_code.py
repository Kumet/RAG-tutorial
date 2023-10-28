from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()


def main():
    # Loading
    repo_path = "/Users/kume/Desktop/tmp/langchain-master"
    loader = GenericLoader.from_filesystem(
        repo_path + "/libs/langchain/langchain",
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()
    print(len(documents))  # 1893

    # Splitting
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=2000,
        chunk_overlap=200,
    )
    texts = python_splitter.split_documents(documents)
    print(len(texts))  # 5354

    # RetrievalQA
    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever(
        search_type="similarity",  # or Similarity
        # search_kwargs={"k", 8},
    )

    # Chat
    llm = ChatOpenAI(model_name="gpt-4")
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    question = "How can I initialize a ReAct agent"
    result = qa(question)
    print(result)


if __name__ == '__main__':
    main()
