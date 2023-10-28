from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()


def main():
    text_path = "../data/jiji.txt"
    loader = TextLoader(text_path)
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=10, separator="\n")

    documents = loader.load()
    docs = text_splitter.split_documents(documents)

    print(docs)
    print(len(docs))


if __name__ == '__main__':
    main()
