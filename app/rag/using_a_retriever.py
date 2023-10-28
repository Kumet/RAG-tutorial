from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()


def chain_type():
    # text_path = "../data/jiji.txt"
    text_path = "../data/jiji_mini.txt"
    loader = TextLoader(text_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    query = "キングオブコントで優勝したのは誰ですか？"
    result = qa.run(query)
    print(result)  # サルゴリラです。


def custom_prompts():
    text_path = "../data/jiji_mini.txt"
    loader = TextLoader(text_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    prompt_template = """以下の文脈を利用して、最後の質問に答えなさい。答えがわからない場合は、答えを作ろうとせず、わからないと答えましょう。
    
    {context}
    
    質問: {question}
    日本語で回答してください。
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        # chain_type="mmr",
        # retriever=docsearch.as_retriever(),
        retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 30}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    query = "キングオブコントで優勝したのは誰ですか？"
    result = qa({"query": query})
    print(result)  # サルゴリラです。


if __name__ == '__main__':
    custom_prompts()
