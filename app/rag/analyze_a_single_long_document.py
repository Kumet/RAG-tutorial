from dotenv import load_dotenv
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

load_dotenv()


def main():
    text_path = "../data/jiji.txt"
    with open(text_path) as f:
        news_text = f.read()
    llm = OpenAI(temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")
    qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
    result = qa_document_chain.run(input_document=news_text, question="キングオブコントを優勝したのは誰ですか？")
    print(result)


if __name__ == '__main__':
    main()
