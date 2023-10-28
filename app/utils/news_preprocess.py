import re

from langchain.document_loaders import PyPDFLoader
from pathlib import Path


def split_news_pdf(pdf_path: str | Path, output_path: str | Path) -> None:
    pdf_path, output_path = Path(pdf_path), Path(output_path)

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load_and_split()
    result = []
    for page in pages:
        sections = re.split(r'(?=10/)', page.page_content.strip())
        sections = sections[1:]
        result += sections

    with open(output_path, "w") as file:
        for item in result:
            item = item.replace("\n", "")
            file.write(f"{item}\n")

    print("saved", output_path)


def complete_news(txt_path: str | Path, output_path: str | Path) -> None:
    txt_path, output_path = Path(txt_path), Path(output_path)

    with open(str(txt_path), "r") as file:
        news_list = [line.strip() for line in file]

    result = []
    for news in news_list:
        filled_text = fill_in_answer(news)
        result.append(filled_text)

    with open(output_path, "w") as file:
        for item in result:
            item = item.replace("\n", "")
            file.write(f"{item}\n")

    print("saved", output_path)


def fill_in_answer(text: str) -> str:
    news, answer = text.split("　①")
    answer = "①" + answer
    answers = answer.split("\u3000")

    for answer in answers:
        index, answer_text = answer[0], answer[1:]

        old_news = news
        news = news.replace(f"【{index}】", answer_text)
        news = news.replace(f"{index}", answer_text)
    return news


def main() -> None:
    pdf_path = "../data/jiji.pdf"
    save_text_path = "../data/separated_news.txt"
    output_path = "../data/jiji.txt"
    # split_news_pdf(pdf_path, save_text_path)
    complete_news(save_text_path, output_path)


if __name__ == '__main__':
    main()
