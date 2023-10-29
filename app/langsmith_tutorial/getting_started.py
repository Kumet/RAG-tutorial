from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()


def main() -> None:
    llm = ChatOpenAI()
    llm.invoke("hello, world!")


if __name__ == '__main__':
    main()
