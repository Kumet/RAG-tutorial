import logging
import os
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import BaseMessage
from langchain.schema.runnable import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import time
from langchain.schema.output_parser import StrOutputParser


load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = f"20231019 Compare Default Role"


def default_chat() -> Runnable[dict, str]:
    system_template = "あなたは、質問者からの言葉に日本語で{max_characters}文字以内で会話してください。"
    human_template = "{question}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    model = ChatOpenAI()
    chain = prompt | model | StrOutputParser()
    return chain


def default_chat_old(chat: ChatOpenAI, text: str, max_characters: int) -> str:
    system_template = "あなたは、質問者からの言葉に{language}で{max_characters}文字以内で会話してください。"
    human_template = "{question}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    prompt_message_list = chat_prompt.format_prompt(
        language="日本語", max_characters=max_characters, question=text
    ).to_messages()
    result = chat(prompt_message_list)
    return result.content


def main() -> None:
    chat = ChatOpenAI()
    max_characters = 50
    chain = default_chat()
    question = "お腹すいたー。"
    time.sleep(3)

    result = default_chat_old(chat, question, max_characters)
    print(result)
    print(chain.invoke({"question": question, "max_characters": max_characters}))


if __name__ == '__main__':
    main()
