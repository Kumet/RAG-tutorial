import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()


def get_latest_news(api_key, country='jp', category="general", page_size=100):
    url = 'https://newsapi.org/v2/top-headlines'
    params = {
        'apiKey': api_key,
        'country': country,
        'category': category,
        'pageSize': page_size,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Failed to retrieve news: {response.status_code}')
        return None


def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    api_key = os.getenv("NEWS_API_KEY")
    category = "science"

    news_data = get_latest_news(api_key, category=category)
    if news_data:
        save_to_file(news_data, f'news_data_{category}.json')
        print(f'News data saved to news_data_{category}.json')
    else:
        print('Failed to retrieve or save news data')
