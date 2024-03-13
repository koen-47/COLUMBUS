import time
import requests

from bs4 import BeautifulSoup


class WebScraper:
    save_dir = "./data/saved"

    def __init__(self):
        pass

    def scrape(self):
        pass

    def create_parser(self, url, max_retries=30, timeout=10):
        page = None
        retry_counter = 1
        while retry_counter < max_retries:
            try:
                page = requests.get(url)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(timeout)
                retry_counter += 1
                continue
        if page is None:
            raise TimeoutError(f"Unable to create BeautifulSoup parser after {max_retries} tries with a timeout "
                               f"interval of {timeout}")
        return BeautifulSoup(page.content, "html.parser")

