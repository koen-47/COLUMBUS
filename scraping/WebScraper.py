import time
import requests

from bs4 import BeautifulSoup


class WebScraper:
    """
    Base class for web scrapers.
    """
    save_dir = "./data/data"

    def __init__(self):
        pass

    def scrape(self):
        """
        Base function for starting to scrape.
        """
        pass

    def create_parser(self, url, max_retries=30, timeout=10):
        """
        Create a BS4 parser for the specified URL and retry in case of failure.

        :param url: URL to create the parser for.
        :param max_retries: maximum number of times to retry creating the parser if there is an error.
        :param timeout: number of seconds to wait between each retry.
        :return:
        """
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

