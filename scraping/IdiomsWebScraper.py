import re
import json

from tqdm import tqdm

from .WebScraper import WebScraper


class IdiomsWebScraper(WebScraper):
    """
    Class to scrape www.theidioms.com"
    """
    def __init__(self):
        super().__init__()
        self._base_url = "https://www.theidioms.com/list/"
        self._n_pages = 165

    def scrape(self):
        """
        Start scraping theidioms.com
        """
        idiom_data = []
        for i in tqdm(range(1, self._n_pages + 1), desc="Collecting idioms (theidioms.com)"):
            url = f"{self._base_url}/page/{i}"
            idiom_data += self._scrape_page(url)
        with open("./theidioms_raw.json", "w+") as file:
            json.dump(idiom_data, file, indent=3)

    def _scrape_page(self, url):
        """
        Scrape each page that contains information on an idiom.
        :param url: url of the page containing idioms.
        :return: list of idioms, their meaning and an example of the idiom being used.
        """
        soup = self.create_parser(url)
        idioms = soup.find(id="phrase").find_all(class_="idiom")
        idiom_data = [re.split("Meaning:|Example:|Read more", idiom.text)[:-1] for idiom in idioms]
        idiom_data = [{"idiom": idiom[0].strip(), "meaning": idiom[1].strip(), "example": idiom[2].strip()} for idiom
                      in idiom_data]
        return idiom_data
