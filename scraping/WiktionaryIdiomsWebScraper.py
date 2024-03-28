import json

from .WebScraper import WebScraper


class WiktionaryIdiomsWebScraper(WebScraper):
    def __init__(self):
        super().__init__()
        self._base_url = "https://en.wiktionary.org/w/index.php?title=Category:English_idioms&from=AA"

    def scrape(self):
        soup = self.create_parser(self._base_url)
        links = soup.find("div", {"id": "mw-pages"}).find_all("a")
        idioms = []
        while len(links) > 2 and links[1].text == "next page":
            url = "https://en.m.wiktionary.org/" + links[1]["href"]
            soup = self.create_parser(url)
            links = soup.find("div", {"id": "mw-pages"}).find_all("a")
            idioms += [element.text for element in links[2:-2]]
        with open("wiktionary_idioms_raw.json", "w") as file:
            json.dump(idioms, file, indent=3)



