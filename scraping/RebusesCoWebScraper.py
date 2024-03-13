import json

from tqdm import tqdm

from .WebScraper import WebScraper


class RebusesCoWebScraper(WebScraper):
    def __init__(self):
        super().__init__()
        self._base_url = "https://www.rebuses.co/free/"
        self._n_pages = 99

    def scrape(self):
        rebus_puzzle_data = []
        for i in tqdm(range(1, self._n_pages + 1), desc="Collecting rebus puzzles (rebuses.co)"):
            url = f"{self._base_url}/page/{i}"
            rebus_puzzle_data += self._scrape_page(url)
        with open("./rebuses_co_raw.json", "w+") as file:
            json.dump(rebus_puzzle_data, file, indent=3)

    def _scrape_page(self, url):
        soup = self.create_parser(url)
        articles = soup.find_all(class_="article-archive")
        rebus_urls = [article.find("a").get("href") for article in articles]

        page_data = []
        for rebus_url in rebus_urls:
            page_data.append(self._scrape_rebus_page(rebus_url))
        return page_data

    def _scrape_rebus_page(self, url):
        soup = self.create_parser(url)
        content = soup.find(class_="content blog-single")
        rebus_img = content.find("img").get("src")
        rebus_answer = content.find_all(class_="toggle-inner")[1].text
        category_tags = set([tag.text for tag in content.find_all("a", {"rel": "category"})])
        type_tag = set([tag.text for tag in content.find_all("a", {"rel": "tag"})])
        type_tag = type_tag.difference(category_tags)

        return {
            "image_url": rebus_img,
            "answer": rebus_answer,
            "category_tags": list(category_tags),
            "type_tags": list(type_tag)
        }
