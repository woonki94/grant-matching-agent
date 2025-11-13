import requests
import xml.etree.ElementTree as ET


def fetch_crossref_abstract(title: str):
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 1}
    r = requests.get(url, params=params).json()

    items = r.get("message", {}).get("items", [])
    if not items:
        return None

    return items[0].get("abstract")


def fetch_openalex_abstract(title: str):
    url = "https://api.openalex.org/works"
    params = {"filter": f"title.search:{title}"}
    res = requests.get(url, params=params).json()
    results = res.get("results")
    if not results:
        return None
    return results[0].get("abstract")

def fetch_arxiv_abstracts(query: str, max_results: int = 5):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f'ti:"{query}"',
        "max_results": max_results,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"arxiv": "http://www.w3.org/2005/Atom"}

    results = []
    for entry in root.findall("arxiv:entry", ns):
        title = entry.find("arxiv:title", ns)
        abstract = entry.find("arxiv:summary", ns)
        if title is None or abstract is None:
            continue
        results.append((title.text.strip(), abstract.text.strip()))

    return results

def fetch_semantic_scholar_abstract(title: str):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "title,abstract,year,authors"
    }
    res = requests.get(url, params=params).json()
    data = res.get("data")
    if not data:
        return None
    return data[0].get("abstract")

if __name__ == '__main__':

    print(fetch_crossref_abstract("Explainable reinforcement learning via reward decomposition"))
    print(fetch_openalex_abstract("Explainable reinforcement learning via reward decomposition"))
    print(fetch_semantic_scholar_abstract("Explainable Reinforcement Learning via Temporal Policy Decomposition"))
    print(fetch_arxiv_abstracts("Explainable Reinforcement Learning via Temporal Policy Decomposition"))