from pprint import pprint

from config import settings
from services.faculty.dynamic_scraper import DynamicFacultyScraperPipeline

BASE_URL = "https://science.oregonstate.edu"
LIST_PATH = "/directory"
PROFILE_LIMIT = 1

pipeline = DynamicFacultyScraperPipeline(
    llm_model_id= (settings.opus or settings.sonnet or settings.haiku or "").strip(),
    sample_profile_count=2,
    timeout=60,
)

scraper = pipeline.generate_scraper(base_url=BASE_URL, list_path=LIST_PATH)

print("=== Generated Scraper Code ===")
print(pipeline.last_generated_code or "<no generated code>")
print("\n=== Profile URLs ===")

list_html = pipeline.fetch_html(f"{BASE_URL}{LIST_PATH}")
profile_urls = scraper.extract_profile_links(list_html, BASE_URL)[:PROFILE_LIMIT]
print("Found", len(profile_urls), "profile URLs")
for url in profile_urls:
    print("-", url)

print("\n=== Parsed Profile Results ===")
results = []
for profile_url in profile_urls:
    profile_html = pipeline.fetch_html(profile_url)
    parsed = scraper.parse_profile(profile_html, profile_url)
    results.append(parsed)

pprint(results)

