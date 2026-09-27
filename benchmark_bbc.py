import urllib.request
import xml.etree.ElementTree as ET
import json
import random

feeds = {
    "World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Sports": "https://feeds.bbci.co.uk/sport/rss.xml",
    "Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Sci/Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml"
}

options_str = "World, Sports, Business, Sci/Tech"
examples = []

for category, url in feeds.items():
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        tree = ET.fromstring(response.read())
        items = tree.findall('.//item')
        
        # Take up to 10 articles from each
        for item in items[:10]:
            title = item.find('title').text
            desc = item.find('description').text if item.find('description') is not None else ""
            text = f"{title}. {desc}".replace("\n", " ").strip()
            
            prompt = (
                f"Classify the following news article into one of these categories: {options_str}.\n\n"
                f"Article: {text}\n\n"
                f"Options: {options_str}\n"
                f"Category:"
            )
            examples.append({"prompt": prompt, "completion": f" {category}", "label": category, "text": text})
    except Exception as e:
        print(f"Error fetching {category}: {e}")

random.seed(42)
random.shuffle(examples)

with open("bbc_test.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Created bbc_test.jsonl with {len(examples)} real BBC articles.")
