import pandas as pd
import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
import time
import random
from urllib.parse import urljoin
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, timeout_handler)

# -----------------------
# CONFIG
# -----------------------

TSV_PATH = "data/corpus.tsv"
OUTPUT_PATH = "data/bias_dataset_v2.csv"

MAX_SOURCES = 150
MAX_ARTICLES_PER_SOURCE = 80
MIN_TEXT_LENGTH = 250

SLEEP_RANGE = (0.05, 1.2)

# -----------------------
# LABEL MAPPING
# -----------------------

def map_bias(label):
    label = str(label).lower()

    if "left" in label:
        return 0
    elif "center" in label:
        return 1
    elif "right" in label:
        return 2
    return None

# -----------------------
# BIAS SIGNAL DETECTION
# -----------------------

LEFT_TERMS = [
    "climate crisis", "gun violence", "systemic racism",
    "inequality", "social justice", "healthcare access"
]

RIGHT_TERMS = [
    "border crisis", "illegal immigration", "tax burden",
    "big government", "second amendment", "law and order"
]

LOADED_WORDS = [
    "attack", "pressure", "demand", "crisis",
    "radical", "extreme", "collapse", "threat",
    "revenge", "failure"
]

def compute_bias_signal(text):
    text_lower = text.lower()

    left_score = sum(term in text_lower for term in LEFT_TERMS)
    right_score = sum(term in text_lower for term in RIGHT_TERMS)
    loaded_score = sum(word in text_lower for word in LOADED_WORDS)

    return left_score, right_score, loaded_score

# -----------------------
# ARTICLE FILTERING
# -----------------------

def is_opinion_or_analysis(url, text):
    url = url.lower()

    keywords = [
        "opinion", "analysis", "editorial",
        "commentary", "column"
    ]

    if any(k in url for k in keywords):
        return True

    # fallback: strong language
    if sum(w in text.lower() for w in LOADED_WORDS) >= 2:
        return True

    return False

# -----------------------
# RSS DISCOVERY
# -----------------------

def find_rss_feed(base_url):
    for path in ["/rss", "/feed", "/rss.xml", "/feed.xml"]:
        try:
            url = base_url.rstrip("/") + path
            feed = feedparser.parse(url)
            if len(feed.entries) > 5:
                return url
        except:
            continue
    return None

# -----------------------
# HOMEPAGE SCRAPING
# -----------------------

def extract_links_from_homepage(base_url):
    links = []

    try:
        r = requests.get(base_url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if href.startswith("/"):
                href = urljoin(base_url, href)

            if base_url in href and len(href) > len(base_url) + 10:
                links.append(href)

    except:
        pass

    return list(set(links))

# -----------------------
# ARTICLE EXTRACTION
# -----------------------

def extract_article(url):
    try:
        article = Article(url)

        # 🔥 ADD TIMEOUT HERE
        article.download(input_html=requests.get(url, timeout=5).text)
        article.parse()

        text = article.text.strip()

        if len(text) < MIN_TEXT_LENGTH:
            return None

        return text

    except Exception as e:
        return None
# -----------------------
# MAIN PIPELINE
# -----------------------

def build_dataset():
    df = pd.read_csv(TSV_PATH, sep="\t")

    df["label"] = df["bias"].apply(map_bias)
    df = df.dropna(subset=["label"])

    df = df.sample(MAX_SOURCES)

    data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        base_url = row["source_url"]
        source_label = int(row["label"])
        count = 0

        print(f"\n🔍 Processing: {base_url}")

        feed_url = find_rss_feed(base_url)
        links = []

        if feed_url:
            feed = feedparser.parse(feed_url)

            links = []
            for entry in feed.entries:
                link = None

                if hasattr(entry, "link"):
                    link = entry.link
                elif hasattr(entry, "links") and len(entry.links) > 0:
                    link = entry.links[0].get("href")
                elif hasattr(entry, "id"):
                    link = entry.id

                if link and link.startswith("http"):
                    links.append(link)

        else:
            links = extract_links_from_homepage(base_url)


# 🔥 THIS WAS MISSING — MAIN LOOP
        for link in links:
            if count >= MAX_ARTICLES_PER_SOURCE:
                break
            
            try:
                signal.alarm(6)  # ⏱ max 6 seconds per article

                text = extract_article(link)

                signal.alarm(0)

            except TimeoutException:
                print(f"⏱ Timeout: {link}")
                continue

            if not text:
                continue

            # Compute signals
            left_score, right_score, loaded_score = compute_bias_signal(text)

            # Filter
            if not is_opinion_or_analysis(link, text) and loaded_score < 1:
                continue

            # Adjust label
            adjusted_label = source_label
            if left_score - right_score >= 2:
                adjusted_label = 0
            elif right_score - left_score >= 2:
                adjusted_label = 2

            data.append({
                "text": text,
                "label": adjusted_label,
                "source": base_url,
                "url": link,
                "left_score": left_score,
                "right_score": right_score,
                "loaded_score": loaded_score
            })

            count += 1
            time.sleep(random.uniform(*SLEEP_RANGE))

        print(f"📄 Collected {count} filtered articles")

    dataset = pd.DataFrame(data)

    print("\n📊 Dataset Distribution:")
    print(dataset["label"].value_counts())

    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"\n💾 Saved to {OUTPUT_PATH}")
    print(f"Total samples: {len(dataset)}")

# -----------------------
# RUN
# -----------------------

if __name__ == "__main__":
    build_dataset()