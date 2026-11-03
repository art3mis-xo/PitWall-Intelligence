import certifi
import feedparser
import requests
import time

RSS_FEEDS = {
    "The Race":   "https://the-race.com/feed/",
    "Autosport":  "https://www.autosport.com/rss/f1/news/",
    "BBC Sport":  "https://feeds.bbci.co.uk/sport/formula1/rss.xml",
    "Motorsport": "https://www.motorsport.com/rss/f1/news/",
}
 
_cache: dict = {"data": [], "fetched_at": 0.0}
CACHE_TTL = 900  # 15 minutes
 
def get_f1_news(max_items: int = 8) -> list[dict]:
    now = time.time()
    if now - _cache["fetched_at"] < CACHE_TTL and _cache["data"]:
        return _cache["data"][:max_items]
 
    items = []
    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "PitWall/1.0 (+F1 news reader)"},
                verify=certifi.where(),
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
            if getattr(feed, "bozo", False) and not feed.entries:
                raise ValueError(str(feed.get("bozo_exception", "invalid feed")))

            for entry in feed.entries[:3]:
                published = entry.get("published", "")
                items.append({
                    "source":    source,
                    "headline":  entry.get("title", ""),
                    "link":      entry.get("link", ""),
                    "published": published,
                    "summary":   entry.get("summary", "")[:200] if entry.get("summary") else "",
                })
        except (requests.RequestException, ValueError) as e:
            print(f"[NEWS] Failed to fetch {source}: {e}")
            continue
 
    items.sort(key=lambda x: x.get("published", ""), reverse=True)
    _cache["data"] = items
    _cache["fetched_at"] = now
 
    return items[:max_items]