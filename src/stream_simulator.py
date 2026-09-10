"""Real-time social media stream simulator for TwitterSentiment-Bridge.

Emulates live Twitter/X streaming feeds across multiple domains (Tech, Crypto,
Aviation, E-Commerce) with timestamps, user handles, and engagement metrics.
"""

import datetime
import random
import time
from typing import Dict, Generator, List, Optional


TOPIC_TWEET_BANKS: Dict[str, List[str]] = {
    "tech": [
        "The new AI features in the IDE are mind-blowing! Coding 5x faster today. 🚀💻",
        "Another day, another broken dependency after updating the package manager. 🤬",
        "Deploying to production on a Friday afternoon... what could go wrong?",
        "Sub-millisecond inference with this hybrid architecture is seriously impressive.",
        "System latency jumped from 20ms to 800ms out of nowhere. Investigating...",
        "Just open-sourced our model evaluation pipeline! Check it out on GitHub. ⭐",
        "Memory leak in the worker process caused our container to OOM crash. 🤦‍♂️",
        "The documentation is super clear and the onboarding was seamless. Kudos to the team!",
        "Why is CSS grid still confusing in 2026? 😅",
        "Huge performance boost after caching intermediate embeddings. Love it!",
    ],
    "crypto": [
        "Bitcoin breaking through resistance levels! Bullish divergence confirmed. 📈💎",
        "Gas fees are insane right now, just spent $45 on a simple swap. Ridiculous.",
        "Smart contract audit passed with zero vulnerabilities found. Ready for mainnet! 🛡️",
        "Flash crash wiped out leverage positions in seconds. Stay safe out there.",
        "Staking rewards deposited successfully. Passive yield compounding nicely.",
        "Project team rugged and deleted their discord. Complete scam. 🚨",
        "Decentralized storage protocol is scaling smoothly under heavy load.",
        "Regulatory uncertainty is killing liquidity across all major exchanges.",
        "The new zero-knowledge rollup is blazingly fast and dirt cheap. Impressive tech! 🔥",
    ],
    "aviation": [
        "Smooth flight and arrived 20 minutes early into Chicago! Great crew. ✈️🌤️",
        "Stuck on the tarmac for 2 hours with no air conditioning. Never flying this airline again.",
        "Upgraded to first class for free! Today is my lucky day. 🥂",
        "Lost my checked luggage with all my business clothes. Unbelievable incompetence.",
        "Flight attendants were so warm and accommodating with the kids. Thank you! ❤️",
        "Another delay due to crew scheduling issues. Third time this month.",
        "Sunset over the clouds at 35,000 feet was breathtaking today. 🌅",
    ],
    "ecommerce": [
        "Package arrived within 24 hours in perfect condition. Incredible service! 📦✨",
        "Received a completely different item and customer support refuses to issue a refund. 😡",
        "High quality material and fits true to size. Will definitely order again.",
        "The checkout button is completely unresponsive on mobile Safari. Lost a sale.",
        "Loved the eco-friendly packaging and personalized thank you note! 🌱",
        "Charged twice on my credit card and still haven't received an order confirmation.",
        "Hands down the best customer support experience I have had all year. 10/10.",
    ],
}

HANDLES = [
    ("Sarah Connor", "sarah_c_tech"),
    ("Alex Rivera", "arivera_dev"),
    ("CryptoWhale", "whale_watcher_99"),
    ("Jordan Lee", "jlee_flyer"),
    ("Morgan Smith", "morgan_reviews"),
    ("TechGuru", "cloud_architect_x"),
    ("David Kim", "dkim_datascience"),
    ("Elena Rostova", "elena_nlp"),
    ("TravelBug", "nomad_chronicles"),
    ("Marcus Vance", "vance_macro"),
]


class SocialStreamSimulator:
    """Generates synthetic, realistic social media posts on a streaming schedule."""

    def __init__(self, default_topic: str = "tech"):
        self.default_topic = default_topic

    def generate_tweet(self, topic: Optional[str] = None) -> Dict:
        """Generate a single realistic tweet with metadata."""
        selected_topic = (topic or self.default_topic).lower()
        bank = TOPIC_TWEET_BANKS.get(selected_topic, TOPIC_TWEET_BANKS["tech"])

        user_name, handle = random.choice(HANDLES)
        text = random.choice(bank)

        # Introduce slight variations to make synthetic stream dynamic
        variants = ["", " #update", " #thoughts", " Really!", " Smh.", " Highly recommend!"]
        if random.random() < 0.3:
            text = f"{text}{random.choice(variants)}"

        tweet_id = f"tw_{random.randint(10000000, 99999999)}"
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return {
            "id": tweet_id,
            "user": user_name,
            "handle": f"@{handle}",
            "text": text,
            "topic": selected_topic,
            "timestamp": timestamp,
            "retweets": random.randint(0, 350),
            "likes": random.randint(1, 1200),
        }

    def stream(
        self,
        topic: Optional[str] = None,
        interval_seconds: float = 1.0,
        max_items: Optional[int] = None,
    ) -> Generator[Dict, None, None]:
        """Stream simulated tweets indefinitely or up to max_items."""
        count = 0
        while max_items is None or count < max_items:
            yield self.generate_tweet(topic=topic)
            count += 1
            if interval_seconds > 0:
                time.sleep(interval_seconds)


if __name__ == "__main__":
    simulator = SocialStreamSimulator(default_topic="tech")
    print("Streaming 5 sample tweets:")
    for t in simulator.stream(max_items=5, interval_seconds=0.2):
        print(f"[{t['handle']} | {t['timestamp']}] {t['text']}")
