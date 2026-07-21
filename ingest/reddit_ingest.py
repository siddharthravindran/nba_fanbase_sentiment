"""Pull posts/comments from NBA team subreddits via PRAW."""
import os
from datetime import datetime, timezone

import praw
from dotenv import load_dotenv

load_dotenv()

TEAM_SUBREDDITS = {
    "Lakers": "lakers",
    "Warriors": "warriors",
    "Celtics": "bostonceltics",
    # TODO: fill in remaining 27 teams
}


def get_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def fetch_recent_posts(reddit: praw.Reddit, subreddit: str, limit: int = 100):
    for submission in reddit.subreddit(subreddit).new(limit=limit):
        yield {
            "source": "reddit",
            "subreddit": subreddit,
            "id": submission.id,
            "text": f"{submission.title}\n{submission.selftext}",
            "url": submission.url,
            "created_utc": datetime.fromtimestamp(
                submission.created_utc, tz=timezone.utc
            ).isoformat(),
        }


if __name__ == "__main__":
    reddit = get_reddit_client()
    for team, sub in TEAM_SUBREDDITS.items():
        for post in fetch_recent_posts(reddit, sub, limit=10):
            print(team, post["id"], post["text"][:60])
