# Custom will also try find the user-agent string in the browsers.json,
# If a match is found, it will use the headers and cipherSuite from that "browser",
# Otherwise a generic set of headers and cipherSuite will be used.

import logging
import random

logging.basicConfig(level=logging.INFO, format="%(message)s")
import re

import cloudscraper
import requests
from requests.adapters import HTTPAdapter


def check_if_nodeJS_is_installed():
    logger = logging.getLogger(__name__)
    try:
        import cloudscraper

        scraper = cloudscraper.create_scraper(interpreter="nodejs")
        _ = scraper.get("https://www.google.com").text
        logger.info("NodeJS is installed")
        return True
    except Exception as e:
        logger.error(f"NodeJS is not installed: {e}")
        return False


def get_scraper(
    template="generic", browser_kwargs: dict = None, headers=None, cipherSuite=None
):
    logger = logging.getLogger(__name__)
    if template.lower() == "requests_only":
        logger.info("Using requests only")

        return requests.Session()
    if template.lower() == "generic":
        logger.info("Using generic scraper")
        scraper = cloudscraper.create_scraper(
            delay=random.SystemRandom().randint(5, 13),
            # interpreter="nodejs" if check_if_nodeJS_is_installed() else "native",
            browser={
                "browser": "chrome",
                "platform": "windows",
                "mobile": False,
            },
        )
        return scraper

    if template.lower() == "custom":
        logger.info("Using custom scraper")
        scraper = cloudscraper.create_scraper(
            browser={
                "custom": "ScraperBot/1.0",
            }
        )
        return scraper

    if browser_kwargs is None:
        browser_kwargs = {}

    if headers is None:
        headers = {}

    if cipherSuite is None:
        cipherSuite = []

    scraper = cloudscraper.create_scraper(
        delay=10,
        interpreter="nodejs" if check_if_nodeJS_is_installed() else "native",
        browser={
            "browser": "chrome",
            "platform": "windows",
            "mobile": False,
            "custom": True,
            "custom_headers": headers,
            "custom_cipherSuite": cipherSuite,
            **browser_kwargs,
        },
    )
    scraper.mount("https://", HTTPAdapter(max_retries=3))
    return scraper
