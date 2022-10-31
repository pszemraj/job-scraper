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


def extract_job_title(
    job_elem, target_html: str = "span", target_class: str = "title", verbose=False
):
    """
    extract_job_title_indeed - extracts the job title from the indeed job element

    Args:
        job_elem (BeautifulSoup object): the job element to extract the title from
        verbose (bool, optional):  if True, then it will print the extracted title

    Returns:
        str: the extracted job title
    """
    title_elem = job_elem.select_one(f"{target_html}[{target_class}]").text
    if verbose:
        logging.info(title_elem)
    try:
        title = title_elem.strip()
    except:
        title = "no title"
    return title


def extract_company(
    job_elem,
    target_html: str = "span",
    target_class: str = "companyName",
):
    """
    extract_company_indeed - extracts the company name from the indeed job element

    Args:
        job_elem (BeautifulSoup object): the job element to extract the company name from

    Returns:
        str: the extracted company name
    """
    company_elem = job_elem.find(target_html, class_=target_class)
    company = company_elem.text.strip()
    return company


def extract_link(job_elem, uURL):
    """
    extract_link_indeedCH - extracts the link to the job posting from the indeed job element
            some manual shenanigans occur here
            working example https://ch.indeed.com/Stellen?q=data&jt=internship&lang=en&vjk=49ed864bd5e422fb
    Args:
        job_elem (BeautifulSoup object): the job element to extract the link from
        uURL (str, optional): the URL of the search query

    Returns:
        str: the extracted link to the job posting
    """
    #

    link = job_elem.find("a")["href"]
    uURL_list = uURL.split("&fromage=last")
    link = uURL_list[0] + "&" + link
    return link.replace("/rc/clk?jk=", "vjk=")


def extract_date(job_elem, target_html: str = "span", target_class: str = "date", verbose=False):
    """
    extract_date_indeed extracts the date the job was posted

    Args:
        job_elem (bs4.element.Tag): bs4 element containing the job information

    Returns:
        date (str): date the job was posted
    """
    date_elem = job_elem.find(target_html, class_=target_class)
    date = date_elem.text.strip()
    if verbose:
        logging.info(date)
    return date


def extract_summary(job_elem, target_html: str = "div", target_class: str = "job-snippet", verbose=False):
    """
    extract_summary_indeed extracts the summary of the job posting

    Args:
        job_elem (bs4.element.Tag): bs4 element containing the job information

    Returns:
        summary (str): summary of the job posting
    """
    summary_elem = job_elem.find(target_html, class_=target_class)
    summary = summary_elem.text.strip()
    if verbose:
        logging.info(summary)
    return summary

