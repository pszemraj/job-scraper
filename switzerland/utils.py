import logging
import os
import sys
from datetime import datetime

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
import pprint as pp
import random
import time
import warnings

import gensim.downloader as api
import pandas as pd
import pyshorteners
import texthero as hero

with warnings.catch_warnings():

    warnings.simplefilter("ignore")
    # set stdout to None to suppress output
    with open(os.devnull, "w") as devnull:

        old_stdout = sys.stdout
        sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        old_logging = logging.getLogger().handlers
        logging.getLogger().handlers = []
        # load the model
        import tensorflow as tf
        import tensorflow_hub as hub

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.getLogger().handlers = old_logging

    logging.info("imported tensorflow and tensorflow_hub")

def save_jobs_to_excel(jobs_list: list, file_path: str, verbose=False):
    """
    save_jobs_to_excel takes a list of dictionaries, each containing a job posting and saves it to an excel file

    Args:
        jobs_list (list): list of dictionaries, each containing a job posting
        file_path (str): path to save the excel file to
        verbose (bool, optional): Defaults to False.

    Returns:
        pd.DataFrame: dataframe of the jobs_list
    """
    df = pd.DataFrame(jobs_list)
    df.to_excel(file_path)
    logging.info("saved the following to excel with filename {}: \n".format(file_path))

    if verbose:

        print(df.info())
    return df


def shorten_URL_bitly(
    long_url: str,
    ACCESS_TOKEN: str = "",
    max_sleep_time: int = 5,
    verbose=False,
):
    """
    shorten_URL_bitly takes a long url and returns a shortened url using the bitly API

                    requires free account / API token. https://bitly.com/

    Args:
        long_url (str): long url to shorten
        ACCESS_TOKEN (str, optional): bitly API token. Defaults to "".
        max_sleep_time (int, optional): max time to sleep between requests. Defaults to 5.
        verbose (bool, optional): Defaults to False.

    Returns:
        str: shortened url
    """

    time.sleep(random.randint(1, max_sleep_time))  # don't overload API

    try:
        s = pyshorteners.Shortener(api_key=ACCESS_TOKEN)
        short_url = s.bitly.short(long_url)

        if verbose:
            logging.info("Short URL is {}".format(short_url))
    except Exception as e:
        print("Error: {}".format(e))
        short_url = long_url

    return short_url


def text_first_N(text, num=40):
    """
    text_first_N takes a string and returns the first N characters

    Args:
        text (str): string to shorten
        num (int, optional): number of characters to return. Defaults to 40.

    Returns:
        str: first N characters of text
    """

    text = " ".join(text) if isinstance(text, list) else str(text)

    return text[:num] + "..." if len(text) > num else text


def load_gensim_word2vec(
    word2vec_model: str = "glove-wiki-gigaword-300", verbose=False
):

    logging.info("loading gensim word2vec model {}".format(word2vec_model))
    loaded_model = api.load(word2vec_model)

    logging.info("loaded data for word2vec - ", datetime.now())

    if verbose:
        # for more info or bug fixing
        wrdvecs = pd.DataFrame(loaded_model.vectors, index=loaded_model.key_to_index)
        logging.info("created dataframe from word2vec data- ", datetime.now())
        logging.info("dimensions of the df: \n", wrdvecs.shape)

    if verbose:
        print("testing gensim model...")
        test_string = "computer"
        vector = loaded_model.wv[test_string]

        print("The shape of string {} is: \n {}".format(test_string, vector.shape))
        print("test complete - ", datetime.now())

    return loaded_model


def load_google_USE(url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
    """
    load_google_USE loads the google USE model from the URL

    Args:
        url (str): URL to the model, defaults to "https://tfhub.dev/google/universal-sentence-encoder/4"

    Returns:
        [type]: [description]
    """
    """helper function to load the google USE model"""
    st = time.perf_counter()
    embed = hub.load(url)
    rt = round((time.perf_counter() - st) / 60, 2)
    logging.info("Loaded Google USE in {} minutes".format(rt))
    return embed



def indeed_postprocess(
    indeed_df, query_term, query_jobtype, verbose=False, shorten_links=False
):
    """
    indeed_postprocess - postprocesses the indeed dataframe

    Args:
        indeed_df (pandas dataframe): the indeed dataframe to postprocess
        query_term (str): the query term used to search indeed
        query_jobtype (str): the job type used to search indeed
        verbose (bool, optional): . Defaults to False.
        shorten_links (bool, optional): if True, then it will shorten the links. Defaults to False.

    Returns:
        pandas dataframe: the postprocessed indeed dataframe
    """
    logging.info("Starting postprocess - ", datetime.now())

    # apply texthero cleaning
    indeed_df["titles"] = hero.clean(indeed_df["titles"])
    indeed_df["summary"] = hero.clean(indeed_df["summary"])

    # use bit.ly to shorten links
    if shorten_links:
        try:
            len(indeed_df["short_link"])
            logging.info("found values for short_link, not-recreating")
        except:
            logging.info("no values exist for short_link, creating them now")
            # there is a random delay to not overload APIs, max rt is 5s * num_rows
            indeed_df["short_link"] = indeed_df["links"].apply(shorten_URL_bitly)
    else:
        indeed_df["short_link"] = "not_created"

    # save file to excel
    rn = datetime.now()
    i_PP_date = rn.strftime("_%m.%d.%Y-%H-%M_")
    indeed_df["date_pulled"] = rn.strftime("%m.%d.%Y")
    indeed_df["time_pulled"] = rn.strftime("%H:%M:%S")
    out_name = (
        "JS_DB_"
        + "query=[term(s)="
        + query_term
        + ", type="
        + query_jobtype
        + "]"
        + i_PP_date
        + ".xlsx"
    )
    indeed_df.to_excel(out_name)
    if verbose:
        logging.info("Saved {} - ".format(out_name), datetime.now())

    # download if requested
    return indeed_df


def indeed_datatable(indeed_df, count_what="companies", freq_n=10):
    """
    indeed_datatable - creates a datatable of the top companies or job titles

    Args:
        indeed_df (pandas dataframe): the indeed dataframe to create the datatable from
        count_what (str, optional): what to count, either "companies" or "titles". Defaults to "companies".
        freq_n (int, optional): how many to count. Defaults to 10.

    Returns:
        datatable: the datatable of the top companies or job titles
    """
    # basically just wrote this to reduce code down below
    # depends on the colab data_table.DataTable()

    logging.info("Count of column '{}' appearances in search:\n".format(count_what))
    comp_list_1 = indeed_df[count_what].value_counts()
    pp.pprint(comp_list_1.head(freq_n), compact=True)

    display_df = indeed_df.copy()
    display_df["summary_short"] = display_df["summary"].apply(text_first_N)
    display_df.drop(columns=["links", "summary"], inplace=True)  # drop verbose columns

    return display_df
