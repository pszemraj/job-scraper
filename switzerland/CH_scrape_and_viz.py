import os
import pprint as pp
import random
import time
import urllib
from datetime import date
from datetime import datetime
from os.path import join

import gensim.downloader as api
import numpy as np
import pandas as pd
import plotly.express as px
import pyshorteners
import requests
import tensorflow_hub as hub
import texthero as hero
from bs4 import BeautifulSoup
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def save_jobs_to_excel(jobs_list, filename, verbose=False):
    # i have no idea what this function does
    jobs = pd.DataFrame(jobs_list)
    jobs.to_excel(filename)

    if verbose:
        print("saved the following to excel with filename {}: \n".format(filename))

        print(jobs.info())
    return jobs


def shorten_URL_bitly(long_url, verbose=False):
    # requires free account / API token. https://bitly.com/
    # generate short URLs from the ones scraped

    time.sleep(random.randint(1, 5))  # don't overload API

    ACCESS_TOKEN = "hahah_get_ur_own"

    # Shorten long URL
    try:
        s = pyshorteners.Shortener(api_key=ACCESS_TOKEN)
        short_url = s.bitly.short(long_url)

        if verbose:
            print("Short URL is {}".format(short_url))
    except:
        print(
            "Error accessing API for key {} and fn shorten_URL_bitly".format(
                ACCESS_TOKEN
            )
        )
        print("Try updating API key / checking fn. Returning original url")
        short_url = long_url

    return short_url


def text_first_N(text, num=40):
    # returns the first N chars in text, i.e. for long job descriptions
    # for use with Pandas .apply() function

    text = str(text)  # convert to string

    if isinstance(text, list):
        text = " ".join(text)

    if len(text) <= num:
        return text
    else:
        short_text = text[:num]
        return short_text + ".."


def optimal_num_clustas(
    input_matrix,
    d_title,
    top_end=11,
    show_plot=False,
    write_image=False,
    output_path_full=None,
):
    # given 'input_matrix' as a pandas series containing a list / vector in each
    # row, find the optimal number of k_means clusters to cluster them using
    # the elbow method

    # 'top_end' is the max number of clusters. If having issues, look at the plot
    # and adjust accordingly

    if output_path_full is None:
        output_path_full = os.getcwd()
    scaler = StandardScaler()
    # texthero input data structure is weird.
    #  stole the below if/else from the source code behind TH kmeans fn
    # https://github.com/jbesomi/texthero/blob/master/texthero/representation.py

    if isinstance(input_matrix, pd.DataFrame):
        # fixes weird issues parsing a texthero edited text pd series
        input_matrix_coo = input_matrix.sparse.to_coo()
        input_matrix_for_vectorization = input_matrix_coo.astype("float64")
    else:
        input_matrix_for_vectorization = list(input_matrix)

    scaled_features = scaler.fit_transform(input_matrix_for_vectorization)
    kmeans_kwargs = {
        "init": "random",
        "n_init": 30,
        "max_iter": 300,
        "random_state": 42,
    }
    # A list holds the SSE values for each k
    sse = []
    for k in range(1, top_end):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    # plot to illustrate (viewing it is optional)
    title_k = "Optimal k-means for" + d_title
    kmeans_opt_df = pd.DataFrame(
        list(zip(range(1, top_end), sse)), columns=["Number of Clusters", "SSE"]
    )
    f_k = px.line(kmeans_opt_df, x="Number of Clusters", y="SSE", title=title_k)
    # find optimum
    kl = KneeLocator(range(1, top_end), sse, curve="convex", direction="decreasing")
    onk = kl.elbow

    if onk is None:
        print("Warning - {} has no solution for optimal k-means".format(d_title))
        print("Returning # of clusters as max allowed ( {} )".format(top_end))
        return top_end

    if onk == top_end:
        print(
            "Warning - {} opt. # equals max value searched ({})".format(
                d_title, top_end
            )
        )

    print("\nFor {}: opt. # of k-means clusters is {} \n".format(d_title, onk))
    f_k.add_vline(x=onk)  # add vertical line to plotly

    if show_plot:
        f_k.show()

    if write_image:
        f_k.write_image(join(output_path_full, title_k + ".png"))

    return onk


def viz_job_data(viz_df, text_col_name, save_plot=False, h=720):
    today = date.today()
    # Month abbreviation, day and year
    td_str = today.strftime("%b-%d-%Y")

    viz_df["tfidf"] = viz_df[text_col_name].pipe(hero.clean).pipe(hero.tfidf)

    viz_df["kmeans"] = viz_df["tfidf"].pipe(hero.kmeans, n_clusters=5).astype(str)

    viz_df["pca"] = viz_df["tfidf"].pipe(hero.pca)

    hv_list = list(viz_df.columns)
    hv_list.remove("tfidf")
    hv_list.remove("pca")
    hv_list.remove("summary")

    plot_title = td_str + " Vizualize Companies by {} Data".format(text_col_name)

    # reformat data so don't have to use built-in plotting

    df_split_pca = pd.DataFrame(viz_df["pca"].to_list(), columns=["pca_x", "pca_y"])
    viz_df.drop(columns="pca", inplace=True)  # drop original PCA column
    viz_df = pd.concat([viz_df, df_split_pca], axis=1)  # merge dataframes

    # plot pca data
    # texthero also features pther ways to reduce dimensions besides pca, see docs

    w = int(h * (4 / 3))

    fig_s = px.scatter(
        viz_df,
        x="pca_x",
        y="pca_y",
        color="kmeans",
        hover_data=hv_list,
        title=plot_title,
        height=h,
        width=w,
        template="plotly_dark",
    )
    fig_s.show()

    if save_plot:
        fig_s.write_html(plot_title + ".html", include_plotlyjs=True)

    print("plot generated - ", datetime.now())


def load_gensim_word2vec(wvmodel="word2vec-google-news-300", verbose=False):
    # another option is the smaller: api.load("word2vec-ruscorpora-300")
    loaded_model = api.load(wvmodel)

    print("loaded data for word2vec - ", datetime.now())

    if verbose:
        # for more info or bug fixing
        wrdvecs = pd.DataFrame(loaded_model.vectors, index=loaded_model.key_to_index)
        print("created dataframe from word2vec data- ", datetime.now())
        print("dimensions of the df: \n", wrdvecs.shape)

    print("testing gensim model...")
    test_string = "computer"
    vector = loaded_model.wv[test_string]

    print("The shape of string {} is: \n {}".format(test_string, vector.shape))
    print("test complete - ", datetime.now())

    return loaded_model


# iterate through all words in the input text generate the word2vec vector for that word, then take the mean of
# those. Only words with length 3 or greater are considered. **note that if a word is not in the google news dataset,
# it is just skipped. If you think the graphs / representations are not good enough, you should check to ensure that
# a large amount of words (or fraction rather) are not being skipped)** * the verbose parameter helps with that


def get_vector_freetext(input_text, model, verbose=0, cutoff=2):
    # verbose = 1 shows you how many words were skipped
    # verbose = 2 tells you each individual skipped word and ^
    # 'cutoff' removes all words with length N or less from the rep. vector

    lower_it = input_text.lower()
    input_words = lower_it.split(" ")  # yes, this is an assumption
    usable_words = [word for word in input_words if len(word) > cutoff]

    list_of_vectors = []
    num_words_total = len(usable_words)
    num_excluded = 0

    for word in usable_words:
        try:
            this_vector = model.wv[word]
            list_of_vectors.append(this_vector)
        except:
            num_excluded += 1
            if verbose == 2:
                print("\nThe word/term {} is not in the model vocab.".format(word))
                print("Excluding from representative vector")

    rep_vec = np.mean(list_of_vectors, axis=0)

    if verbose > 0:
        print(
            "Computed representative vector. Excluded {} words out of {}".format(
                num_excluded, num_words_total
            )
        )

    return rep_vec


def viz_job_data_word2vec(
    viz_df, text_col_name, save_plot=False, h=720, query_name="", show_text=False
):
    today = date.today()
    # Month abbreviation, day and year
    td_str = today.strftime("%b-%d-%Y")

    # compute word2vec avg vector for each row of text
    viz_df["avg_vec"] = viz_df[text_col_name].apply(
        get_vector_freetext, args=(w2v_model,)
    )

    # get optimal number of kmeans. limit max to 15 for interpretability
    max_clusters = 15
    if len(viz_df["avg_vec"]) < max_clusters:
        max_clusters = len(viz_df["avg_vec"])

    kmeans_numC = optimal_num_clustas(
        viz_df["avg_vec"], d_title="word2vec-" + query_name, top_end=max_clusters
    )

    # complete k-means clustering + pca dim red. w/ avg_vec
    if kmeans_numC is None:
        kmeans_numC = 5

    viz_df["kmeans"] = (
        viz_df["avg_vec"]
        .pipe(
            hero.kmeans,
            n_clusters=kmeans_numC,
            algorithm="elkan",
            random_state=42,
            n_init=30,
        )
        .astype(str)
    )
    # texthero has other algs to reduce dimensions besides pca, see docs
    viz_df["pca"] = viz_df["avg_vec"].pipe(hero.pca)

    # generate list of column names for hover_data
    hv_list = list(viz_df.columns)
    hv_list.remove("avg_vec")
    hv_list.remove("pca")
    if "tfidf" in hv_list:
        hv_list.remove("tfidf")
    if "summary" in hv_list:
        hv_list.remove("summary")

    # reformat data so don't have to use texthero built-in plotting
    df_split_pca = pd.DataFrame(viz_df["pca"].to_list(), columns=["pca_x", "pca_y"])
    viz_df.drop(columns="pca", inplace=True)  # drop original PCA column
    viz_df = pd.concat([viz_df, df_split_pca], axis=1)  # merge dataframes

    # set up plot pars (width, title, text)
    w = int(h * (4 / 3))

    if len(query_name) > 0:
        # user provided query_name so include
        plot_title = (
            td_str
            + " viz Jobs by '{}' via word2vec + pca".format(text_col_name)
            + " | "
            + query_name
        )
    else:
        plot_title = td_str + " viz Jobs by '{}' via word2vec + pca".format(
            text_col_name
        )

    if show_text:
        # adds company names to the plot if you want
        viz_df["companies_abbrev"] = viz_df["companies"].apply(text_first_N, num=15)
        graph_text_label = "companies_abbrev"
    else:
        graph_text_label = None

    # plot dimension-reduced data
    fig_w2v = px.scatter(
        viz_df,
        x="pca_x",
        y="pca_y",
        color="kmeans",
        hover_data=hv_list,
        title=plot_title,
        height=h,
        width=w,
        template="plotly_dark",
        text=graph_text_label,
    )
    fig_w2v.show()

    # save if requested

    if save_plot:
        # saves the HTML file
        # auto-saving as a static image is a lil difficult so just click on the interactive
        # plot it generates
        fig_w2v.write_html(
            plot_title + query_name + "_" + text_col_name + ".html",
            include_plotlyjs=True,
        )

    print("plot generated - ", datetime.now())


def load_google_USE():
    st = time.time()
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    rt = (time.time() - st) / 60
    print("loaded google USE embeddings in {} minutes".format(round(rt, 2)))

    return embed


def vizjobs_googleUSE(
    viz_df,
    text_col_name,
    USE_embedding,
    save_plot=False,
    h=720,
    query_name="",
    show_text=False,
    viz_type="TSNE",
):
    today = date.today()
    # Month abbreviation, day and year
    td_str = today.strftime("%b-%d-%Y")

    # generate embeddings for google USE. USE_embedding MUST be passed in
    embeddings = USE_embedding(viz_df[text_col_name])  # create list from np arrays
    use = np.array(embeddings).tolist()  # add lists as dataframe column
    viz_df["use_vec"] = use

    # get optimal number of kmeans. limit max to 15 for interpretability
    max_clusters = 15
    if len(viz_df["use_vec"]) < max_clusters:
        max_clusters = len(viz_df["use_vec"])

    kmeans_numC = optimal_num_clustas(
        viz_df["use_vec"], d_title="google_USE-" + query_name, top_end=max_clusters
    )

    # complete k-means clustering + pca dim red. w/ use_vec
    if kmeans_numC is None:
        kmeans_numC = 5

    viz_df["kmeans"] = (
        viz_df["use_vec"]
        .pipe(
            hero.kmeans,
            n_clusters=kmeans_numC,
            algorithm="elkan",
            random_state=42,
            n_init=30,
        )
        .astype(str)
    )

    # use the vector for dimensionality reduction

    if viz_type.lower() == "tsne":
        viz_df["TSNE"] = viz_df["use_vec"].pipe(hero.tsne, random_state=42)
    else:
        viz_df["pca"] = viz_df["use_vec"].pipe(hero.pca)

    # generate list of column names for hover_data in the html plot

    hv_list = list(viz_df.columns)
    hv_list.remove("use_vec")

    if "tfidf" in hv_list:
        hv_list.remove("tfidf")
    if "pca" in hv_list:
        hv_list.remove("pca")
    if "TSNE" in hv_list:
        hv_list.remove("TSNE")
    if "summary" in hv_list:
        hv_list.remove("summary")

    # reformat data so don't have to use texthero built-in plotting

    if viz_type.lower() == "tsne":
        # TSNE reformat
        df_split_tsne = pd.DataFrame(
            viz_df["TSNE"].to_list(), columns=["tsne_x", "tsne_y"]
        )
        viz_df.drop(columns="TSNE", inplace=True)  # drop original PCA column
        viz_df = pd.concat([viz_df, df_split_tsne], axis=1)  # merge dataframes
    else:
        # PCA reformat
        df_split_pca = pd.DataFrame(viz_df["pca"].to_list(), columns=["pca_x", "pca_y"])
        viz_df.drop(columns="pca", inplace=True)  # drop original PCA column
        viz_df = pd.concat([viz_df, df_split_pca], axis=1)  # merge dataframes

    # set up plot pars (width, title, text)
    w = int(h * (4 / 3))

    if len(query_name) > 0:
        # user provided query_name so include
        plot_title = (
            td_str
            + " viz Jobs by '{}' via google USE + {}".format(text_col_name, viz_type)
            + " | "
            + query_name
        )
    else:
        plot_title = td_str + " viz Jobs by '{}' via google USE {}".format(
            text_col_name, viz_type
        )
    if show_text:
        # adds company names to the plot if you want
        viz_df["companies_abbrev"] = viz_df["companies"].apply(text_first_N, num=15)
        graph_text_label = "companies_abbrev"
    else:
        graph_text_label = None

    # setup labels (decides pca or tsne)
    if viz_type.lower() == "tsne":
        plt_coords = ["tsne_x", "tsne_y"]
    else:
        plt_coords = ["pca_x", "pca_y"]

    # plot dimension-reduced data

    viz_df.dropna(inplace=True)

    fig_use = px.scatter(
        viz_df,
        x=plt_coords[0],
        y=plt_coords[1],
        color="kmeans",
        hover_data=hv_list,
        title=plot_title,
        height=h,
        width=w,
        template="plotly_dark",
        text=graph_text_label,
    )
    fig_use.show()

    # save if requested

    if save_plot:
        # saves the HTML file
        # auto-saving as a static image is a lil difficult so just click on the interactive
        # plot it generates
        fig_use.write_html(
            plot_title + query_name + "_" + text_col_name + ".html",
            include_plotlyjs=True,
        )

    print("plot generated - ", datetime.now())


def find_CHjobs_from(
    website,
    desired_characs,
    job_query,
    job_type=None,
    language=None,
    verbose=False,
    filename=date.today().strftime("%b-%d-%Y") + "_[raw]_scraped_jobs_CH.xls",
):
    if website == "indeed":
        sp_search = load_indeed_jobs_CH(job_query, job_type=job_type, language=language)
        job_soup = sp_search.get("job_soup")
        URL_used = sp_search.get("query_URL")

        if verbose:
            print("\n The full HTML docs are: \n")
            pp.pprint(job_soup, compact=True)
        jobs_list, num_listings = extract_job_information_indeedCH(
            job_soup, desired_characs, uURL=URL_used
        )
    elif website == "indeed_default":
        sp_search = load_indeed_jobs_CH(job_query, run_default=True)
        job_soup = sp_search.get("job_soup")
        URL_used = sp_search.get("query_URL")
        if verbose:
            print("\n The full HTML docs are: \n")
            pp.pprint(job_soup, compact=True)

        jobs_list, num_listings = extract_job_information_indeedCH(
            job_soup, desired_characs, uURL=URL_used
        )

    job_df = save_jobs_to_excel(jobs_list, filename)

    print(
        "{} new job postings retrieved from {}. Stored in {}.".format(
            num_listings, website, filename
        )
    )

    return job_df


def load_indeed_jobs_CH(job_query, job_type=None, language=None, run_default=False):
    i_website = "https://ch.indeed.com/Stellen?"
    def_website = "https://ch.indeed.com/Stellen?q=Switzerland+English&jt=internship"
    if run_default:
        # switzerland has a unique page shown below, can run by default
        # website = "https://ch.indeed.com/Switzerland-English-Jobs"

        getVars = {"fromage": "last", "limit": "50", "sort": "date"}

        url = def_website + urllib.parse.urlencode(getVars)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        job_soup = soup.find(id="resultsCol")
    else:
        getVars = {
            "q": job_query,
            "jt": job_type,
            "lang": language,
            "fromage": "last",
            "limit": "50",
            "sort": "date",
        }

        # if values are not specified, then remove them from the dict (and URL)
        if job_query is None:
            del getVars["q"]
        if job_type is None:
            del getVars["jt"]
        if language is None:
            del getVars["lang"]

        url = i_website + urllib.parse.urlencode(getVars)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        job_soup = soup.find(id="resultsCol")

    # return the job soup

    soup_results = {"job_soup": job_soup, "query_URL": url}
    return soup_results


def_URL = "https://ch.indeed.com/Stellen?" + "ADD_queries_here"


def extract_job_information_indeedCH(
    job_soup, desired_characs, uURL=def_URL, verbose=False, print_all=False
):
    # job_elems = job_soup.find_all('div', class_='mosaic-zone-jobcards')
    job_elems = job_soup.find_all("div", class_="job_seen_beacon")

    if print_all:
        print("\nAll found 'job elements' are as follows: \n")
        pp.pprint(job_elems, compact=True)

    with open("job_elements.txt", "w") as f:
        # save to text file for investigation
        print(job_elems, file=f)

    cols = []
    extracted_info = []

    if "titles" in desired_characs:
        titles = []
        cols.append("titles")
        for job_elem in job_elems:
            titles.append(extract_job_title_indeed(job_elem, verbose=verbose))
        extracted_info.append(titles)

    if "companies" in desired_characs:
        companies = []
        cols.append("companies")
        for job_elem in job_elems:
            companies.append(extract_company_indeed(job_elem))
        extracted_info.append(companies)

    if "date_listed" in desired_characs:
        dates = []
        cols.append("date_listed")
        for job_elem in job_elems:
            dates.append(extract_date_indeed(job_elem))
        extracted_info.append(dates)

    if "summary" in desired_characs:
        summaries = []
        cols.append("summary")
        for job_elem in job_elems:
            summaries.append(extract_summary_indeed(job_elem))
        extracted_info.append(summaries)

    if "links" in desired_characs:
        links = []
        cols.append("links")
        for job_elem in job_elems:
            links.append(extract_link_indeedCH(job_elem, uURL))
        extracted_info.append(links)

    jobs_list = {}

    for j in range(len(cols)):
        jobs_list[cols[j]] = extracted_info[j]

    num_listings = len(extracted_info[0])

    return jobs_list, num_listings


def extract_job_title_indeed(job_elem, verbose=False):
    title_elem = job_elem.select_one("span[title]").text
    if verbose:
        print(title_elem)
    try:
        title = title_elem.strip()
    except:
        title = "no title"
    return title


def extract_company_indeed(job_elem):
    company_elem = job_elem.find("span", class_="companyName")
    company = company_elem.text.strip()
    return company


def extract_link_indeedCH(job_elem, uURL):
    # some manual shenanigans occur here
    # working example https://ch.indeed.com/Stellen?q=data&jt=internship&lang=en&vjk=49ed864bd5e422fb

    link = job_elem.find("a")["href"]
    uURL_list = uURL.split("&fromage=last")
    link = uURL_list[0] + "&" + link
    # replace some text so that the link has a virtual job key. Found via trial and error
    return link.replace("/rc/clk?jk=", "vjk=")


def extract_date_indeed(job_elem):
    date_elem = job_elem.find("span", class_="date")
    date = date_elem.text.strip()
    return date


def extract_summary_indeed(job_elem):
    summary_elem = job_elem.find("div", class_="job-snippet")
    summary = summary_elem.text.strip()
    return summary


def indeed_postprocess(
    i_df, query_term, query_jobtype, verbose=False, shorten_links=False
):
    print("Starting postprocess - ", datetime.now())

    # apply texthero cleaning
    i_df["titles"] = hero.clean(i_df["titles"])
    i_df["summary"] = hero.clean(i_df["summary"])

    # use bit.ly to shorten links
    if shorten_links:
        try:
            len(i_df["short_link"])
            print("found values for short_link, not-recreating")
        except:
            print("no values exist for short_link, creating them now")
            # there is a random delay to not overload APIs, max rt is 5s * num_rows
            i_df["short_link"] = i_df["links"].apply(shorten_URL_bitly)
    else:
        i_df["short_link"] = "not_created"

    # save file to excel
    rn = datetime.now()
    i_PP_date = rn.strftime("_%m.%d.%Y-%H-%M_")
    i_df["date_pulled"] = rn.strftime("%m.%d.%Y")
    i_df["time_pulled"] = rn.strftime("%H:%M:%S")
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
    i_df.to_excel(out_name)
    if verbose:
        print("Saved {} - ".format(out_name), datetime.now())

    # download if requested
    return i_df


def indeed_datatable(i_df, count_what="companies", freq_n=10):
    # basically just wrote this to reduce code down below
    # depends on the colab data_table.DataTable()

    print("Count of column '{}' appearances in search:\n".format(count_what))
    comp_list_1 = i_df[count_what].value_counts()
    pp.pprint(comp_list_1.head(freq_n), compact=True)

    i_df_disp = i_df.copy()
    i_df_disp["summary_short"] = i_df_disp["summary"].apply(text_first_N)
    i_df_disp.drop(columns=["links", "summary"], inplace=True)  # drop verbose columns

    return i_df_disp


# define whether or not to shorten links
shorten_key = False  # @param {type:"boolean"}
# define whether or not to download excel versions of the files
# download_key = for specific searches deemed relevant
# download_all includes all searches
download_key = True  # @param {type:"boolean"}
download_all = False  # @param {type:"boolean"}

# determines columns in output dataframe post-scraping
desired_characs = ["titles", "companies", "links", "date_listed", "summary"]

if __name__ == "__main__":
    output_folder_path = os.getcwd()

    using_google_USE = True
    using_gensim_w2v = False

    # only load models declared as used
    today = date.today()
    # Month abbreviation, day and year
    d4 = today.strftime("%b-%d-%Y")

    default_filename = d4 + "_[raw]_scraped_jobs_CH.xls"

    if using_google_USE:
        meine_embeddings = load_google_USE()

    if using_gensim_w2v:
        w2v_model = load_gensim_word2vec()

    """## Swiss Jobs - Indeed

    ```
    This function extracts all the desired characteristics of all new job postings
        of the title and location specified and returns them in single file.
    The arguments it takes are:

        - Website: to specify which website to search
            - (options: 'indeed' or 'indeed_default')
        - job_query: words that you want to narrow down the jobs to.
            - for example 'data'
        - job_type:
            - 'internship' or 'fulltime' or 'permanent'
        - language:
            - 'en' or 'de' or other languages.. 'fr'? ew
        - Desired_characs: what columns of data do you want to extract? options are:
            - 'titles', 'companies', 'links', 'date_listed', 'summary'
        - Filename: default is "JS_test_results.xls", can be changed to whatever
    ```

    ### query 1 - internship in "data"
    """

    jq1 = "data"  # @param {type:"string"}
    jt1 = "internship"  # @param {type:"string"}
    lan = "en"  # @param {type:"string"}

    # variables for fn defined in form above

    chdf1 = find_CHjobs_from(
        website="indeed",
        desired_characs=desired_characs,
        job_query=jq1,
        job_type=jt1,
        language=lan,
    )

    q1_processed = indeed_postprocess(
        chdf1, query_term=jq1, query_jobtype=jt1, shorten_links=shorten_key
    )

    indeed_datatable(q1_processed)

    """**Viz Query 1**"""

    viz1_df = q1_processed.copy()
    viz1_df.drop(columns=["links", "short_link"], inplace=True)

    if using_google_USE:

        # general rule - if # of jobs returned > 25 may want to turn off text in
        # one or both plots (summary text and title text)

        vizjobs_googleUSE(
            viz1_df,
            "summary",
            meine_embeddings,
            save_plot=True,
            show_text=True,
            query_name=jt1 + " in " + jq1,
            viz_type="pca",
        )

        vizjobs_googleUSE(
            viz1_df,
            "titles",
            meine_embeddings,
            save_plot=True,
            show_text=False,
            query_name=jt1 + " in " + jq1,
            viz_type="pca",
        )
    else:
        viz_job_data_word2vec(
            viz1_df,
            "summary",
            save_plot=True,
            h=720,
            query_name=jt1 + " in " + jq1,
            viz_type="pca",
        )
        viz_job_data_word2vec(
            viz1_df,
            "titles",
            save_plot=True,
            h=720,
            query_name=jt1 + " in " + jq1,
            viz_type="pca",
        )

    # query 2 - all jobs in Switzerland for English Speakers

    jq2 = "indeed_default"  # passing this phrase in causes it to search for all en jobs
    jt2 = "all"
    # in the case of "run the special case on Indeed" query terms don't matter
    chdf2 = find_CHjobs_from(
        website="indeed_default", job_query="gimme", desired_characs=desired_characs
    )

    q2_processed = indeed_postprocess(
        chdf2, query_term=jq2, query_jobtype=jt2, shorten_links=False
    )

    indeed_datatable(q2_processed)

    """**Viz Query 2**"""

    viz_q2 = q2_processed.copy()
    viz_q2.drop(columns="links", inplace=True)

    if using_google_USE:

        # general rule - if # of jobs returned > 25 may want to turn off text in
        # one or both plots (summary text and title text)

        vizjobs_googleUSE(
            viz_q2,
            "summary",
            meine_embeddings,
            save_plot=True,
            show_text=True,
            query_name="all listings for CH eng. jobs",
            viz_type="tsne",
        )

        vizjobs_googleUSE(
            viz_q2,
            "titles",
            meine_embeddings,
            save_plot=True,
            show_text=False,
            query_name="all listings for CH eng. jobs",
            viz_type="tsne",
        )
    else:
        viz_job_data_word2vec(
            viz_q2,
            "summary",
            save_plot=False,
            h=720,
            query_name="all listings for CH eng. jobs",
            viz_type="tsne",
        )
        viz_job_data_word2vec(
            viz_q2,
            "titles",
            save_plot=False,
            h=720,
            query_name="all listings for CH eng. jobs",
            viz_type="tsne",
        )
