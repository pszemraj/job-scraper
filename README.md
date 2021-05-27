
# job-scraper *Swiss Edition*
<!-- TOC -->

- [job-scraper *Swiss Edition*](#job-scraper-swiss-edition)
  - [Updates w.r.t. Original](#updates-wrt-original)
    - [Added Features](#added-features)
      - [1 - Integration with Google Drive](#1---integration-with-google-drive)
      - [2 - Added Visualization](#2---added-visualization)
      - [3 - Google Colab Tables](#3---google-colab-tables)
      - [4 - Link Shortening](#4---link-shortening)
  - [Example](#example)
  - [Details on Querying](#details-on-querying)
  - [Source](#source)
  - [Original Repo](#original-repo)
    - [Scraping jobs from Indeed or CWjobs](#scraping-jobs-from-indeed-or-cwjobs)
    - [Terms and conditions](#terms-and-conditions)
    - [Using the selenium web driver](#using-the-selenium-web-driver)
    - [Accompanying blog post](#accompanying-blog-post)

<!-- /TOC -->
---

## Updates w.r.t. Original

- In the "switzerland" folder is a link to an .ipynb file that also links to Colab. It merges the job_scraper.py code with Demo.ipynb from the original project, and makes relevant adjustments for [the Swiss version of Indeed](https://ch.indeed.com/?from=gnav-jobsearch--jasx), which is mostly just URL syntax.
- Currently, the CH version only scrapes data from Indeed
- A link to a Colab version is [here.](https://colab.research.google.com/drive/1kLxtsvL9uDZfRrzd9MC15libyMnc1Ear) Copy a version to your drive to try out.

### Added Features

As the original just pulled and saved an excel file, additional features have been added to make the script more useful:

#### 1 - Integration with Google Drive
- files are now auto-saved to google drive folder as specified, includes the day's date to keep track
- files have columns added for date and time pulled incase some sort of larger-scale database creation is useful

#### 2 - Added Visualization

- k-means visualization: text in field of choice (typically either the job title or the summary) is vectorized and then clustered via unsupervised k-means.
- Current options for vectorization are TF-IDF or word2vec via the Google News pretrained dataset (available through Gensim)
- optimal number of k-means clusters determined via elbow method
- Jobs are then plotted by their dimensionality reduced representation (currently: PCA) and colored by cluster. A custom plotting function (roughly analogous to that [TextHero](https://texthero.org/) includes built-in but with more features) displays the job data.

![viz_sans_labels](https://user-images.githubusercontent.com/74869040/119703148-87382280-be56-11eb-94d9-b5627947cf4b.png)


- Text with company name can be added to see distributions (*Note: README has static images, but the graphs are plotly scatterplots in HTML and interactive with tooltips*)

![viz_w_labels](https://user-images.githubusercontent.com/74869040/119703209-9d45e300-be56-11eb-88c2-453c395a60f3.png)


#### 3 - Google Colab Tables

- uses Google Colab's built-in table feature for dataframes, allowing the user to filter/sort/see job data without needing to exit the notebook

![table_example](https://user-images.githubusercontent.com/74869040/119703251-a46cf100-be56-11eb-9c42-e0381b82be3b.png)

#### 4 - Link Shortening

- Allows integration with the pyshorteners package for shortening scraped links (to use for the actual app)
- Works with bit.ly

## Example

In the section below all the function definitions (i.e. *main*), the code following will return 50 job postings for language = en, job type = internship, and job query = "data":

```
# define input params for query
desired_characs = ['titles', 'companies', 'links', 'date_listed', 'summary']
jq1="data"
jt1 = "internship"
lan = "en"

# scrape data
chdf1 = find_CHjobs_from(website="indeed", desired_characs=desired_characs,
                         job_query=jq1, job_type=jt1, language=lan)
# process output scraped data
q1_processed = indeed_postprocess(chdf1, query_term=jq1, query_jobtype=jt1,
                       shorten_links=False, download_excel=True)
# display Colab data table
data_table.DataTable(indeed_datatable(q1_processed),
                     include_index=False, num_rows_per_page=20)

# generate viz
viz1 = q1_processed.copy()
viz1.drop(columns=["links", "short_link"], inplace=True)
viz_job_data_word2vec(viz1, "summary", save_plot=True, show_text=True,
                      query_name=jt1 + " in " + jq1)
```
## Details on Querying

The following describes possible input params to **find_CHjobs_from()**:
```
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

## Source

Credit to the original repo and medium post - see below.


---
*Everything below here is a copy of the original repo README*

## Original Repo

### Scraping jobs from Indeed or CWjobs

The module job-scraper.py enables you to web scrape job postings from Indeed.co.uk or CWjobs.co.uk.

Both require the package Beautiful Soup. For CWjobs, the Selenium web driver is also required. These can be installed as follows:

```bash
$ pip install beautifulsoup4
$ pip install selenium
```

To use this module, import the job_scraper.py file and call the funciton "find_jobs_from()", which takes in several arguments. For an explanation and demonstration of the required arguments, see Demo.ipynb.

### Terms and conditions
I do not condone scraping data from Indeed or CWjobs in any way. Anyone who wishes to do so should first read their statements on scraping software [here](https://www.indeed.co.uk/legal) and [here](https://www.cwjobs.co.uk/recruiters/terms).


### Using the selenium web driver
At present, the default browser is set as Google Chrome. This can be modified within job_scraper.py.

In order to extract jobs from CWjobs using Selenium, the appropriate driver must be installed. The driver in this repository is for Google Chrome version 81. See [this link](https://sites.google.com/a/chromium.org/chromedriver/downloads) to download an appropriate driver for the Google Chrome browser, if required, and place it in the same directory as the job-scraper.py function.

### Accompanying blog post
A full description of this code and the process I followed to write it is available [here](https://medium.com/@Chris.Lovejoy/automating-my-job-search-with-python-ee2b465c6a8f).
