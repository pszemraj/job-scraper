
# job-scraper *Swiss Edition*

---

## Updates w.r.t. Original

In the "switzerland" folder is a link to an .ipynb file that also links to Colab. It merges the job_scraper.py code with Demo.ipynb from the original project, and makes relevant adjustments for [the Swiss version of Indeed](https://ch.indeed.com/?from=gnav-jobsearch--jasx), which is mostly just URL syntax.

### Added Features

As the original just pulled and saved an excel file, additional features have been added to make the script more useful:

#### 1 - Integration with Google Drive
- files are now auto-saved to google drive with the day's date to keep track 
- files have columns added for date and time pulled incase some sort of larger-scale database creation is useful

#### 2 - Added Visualization

- k-means visualization: text in field of choice (typically either the job title or the summary) is vectorized and then clustered via unsupervised k-means.
- optimal number of k-means clusters determined via elbow method
- Jobs are then plotted by their dimensionality reduced representation (currently: PCA) and colored by cluster. A custom plotting function (roughly analogous to that [TextHero](https://texthero.org/) includes built-in but with more features) displays the job data.
- 
![viz_sans_labels](https://user-images.githubusercontent.com/74869040/119703148-87382280-be56-11eb-94d9-b5627947cf4b.png)


- Text with company name can be added to see distributions (*Note: README has static images, but the graphs are plotly scatterplots in HTML and interactive with tooltips*)

![viz_w_labels](https://user-images.githubusercontent.com/74869040/119703209-9d45e300-be56-11eb-88c2-453c395a60f3.png)


#### 2 - Google Colab Tables

- uses Google Colab's built-in table feature for dataframes, allowing the user to filter/sort/see job data without needing to exit the notebook

![table_example](https://user-images.githubusercontent.com/74869040/119703251-a46cf100-be56-11eb-9c42-e0381b82be3b.png)

### Source 

Credit to the original of course, see below.


---
## Original Repo

### Scraping jobs from Indeed or CWjobs

The module job-scraper.py enables you to web scrape job postings from Indeed.co.uk or CWjobs.co.uk.

Both require the package Beautiful Soup. For CWjobs, the Selenium web driver is also required. These can be installed as follows:

```bash
$ pip install beautifulsoup4
$ pip install selenium
```

To use this module, import the job_scraper.py file and call the funciton "find_jobs_from()", which takes in several arguments. For an explanation and demonstration of the required arguments, see Demo.ipynb.

## Terms and conditions
I do not condone scraping data from Indeed or CWjobs in any way. Anyone who wishes to do so should first read their statements on scraping software [here](https://www.indeed.co.uk/legal) and [here](https://www.cwjobs.co.uk/recruiters/terms).


## Using the selenium web driver
At present, the default browser is set as Google Chrome. This can be modified within job_scraper.py.

In order to extract jobs from CWjobs using Selenium, the appropriate driver must be installed. The driver in this repository is for Google Chrome version 81. See [this link](https://sites.google.com/a/chromium.org/chromedriver/downloads) to download an appropriate driver for the Google Chrome browser, if required, and place it in the same directory as the job-scraper.py function.

## Accompanying blog post
A full description of this code and the process I followed to write it is available [here](https://medium.com/@Chris.Lovejoy/automating-my-job-search-with-python-ee2b465c6a8f).

