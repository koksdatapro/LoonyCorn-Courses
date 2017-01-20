import requests
import urllib2
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def getWashPostText(url,token):
    # THis function takes the URL of an article in the 
    # Washington Post, and then returns the article minus all 
    # of the crud - HTML, javascript etc. How? By searching for
    # everything that lies between the tags titled 'token'
    # Like most web-scraping, this will only work for urls where
    # we know the structure (eg say all articles in the WashPo are
    # enclosed in <article></article> tags). This will also change from
    # time to time as different HTML formats are employed in the website
    try:
        page = urllib2.urlopen(url).read().decode('utf8')
    except:
        # if unable to download the URL, return title = None, article = None
        return (None,None)
    soup = BeautifulSoup(page)
    if soup is None:
        return (None,None)
    # If we are here, it means the error checks were successful, we were
    # able to parse the page
    text = ""
    if soup.find_all(token) is not None:
        # Search the page for whatever token demarcates the article
        # usually '<article></article>'
        text = ''.join(map(lambda p: p.text, soup.find_all(token)))
        # mush together all the text in the <article></article> tags
        soup2 = BeautifulSoup(text)
        # create a soup of the text within the <article> tags
        if soup2.find_all('p')!=[]:
            # now mush together the contents of what is in <p> </p> tags
            # within the <article>
            text = ''.join(map(lambda p: p.text, soup2.find_all('p')))
    return text, soup.title.text

def getNYTText(url, token):

	response = requests.get(url)
	soup = BeautifulSoup(response.content)


	page = str(soup)
	title = soup.find('title').text


	mydivs = soup.find_all("p", {"class":"story-body-text story-content"})
	text = ''.join(map(lambda p: p.text, mydivs))

	return text, title

def scrapeSource(url, magicFrag = '2016', scraperFunction= getNYTText, token = None):
	hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
	urlBodies = {}
	request = urllib2.Request(url, headers = hdr)
	response = urllib2.urlopen(request,)
	soup = BeautifulSoup(response)

	numErrors = 0
	for a in soup.find_all('a'):
		try:
			url = a['href']
			if ((url not in urlBodies) and ((magicFrag is not None and magicFrag in url) or magicFrag is None)):
				body = scraperFunction(url,token)

				if body and len(body) > 0:
					urlBodies[url] = body
				print url
		except:
			numErrors += 1

	return urlBodies


class FrequencySummarizer:
	def __init__(self, min_cut = 0.1, max_cut = 0.9):

		self.min_cut = min_cut
		self.max_cut = max_cut
		self._stopwords =  set(stopwords.words('english') + list(punctuation) +
								[u"'s'",'"'])

	def _compute_frequencies(self,word_sent,customStopWords = None):

		freq = defaultdict(int)

		if customStopWords is None:
			stopwords = set(self._stopwords)

		else:
			stopwords = set(customStopWords).union(self._stopwords)

		for sentence in word_sent:
			for word in sentence:
				if word not in stopwords:
					freq[word] += 1

		m = float(max(freq.values()))

		for word in freq.keys():
			freq[word] = freq[word]/m

			if freq[word] >= self.max_cut or freq[word] <= self.min_cut:
				del freq[word]

		return freq

	def extractFeatures(self, article, n, customStopWords = None):
		text = article[0]
		title = article[1]

		sentences = sent_tokenize(text)
		word_sent = [word_tokenize(s.lower()) for s in sentences]

		self.freq = self._compute_frequencies(word_sent,customStopWords)

		if n < 0:
			return nlargest(len(self.freq.keys()), self.freq, key = self.freq.get)
		else:
			return nlargest(n, self.freq, key = self.freq.get)

	def extractRawFrequencies(self, article):


		text = article[0]
		title = article[1]

		sentences = sent_tokenize(text)
		word_sent = [word_tokenize(s.lower()) for s in sentences]
		freq = defaultdict(int)
		for s in word_sent:
			for word in s:
				if word not in self._stopwords:
					freq[word] += 1

		return freq

	def summarize(self, article,n):

		text = article[0]
		title = article[1]

		sentences = sent_tokenize(text)
		word_sent = [word_tokenize(s.lower()) for s in sentences]

		self._freq = self._compute_frequencies(word_sent)
		ranking = defaultdict(int)

		for i, sentence in enumerate(word_sent):
			for word in sentence:
				if word in self._freq:
					ranking[i] += self._freq[word]

		sentence_index = nlargest(n, ranking, key = ranking.get)

		return [sentences[j] for j in sentences_index]


urlWashingtonPostNonTech = "https://www.washingtonpost.com/sports/"
urlNewYorkTimesNonTech = "https://www.nytimes.com/pages/sports/index.html"
urlWashingtonPostTech = "https://www.washingtonpost.com/business/technology/"
urlNewYorkTimesTech = "http://www.nytimes.com/pages/technology/index.html"

print "WP Tech"
print "*"*25
washingtonPostTechArticles = scrapeSource(urlWashingtonPostTech, '2016', getWashPostText, 'article')

print "WP Non-Tech"
print "*"*25
washingtonPostNonTechArticles = scrapeSource(urlWashingtonPostNonTech, '2016', getWashPostText, 'article')
print "NYT Tech"
print "*"*25
NewYorkTimesTechArticles = scrapeSource(urlNewYorkTimesTech, '2016', getNYTText)
print "NYT Non-Tech"
print "*"*25
NewYorkTimesNonTechArticles = scrapeSource(urlNewYorkTimesNonTech, '2016', getNYTText)

#print (len(washingtonPostTechArticles) + len(NewYorkTimesTechArticles), len(washingtonPostNonTechArticles) + len(NewYorkTimesNonTechArticles))



articleSummaries = {}

for techUrlDictonary in [NewYorkTimesTechArticles, washingtonPostTechArticles]:
	for articleUrl in techUrlDictonary:
		if techUrlDictonary[articleUrl][0] is not None:
			if len(techUrlDictonary[articleUrl][0]) > 0:
				fs = FrequencySummarizer()

				summary = fs.extractFeatures(techUrlDictonary[articleUrl],25)

				articleSummaries[articleUrl] = {'feature-vector' : summary, 'label' : 'Tech'}

for nontechUrlDictonary in [NewYorkTimesNonTechArticles, washingtonPostNonTechArticles]:
	for articleUrl in nontechUrlDictonary:
		if nontechUrlDictonary[articleUrl][0] is not None:
			if len(nontechUrlDictonary[articleUrl][0]) > 0:
				fs = FrequencySummarizer()

				summary = fs.extractFeatures(nontechUrlDictonary[articleUrl],25)

				articleSummaries[articleUrl] = {'feature-vector' : summary, 'label' : 'Non - Tech'}

def getDoxyDonkeyText(testUrl, token):
	hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
	urlBodies = {}
	request = urllib2.Request(testUrl, headers = hdr)
	response = urllib2.urlopen(request,)
	soup = BeautifulSoup(response)
	page = str(soup)

	title = soup.find("title").text
	mydivs = soup.find_all("div", {"class":token})
	text = ''.join(map(lambda p: p.text, mydivs))

	return text, title


testUrl = "http://doxydonkey.blogspot.in"

testArticle = getDoxyDonkeyText(testUrl, "post-body")

fs = FrequencySummarizer()

testArticleSummary = fs.extractFeatures(testArticle,25)


def knearest(articleSummaries, testArticleSummary):
	similarities = {}
	for articleUrl in articleSummaries:
		oneArticleSummary = articleSummaries[articleUrl]['feature-vector']
		similarities[articleUrl] = len(set(testArticleSummary).intersection(set(oneArticleSummary)))

	labels = defaultdict(int)

	knn = nlargest(5, similarities, key = similarities.get)

	for oneNeighbor in knn:
		labels[articleSummaries[oneNeighbor]['label']] += 1

	x = nlargest(1,labels, key= labels.get)
	
	return x

cumulativeRawFrequencies = {'Tech':defaultdict(int), 'Non-Tech':defaultdict(int)}

trainingData = {'Tech':NewYorkTimesTechArticles, 'Non-Tech':NewYorkTimesNonTechArticles}

for label in trainingData:
	for articleUrl in trainingData[label]:
		if trainingData[label][articleUrl][0] is not None:
			if len(trainingData[label][articleUrl][0]) > 0:

				fs = FrequencySummarizer()

				rawFrequencies = fs.extractRawFrequencies(trainingData[label][articleUrl])

				for word in rawFrequencies:
					cumulativeRawFrequencies[label][word] += rawFrequencies[word]


techiness = 1.0
nontechiness = 1.0

for word in testArticleSummary:

	if word in cumulativeRawFrequencies['Tech']:
		techiness *= 1e3*cumulativeRawFrequencies['Tech'][word]/float(sum(cumulativeRawFrequencies['Tech'].values()))

	else:
		techiness /= 1e3


for word in testArticleSummary:

	if word in cumulativeRawFrequencies['Non-Tech']:
		nontechiness *= 1e3*cumulativeRawFrequencies['Non-Tech'][word]/float(sum(cumulativeRawFrequencies['Non-Tech'].values()))

	else:
		nontechiness /= 1e3




techiness *= float(sum(cumulativeRawFrequencies['Tech'].values())) / (float(sum(cumulativeRawFrequencies['Tech'].values())) + float(sum(cumulativeRawFrequencies['Non-Tech'].values())))
nontechiness *= float(sum(cumulativeRawFrequencies['Non-Tech'].values())) / (float(sum(cumulativeRawFrequencies['Tech'].values())) + float(sum(cumulativeRawFrequencies['Non-Tech'].values())))
if techiness > nontechiness:
    label = 'Tech'
else:
    label = 'Non-Tech'
print label, techiness, nontechiness








