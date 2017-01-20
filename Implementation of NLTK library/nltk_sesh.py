
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from collections import defaultdict

from string import punctuation

from heapq import nlargest

import requests

class FrequencySummarizer:

	def __init__(self, min_cut=0.1, max_cut=0.9):

		self.min_cut = min_cut
		self.max_cut = max_cut

		self.stopwords = set(stopwords.words('english') + list(punctuation))

	def _compute_frequencies(self, word_sent):
		freq = defaultdict(int)

		for sentence in word_sent:

			for word in sentence:
				
				freq[word] += 1


		max_freq = float(max(freq.values()))

		for word in freq.keys():

			freq[word] = freq[word]/max_freq

			if freq[word] >= self.max_cut or freq[word] <= self.min_cut:

				del freq[word]

		return freq

	def summarize(self, text, n):

		sents = sent_tokenize(text)

		assert n <= len(sents)

		word_sent = [word_tokenize(s.lower()) for s in sents]

		self._freq = self._compute_frequencies(word_sent)

		rankings = defaultdict(int)

		for i,sent in enumerate(word_sent):

			for word in sent:

				if word in self._freq:
					rankings[i] += self._freq[word]


		sents_idx = nlargest(n,rankings, key = rankings.get)

		return [sents[j] for j in sents_idx]

import urllib2

from bs4 import BeautifulSoup


def get_only_text_washington_post_url(url):

	try:
		page = urllib2.urlopen(url).read().decode('utf8')
	except:
		return (None,None)


	soup = BeautifulSoup(page)

	

	text = ""

	if soup.find_all('article') is not None:
		text = ''.join(map(lambda p: p.text, soup.find_all('article')))
			


	return text, soup.title.text


def getNYTText(url, token):

	response = requests.get(url)
	soup = BeautifulSoup(response.content)


	page = str(soup)
	title = soup.find('title').text


	mydivs = soup.find_all("p", {"class":"story-body-text story-content"})
	text = ''.join(map(lambda p: p.text, mydivs))

	return text, title
			


## the article we would like to summarize
nyturl = "https://www.nytimes.com/2017/01/07/opinion/sunday/how-to-destroy-the-business-model-of-breitbart-and-fake-news.html?action=click&pgtype=Homepage&clickSource=story-heading&module=opinion-c-col-top-region&region=opinion-c-col-top-region&WT.nav=opinion-c-col-top-region&_r=0"
#page = urllib2.urlopen(someUrl).read().decode('utf8')
#request = urllib2.Request(someUrl)
#response = urllib2.urlopen(request)
#souper = BeautifulSoup(response)

#textOfUrl = get_only_text_washington_post_url(someUrl)
# get the title and text'
textOfUrl = getNYTText(nyturl,None)


fs = FrequencySummarizer()
# instantiate our FrequencySummarizer class and get an object of this class
summary = fs.summarize(textOfUrl[0], 3)


print summary







