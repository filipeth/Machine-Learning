from bs4 import BeautifulSoup
import requests
import urllib3

# html_doc = """<html>
# <body>
# <h1> My first Heading</h1>
# <b><!--This is a comment line--></b>
# <p title="About Me" class="test">My first paragraph.</p>
# <div class="cities">
# <h2>London</h2>
# </div>
# </body>
# </html>"""
#
#
# soup = BeautifulSoup(html_doc, 'html.parser')
# # print(type(soup))
# # print(soup)
# tag = soup.p
# print(tag)
# comment = soup.b.string
# print(comment)
# print(tag.attrs)
# print(type(tag))

# url = 'http://www.simplilearn.com/resources'
# result = requests.get(url)
# webpage = result.content
# s1_soup = BeautifulSoup(webpage, 'html.parser')
# result.close()
# print(s1_soup)
# print(s1_soup.prettify())
# print(s1_soup.head)
# print(s1_soup.title)
# for href in s1_soup.findAll('a', href=True):
#     print(href['href'])


http = urllib3.PoolManager()
url = 'http://www.simplilearn.com/resources'
webpage = http.request('GET', url)
soup = BeautifulSoup(webpage.data, 'html.parser')
webpage.close()
# print(soup.title)
print(soup.contents)
# print(soup.head)

# for href in soup.findAll('a', href=True):
#     print(href['href'])

# for article in soup.findAll(class_='col-lg-16 free_cources_div'):
#     print(article.h2)
