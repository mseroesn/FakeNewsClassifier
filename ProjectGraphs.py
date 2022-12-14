#importing libraries
import seaborn as sns
import matplotlib.pyplot as plot
import pandas
from wordcloud import WordCloud
from collections import Counter


dataSet = pandas.read_csv("DataSet.csv")

dataSet.drop_duplicates(keep="first")
dataSet.dropna(axis=0, inplace=True)

dataSet["content"] = dataSet["author"] + " " + dataSet["title"]

# WordCloud for True News
wc = WordCloud(background_color='white', min_font_size=10,width=500,height=500)
true_news_wc = wc.generate(dataSet[dataSet['label'] == 0]['content'].str.cat(sep=" "))
plot.figure(figsize=(8, 6))
plot.imshow(true_news_wc)
plot.show()


# create list of True News words
true_news_words_list = dataSet[dataSet['label'] == 0]['content'].str.cat(sep=" ").split()

# create DataFrame of that
true_news_words_df = pandas.DataFrame(Counter(true_news_words_list).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=true_news_words_df[0],y=true_news_words_df[1])
plot.xticks(rotation='vertical')
plot.xlabel('Words')
plot.ylabel('Counts')
plot.title('True News Words Count')
plot.show()

# create list of Fake News words
fake_news_words_list = dataSet[dataSet['label'] == 1]['content'].str.cat(sep=" ").split()

# create DataFrame of that
fake_news_words_df = pandas.DataFrame(Counter(fake_news_words_list).most_common(20))

# Now Let's Plot barplot of this words
sns.barplot(x=fake_news_words_df[0], y=fake_news_words_df[1])
plot.xticks(rotation='vertical')
plot.xlabel('Words')
plot.ylabel('Counts')
plot.title('Fake News Words Count')
plot.show()
