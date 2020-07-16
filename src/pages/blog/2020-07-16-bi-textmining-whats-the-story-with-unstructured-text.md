---
templateKey: blog-post
title: BI TextMining - Whats the story with unstructured text
date: 2020-07-15T13:38:04.849Z
description: Applying technology to understand values of unstructured text
featuredpost: true
featuredimage: /img/losty.png
tags:
  - ML
  - BI
  - TextMining
  - Python
---
According to a study by Merril Lynch & Gartner, 85% of all corporate data is unstructured

#### Applications of text mining

* Information Extraction: using pattern matching to identify key words and sequences
* Topic Tracking: predicting documents of interest based on user profile and activity
* Summerization: assuming its a real word, this process summarises data to save end users time
* Categorization: identify main themes and place into a set of predefined categories
* Clustering: grouping similar documents without having a predefined set of categories
* Concept Linking: connect related documents by identifying shared concepts
* Question Answering: find the best answer to a given question through knowledge driven pattern matching

#### Algorithms

* Bag of Words: use methods that treat words as tokens without understanding or context
* Focus on Meaning: identify parts of speech associated with sentiment, use the meaning of words to analyse text

#### Natural Language Processing

* Utterance - this is any input or sentence passed to a computer program to be interpreted
* Intent - this is the intention of the utterance - if an utterance is 'pass me the cans', the intent could be 'getCans' or 'drinkCans'
* Entity - sometimes referred to as a slot, an entity typically represents the key words, usually nouns, in an utterance in the example 'buy me cans on friday', the entities could be who - 'me', what - 'cans', and when - 'friday'
* Slot filling - term used regularly when talking about chatbots, slot filling involves prompting a user to provide entities for example if the user says 'buy me cans', the 'when' slot/entity still needs to be clarified so the chatbot might ask 'when should i buy you cans?'
* [Lex](https://aws.amazon.com/lex/)- NLP with AWS
* [DialogFlow](https://dialogflow.com/)- NLP with Google
* [Rasa](https://rasa.com/)- Open Source NLP
* [Servisbot](https://servisbot.com/)- Industry leaders in chatbots and conversational AI
* [Wordnet](https://wordnet.princeton.edu/)- Big DB of words

#### Github

In the linked [github](https://github.com/shanenolanwit/tunnel-snakes-101/blob/master/text_density.ipynb) project I took a csv that lists utterances and intents and done some analysis on the data.

#### **Requirements**

* import data from a csv
* clean/transform the data
* count the number of unique utterances and intents
* investigate patterns such as min and max length in utterances associated with each intent
* find and graph the most frequent words across all intents
* find and graph the most frequent words for each intent

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from statistics import mean, median
from collections import Counter
import itertools
```

```
locale = 'english'
stop_words = stopwords.words(locale)
print("loading {} stop words such as {}".format(locale, stop_words[:5]))
dataset = pd.read_csv('frequency.csv', header=None, usecols=[0,1])
dataset.columns = ['utterance', 'intent']
```

loading english stop words such as \['i', 'me', 'my', 'myself', 'we']

```
clean_dataset = dataset.dropna(axis=0, how='any')
print("Dataset row count : {}".format(len(clean_dataset)))
clean_dataset.describe()
```

Dataset row count : 30

|        | utterance              | intent     |
| ------ | ---------------------- | ---------- |
| count  | 30                     | 30         |
| unique | 30                     | 2          |
| top    | is the mocha expensive | coffeeshop |
| freq   | 1                      | 18         |

```
def remove_stop_words(vocab):
    return [word for word in vocab if word not in stop_words]
```

## Find the most frequent words

* join all data from the utterances column
* remove the stop words
* use the [counter](https://pymotw.com/2/collections/counter.html) module to count the most common occurrences
* use a list to collect your data, then initialise a DataFrame when you are ready, never use append if you can avoid it

  * as per the official [dataframes docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html)
  * and as per [some guy](https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it) on stackoverflow

```
complete_vocab = " ".join(clean_dataset["utterance"]).split(' ') 

clean_vocab = remove_stop_words(complete_vocab)
print("Word count reduced from {} to {}".format(len(complete_vocab), len(clean_vocab)))
counter = Counter(clean_vocab)
top_words = counter.most_common(10)

word_count = []
for top_word in top_words:
    word_count.append({"word": top_word[0], "count": top_word[1]}) 
   
word_count_df=pd.DataFrame(word_count,columns=['word','count'])
```

Word count reduced from 127 to 74

## Create a barchart showing frequency across all intents

```
print("Most frequent word is {}".format(top_words[0][0]))
# in this notebook, graphs are displayed inline, to change to file downloads use the following code
# plt.savefig('pathtowhereyouwantyourimagesaved.png')
fig, ax = plt.subplots()
word_count_df.sort_values(by="count",ascending=False)['count'].plot.bar();
ax.set_xticklabels(word_count_df.sort_values(by="count",ascending=False)['word'])
plt.title("Most Frequent Words")
plt.ylabel("Count")
plt.xlabel("Word")
plt.show()
```

Most frequent word is coffee

![](data:image/png;base64,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htpue6rtybYnDwwMebVtREQshja6aF4LzLR9TwttRURErY2A358FdM9ERERzGg34+o7tuwOnN9lOREQ8U6MnWW3/FXh2k21ERMTQciVrREShEvAREYVKwEdEFCoBHxFRqAR8REShEvAREYVKwEdEFCoBHxFRqAR8REShEvAREYVKwEdEFCoBHxFRqAR8REShEvAREYVKwEdEFCoBHxFRqAR8REShEvAREYVKwEdEFCoBHxFRqEYDXtLqkk6VdJOkGyVt12R7ERExz/iG9/814Bzb+0haDlix4fYiIqLWWMBLWhV4BXAQgO0ngCeaai8iIubXZBfN+sAc4FhJv5X0PUkrNdheRER0aTLgxwNbAd+yvSXwKHDk4CdJmiJpuqTpc+bMabCciIixpcmAvxO40/ZV9fKpVIE/H9tTbU+2PXlgYKDBciIixpbGAt72n4E7JG1Ur9oNuKGp9iIiYn5Nj6J5P3BiPYLmVuDghtuLiIhaowFv+xpgcpNtRETE0HIla0REoRLwERGFSsBHRBQqAR8RUagEfEREoRLwERGFSsBHRBQqAR8RUagEfEREoRLwERGFSsBHRBQqAR8RUagEfEREoRLwERGFSsBHRBQqAR8RUagEfEREoRLwERGFSsBHRBQqAR8RUagEfEREocY3uXNJtwNzgaeBp2xPbrK9iIiYp9GAr+1i+74W2omIiC7poomIKFTTAW/gV5JmSJrScFsREdGl6S6aHWzfJWkt4DxJN9me1v2EOvinAEycOLHhcpbcpCPPXuJ93H70XqNQSUTE8Bo9grd9V/3vvcAZwLZDPGeq7cm2Jw8MDDRZTkTEmNJYwEtaSdIqna+BVwOzmmovIiLm12QXzXOAMyR12jnJ9jkNthcREV0aC3jbtwJbNLX/iIgYXoZJRkQUKgEfEVGoBHxERKES8BERhUrAR0QUKgEfEVGoBHxERKES8BERhUrAR0QUKgEfEVGoBHxERKES8BERhUrAR0QUKgEfEVGoBHxERKES8BERhUrAR0QUKgEfEVGoBHxERKES8BERhWo84CWNk/RbSWc13VZERMwzooCXtMNI1i3AocCNi1JUREQsuZEewX99hOvmI2ldYC/ge4tSVERELLnxw22UtB2wPTAg6fCuTasC40aw/68CHwFWWewKIyJisQwb8MBywMr187pD+hFgn+FeKOl1wL22Z0ja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%0A)

## group the intents and get a count of how many utterances each intent has

```
grouped_dataset = clean_dataset.groupby(['intent'])
grouped_dataset.groups.keys()
grouped_dataset.count()
```

|            | **utterance** |
| ---------- | ------------- |
| **intent** |               |
| coffeeshop | 18            |
| greeting   | 12            |

## display the distribution of utterances across intents as a pie chart

```
fig1, ax1 = plt.subplots()
labels = grouped_dataset.groups.keys()
sizes = grouped_dataset.count()["utterance"]
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
```

![](data:image/png;base64,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%0A)

## Generate some statistics about each intent

* iterate the intents
* create a list of all the words associated with each intent (removing stop words)
* produce a new dataframe displaying details about each intent

  * intent name
  * number of utterances associated with the intent
  * number of words in longest utterance associated with the intent
  * number of words in shortest utterance associated with the intent
  * average number of words per utterance associated with the intent
  * median number of words per utterances associated with the intent

```
word_count = lambda sentence: len(word_tokenize(sentence))
top_words_by_intent = []
my_list = []

for name, group in grouped_dataset:
    # create a single vocab from all the utterances associated with this intent
    group_vocab = " ".join(group["utterance"]).split(" ")
    clean_group_vocab = remove_stop_words(group_vocab)
    group_counter = Counter(clean_group_vocab)
    # get the top 3 most frequent words associated with this intent
    most_common_in_group = group_counter.most_common(3)
    # this will make creating a graphical representation much easier later
    top_words_by_intent.append({
        "intent": name, 
        "top": list(map(lambda x: x[0], most_common_in_group)),
        "1": most_common_in_group[0],
        "2": most_common_in_group[1],
        "3": most_common_in_group[2]
    })
    my_list.append({"intent_name": name, 
                    "utterances": len(group),
                    "max_words_ut": len(max(group["utterance"], key=word_count).split(' ')),
                    "min_words_ut": len(min(group["utterance"], key=word_count).split(' ')),
                    "avg" : mean(list(map(lambda x: len(x.split(' ')), group["utterance"]))),
                    "median" : median(list(map(lambda x: len(x.split(' ')), group["utterance"]))),
                   })
```

## lets print our table

```
dfx=pd.DataFrame(my_list,columns=['intent_name','utterances', 'max_words_ut','min_words_ut', 'avg', 'median'])
print(dfx.to_string())
```

|     | intent_name | utterances | max_words_ut | min_words_ut | avg      | median |
| --- | ----------- | ---------- | ------------ | ------------ | -------- | ------ |
| 0   | coffeeshop  | 18         | 9            | 3            | 5.611111 | 5.0    |
| 1   | greeting    | 12         | 5            | 1            | 2.166667 | 2.0    |

## lets now look at each intents top 3 words and their frequency

```
dftw=pd.DataFrame(top_words_by_intent,columns=['intent','top','1','2','3'])
print(dftw)
```

|     | intent     | top                   | 1           | 2          | 3        |
| --- | ---------- | --------------------- | ----------- | ---------- | -------- |
| 0   | coffeeshop | \[coffee, large, hot] | (coffee, 7) | (large, 5) | (hot, 4) |
| 1   | greeting   | \[well, hello, hi]    | (well, 4)   | (hello, 2) | (hi, 2)  |

## format our data to be charted

because of the weird way matplotlib does [grouped bar chart](https://matplotlib.org/3.1.1/gallery/units/bar_unit_demo.html) we need to format our data a little differently, we need to have the first bar of each group together, the second bar of each group together and so forth.

the code also uses the itertools chain to concat the lists of labels, see [this](https://stackoverflow.com/questions/1720421/how-do-i-concatenate-two-lists-in-python) stackoverflow post for why this is the most efficient way to concat lists

```
def generate_labels_and_counts(data):
    labels = list(map(lambda x: x[0], data))
    counts = list(map(lambda x: x[1], data))
    return { "labels": labels, "counts": counts }

data = generate_labels_and_counts(dftw['1'].tolist())
number_one_labels = data["labels"]
number_ones = data["counts"]

data = generate_labels_and_counts(dftw['2'].tolist())
number_two_labels = data["labels"]
number_twos = data["counts"]

data = generate_labels_and_counts(dftw['3'].tolist())
number_three_labels = data["labels"]
number_threes = data["counts"]

top_three_list = [number_ones, number_twos, number_threes]
top_three_labels = [number_one_labels, number_two_labels, number_three_labels]
# https://stackoverflow.com/questions/1720421/how-do-i-concatenate-two-lists-in-python
labels = list(itertools.chain.from_iterable(top_three_labels))

# formatted data for our grouped bar charts
print(top_three_list)
print(labels)
```

\[[7, 4], \[5, 2], \[4, 2]] 

\['coffee', 'well', 'large', 'hello', 'hot', 'hi']

## lets finally graph the most frequent words per intent

```
intents = dftw["intent"]

number_groups = len(intents)+1
bin_width = 1.0/(number_groups+1)
fig, ax = plt.subplots(figsize=(6,6))
for i in range(number_groups):
    ax.bar(x=np.arange(len(intents)) + i*bin_width, 
           height=top_three_list[i],
           width=bin_width,
           align='center')
ax.set_xticks(np.arange(len(categorical_1)) + number_groups/(2*(number_groups+1)))
ax.set_xticklabels(intents)

rects = ax.patches

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label,
            ha='center', va='bottom')
    
plt.title("Most Frequent Words x Intent")
plt.ylabel("Count")
plt.xlabel("Word x Intent")
plt.show()
```

![](data:image/png;base64,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%0A)

Taken from original [blog](https://chickengoujons.ie/ml/textmining/intro) at chickengoujons.ie