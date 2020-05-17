---
templateKey: blog-post
title: Lookahead lads
date: 2020-05-17T22:19:47.123Z
description: Some fun with positive lookaheads
featuredpost: true
featuredimage: /img/losty.png
tags:
  - regex
  - node
  - lookahead
  - capture groups
---
This blog will talk about how useful lookaheads are, and will give some simple examples of how to use them. But Here's some `justsendit` examples that we'll use.

I used [rubular](https://rubular.com/r/kJ2HovCHeZWRw2) to build some examples. I've used rubular for years and it's not the most user friendly regex builder, and it was built for testing ruby regular expressions but it's pretty quick and easy. Who doesn't like quick and easy. When your regex jobs get a bit more complex and language specific there are better sites out there. I'll some examples later.

![](/img/rubular17may2020.png)

So lets take that regular expression and use it in a wee javascript script. I just used [Repl.it](https://repl.it/repls/UniqueUnwittingTrials) to put together a quick example of taking the matches from our lookahead and printing them out. Pretty kewl.

```
const matches = "justsendit.ie".match(/(?=(just(?=(sendit(?=(\.(?=(ie)))))))).*/)
const parts = matches.splice(1,4);
console.log(parts);
```

![repl executing the snippet](/img/repl17may2020.png "Running the code in REPL")
