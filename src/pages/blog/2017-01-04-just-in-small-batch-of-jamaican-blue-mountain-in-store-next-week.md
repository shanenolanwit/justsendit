---
templateKey: blog-post
title: Lookahead lads
date: 2020-05-17T21:44:30.499Z
description: Some fun with positive lookaheads
featuredpost: true
featuredimage: /img/rubularjustsendit.png
tags:
  - regex
  - node
  - lookahead
  - capture groups
---
```
const matches = "justsendit.ie".match(/(?=(just(?=(sendit(?=(.(?=(ie)))))))).*/)
const parts = matches.splice(1,4);
console.log(parts);
```