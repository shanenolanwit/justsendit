---
templateKey: blog-post
title: Google Sheets Party Tricks
date: 2020-05-18T20:37:11.914Z
description: This post looks at some cool Google sheets tricks to show off at
  your next party
featuredpost: true
featuredimage: /img/losty.png
tags:
  - sheets
  - excel
  - google
  - etl
---
We'll start with a sheet that has some data about the characters from Stephen Kings `The Stand`, with some extra columns for fun.

![google sheets data](/img/sheets1may182020.png "The Stand - raw data")

So first thing we want to do is count how many characters are in each surname. So how would we go about this in Excel? Split the name into multiple columns using a space as the delimiter, then use a length function to get the length of the surname column?\
**nah**\
Using sheets we can inject javascript into our sheet and do all sorts of crazy shit. Go to `Tools > Script Editor` and add some of our own functions. To make a function private just don't bother adding a js doc marked as `@customfunction`.

![google sheets data](/img/sheets2may182020.png "Tools > Script Editor")

So we've defined a function that takes a name, splits it using a space and returns the length of the second part of the name. Next lets use this function in our sheet. Using the \`=\` key to get some autocomplete going on your functions.

![google sheets data](/img/sheets3may182020.png "auto complete on custom functions")

Awesome

![google sheets data](/img/sheets4may182020.png "same as any other function")

Once you've passed in your cell, you'll see your answer appear in the cell. Complex functions may take a second to load, but thats no different to excel.

![google sheets data](/img/sheets5may182020.png "success it returned 7")

Perfect, worked for a single row. You can now drag the bottom right corner of the cell down to autocomplete the function for the rest of your rows. . . . But heres the craic with that, if you have a really complex function, and a billion rows, you end up invoking that function a billion times. Wouldn't it be better if our function could just execute against an entire column of names at once ? Well it can. If we create another function that takes in an array, we can highlight a column and process the entire column at once.

![google sheets data](/img/sheets6may182020.png "using map and arrow functions")

Our new function appears by magic

![google sheets data](/img/sheets7may182020.png "selecting an entire column")

And we can see that the returned array has resulted in the creations of a column

![google sheets data](/img/sheets8may182020.png "awesome")

Next party trick is using functions to populate two columns, so we want each name in its own column, unless its a pregnant female, in which case we want to redact the names. So the input will be four columns: name, gender, pregnant and redaction string, the output will be two columns: first name and surname.

![google sheets data](/img/sheets9may182020.png "add params to functions using jsdoc")

This is now available to our sheet

![google sheets data](/img/sheets91may182020.png "awesome")

Add in our variables

![google sheets data](/img/sheets92may182020.png "redacto !")

And we should see our columns created with the redacted names redacted

![google sheets data](/img/sheets93may182020.png "it worked !")

Last party trick for now is using SQL type queries instead of messy formatting and vlookups, you can use the `QUERY` function to select a table of data, and write SQL queries against it. You select select single columns, multiple columns, and return single rows or multiple rows.

![google sheets data](/img/sheets94may182020.png "single return value")

Multiple values

![google sheets data](/img/sheets95may182020.png "two results, single column")

Multiple rows and columns

![google sheets data](/img/sheets96may182020.png "multiple everything")



Thats all the party tricks for now