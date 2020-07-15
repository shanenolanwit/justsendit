---
templateKey: blog-post
title: ML Classification - Whats the story with Neural Networks
date: 2020-07-15T13:26:04.352Z
description: "Neural networks can be a tricky concept to grasp. This post breaks
  down a simple example to help the grasp the fundamentals of this topic. "
featuredpost: true
featuredimage: /img/losty.png
tags:
  - ML
  - Neural Networks
  - Classification
---
Think of neural networks as a collection of simple nodes (neurons) that represent a binary state: on or off (0 or 1).\
This activation state is determined by an **activation function**.\
An activation function typically

* multiplies each input by a given weight
* adds the weighted input values to a given bias
* converts the answer to a sigmoid value - 0 or 1 representing the active state

The outputs of these nodes may become inputs into other nodes, depending on the complexity of the problem.\
This concept is better explained with examples...

#### Cans with the lads - cans and truth tables

**AND**\
Say you want to go drinking cans, but you only want to go drinking cans if your two friends Jim and Jack are going too.\
This is the classic case for using an **AND** type truth table.

| x   | y   | result |
| --- | --- | ------ |
| 0   | 0   | 0      |
| 0   | 1   | 0      |
| 1   | 0   | 0      |
| 1   | 1   | 1      |

Applying this logic to our neural network, we say that our activation node represents if we are going drinking or not. The inputs to this node are x and y values representing if Jim is going drinking and if Jack is going drinking.

\
You set a threshold of 2 meaning you only go drinking (activate) if the sum of the inputs is greater than or equal to 2.\
More formally we call this threshold our bias and use simple algebra to structure the function as if the sum of inputs plus bias is less than zero.\
(x + y >= 2) => (x + y - 2 <= 0)Later you will see weights being applied to certain inputs, this means that some inputs are more significant than others.\
To keep using this example, you decide you will go for cans if Jack and Jim go, or if Jack goes, but not if neither or only Jim goes.\
You keep the threshold score as 2, but give Jacks answer a weight of 2.

**OR**\
Say you still want to go drinking cans, but now you're happy to go as long as anyone else goes.\
This is the classic case for using an **OR** type truth table.

| x   | y   | result |
| --- | --- | ------ |
| 0   | 0   | 0      |
| 0   | 1   | 1      |
| 1   | 0   | 1      |
| 1   | 1   | 1      |

We apply this logic to our neural network by modifying our activation function and lowering the threshold to 1.

**XOR**\
Jack and Jim are dead. Your other options for drinking partners are Alice and Bob. Alice and Bob are a couple who are grand on their own but annoying together.\
You decide to go if either is free, but not if neither or both are free. This is the classic case for using an **XOR** type truth table.

| x   | y   | result |
| --- | --- | ------ |
| 0   | 0   | 0      |
| 0   | 1   | 1      |
| 1   | 0   | 1      |
| 1   | 1   | 0      |

There is a major problem applying this logic to our activation function because now we have two threshold values - 0 and 2\
We need to add weights and hidden neuron layers to our network to get the values we want.\
So what weights do we assign our inputs and biases to make sure we correctly classify our inputs into a 1 or 0?\
Great question, the answer is we don't, although we can work it out by hand for small examples, it quickly becomes impossible with the addition of extra neurons and hidden layers.\
Luckily neural networks are very good at assigning weights, checking how accurate their answers are, and then going back and adjusting their weights and thresholds to make their predictions more accurate.\
There is always a danger that you will overfit the data, and just have weights and biases that will only fit your exact data set but is not flexible to new data. Techniques such as randomly dropping neurons prevent this overfitting.

In the diagram below we can see that it is hard to solve the xor problem using a linear classifier\
This simply means we can't separate our positive results from negative results using a single hyperplane\
We take our set of possible inputs/coordinates { (0,0),(0,1),(1,0),(1,1) } and pass them through two activation functions.\
For each input pair (x1, x2) we pass each value through two functions representing possible outputs.\
h1 takes each input and weights them (20), adds the bias (-10) and gets the sigmoid of this number to reduce it to 0 or 1.\
h2 takes each input and weights them (-20), adds the bias (30) and gets the sigmoid of this number to reduce it to 0 or 1\
The output of h1 and h2 are then weighted and passed to the activation function y, which adds the values to the bias, reduces it to a 0 or 1

![solving xor with neural networks](https://chickengoujons.ie/images/neuralxor.png)

| x   | y   | activation 1                 | activation 2                  | activation 3                 |
| --- | --- | ---------------------------- | ----------------------------- | ---------------------------- |
| 0   | 0   | sig(0\*20 + 0\*20 + -10) = 0 | sig(0\*-20 + 0\*-20 + 30) = 1 | sig(0\*20 + 1\*20 + -30) = 0 |
| 0   | 1   | sig(0\*20 + 1\*20 + -10) = 1 | sig(0\*-20 + 1\*-20 + 30) = 1 | sig(1\*20 + 1\*20 + -30) = 1 |
| 1   | 0   | sig(1\*20 + 0\*20 + -10) = 1 | sig(1\*-20 + 0\*-20 + 30) = 1 | sig(1\*20 + 1\*20 + -30) = 1 |
| 1   | 1   | sig(1\*20 + 1\*20 + -10) = 1 | sig(1\*-20 + 1\*-20 + 30) = 0 | sig(1\*20 + 0\*20 + -30) = 0 |

\
(0,0) =>sig(0\*20 + 0\*20 + -10) = 0,sig(0\*-20 + 0\*-20 + 30) = 1,sig(0\*20 + 1\*20 + -30) = 0\
(0,1) =>sig(0\*20 + 1\*20 + -10) = 1,sig(0\*-20 + 1\*-20 + 30) = 1,sig(1\*20 + 1\*20 + -30) = 1\
(1,0) =>sig(1\*20 + 0\*20 + -10) = 1,sig(1\*-20 + 0\*-20 + 30) = 1,sig(1\*20 + 1\*20 + -30) = 1\
(1,1) =>sig(1\*20 + 1\*20 + -10) = 1,sig(1\*-20 + 1\*-20 + 30) = 0,sig(1\*20 + 0\*20 + -30) = 0\
\
I designed a primitive neural network to solve the XOR problem, and print out the required values. Download the source [here](https://github.com/shanenolanwit/neural-xor)\
The network does not learn from its mistakes, it simply assigns random values to weights and biases and checks if they satisfy the requirements. We can see from the following output, that after 4 runs of the program it takes an average of 291 iterations to get the values correct.

```
Shanes-MacBook-Pro:xor shane$ node index.js 
found a result after 231 tries
Weight X: -16 Bias X: 25 Weight Y: 9 Bias Y: -7 Weight Z: 4 Bias Z: -8
(0,0) => sig(0*-16 + 0*-16 + 25) 
         sig(0*9 + 0*9 + -7) 
         => sig(0*4 + 1*4 + -8) => 0
(0,1) => sig(0*-16 + 1*-16 + 25) 
         sig(0*9 + 1*9 + -7) 
         => sig(0*4 + 1*4 + -8) => 1
(1,0) => sig(1*-16 + 0*-16 + 25) 
         sig(1*9 + 0*9 + -7) 
         => sig(0*4 + 1*4 + -8) => 1
(1,1) => sig(1*-16 + 1*-16 + 25) 
         sig(1*9 + 1*9 + -7) 
         => sig(0*4 + 1*4 + -8) => 0
Shanes-MacBook-Pro:xor shane$ node index.js 
found a result after 160 tries
Weight X: 20 Bias X: -25 Weight Y: -8 Bias Y: 1 Weight Z: -27 Bias Z: 18
(0,0) => sig(0*20 + 0*20 + -25) 
         sig(0*-8 + 0*-8 + 1) => 
         sig(1*-27 + 0*-27 + 18) => 0
(0,1) => sig(0*20 + 1*20 + -25) 
         sig(0*-8 + 1*-8 + 1) => 
         sig(1*-27 + 0*-27 + 18) => 1
(1,0) => sig(1*20 + 0*20 + -25) 
         sig(1*-8 + 0*-8 + 1) => 
         sig(1*-27 + 0*-27 + 18) => 1
(1,1) => sig(1*20 + 1*20 + -25) 
         sig(1*-8 + 1*-8 + 1) => 
         sig(1*-27 + 0*-27 + 18) => 0
Shanes-MacBook-Pro:xor shane$ node index.js 
found a result after 107 tries
Weight X: 25 Bias X: -17 Weight Y: -17 Bias Y: 26 Weight Z: -15 Bias Z: -26
(0,0) => sig(0*25 + 0*25 + -17) 
         sig(0*-17 + 0*-17 + 26) 
         => sig(1*15 + 0*15 + -26) => 0
(0,1) => sig(0*25 + 1*25 + -17) 
         sig(0*-17 + 1*-17 + 26) 
         => sig(1*15 + 0*15 + -26) => 1
(1,0) => sig(1*25 + 0*25 + -17) 
         sig(1*-17 + 0*-17 + 26) 
         => sig(1*15 + 0*15 + -26) => 1
(1,1) => sig(1*25 + 1*25 + -17) 
         sig(1*-17 + 1*-17 + 26) 
         => sig(1*15 + 0*15 + -26) => 0
Shanes-MacBook-Pro:xor shane$ node index.js 
found a result after 667 tries
Weight X: -21 Bias X: 12 Weight Y: 16 Bias Y: -24 Weight Z: -24 Bias Z: 7
(0,0) => sig(0*-21 + 0*-21 + 12) 
         sig(0*16 + 0*16 + -24) 
         => sig(0*-24 + 1*-24 + 7) => 0
(0,1) => sig(0*-21 + 1*-21 + 12) 
         sig(0*16 + 1*16 + -24) 
         => sig(0*-24 + 1*-24 + 7) => 1
(1,0) => sig(1*-21 + 0*-21 + 12) 
         sig(1*16 + 0*16 + -24) 
         => sig(0*-24 + 1*-24 + 7) => 1
(1,1) => sig(1*-21 + 1*-21 + 12) 
         sig(1*16 + 1*16 + -24) 
         => sig(0*-24 + 1*-24 + 7) => 0
```

This content was taken from my personal [blog](https://chickengoujons.ie/ml/neural) at [chickengoujons.ie](https://chickengoujons.ie)