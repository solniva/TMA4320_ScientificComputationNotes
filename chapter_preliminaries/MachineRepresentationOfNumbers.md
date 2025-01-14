---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"slideshow": {"slide_type": "slide"}}

# Lecture 2

+++ {"slideshow": {"slide_type": "slide"}}

## 1. General Sources of Errors

$$
% \DeclareMathOperator{\Div}{div}
% \DeclareMathOperator{\Grad}{grad}
% \DeclareMathOperator{\Curl}{curl}
% \DeclareMathOperator{\Rot}{rot}
% \DeclareMathOperator{\ord}{ord}
% \DeclareMathOperator{\Kern}{ker}
% \DeclareMathOperator{\Image}{im}
% \DeclareMathOperator{\spann}{span}
% \DeclareMathOperator{\dist}{dist}
% \DeclareMathOperator{\diam}{diam}
% \DeclareMathOperator{\sig}{sig}
% \DeclareMathOperator{\fl}{fl}
\newcommand{\RR}{\mathbb{R}}
\newcommand{\NN}{\mathbb{N}}
\newcommand{\VV}{\mathbb{V}}
\newcommand{\FF}{\mathbb{F}}
\newcommand{\dGamma}{\,\mathrm{d} \Gamma}
\newcommand{\dGammah}{\,\mathrm{d} \Gamma_h}
\newcommand{\dx}{\,\mathrm{d}x}
\newcommand{\dy}{\,\mathrm{d}y}
\newcommand{\ds}{\,\mathrm{d}s}
\newcommand{\dt}{\,\mathrm{d}t}
\newcommand{\dS}{\,\mathrm{d}S}
\newcommand{\dV}{\,\mathrm{d}V}
\newcommand{\dX}{\,\mathrm{d}X}
\newcommand{\dY}{\,\mathrm{d}Y}
\newcommand{\dE}{\,\mathrm{d}E}
\newcommand{\dK}{\,\mathrm{d}K}
\newcommand{\dM}{\,\mathrm{d}M}
\newcommand{\cd}{\mathrm{cd}}
\newcommand{\onehalf}{\frac{1}{2}}
\newcommand{\bfP}{\boldsymbol P}
\newcommand{\bfx}{\boldsymbol x}
\newcommand{\bfy}{\boldsymbol y}
\newcommand{\bfa}{\boldsymbol a}
\newcommand{\bfu}{\boldsymbol u}
\newcommand{\bfv}{\boldsymbol v}
\newcommand{\bfe}{\boldsymbol e}
\newcommand{\bfb}{\boldsymbol b}
\newcommand{\bff}{\boldsymbol f}
\newcommand{\bfp}{\boldsymbol p}
\newcommand{\bft}{\boldsymbol t}
\newcommand{\bfj}{\boldsymbol j}
\newcommand{\bfB}{\boldsymbol B}
\newcommand{\bfV}{\boldsymbol V}
\newcommand{\bfE}{\boldsymbol E}
\newcommand{\bfzero}{\boldsymbol 0}
$$

+++ {"slideshow": {"slide_type": "slide"}}

Recall the 6 steps in Scientific Computing

1. Mathematical Modeling
2. Analysis of the mathematical model (Existence, Uniqueness, Continuity)
3. Numerical methods (computational complexity, stability, accuracy)
4. Realization (implemententation) 
5. Postprocessing 
6. Validation

+++ {"slideshow": {"slide_type": "slide"}}

__Discussion__:

:::{admonition} TODO
:class: danger dropdown
Add mentimeter
:::

+++ {"slideshow": {"slide_type": "fragment"}}

Today we will talk about one important and unavoidable source of errors,
namely the way, a computer deals with numbers.

+++ {"slideshow": {"slide_type": "slide"}}

## 2. Machine Representation of Numbers

+++ {"slideshow": {"slide_type": "slide"}}

Let's start with two simple tests. 
* Define two numbers $a=0.2$ and $b=0.2$ and test whether their sum is equal to $0.4$.
* Now define two numbers $a=0.2$ and $b=0.1$ and test whether their sum is equal to $0.3$.

+++ {"slideshow": {"slide_type": "slide"}}

__Solution.__

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
# Write your code here
a = 0.2
b = 0.1
sum = 0.3

if (a+b) == sum:
    print("That is what I expected!!")
else:
    print("What the hell is going on??")

diff = a+b
diff = diff - sum
print(f"{diff}")
```

+++ {"slideshow": {"slide_type": "slide"}}

Why is that? The reason is the way numbers are represent on a computer, which will
be the topic of the first part of the lecture.

After the lecture I recommed you to take a look 
[](https://0.30000000000000004.com) which discusses the phenomena we just observed in some detail.

+++ {"slideshow": {"slide_type": "slide"}}

## Positional System

+++ {"slideshow": {"slide_type": "subslide"}}

On everyday base, we represent numbers using the __positional system__. For instance, when we write $145397.2346$ to denote the number

$$
145397.2346 = 
  1 \cdot 10^5 
+ 4 \cdot 10^4 
+ 5 \cdot 10^3
+ 3 \cdot 10^2 
+ 9 \cdot 10^1
+ 7 \cdot 10^0
+ 2 \cdot 10^{-1}
+ 3 \cdot 10^{-2}
+ 4 \cdot 10^{-3}
+ 6 \cdot 10^{-4}
$$

using $10$ as __base__. This is also known a __decimal system__. 

+++ {"slideshow": {"slide_type": "slide"}}

$$ 
1234.987 = 1 \cdot 10^{3} + 2\cdot 10^2 + 3 \cdot 10^1 + 4 \cdot 10^0 
+ 9 \cdot 10^{-1} + 8 \cdot 10^{-2} + 7 \cdot 10^{-3}
$$

In general for any $\beta \in \mathbb{N}$, $\beta \geqslant 2$, we use
the __positional representation__
$$
x_{\beta} = (-1)^s [ a_n a_{n-1}\ldots a_0.a_{-1}a_{-2}\ldots a_{-m} ]_{\beta}
$$

with $a_n \neq 0$ to represent the number 

$$
x_{\beta} = \sum_{k=-m}^n a_k \beta^{k}. 
$$

+++ {"slideshow": {"slide_type": "subslide"}}

Here,
* $\beta$ is called the __base__
* $a_k \in [0, \beta-1]$ are called the __digits__
* $s \in \{0,1\}$ defines the __sign__
* $a_n a_{n-1}\ldots a_0$ is the __integer__ part
* $a_{-1}a_{-2}\ldots a_{-m}$ is called the __fractional__ part
* The point between $a_0$ and $a_{-1}$ is generally called the __radix point__
 

+++ {"slideshow": {"slide_type": "slide"}}

:::{exercise}  
:label: ex-pos-repr 
Write down the position representation of the number $3\frac{2}{3}$ for
both the base $\beta=10$ and $\beta=3$.

:::

+++ {"slideshow": {"slide_type": "slide"}}

```{solution} ex-pos-repr
:class: dropdown
* $\beta = 10: [3.666666666\cdots]_{10}$
* $\beta = 3:   1 \cdot 3^{1} + 0 \cdot 3^{0} + 2 \cdot 3^{-1} = [10.2]_{3}$

```

+++ {"slideshow": {"slide_type": "slide"}}

To represent numbers on a computer, the most common bases are 
* $\beta = 2$ (binary),
* $\beta=10$ (decimal)
* $\beta = 16$ (hexidecimal). 

For the latter one, one uses $1,2,\ldots, 9$, A,B,C,D,E,F to represent the digits.
For $\beta = 2, 10, 16$ is also called the binary point, decimal point and hexadecimal point, respectively.

+++ {"slideshow": {"slide_type": "slide"}}

We have already seen that for many (actuall most!) numbers, the fractional part can be infinitely long in order to represent the number exactly. But on a computer, only a finite amount of storage is available, so to represent numbers, only a fixed numbers of digits can be kept in storage for each number we wish to represent.

This will of course automatically introduces errors whenever our number can not represented exactly by
the finite number of digits available.

+++ {"slideshow": {"slide_type": "slide"}}

### Fix-point system

Use $N=n+1+m$ digits/memory locations to store the number $x$ written as above.
Since the binary/decimal point is _fixed_ , it is difficult to represent large numbers $\geqslant \beta^{n+1}$ or small numbers $ < \beta^{-m}$.

E.g. nowdays we often use 16 (decimal) digits in a computer, if you distributed that evenly 
to present same number of digits before and after the decimal point, the range or representable numbers is between
$10^8$ and $10^{-8}$ __This is very inconvenient__! 

Also, small numbers which are located
towards the lower end of this range cannot be as accuractely represented as number close
to the upper end of this range.

As a remedy, an modified representation system for numbers was introduced, known as __normalized floating point system__.

+++ {"slideshow": {"slide_type": "slide"}}

### Normalized floating point system

Returning to our first example: 
$$
145397.2346 
= 0.1453972346 \cdot 10^{6}
= 1453972346 \cdot 10^{6-10}
$$

In general we write
$$
x = (-1)^s 0.a_1 a_2 \ldots a_t \beta^e 
  = (-1)^s \cdot m \cdot \beta^{e - t} 
$$

Here,
* $t \in \mathbb{N}$ is the number of _significant digits_ 
* $e$ is an integer called the _exponent_ 
* $m = a_1 a_2 \ldots a_t \in \mathbb{N}$ is known as the _mantissa_.

* Exponent $e$ defines the _scale_ of the represented number,
  typically, $e \in \{e_{\mathrm{min}}, \ldots, e_{\mathrm{max}}\}$,
  with $e_{\mathrm{min}} < 0$ and $e_{\mathrm{max}} > 0$.
* Number of significant digits $t$ defines the __relative accuracy__.

+++ {"slideshow": {"slide_type": "slide"}}

We define the __finite__ set of available __floating point numbers__

$$
\mathbb{F}(\beta,t, e_{\mathrm{min}}, e_{\mathrm{max}})
 = \{0 \} \cup \left\{  x \in \mathbb{R}: x = (-1)^s\beta^e \sum_{i=1}^t a_i \beta^{-i}, e_{\mathrm{min}} \leqslant e \leqslant e_{\mathrm{max}}, 0 \leqslant a_i \leqslant \beta - 1  \right\}
$$       

* Typically to enforce a unique representation and to ensure maximal relative accuracy, one requires that $a_1 \neq 0$ for non-zero numbers.

+++ {"slideshow": {"slide_type": "slide"}}

:::{exercise}
:label: ex-float-num

What is the smallest (non-zero!) and the largest number you can represent with $\mathbb{F}$?

:::

+++ {"slideshow": {"slide_type": "fragment"}}

:::{solution} ex-float-num
:class: dropdown

$$
\beta^{e_{\mathrm{min}}-1} 
\leqslant |x| 
\leqslant \beta^{e_{\mathrm{max}}}(1-\beta^{-t})
\quad\text{for } x \in \mathbb{F}.
$$

:::

+++ {"slideshow": {"slide_type": "slide"}}

__Conclusion:__

* Every number $x$ satifying $\beta^{e_{\mathrm{min}}-1} 
\leqslant |x| 
\leqslant \beta^{e_{\mathrm{max}}}(1-\beta^{-t})$ but which is __not__ in $\mathbb{F}$
can be represented by a floating point number $\mathrm{fl}(x)$ by rounding off to the closest number in $\mathbb{F}$.

* Relative _machine precision_ is 
$$
\dfrac{|x-\mathrm{fl}(x)|}{|x|} \leqslant \epsilon := \frac{\beta^{1-t}}{2}
$$

* $|x| < \beta^{e_{\mathrm{min}}-1}$ leads to __underflow__.
* $|x| > \beta^{e_{\mathrm{max}}}(1-\beta^{-t})$ leads to __overflow__.

Standard machine presentations nowadays using
* Single precision, allowing for 7-8 sigificant digits
* Double precision, allowing for 16 sigificant digits

+++ {"slideshow": {"slide_type": "slide"}}

### Things we don't discuss in this but which are important in numerical mathematics

We see that already by entering data from our model into the computer, we make an unavoidable error.
The same also applied for the realization of basics mathematical operations $\{+, -, \cdot, /\}$ etc. on a computer.

Thus it is of importance to understand how errors made in a numerical method are propagated through the numerical algorithms. Keywords for the interested are
* Forward propagation: How does an initial error and the algorithm affect the final solution?
* Backward propagation: If I have certain error in my final solution, how large was the initial error?
