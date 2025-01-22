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

# Numerical integration: Part IV

## Gauß quadrature formulas

**Anne Kværnø, André Massing**


+++ {"slideshow": {"slide_type": "slide"}}

# Gauß quadrature

+++ {"slideshow": {"slide_type": "slide"}}

Last lecture, when comparing the trapezoidal rule with Gauß-Legendre
quadrature rule, both based on two quadrature nodes,
we observed that
* the Gauß-Legendre quadrature was much more accurate than the
  trapezoidal rule,

* the Gauß-Legendre quadrature *has degree of exactness equal to $3$
  and not only $1$*.

So obviously the position of the nodes matters!

+++ {"slideshow": {"slide_type": "slide"}}

**Questions:**
* Is there a general approach to construct quadrature rules
  $Q[\cdot](\{x_i\}_{i=0}^n,\{w_i\}_{i=0}^n)$ based on $n+1$ nodes with
  a degree of exactness $> n$?

* What is the maximal degree of exactness we can achieve?

+++ {"slideshow": {"slide_type": "fragment"}}

**Intuition:** If we don't predefine the quadrature nodes,
we have $2n+2$ parameters ($n+1$ nodes and $n+1$ weights)
in total.

With $2n+2$ parameters, we might hope that we can construct
quadrature rules which are exact for $p \in \mathbb{P}_{2n+1}$.

+++ {"slideshow": {"slide_type": "slide"}}

## Definition 1: Gaussian quadrature

A quadrature rule 
$Q[\cdot](\{x_i\}_{i=0}^n,\{w_i\}_{i=0}^n)$ based on $n+1$ nodes
which has degree of exactness equals to $2n+1$ is called a
**Gaussian (Legendre) quadrature** (GQ).

+++ {"slideshow": {"slide_type": "slide"}}

## Orthogonal polynomials
To construct Gaussian quadrature rule, we need to briefly review
the concept of orthogonality, which we introduced when we learned
about Fourier series.

+++ {"slideshow": {"slide_type": "slide"}}

Two functions $f, g: [a,b] \to \mathbb{R}$ are orthogonal if

$$
{\langle f, g \rangle} := \int_a^b f(x) g(x) {\,\mathrm{d}x} = 0.
$$

Usually, it will be clear from the context which interval $[a,b]$
we picked.

+++ {"slideshow": {"slide_type": "slide"}}

## Theorem 1: Orthogonal polynomials on $[a,b]$

<div id="quad:thm:orthopolys"></div>
There is a sequence of $\{p_k\}_{k=1}^{\infty}$
of polynomials satisfying

+++ {"slideshow": {"slide_type": "fragment"}}

<!-- Equation labels as ordinary links -->
<div id="_auto1"></div>

\begin{equation}
p_0(x) = 1,
\label{_auto1} \tag{1}
\end{equation}

+++ {"slideshow": {"slide_type": "fragment"}}

<!-- Equation labels as ordinary links -->
<div id="quad:eq:poly_normalization"></div>

\begin{equation}  
p_k(x) = x^k + r_{k-1}(x) \quad \text{for } k=1,2,\ldots
\label{quad:eq:poly_normalization} \tag{2}
\end{equation}

with $r_{k-1} \in \mathbb{P}_{k-1}$ and ...

+++ {"slideshow": {"slide_type": "slide"}}

satisfying the
*orthogonality property*

<!-- Equation labels as ordinary links -->
<div id="_auto2"></div>

\begin{equation}
{\langle p_k, p_l \rangle} = \int_a^b p_k(x) p_l(x) dx = 0
\quad \text{for } k \neq l,
\label{_auto2} \tag{3}
\end{equation}

+++ {"slideshow": {"slide_type": "fragment"}}

and that every polynomial $q_n \in \mathbb{P}_{n}$
can be written as a linear combination of
those orthogonal polynomials up to order $n$.
In other words

$$
\mathbb{P}_{n} = \mathrm{Span} \{p_0,\ldots, p_n\}
$$

+++ {"slideshow": {"slide_type": "slide"}}

**Proof.**
We start from the sequence $\{\phi_k\}_{k=0}^{\infty}$ of monomials
$\phi_k(x) = x^k$ and apply the Gram-Schmidt orthogonalization
procedure:

+++ {"slideshow": {"slide_type": "fragment"}}

\begin{align*}
 \widetilde{p}_0
 &:= 1 = \phi_0
\\ 
 \widetilde{p}_1
 &:= \phi_1 - \dfrac{{\langle \phi_1, \widetilde{p_0} \rangle}}{\|\widetilde{p_0}\|^2} \widetilde{p}_0
 \\ 
 \widetilde{p}_2
 &:= \phi_2
 - \dfrac{{\langle \phi_2, \widetilde{p }_0\rangle }}{\|\widetilde{p}_0\|^2} \widetilde{p}_0
 - \dfrac{{\langle \phi_2, \widetilde{p}_1 \rangle}}{\|\widetilde{p}_1\|^2} \widetilde{p}_1
 \\ 
 \ldots
 \\ 
\widetilde{p}_k
&= \phi_k - \sum_{j=0}^{k-1}\frac{
{\langle \phi_k, \widetilde{p \rangle}_j}}
{\|\widetilde{p}_j\|^2} \widetilde{p}_j
\end{align*}

+++ {"slideshow": {"slide_type": "slide"}}

By construction, $\widetilde{p}_n \in \mathbb{P}_{n}$ and
${\langle p_k, p_l \rangle} = 0$ for $k\neq l$.
Since $\widetilde{p}_k(x) = a_k x^k + a_{k-1} x^{k-1} + \ldots a_0$,
we simply define
$p_k(x) = \widetilde{p}_k/a_k$
to satisfy ([2](#quad:eq:poly_normalization)).

+++ {"slideshow": {"slide_type": "slide"}}

## Theorem 2: Roots of orthogonal polynomials

Each of the polynomials $p_n$ defined in
[Theorem 1: Orthogonal polynomials on $[a,b]$](#quad:thm:orthopolys)
has **$n$ distinct real roots**.


**Proof.**
Without proof, will be added later for the curious among you.

+++ {"slideshow": {"slide_type": "slide"}}

## Theorem 3: Construction of Gaussian quadrature

Let $p_{n+1} \in \mathbb{P}_{n+1}$ be a polynomial
on $[a,b]$
satisfying

$$
{\langle p_{n+1}, q \rangle} = 0 \quad {\forall\;} q \in \mathbb{P}_{n}.
$$

Set $\{x_i\}_{i=0}^n$ to be the $n+1$ real roots of $p_{n+1}$
and define the weights $\{w_i\}_{i=0}^n$ by

$$
w_i = \int_{a}^{b} \ell_i(x) {\,\mathrm{d}x}.
$$

where $\{\ell_i\}_{i=0}^n$ are the $n+1$ cardinal functions
associated with
$\{x_i\}_{i=0}^n$.
The resulting quadrature rule is
a Gaussian quadrature.

**Proof.** Without proof, will be added later for the curious among you.

+++ {"slideshow": {"slide_type": "slide"}}

**Recipe 1 to construct a Gaussian quadrature.**

To construct a Gaussian formula on $[a,b]$ based
on $n+1$ nodes you proceed as follows

+++ {"slideshow": {"slide_type": "fragment"}}

1. Construct a polynomial $p_{n+1} \in \mathbb{P}_{n+1}$
  on the interval $[a, b]$
  which satisfies

$$
\int_{a}^b p_{n+1}(x) q(x) {\,\mathrm{d}x} \quad {\forall\;} q \in \mathbb{P}_{n}.
$$

  You can start from the monomials $\{1,x, x^2, \ldots, x^{n+1}\}$
  and use Gram-Schmidt to orthogonalize them.

+++ {"slideshow": {"slide_type": "fragment"}}

2. Determine the $n+1$ **real** roots $\{x_i\}_{i=0}^n$
  of $p_{n+1}$ which serve then as quadrature
  nodes.

+++ {"slideshow": {"slide_type": "fragment"}}

3. Calculate the cardinal functions $\ell_i(x)$ associated
  with $n+1$ nodes $\{x_i\}_{i=0}^n$ and then the weights are given
  by
  $\displaystyle w_i = \int_{a}^{b} \ell_i(x) {\,\mathrm{d}x}.$

+++ {"slideshow": {"slide_type": "slide"}}

This is the recipe you are asked to use in Exercise set 3.
Alternatively one can start from a reference interval,
leading  to

+++ {"slideshow": {"slide_type": "slide"}}

**Recipe 2 to construct a Gaussian quadrature.**

To construct a Gaussian formula on $[a,b]$ based
on $n+1$ nodes you proceed as follows

1. Construct a polynomial $p_{n+1} \in \mathbb{P}_{n+1}$
  on the reference interval $[-1, 1]$
  which satisfies

$$
\int_{-1}^1 p_{n+1}(x) q(x) {\,\mathrm{d}x} \quad {\forall\;} q \in \mathbb{P}_{n}.
$$

2. You determine the $n+1$ **real** roots $\{\hat{x}_i\}_{i=0}^n$
  of $p_{n+1}$ which serve then as quadrature
  nodes.

3. Calculate the cardinal functions $\ell_i(x)$ associated
  with $n+1$ nodes $\{\hat{x}_i\}_{i=0}^n$ and then the weights are given
  by
  $\displaystyle \hat{w}_i = \int_{-1}^{1} \ell_i(x) {\,\mathrm{d}x}.$

4. Finally, transform the resulting Gauß quadrature formula
  to the desired interval $[a,b]$ via

$$
x_i = \frac{b-a}{2}\widehat{x}_i + \frac{b+a}{2}, \quad  w_i = \frac{b-a}{2}\widehat{w}_i
\quad \text{for } i = 0, \ldots n.
$$

+++ {"slideshow": {"slide_type": "slide"}}

## Example: Revisiting Gauß-Legendre quadrature with 2 nodes

We will now derive the Gauß-Legendre quadrature with 2 nodes
we encountered in the previous lectures

Today we will use the `sympy` quite a bit,
and start with the snippets

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from sympy.abc import x # Denote our integration variable x
from sympy import integrate
```

+++ {"slideshow": {"slide_type": "fragment"}}

Spend a minute and have look at [integrate](https://docs.sympy.org/latest/modules/integrals/integrals.html) submodule.

+++ {"slideshow": {"slide_type": "slide"}}

First we construct the first 3 orthogonal polynomials
(order 0, 1, 2) on $[0,1]$. Spend 2 minutes to understand the
code below:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Interval
a, b = 0, 1

# Define scalar product
def scp(p,q):
    return integrate(p*q, (x, a, b))

# Define monomials up to order 2
mono = lambda x,m: x**m
def mono(x,m):
    return x**m

phis = [ mono(x,m) for m in range(3)]
print(phis)
```

+++ {"slideshow": {"slide_type": "slide"}}

Construct orthogonal polynomials (not normalized)

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Insert code here
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
ps = []
# Use Gram-Schmidt
for phi in phis:
    ps.append(phi)
    for p in ps[:-1]:
        ps[-1] = ps[-1] - scp(p, ps[-1])/scp(p, p)*p
        
print("ps")
print(ps)
```

+++ {"slideshow": {"slide_type": "slide"}}

Now write a code snippet to check whether they are actually orthogonal

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Insert code here
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
---
for p in ps:
    for q in ps:
        int_p_q = scp(p,q)
        print("int_p_q = {}".format(int_p_q))
```

+++ {"slideshow": {"slide_type": "slide"}}

Compute the roots of the second order polynomial. 
Of course you can do it by hand
but le'ts us `sympy` for it.
Spend a minute a have a look at [solve](https://docs.sympy.org/latest/modules/solvers/solvers.html) submodule.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from sympy.solvers import solve

# Insert code here
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(ps[-1])
xqs = solve(ps[-1])
print(xqs)
```

+++ {"slideshow": {"slide_type": "slide"}}

Next constructe the cardinal functions
$\ell_0$ and $\ell_1$ associated with the 2 roots.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Non-normalized version
L_01 = (x-xqs[1])
print(L_01)
print(L_01.subs(x, xqs[1]))
print(L_01.subs(x, xqs[0]))
# Normalize
L_01 /= L_01.subs(x, xqs[0])
print(L_01.subs(x, xqs[0]))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Non-normalized version
L_11 = (x-xqs[0])
# Normalize
L_11 /= L_11.subs(x, xqs[1])

Ls = [L_01, L_11]
print(Ls)
```

+++ {"slideshow": {"slide_type": "slide"}}

Finally, compute the weights.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
# Insert code here
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
ws = [integrate(L, (x, a, b)) for L in Ls ] 
print(ws)
```

## Exercise: Now construct a Gaussian quadrature for n=3 or n=4 points.
