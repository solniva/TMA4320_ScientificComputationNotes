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

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Numerical solution of ordinary differential equations: Error estimation and step size control

+++ {"slideshow": {"slide_type": "slide"}}

As always, we start by import some important Python modules.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import numpy as np
from numpy import pi
from numpy.linalg import solve, norm    
import matplotlib.pyplot as plt

# Do a pretty print of the tables using panda
import pandas as pd
from IPython.display import display

# Use a funny plotting style
plt.xkcd()
newparams = {'figure.figsize': (6.0, 6.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 14}
plt.rcParams.update(newparams)
```

+++ {"slideshow": {"slide_type": "slide"}}

This goal of this section is to develop Runge Kutta methods with
automatic adaptive time-step selection.

Adaptive time-step selection aims to
dynamically adjust the step size during the numerical integration
process to balance accuracy and computational efficiency. By
increasing the step size when the solution varies slowly and
decreasing it when the solution changes rapidly, adaptive methods
ensure that the local error remains within a specified tolerance. This
approach not only enhances the precision of the solution but also
optimizes the computational resources, making it particularly valuable
for solving complex and stiff ODEs where fixed step sizes may either
fail to capture important dynamics or result in unnecessary
computations.

+++ {"slideshow": {"slide_type": "slide"}}

:::{admonition} TODO
:class: danger dropdown
Add solution of three-body problem as an motivational example.
:::

+++ {"slideshow": {"slide_type": "slide"}}

### Error estimation
Given two methods, one of order $p$ and the other of order $p+1$ or higher. Assume we have
reached a point $(t_n,\mathbf{y}_n)$. One step forward with each of these methods can be written as

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

\begin{align*}
  \mathbf{y}_{n+1} &= \mathbf{y}_n + \tau \mathbf{\Phi}(t_n, \mathbf{y}_n; \tau), && \text{order $p$}, \\ 
  \widehat{\mathbf{y}}_{n+1} &= \mathbf{y}_n + \tau \widehat{\mathbf{\Phi}}(t_n, \mathbf{y}_n; \tau), && \text{order $\widehat{p} = p+1$ or more}. \\ 
\end{align*}

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

Let $\mathbf{y}(t_{n+1};t_n,\mathbf{y}_n)$ be the exact solution of the ODE through $(t_n,\mathbf{y}_n)$.
We would like to find an estimate for *the local error* $\mathbf{l}_{n+1}$, that is, the error in one step starting from  $(t_n, \mathbf{y}_n)$,

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\mathbf{l}_{n+1} = \mathbf{y}(t_{n+1};t_n,\mathbf{y}_n) - \mathbf{y}_{n+1}.
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

As we have already seen, the local error is determined by finding the power series in $\tau$ for both the exact and numerical solutions. The local error is of order $p$ if the lowest order terms in the series, where the exact and numerical solutions differ, are of order $p+1$. Therefore, the local errors of the two methods are:

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

\begin{align*}
\mathbf{y}(t_{n+1};t_n,\mathbf{y}_n) - \mathbf{y}_{n+1} &= \mathbf{\Psi}(t_n,y_n)\tau^{p+1}  +\dotsc, \\ 
\mathbf{y}(t_{n+1};t_n,\mathbf{y}_n) - \widehat{\mathbf{y}}_{n+1} &= \widehat{\mathbf{\Psi}}(t_n,y_n)\tau^{p+2} + \dotsc,
\end{align*}

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

where $\Psi(t_n,y_n)$ is a term consisting of method parameters and differentials of $\mathbf{f}$ and
$\dotsc$ contains all the terms of the series of order $p+2$ or higher.
Taking the difference gives

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\widehat{\mathbf{y}}_{n+1} - \mathbf{y}_{n+1} = \mathbf{\Psi}(t_n,\mathbf{y}_n)\tau^{p+1} + \ldots.
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

Assume now that $\tau$ is small, such that the *principal error term* $\mathbf{\Psi(t_n,y_n)}\tau^{p+1}$ dominates the error series. Then a reasonable approximation to the unknown local error $\mathbf{l}_{n+1}$ is the *local error estimate* $\mathbf{le}_{n+1}$:

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\mathbf{le}_{n+1} = \widehat{\mathbf{y}}_{n+1} - \mathbf{y}_{n+1} \approx \mathbf{y}(t_{n+1};t_n,\mathbf{y}_n) - \mathbf{y}_{n+1}.
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Stepsize control
The next step is to control the local error, that is, choose the step size so that $\|\mathbf{le}_{n+1}\| \leq \text{Tol}$ for some given tolerance Tol, and for some chosen norm $\|\cdot\|$.

Essentially:
Given $t_n, \mathbf{y}_n$ and a step size $\tau_n$.
* Do one step with the method of choice, and find an error estimate $\mathbf{le}_{n+1}$.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

* if  $\|\mathbf{le}\|_{n+1} < \text{Tol}$

    * Accept the solution $t_{n+1}, \mathbf{y}_{n+1}$.

    * If possible, increase the step size for the next step.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

* else

    * Repeat the step from $(t_n,\mathbf{y}_n)$ with a reduced step size $\tau_{n}$.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

In both cases, the step size will change. But how?
From the discussion above, we have that

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\| \mathbf{le}_{n+1} \| \approx D  \tau_{n}^{p+1}.
$$

where $\mathbf{le}_{n+1}$ is the error estimate we can compute, $D$ is
some unknown quantity, which we assume almost constant from one step
to the next.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

What we want is a step size $\tau_{new}$ such that

$$
\text{Tol} \approx D \tau _{new}^{p+1}.
$$

+++ {"slideshow": {"slide_type": "skip"}, "editable": true}

From these two approximations we get:

$$
\frac{\text{Tol}}{\|\mathbf{le}_{n+1}\|} \approx \left(\frac{\tau _{new}}{\tau _n}\right)^{p+1}
\qquad \Rightarrow \qquad
\tau_{new} \approx \left( \frac{\text{Tol}}{\|\mathbf{le}_{n+1}\|} \right)^{\frac{1}{p+1}} \tau _{n}.
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

That is, if the current step $\tau_n$ was rejected, we try a new step $\tau _{new}$
with this approximation.
However, it is still possible that this new step will be rejected as well.
To avoid too many rejected steps, it is therefore common to be a bit restrictive when choosing the new
step size, so the following is used in practice:

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\tau _{new} = P\cdot \left( \frac{\text{Tol}}{\|\mathbf{le}_{n+1}\|} \right)^{\frac{1}{p+1}} \tau _{n}.
$$

where the *pessimist factor* $P<1$ is some constant, normally chosen between 0.5 and 0.95.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

### Implementation
We have all the bits and pieces for constructing an adaptive ODE solver based on Euler's and Heuns's methods. There are still some practical aspects to consider:

* The combination of the two methods, can be written as

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

\begin{align*}
      \mathbf{k}_1 &= \mathbf{f}(t_n, \mathbf{y}_n), \\ 
      \mathbf{k}_2 &= \mathbf{f}(t_n+\tau, \mathbf{y}_n+\tau \mathbf{k}_1), \\ 
      \mathbf{y}_{n+1} &= \mathbf{y}_n + \tau \mathbf{k}_1, && \text{Euler} \\ 
      \widehat{\mathbf{y}}_{n+1} &= \mathbf{y}_n + \frac{\tau}{2}(\mathbf{k}_1 + \mathbf{k}_2), && \text{Heun} \\ 
      \mathbf{le}_{n+1} &= \|\widehat{\mathbf{y}}_{n+1} - \mathbf{y}_{n+1}\| = \frac{\tau}{2}\|\mathbf{k}_2-\mathbf{k}_1 \|.
\end{align*}

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

* Even if the error estimate is derived for the lower order method, in this case Euler's method, it is common to advance the solution with the higher order method, since the additional accuracy is for free.

+++ {"slideshow": {"slide_type": "fragment"}}

* Adjust the last step to be able to terminate the solutions exactly in $T$.

+++ {"slideshow": {"slide_type": "fragment"}}

* To avoid infinite loops, add some stopping criteria. In the code below, there is a maximum number of allowed steps (rejected or accepted).

+++ {"slideshow": {"slide_type": "slide"}}

A Runge - Kutta methods with an error estimate are usually called **embedded Runge - Kutta methods** or **Runge - Kutta pairs**, and
the coefficients can be written in a Butcher tableau as follows

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\begin{array}{c|ccccl}
    c_1 & a_{11} & a_{12} & \cdots & a_{1s} \\ 
    c_2 & a_{21} & a_{22} & \cdots & a_{2s} \\ 
    \vdots & \vdots &&&\vdots \\ 
    c_s & a_{s1} & a_{s2} & \cdots & a_{ss} \\ \hline
        & b_1 & b_2 & \cdots & b_s  & \qquad\text{Order $p$}\\ \hline
        & \widehat{b}_1 & \widehat{b_2} & \cdots & \widehat{b}_s  & \qquad\text{Order $\widehat{p}= p+1$}
   \end{array}.
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

Since
 * $\mathbf{y}_{n+1} = \mathbf{y}_n + \tau_n\sum_{i=1}^s b_i \mathbf{k}_i$

 * $\widehat{\mathbf{y}}_{n+1} = \mathbf{y}_n + \tau_n\sum_{i=1}^s \widehat{b}_i \mathbf{k}_i$

the error estimate is simply given by

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

$$
\mathbf{le}_{n+1} = \tau_n\sum_{i=1}^s (\widehat{b}_i - b_i)\mathbf{k}_i.
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

Recalling Euler and Heun,

$$
\begin{array}{ccccccc}
    \displaystyle
    \begin{array}{c|c}
      0 & 0 \\ \hline & 1
    \end{array}
    & \qquad  &
    \displaystyle
    \begin{array}{c|cc}
      0 & 0 & 0\\ 1 & 1 &0 \\ \hline & \frac{1}{2} & \frac{1}{2}
    \end{array}
    \\ 
    \text{Euler} && \text{Heun}
  \end{array}
$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

and the Heun-Euler pair can be written as

$$
\begin{array}{c|cc} 0 & & \\ 1 & 1 &   \\ \hline & 1 & 0 \\ \hline \displaystyle & \frac{1}{2} &  \frac{1}{2}
 \end{array}
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

A particular mention deserves also the classical *4-stage Runge-Kutta method*
from a previous notebook, which can be written as

$$
\begin{array}{c|cccc}
      0 & 0 & 0 & 0 & 0\\ \frac{1}{2} &  \frac{1}{2} & 0 & 0 & 0\\ \frac{1}{2} & 0 & \frac{1}{2} & 0 & 0\\ 1 &  0 & 0 & 1 & 0 \\ \hline & \frac{1}{6} & \frac{1}{3} & \frac{1}{3} & \frac{1}{6}
    \end{array}
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

See this [list of Runge - Kutta methods](https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods) for more.
For the last one there exist also a embedded Runge-Kutta 4(3) variant
due to **Fehlberg**:

$$
\begin{array}{c|ccccc}
      0 & 0 & 0 & 0 & 0 & 0
      \\ 
      \frac{1}{2} & \frac{1}{2} & 0 & 0 & 0 & 0
      \\ 
      \frac{1}{2} & 0 & \frac{1}{2} & 0 & 0 & 0
      \\ 
      1 &  0 & 0 & 1 & 0 & 0
      \\ 
      1 & \frac{1}{6} & \frac{1}{3} & \frac{1}{3} & \frac{1}{6} & 0
      \\ 
      \hline
      & \frac{1}{6} & \frac{1}{3} & \frac{1}{3} & 0 & \frac{1}{6}
      \\ 
     \hline
      & \frac{1}{6} & \frac{1}{3} & \frac{1}{3} & \frac{1}{6} & 0
\end{array}
$$

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

**Outlook.** In your homework/project assignment, you will be asked to implement automatic adaptive time step selection
based on embedded Runge-Kutta methods. You can either develop those from scratch or start from the `ExplicitRungeKutta` class
we presented earlier and incorporate code for error estimation and time step selection.
