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

# Numerical solution of ordinary differential equations

The topic of this note is the numerical solution of systems of
ordinary differential equations (ODEs).  This has been discussed in
previous courses, see for instance the web page
[Differensialligninger](https://wiki.math.ntnu.no/tma4100/tema/differentialequations)
from Mathematics 1, as well as in Part 1 of this course, where the
Laplace transform was introduced as a tool to solve ODEs analytically.

Before we present the first numerical methods to solve ODEs, we want to whet your appetite by quickly
looking at a number of examples which hopefully will will serve as test examples
throughout this topic.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Scalar first order ODEs
A scalar, first-order ODE is an equation on the form

$$
y'(t) = f(t,y(t)), \qquad y(t_0)=y_0,
$$

where $y'(t)=\frac{dy}{dx}$.
The *inital condition* $y(t_0)=y_0$ is required for a unique
solution.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

**Notice.**

It is common to use the term *initial value problem (IVP)* for an ODE
for which the inital value $y(t_0)=y_0$ is given, and we only are
interested in the solution for $x>t_0$. In these lecture notes, only
initial value problems are considered.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

:::{prf:example} Population growth and decay processes
:label: exa-pop-growth-ode

One of the simplest possible IVP is given by

$$
y'(t) = \lambda y(t), \quad y(t_0)=y_0.
$$(ode:eq:exponential)

For $\lambda > 0$ this equation can present a simple model for the growth of
some population, e.g., cells, humans, animals, with unlimited resources
(food, space etc.). The constant $\lambda$ then corresponds to the
*growth rate* of the population.

Negative $\lambda < 0$
typically appear in decaying processes, e.g., the decay of a radioactive
isotopes, where $\lambda$ is then simply called the *decay constant*.

The analytical solution to {ref}`ode:eq:exponential` 

$$
y(t) = y_0 e^{\lambda(t-t_0)}
$$(ode:eq:exponential_sol)

and will serve us at several occasions as reference solution to assess
the accuracy of the numerical methods to be introduced.

:::

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

:::{prf:example}  Time-dependent coefficients
:label: ode:exa:time-dep-coef

Given the ODE

$$
y'(t) = -2ty(t), \quad y(0) = y_0.
$$

for some given initial value $y_0$.
The general solution of the ODE is

$$
y(t) = C e^{-t^2},
$$

where $C$ is a constant. To determine the constant $C$,
we use the initial condition $y(0) = y_0$
yielding the solution

$$
y(t) = y_0 e^{-t^2}.
$$

:::

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Systems of ODEs
A system of $m$ ODEs are given by

\begin{align*}
y_1' &= f_1(t,y_1,y_2,\dotsc,y_m), & y_1(t_0) &= y_{1,0} \\ 
y_2' &= f_2(t,y_1,y_2,\dotsc,y_m), & y_2(t_0) &= y_{2,0} \\ 
     & \vdots                      &          &\vdots    \\ 
y_m' &= f_m(t,y_1,y_2,\dotsc,y_m), & y_m(t_0) &= y_{m,0} \\ 
\end{align*}

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

Or more compactly by

$$
\mathbf{y}'(t) = \mathbf{f}(t, \mathbf{y}(t)),  \qquad \mathbf{y}(t_0) = \mathbf{y}_0
$$

where we use boldface to denote vectors in $\mathbb{R}^m$,

$$
\mathbf{y}(t) =
\left(
\begin{array}{c}
y_1(t) 
\\ y_2(t) 
\\ \vdots 
\\ y_m(t)
\end{array}
\right),
\qquad
\mathbf{f}(t,\mathbf{y}) =
\left(
\begin{array}{c}
f_1(t,y_1,y_2,\dotsc,y_m), 
\\ f_2(t,y_1,y_2,\dotsc,y_m), 
\\ \vdots 
\\ f_m(t,y_1,y_2,\dotsc,y_m),
\end{array}
\right),
\qquad
\mathbf{y}_0 =
\left(
\begin{array}{c}
y_{1,0} 
\\ y_{2,0} 
\\ \vdots 
\\ y_{m,0}
\end{array}
\right).
$$

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

:::{prf:example} Lotka-Volterra equation
:label: ode:exa:lotka-volterra

The [Lotka-Volterra equation](https://en.wikipedia.org/wiki/Lotka-Volterra_equations) is
a system of two ODEs describing the interaction between preys and
predators over time. The system is given by

\begin{align*}
y'(t) &= \alpha y(t) - \beta y(t) z(t) \\ 
z'(t) &= \delta y(t)z(t) - \gamma z(t)
\end{align*}

where $x$ denotes time, $y(t)$ describes the population of preys and
$z(t)$ the population of predators.  The parameters $\alpha, \beta,
\delta$ and $\gamma$ depends on the populations to be modeled.

:::

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

:::{prf:example} Spreading of diseases
:label: ode:exa:spreading-disease

Motivated by the ongoing corona virus pandemic, we consider
a (simple!) model for the spreading of an infectious disease,
which goes under the name [SIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model).

:::

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

The SIR models divides the population into three
population classes, namely
* S(t): number individuals  **susceptible** for infection,
* I(t): number **infected** individuals, capable of transmitting the disease,
* R(t): number  **removed** individuals who cannot be infected due death or to immunity  
  after recovery

The model is of the spreading of a disease is based
on moving individual from $S$ to $I$ and then to $R$.
A short derivation can be found in  {cite}`Ch. 4.2 of <LangtangenLinge2016>`.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

The final ODE system is given by

\begin{align}
S' &= - \beta S I
\\
I' &= \beta S I - \gamma I
\\
R' &= \gamma I,
\end{align}

where $\beta$ denotes the infection rate, and $\gamma$ the removal rate.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Higher order ODEs
An initial value  ODE of order $m$ is given by

$$
u^{(m)} = f(t,u,u',\dotsc,u^{(m-1)}), \qquad u(t_0)=u_0, \quad
u'(t_0)=u'_0,\quad  \dotsc, \quad u^{(m-1)}(t_0) = u^{(m-1)}_0.
$$

Here $u^{(1)} =u'$ and $u^{(m+1)}=\frac{du^{(m)}}{dx}$, for $m>0$.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

:::{prf:example} Van der Pol's equation
:label: ode:exa:van-der-pol

[Van der Pol's equation](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)
is a second order differential equation, given by

$$
u^{(2)} = \mu (1-u^2)u' - u, \qquad u(0)=u_0, \quad u'(0)=u'_0,
$$

where $\mu>0$ is some constant.  As initial values $u_0=2$ and
$u'_0=0$ are common choices.

:::

Later in this module we will see how such equations can be rewritten as a
system of first order ODEs.  Systems of higher order ODEs can be treated similarly.
