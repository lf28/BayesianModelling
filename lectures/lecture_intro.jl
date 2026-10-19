### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f7ad60c5-aaa1-46b0-9847-5e693503c5ca
begin
    using PlutoTeachingTools
    using PlutoUI
    using HypertextLiteral
    using Distributions, LinearAlgebra
    using StatsPlots
    using LogExpFunctions
    # using Statistics
    using StatsBase
    using LaTeXStrings
    using Latexify
    using Random
    using GLM
end

# ╔═╡ e56e1a50-2663-4220-8bf0-8df93633d31a
begin
    using Plots
    default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
end

# ╔═╡ 6b33f22b-02c4-485a-a8e0-8329e31f49ea
TableOfContents()

# ╔═╡ 263003d7-8f36-467c-b19d-1cd92235107d
# figure_url = "https://sites.cs.st-andrews.ac.uk/people/lf28/figs/";

figure_url = "https://sites.cs.st-andrews.ac.uk/people/lf28/figs/";

# ╔═╡ e03e585c-23ab-4f1b-855f-4dcebee6e63c
# bayes_figure_url = "https://sites.cs.st-andrews.ac.uk/people/lf28/figs/bayes/";
bayes_figure_url = "https://sites.cs.st-andrews.ac.uk/people/lf28/figs/bayes/";

# ╔═╡ fdc69b6d-247c-49f4-8642-0345281c2370
ChooseDisplayMode()

# ╔═╡ 45b8e008-cfdc-11ed-24de-49f79ef9c19b
md"""

# Bayesian modelling


**Lecture 1. Probability theory review & Bayesian inference** 


$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang (lf28@st-andrews.ac.uk)

*School of Computer Science*

*University of St Andrews, UK*

*March 2023*

"""

# ╔═╡ 88621eb4-0d63-4cb8-808c-78494cdfb0ad
md"""
# Welcome






## Who am I


* ##### *name*: *Lei Fang*
  * Lecturer with the School of Computer Science, University of St Andrews
  * Research interests: Bayesian machine learning


* ##### *Email*: lf28 @ st-andrews.ac.uk
"""

# ╔═╡ 473eadf8-e1f7-4bdc-b937-da69aac35be5
md"""
## Target audience


##### Target audience

Researchers who are interested in learning Bayesian inference and applying statistical inference in their work


##### Pre-requisite

A basic knowledge of programming in R, Matlab, Mathematica, Python, C++, or similar

* we will use `Julia` in this course
* a popular language for numerical computing


If you’re not comfortable with linear algebra, calculus, don’t worry. 
* we will give intuitive explanations
* but it might be worth looking at an A-level textbook to brush up on your skills.
"""

# ╔═╡ 1371e5cd-cfd9-40ee-ad96-b97abdd30525
md"""

## Learning outcomes


By the end of this course, you should be able to


* Understand the basics and benefits of Bayesian inference
* Understand the workflow of Bayesian inference
* Construct and modify statistical generative models
* Appreciate why we often need to use MCMC in Bayesian inference
* Understand how MCMC algorithms work intuitively
* Know how to do applied Bayesian inference in Julia's `Turing.jl`
  * Bayesian linear regression; logistic regression; 
  * Bayesian non-linear predictive models
  * Bayesian unsupervised learning
"""

# ╔═╡ fd787b38-6945-49cf-becf-1d37847a88fe
md"""

## A couple of things



* #### Lecture notes/slides available from Moodle

\

* #### All source code available in [https://github.com/lf28/BayesianModelling](https://github.com/lf28/BayesianModelling)



"""

# ╔═╡ 0e80b64a-22ad-41d2-b6be-5d9b88af06c9
# md"""

# ## Agenda


# #### First day:

# * **Lecture 1** Introduction and overview
#   * probability theory review
#   * Bayesian inference workflow
#   * Graphical models


# * **Lecture 2** Bayesian inference with i.i.d. model 
#   * Priors; conjugate prior
#   * Bernoulli and Gaussian examples
#   * Model checking: predictive analysis
#   * Beyond i.i.d. model


# * **Lecture 3** MCMC 
#   * Metropolis-Hastings (MH) and Gibbs sampler
#   * Hamiltonian and NUTS sampler


# #### Second day:


# * **Lecture 4** Practical Bayesian inference with Turing.jl
#   * Quick introduction to Julia and *Turing.jl*


# * **Lecture 5** Bayesian linear regression 
#   * and their extensions and variants

# * **Lecture 6** Bayesian logistic regression
#   * Bayesian fixed basis models (sparsity)





# #### Third day:

# A mix of applied and theoretical problems (all applied)

# * learn how to implement what we have learnt in the morning
#   * involves a bit programming
# * how to use `Turing.jl`
# * I will be around to help 
# """

# ╔═╡ 692cb743-d345-4b1b-b71f-3e26d1c7c80a
md"""

## Some motivating examples - 1
#### Example 1. Coughing


!!! note "Coughing example"
	##### Someone **coughed** in our class. What is the chance he/she has got **cancer**?

## Some motivating examples - 1 (cont.)
#### Example 1. Coughing


!!! note "Coughing example"
	##### Someone **coughed** in our class. What is the chance he/she has got **cancer**?

* the chance is probably low
* if agree, you might have already *implicitly* applied Bayesian inference
"""

# ╔═╡ 8bf95702-90ac-4b3c-a027-e1b673359a60
md"""

## Some motivating examples - 2 
#### Example 2. Amazon sellers

!!! note "Amazon seller"
	##### Which seller to choose?
	* the first with **400k** reviews
	* the second with **2k** reviews
"""

# ╔═╡ 3a80d31d-9f32-48b1-b5fd-788b588d96cb
Resource(figure_url * "/figs4CS5010/amazonreviews.png", :width=>700, :align=>"left")

# ╔═╡ dede5ea1-c67d-4344-b6d0-c4a9b8d8dbeb
md"""

## Some motivating examples - 3
#### Example 3. Regression



##### *Regression problem*: find the relationship between input $X$ and $Y$ (when $Y$ is continuous)


"""

# ╔═╡ e7242619-19fc-461b-a5e0-fc50361d2565
TwoColumnWideRight(
    md"""
    \
    \
    \

    #### For example, *linear regression*

    ```math
    \Large f(X) = \beta_0 + \beta_1 X 
    ```
    """,
    html"""<center><img src="https://sites.cs.st-andrews.ac.uk/people/lf28/figs/slrpop1-output-1.png" width = "700"/></center>
    """
)

# ╔═╡ 63b92973-1031-439c-aa68-60f00552ab18
md"""
##


##### Now consider a non-linear dataset
* the relationship is non-linear
* pay attention to the gaps in the middle
"""

# ╔═╡ 926be45c-84a4-4bb5-8cc0-56bdf74bc581
md"""

## Some motivating examples - 4
#### Example 4. Classification



##### *Classification problem*: find the relationship between input $X$ and $Y$ (when $Y$ is categorical):

$\large P(Y = \text{class 1} | X=x) = \sigma(x)$

* what is the probability that an instance belongs to class 1 or  2
"""

# ╔═╡ 9066781d-b542-4c95-8a5b-d244df217650
TwoColumnWideRight(
    md"""
    \


    !!! note "Question"
    	##### How would you classify data at the following input locations?

    """,
    Resource(bayes_figure_url * "logis_data.png", :width=>600)
)

# ╔═╡ c9b2fa82-841d-47cf-b7b4-23f8a86e60a7
md"""

## Some motivating examples - 4 (cont.)

> ##### _Which one do you prefer?_
"""

# ╔═╡ 2bc778d0-6974-4a97-8769-3cb66100fd80
TwoColumn(Resource(bayes_figure_url * "freq_logis.png"), Resource(bayes_figure_url * "bayes_logis.png"))

# ╔═╡ 2fba13a9-a0ab-44a6-ba4d-b828ebe96274
md"""

## We are going to see 



> #### How *Bayesian inference* handles the above problems in a **uniform** procedure
> * **reasonable** inference results emerge automatically without ad hoc remedies

## What is Bayesian procedure (a preview)?

!!! tip "Bayesian procedure"
	#### In a nutshell, the Bayesian inference procedure has two stages
	> * ##### (*modelling*) Tell a story about your data (how your data is generated)

	> * ##### (*computation/inference*) Leave the inference to probability rules


"""

# ╔═╡ db1f6632-f894-4ce7-a5b4-493158a13860
md"""

# Probability Theory 

"""

# ╔═╡ f05dcb39-84c9-42e5-95f6-67238c574609
md"""

## Random variable 


#### Let's consider ``\cancel{\textbf{random}} \textbf{variable}``  first, 

> ##### e.g. a variable ``\large X= 5``

* ``X``: a _deterministic_ variable
* *with ``100\%`` certainty*, ``X`` takes the value of 5


##


#### ``{\textbf{Random}}\; \textbf{variable}``: the value is _**random**_ rather than _**certain**_

\

##### *For example*: $X$ is the rolling realisation of a 6--faced die 🎲
* ``X \in \{1,2,3,4,5,6 \}``

* and ``X`` has a *probability distribution* ``P(X)``
```math
\large
\begin{equation}  \begin{array}{c|cccccc} 
 & X = 1 & X = 2 & X = 3 & X = 4 & X = 5 & X = 6 \\
\hline
P(X) & 1/6 & 1/6 & 1/6 & 1/6 & 1/6 & 1/6

\end{array} \end{equation} 

```
"""

# ╔═╡ 0cc7386d-4584-4a34-bbf7-546bf9ac1134
md"""
## Probability distributions


##### Random variables' uncertainties is quantified by **probability distributions**

$$\large P(X=x) \geq 0, \forall x\;\; \text{and}\;\; \sum_x{P(X=x)}=1$$
* non-negative and sum to one


##### **Temperature** $T: P(T)$

```math

\begin{equation}  \begin{array}{c|c} 
T & P(T)\\
\hline
hot & 0.5 \\
cold & 0.5
\end{array} \end{equation} 
```


##### **Weather** $W: P(W)$

```math

\begin{equation}  \begin{array}{c|c} 
W & P(W)\\
\hline
sun & 0.6 \\
rain & 0.1 \\
fog & 0.2 \\
snow & 0.1
\end{array} \end{equation} 
```
"""

# ╔═╡ b52911ff-886a-4113-85a5-af508dc0cb8e
md"""
## Discrete *vs* continuous r.v.s



#### *Discrete random variable*: ``\mathcal A`` is discrete, *e.g.* ``\mathcal A=\{\texttt{true},\texttt{false}\}, \mathcal A=\{1,2,3,\ldots, \infty\}``
  * ``P:`` *probability mass function (p.m.f)* satisfies 
    * ``\large 0\leq P(x) \leq 1``
    * ``\large\sum_{x\in \mathcal A} P(x) = 1``
  * *e.g.* Bernoulli, Binomial, Poisson and so on


\


#### *Continuous random variable*: ``\mathcal A`` is continuous, *e.g.* ``\mathcal A = [0,1], \mathcal A= (-\infty , +\infty)``
  * ``p:`` *probability density function (p.d.f)* satisfies (smaller case ``p``)
    * ``\large p(x) > 0``
    * ``\large \int_{x\in \mathcal A} p(x) \mathrm{d}x=1``
  * *e.g.* Gaussian, Beta distributions
"""

# ╔═╡ 34a4fdfa-df3f-426c-acbd-d18056246714
md"""


| Variable kind | Discrete or continous| Target space ``\mathcal A`` |
| :---|:---|:---|
| Toss of a coin | Discrete | ``\{0,1\}`` |
| Roll of a die | Discrete |``\{1,2,\ldots, 6\}`` |
| Outcome of a court case | Discrete |``\{0,1\}`` |
| Number of heads of 100 coin tosses| Discrete|``\{0,1, \ldots, 100\}`` |
| Number of covid cases | Discrete|``\{0,1,\ldots\}`` |
| Height of a human | Continuous |``R^+=(0, +\infty)`` |
| The probability of coin's bias ``\theta``| Continuous|``[0,1]`` |
| Measurement error of people's height| Continuous|``(-\infty, \infty)`` |
"""

# ╔═╡ 910399dd-ebe1-447f-967f-7a38e4913b56
md"""
## 	Examples: discrete r.v. -- Bernoulli


#### ``X``: a random variable taking binary values, the domain ``\mathcal{A} = \{1, 0\}``

* the probability distribution (probability mass function) is 

```math
\large P(X ) =\begin{cases}\theta & X= 1 \\ 1-\theta & X=0 \end{cases}
```

* ``0\leq \theta \leq 1`` is the parameter of the distribution: called *bias*



"""

# ╔═╡ 10beeb49-277a-489a-969d-140d51fcfde3
@bind θ Slider(0:0.1:1, default=0.5)

# ╔═╡ e11b9b91-a912-45a1-913c-aeb3e78e60fc
md""" ``θ`` = $(θ)"""

# ╔═╡ 17f73ee7-e842-4f10-9118-47399e67f047
let
    bar(Bernoulli(θ), xticks=[0, 1], xlabel=L"X", ylabel=L"P(X)", label="", ylim=[0, 0.8])
end

# ╔═╡ b1ff85ec-6a23-4276-a7e8-6303e294f301
md"Total trials: ``n``"

# ╔═╡ 6a3217fb-47c3-400c-9fbd-5c524a1e54fb
@bind nb Slider(2:1:100, default=10)

# ╔═╡ 5fd8cfe7-258d-4c8d-a482-c439d7755a72
md"Bias of each trial ``\theta``"

# ╔═╡ 5a73ebc0-2bc1-425a-b3e6-18afd2247cb6
@bind θ_bi Slider(0:0.05:1, default=0.7)

# ╔═╡ 64b0c1fb-4758-4536-913b-3816ca431608
md"""

## Examples*: discrete r.v. -- Binomial (conti.)


#### *Binomial* r.v.: the number of successes for ``n`` independent Bernoulli trials (with bias ``\theta``),  $\mathcal{A}=?$

$$\large P(X=x) = \binom{n}{x} (1-\theta)^{n-x} \theta^{x},\;\; \text{for } x= 0,1,\ldots, n$$
  * ``\binom{n}{x}``: *binomial coefficient*, e.g. $\binom{3}{2} = 3$ 


##### e.g. A binomial r.v. with ``n``= $(nb), ``\theta`` = $(θ_bi) is plotted below

* the number of heads: tossing a coin (with bias $(θ_bi)) $(nb) times

"""

# ╔═╡ b680c9da-2cb5-432f-8032-e2a37fc3e738
let
    binom = Binomial(nb, θ_bi)
    bar(binom, label="", xlabel=L"X", xticks=0:nb, ylabel=L"P(X)", title=L"n=%$(nb),\;\;\theta = %$(θ_bi)", legend=:topleft)

    # p_more_heads = exp(logsumexp(logpdf.(binom, floor(nb/2)+1:1:nb)))
    # vspan!([floor(nb/2)+0.5, nb+0.5], alpha=0.5, label=L"P(X \geq \lfloor n/2 \rfloor +1)\approx%$(round(p_more_heads; digits=2))")
end

# ╔═╡ 1e4db58a-e57d-457e-9bf4-20961a4775b1
md"""

## 	Examples: continuous r.v. -- Uniform distribution


#### ``X`` is a **uniform distribution** over interval ``[a, b]``

```math
\large p(x) = \begin{cases}\frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise}\end{cases}
```


##### As an example, a uniform distribution ``x\in [0,1]``
```math
\large p(x) = \begin{cases}1 & 0 \leq x \leq 1 \\ 0 & \text{otherwise}\end{cases}
```

"""

# ╔═╡ 42466480-2d4d-4a62-be06-ede102d818f0
let
    a, b = 0, 1
    plot(Uniform(0, 1), fill=true, alpha=0.5, lw=2, ylim=[0, 2], xlabel=L"X", ylabel=L"p(X)", label="", title=L"p(X) = \texttt{Uniform}(0,1)")
end

# ╔═╡ d1e63621-5e4a-4f43-aa4f-a41f12fe632a
md"""
## Examples: continuous r.v. -- Gaussian distribution



"""

# ╔═╡ 7750287e-96e3-4920-9e3e-0dd461e2f2ea
TwoColumn(md"""
\

#### Gaussian random variable $X$


$(Resource("https://sites.cs.st-andrews.ac.uk/people/lf28/figs/CS5914/gaussian_eq_1d.png", :width=>450, :align=>"middle"))

\

* ``\mu``: mean or location
* ``\sigma^2``: variance or scale, controls the spread



""", let

    μs = [-3, 0, 3]
    σ²s = [1, 2, 5]
    plt_gaussian = Plots.plot(title="Gaussian distributions", xlabel=L"X", ylabel=L"p(X)", size=(300, 300))
    for i in 1:3
        plot!(plt_gaussian, Normal(μs[i], sqrt(σ²s[i])), fill=true, alpha=0.5, label=L"\mathcal{N}(μ=%$(μs[i]), σ^2=%$(σ²s[i]))")
        vline!([μs[i]], color=i, label="", linewidth=2)
    end
    plt_gaussian
end)

# ╔═╡ 7416236c-5ff0-4eb8-b448-e50acb0c016b
md"""
## Joint distributions


#### A **joint distribution** over a set of random variables: 

$\Large X_1, X_2, \ldots, X_n$
* specifies a real number for each assignment (or outcome): 


```math
\large\begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n) = P(x_1, x_2, \ldots, x_n) \end{equation} 
```

* Must still statisfy

$\large P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 

##

##### Example: joint distribution of temperature (``T``) and weather (``W``): ``P(T,W)``
```math
\large\begin{equation}
\begin{array}{c c |c} 
T & W & P\\
\hline
hot & sun & 0.4 \\
hot & rain & 0.1 \\
cold & sun  & 0.2\\
cold & rain & 0.3\end{array} \end{equation} 
```



"""

# ╔═╡ c2f8d8e7-16d5-46c5-a66d-664dbaeb4c91
md"""

## Conditional probability

#### Conditional probability is defined as

$$\large P(A=a|B=b) = \frac{P(A=a, B=b)}{P(B=b)}$$

* read: *probability of ``A`` given ``B``*
* the probability of $A=a$ given $B=b$ is true



"""

# ╔═╡ be70c96c-685d-4136-8828-c8a35a5cbe0b
Resource(figure_url * "figs4CS5010/condiprob.png", :width=>800, :align=>"left")

# ╔═╡ 48484810-1d47-473d-beb3-972ca8b934d5
md"""
## Two probability rules


#### There are only two probability rules 

* #### *sum rule*

 

* #### *product rule*




"""

# ╔═╡ 559b0936-a051-4f2e-b0ca-81becd631bce
md"""
## Probability rule 1: sum rule



$$\large P(X_1) = \sum_{x_2}P(X_1, X_2=x_2);\;\; P(X_2) = \sum_{x_1}P(X_1=x_1, X_2),$$

* ``P(X_1), P(X_2)`` are called **marginal probability distribution**

"""

# ╔═╡ 1a930c29-0b6e-405b-9ada-08468c69173c
md"""
## Example -- sum rule

"""

# ╔═╡ ddc281f3-6a90-476a-a9b9-34d159cca80a
Resource(figure_url * "/figs4CS5010/sumrule_.png", :width=>800, :align=>"left")

# ╔═╡ 05e3c6d0-f59b-4e1e-b631-11eb8f8b0945
md"""
## Probability rule 2: product rule


#### Product rule (*a.k.a* **chain rule**)

$$\large P(X, Y) = P(X)P(Y|X);\;\; P(X, Y) = P(Y)P(X|Y)$$

  * ##### the chain order doesn't matter
  * joint distribution factorised as a product
    * between marginal and conditional

"""

# ╔═╡ 32bd86a4-0396-47f5-9025-6df83268850e
md"""

## Product rule: generalisation


#### Chain rule for higher-dimensional random variables

* ##### for three
$\large P(x_1, x_2, x_3) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)$

* ##### for any ``n``:

$\large \begin{align}P(x_1, x_2, \ldots, x_n) &= P(x_1)P(x_2|x_1)P(x_3|x_1,x_2) \ldots \\ &=\prod_{i=1}^n P(x_i|x_1, \ldots, x_{i-1})\end{align}$

"""

# ╔═╡ e12f0e44-2b7d-4075-9986-6322fb1498c4
md"""

## Bayes' theorem



"""

# ╔═╡ e4044dd5-f1a3-4ec6-b0e6-2e4a3cfaf0ce
@htl("<center><img src=$(figure_url * "CS5914/bayes.png") height='200' /></center>")

# ╔═╡ f60b1407-df78-41b5-959c-742910364fa6
md"""

* #### *Bayes' rule* provides us a way to find out the posterior (or the inverse probability);
  * ##### given the prior and likelihood (or the forward probability)




## Bayes' theorem -- the evidence term


"""

# ╔═╡ b8677457-b952-4dad-9f11-ae197c126287
@htl("<center><img src=$(figure_url * "CS5914/bayes4.png") height='200' /></center>")

# ╔═╡ 01231683-61fc-4f36-b004-c005723c53e0
md"""


* ##### note the **Evidence** is the sum of the numerator 

* ##### it serves the purpose of **normalising constant** such that the posterior sum to 1

```math
\large
\sum_h p(h|\mathcal{D}) = 1
```

##

##### Since the *Evidence* ``p(\mathcal{D})`` is a constant 
* it does **not** depend on ``h``
"""

# ╔═╡ dabd0bef-40dc-49d8-bfef-0ecfd2efcae3
@htl("<center><img src=$(figure_url * "CS5914/bayes2.png") height='200' /></center>")

# ╔═╡ cf969c4c-909d-4dc7-ad95-74352f987c1f
md"""
## Why chain rule (or Bayes' rule) is useful

"""

# ╔═╡ 571a31d9-77fe-4457-827a-6a746a529a4a
TwoColumn(md"""
#### It is always easier for us to think generatively or causally

$$\large H \xRightarrow[\text{probability}]{\text{forward}} \mathcal{D}$$ 


* ##### from the hidden cause ``H`` to the observed effects ``\mathcal{D}``

""", @htl("<center><img src=$(figure_url * "cough_bn.png") height='200' /></center>")
)

# ╔═╡ a602e848-8438-4a41-ab76-ec3f6497de3e
md"""



## Example: Cough example
#### first Bayesian inference problem

* ##### ``H``: the unknown cause ``h \in \mathcal{A}_H=\{\texttt{cold}, \texttt{cancer}, \texttt{healthy}\}`` 

* ##### ``\mathcal{D}``: the observed data is ``\mathcal{D} = \texttt{Cough}``
 

#### The chain rule allows us to specify the full joint:

```math
\large P(H, \mathcal{D}) = P(H) P(\mathcal{D}|H)
```

* ``\mathcal{D}`` depends on ``H``: the cough observation depends on the hidden cause

* Bayes' rule: compute the reverse probability: ``P(H|\mathcal{D})``

"""

# ╔═╡ 6c0333fd-5b6f-4a5c-b809-781614a7fe63
@htl("<center><img src=$(figure_url * "cough_bn.png") height='200' /></center>")

# ╔═╡ b7973f17-7f7f-4cb2-bfd2-9defc4b5910b
md"""
## Example -- Cough example (cont.)

#### **Posterior**: ``P(H|\mathcal{D})``

* routinely apply Bayes' theorem

$$\begin{align}P(h|\texttt{cough}) &= \frac{P(h) P(\texttt{cough}|h)}{P(\texttt{cough})}\\

&\propto \begin{cases} 0.8 \times 0.05 & h=\texttt{healthy} \\ 0.18 \times 0.85 & h=\texttt{cold} \\ 0.02 \times 0.9& h=\texttt{cancer}  \end{cases}
\end{align}$$



* note that the evidence term 

  $$\begin{align}
  P(\texttt{cough}) &= \sum_{h\in \mathcal{A}_H} P(h) P(\texttt{cough}|h) \\&= 0.8 \times 0.05 + 0.18 \times 0.85 + 0.02 \times 0.9 = 0.211\end{align}$$
  is a **normalising constant**: it normalises ``\text{prior} \times \text{likelihood}`` such that the posterior becomes a valid distribution (*i.e.* sum to one)

$$P(h|\texttt{cough}) \approx \begin{cases} 0.19 & h=\texttt{healthy} \\
0.725 & h=\texttt{cold}\\ 
0.085 & h=\texttt{cancer}\end{cases}$$
"""

# ╔═╡ 52df31d8-f439-4b44-a5f3-0b2c27db7522
begin
    Hs = ["healthy", "cold", "cancer"]
    prior_p = [0.8, 0.18, 0.02]
    liks = [0.05, 0.85, 0.9]
    x_ = 1
    post_p = prior_p .* liks
    marg_lik = sum(post_p)
    post_p = post_p/marg_lik
end;

# ╔═╡ fea3395d-250d-406c-9a8c-646f5c48ab12
md"""

## Example -- Cough example (cont.)


##### **Prior:** ``P(H)``

* three possible causes ``h \in \{\texttt{cold}, \texttt{cancer}, \texttt{healthy}\}`` with a prior


$$P(h) = \begin{cases} 0.8 & h=\texttt{healthy} \\
0.18 & h=\texttt{cold}\\ 
0.02 & h=\texttt{cancer}\end{cases}$$

$(begin
prior_plt=plot(prior_p, fill=true, st=:bar, xticks=(1:3, Hs), markershape=:circ, ylim =[0, 1.0], label="", title="Prior: "*L"P(h)")
end)


"""

# ╔═╡ 265bdf7b-2d31-4ff0-8af7-a5e9c5e881bd
like_plt = plot(liks, fill=true, st=:bar, xticks=(1:3, Hs), markershape=:circ, ylim=[0, 1.0], label="", title="Likelihood: "*L"P(\texttt{cough}|h)", c=2, lw=2);

# ╔═╡ 602bfca9-b90e-48cc-949f-2909aacf3546
md"""
## Example -- Cough example (cont.)
 
##### *Likelihood*: ``P(\mathcal{D}|H)``


* the observed ``\mathcal{D} = \texttt{cough}`` (``\neg \texttt{cough}`` means cough is not observed)

|  | ``h`` | ``p(\texttt{cough}\|h)``|``p(\neg\texttt{cough}\|h)``|
|---|:---:|:---:|:---:|
|  | ``\texttt{healthy}`` | 0.05 | 0.95 |
|  | ``\texttt{cold}`` | 0.85 | 0.15|
|  | ``\texttt{cancer}`` | 0.9 | 0.1|


$(begin
like_plt
end)
"""

# ╔═╡ dd011a05-7cea-4793-b5f0-770e20f3a503
md"""
## Aside: maximum likelihood estimation (MLE)

#### *Maximum likelihood estimation (MLE)* is a popular frequentist method

```math
\large
\hat{h}_{\text{ML}} \leftarrow \arg\max_{h} P(\mathcal D|h)
```

* "``\arg\max``": find the variable ``h`` that maximise ``P(\mathcal D|h)``
* ``\hat h_{\text{ML}}`` is called **maximum likelihood estimator**
* do not involve prior
* but it suffers *overfiting*!


$(begin
like_plt
end)
* e.g. ``h_{\text{ML}}`` is ``\texttt{cancer}``!

"""

# ╔═╡ b5ed7206-7d64-4845-a56d-2bbe35da9a38
begin
    # l = @layout [a b; c]
    post_plt = Plots.plot(post_p, seriestype=:bar, markershape=:circle, label="", color=3, xticks=(1:3, Hs), ylim=[0, 1], ylabel="", title="Posterior: "*L"P(h|\texttt{cough})", legend=:outerleft)
    # plt_cough = Plots.plot(prior_plt, like_plt, post_plt, layout=l)
end

# ╔═╡ 2ca8a673-9eeb-4295-91ed-27c173a91db7
md"""
## Bayesian inference as Belief update


#### Bayesian inference: a belief update procedure 

```math
\large
\text{prior } P(H) \Rightarrow \text{posterior } P(H|\mathcal{D})
```

"""

# ╔═╡ 9c2086db-b108-4b15-87c8-8f38e53767fb
let
    l = @layout [a b]
    plt_cough = Plots.plot(prior_plt, like_plt, layout=l, size=(700, 250))
end

# ╔═╡ c23e3616-82fc-4316-a561-93decb98cc9c
md" ```math
\Big\Downarrow \text{belief update}
```
"

# ╔═╡ 302a0551-dea3-49e9-bcca-409126571105
plot(post_plt)

# ╔═╡ 65d9e2ed-1702-43aa-aafd-e2b38d0b403a
md"""

## Another example -- change point detection*


> #### Your friend has two coins: 
> * ##### one fair ``p_1= 0.5`` 
> * ##### the other is bent with bias $p_2= 0.2$ 
> ##### He always uses the fair coin at the beginning then switches to the bent coin after some unknown switch

> #### Given tosses ``\mathcal{D}= [t,h,h,t,h,t,t]``
> #### When did he switch?
"""

# ╔═╡ 5920cd07-30b9-44a7-8f5c-999c1f8ef043
md"""
## Forward generative modelling*

#### Let's identify the random variables first

* ##### hypothesis: ``S\in \{1,2,\ldots, n-1\}``, the unknown switching point




* ##### data (observation): ``\mathcal{D}=\{d_1, d_2,\ldots\}``


#### Let's think **forwardly**, or **generatively**

```math
\Large S \Rightarrow {d_1, d_2, \ldots, d_n}
```

* ##### assume knowing ``S\in \{1,2,\ldots, n-1\}`` 

* ##### the likelihood ``P(\mathcal{D}|S)`` is ? 




"""

# ╔═╡ 5acefb1f-209e-489f-afb9-dce517ad6225
md"""
## Forward generative modelling (cont.)*


$$\large P(\mathcal D|S) = \underbrace{\prod_{i=1}^S P(d_i|p_1 =0.5)}_{\text{fair coin}} \underbrace{\prod_{j=S+1}^N P(d_j|p_2 = 0.2)}_{\text{bent coin}};$$ 



  * *e.g.* if ``S=1``

$$P(\{0,1,1,0,1,0,0\}|S=1) = \underbrace{0.5}_{\text{before switch: }\{0\}} \times \underbrace{0.2\cdot 0.2\cdot(1-0.2)\cdot 0.2\cdot (1-0.2)^2}_{\text{after switch: }\{1,1,0,1,0,0\}}$$

  * *e.g.* if ``S=2``

$$P(\{0,1,1,0,1,0,0\}|S=2) = \underbrace{0.5\cdot 0.5}_{\text{before switch: }\{0,1\}} \times \underbrace{0.2\cdot(1-0.2)\cdot 0.2\cdot (1-0.2)^2}_{\text{after switch: }\{1,0,1,0,0\}}$$

* then we can calculate the likelihood for ``S=1,2,\ldots, n-1`` 

"""

# ╔═╡ 866cd5fd-9d9d-42da-b05e-b4b3ad958ecb
𝒟 = [0, 1, 1, 0, 1, 0, 0];

# ╔═╡ 6593bd28-f483-4168-ac69-eae469fbcb27
md"""
## Forward generative modelling: Prior for P(S)*


##### To reflect our ignorance, we use a uniform flat prior

$$\large P(S=s) = \frac{1}{n-1}; \; \text{for }s \in \{1,2,\ldots, n-1\}$$

"""

# ╔═╡ 9e0953ee-3451-418a-afe2-b2177a2a8898
begin
    # to type 𝒟, type "\scrD"+ "tab"
    n1 = length(𝒟)-1
    Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1, marker=:circle, label="", ylim=[0, 1.0], xlabel=L"S", title=L"P(S)")
    Plots.plot!(1:n1, 1/n1 * ones(n1), st=:path, fill=true, color=1, alpha=0.3, label="Prior")
end

# ╔═╡ 9b925cf9-fb2f-403e-8208-6f1358a598d9
function ℓ_switch(D, p₁=0.5, p₂=0.2)
    likes = zeros(length(D)-1)
    for t in 1:length(likes)
        # 		Bernoulli(p) returns an instance of Bernoulli r.v.; coin tosses are Bernoullis!
        # 		pdf(Bernoulli(p), y) return the probability, either p or 1-p depends on y
        # 		prod times everything together
        likes[t] = prod(pdf.(Bernoulli(p₁), D[1:t])) * prod(pdf.(Bernoulli(p₂), D[(t+1):end]))
    end
    return likes, sum(likes)
end;

# ╔═╡ 904fc5dc-17b1-4500-9cb8-d7e472c90212
begin
    Plots.plot(1:(length(𝒟)-1), ℓ_switch(𝒟)[1], xlabel=L"S", ylabel=L"p(\mathcal{D}|S)", title="Likelihood "*L"p(\mathcal{D}|S)", st=:sticks, marker=:circle, label="")
end

# ╔═╡ 72fb8fa0-8263-4729-af19-ff6047865a61
md"""
## Make the inference*

##### The rest is very routine: **apply Bayes' rule** mechanically

$$\Large P(S|\mathcal D) \propto P(S) P(\mathcal D|S)$$
  * then normalise by dividing by the evidence: ``P(\mathcal D) = \sum_s  P(S)P(\mathcal D|S)``



"""

# ╔═╡ baf8e5b2-4473-4e22-bb4d-2c1fd8cccfaa
begin
    post_p_switch = ℓ_switch(𝒟)[1] ./ ℓ_switch(𝒟)[2]
end;

# ╔═╡ 691c897a-a282-4023-8b7a-d7df665bf53c
begin
    # n1 = length(𝒟)-1
    Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1, marker=:circle, label="", legend=:outerbottom, size=(600, 500))
    Plots.plot!(1:n1, 1/n1 * ones(n1), st=:path, fill=true, color=1, alpha=0.3, label="Prior")
    Plots.plot!(1:n1, post_p_switch, xlabel=L"S", ylabel=L"p(S|\mathcal{D})", title="Posterior after update: "*L"P(S|\mathcal{D})", st=:sticks, marker=:circle, color=2, label="")
    Plots.plot!(1:n1, post_p_switch, st=:path, fill=true, alpha=0.3, color=2, label="Posterior")
end

# ╔═╡ 4d398cf7-c3dd-49cf-b0c3-3731e50c1c37
md"""


## Another dataset ?*

"""

# ╔═╡ 2ca2d909-7499-45fc-864e-cf22486896dc
𝒟₂ = [0, 1, 1, 0, 1, 0, 0]

# ╔═╡ 9d9e9537-8194-4716-8f1f-fd1013c7e040
begin

    plot(findall(𝒟₂ .== 0), 𝒟₂[𝒟₂ .== 0], seriestype=:scatter, label=L"\texttt{tail}", legend=:outerbottom, xlabel="time")
    plot!(findall(𝒟₂ .== 1), 𝒟₂[𝒟₂ .== 1], seriestype=:scatter, label=L"\texttt{head}")
end;

# ╔═╡ facc40e6-87cd-4884-b20e-28d84e4337c6
begin
    post_p₂ = ℓ_switch(𝒟₂)[1] ./ ℓ_switch(𝒟₂)[2]
end;

# ╔═╡ 7652aa5c-47c7-4263-a673-00f073f91018
let
    post_p = post_p₂
    n1 = length(𝒟₂)-1
    Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1, marker=:circle, label="", legend=:outerbottom, size=(600, 500))
    Plots.plot!(1:n1, 1/n1 * ones(n1), st=:path, fill=true, color=1, alpha=0.3, label="Prior")
    Plots.plot!(1:n1, post_p, xlabel=L"S", ylabel=L"p(S|\mathcal{D})", title="Posterior after update: "*L"P(S|\mathcal{D})", st=:sticks, marker=:circle, color=2, label="")
    Plots.plot!(1:n1, post_p, st=:path, fill=true, alpha=0.3, color=2, label="Posterior")
end

# ╔═╡ 0198a401-ed47-4f5c-8aca-441cccff472b
md"""

# Conditional independence and Graphical models
"""

# ╔═╡ 6db0a9fa-0137-413e-b122-24a51c4319e0
md"""

## Independence

#### Random variable $X,Y$ are **independent**, if 

$$\Large \forall x,y : P(x,y) = P(x)P(y)$$


#### *Alternatively*

$$\Large \forall x,y: P(x|y) = P(x)$$
  * intuition: knowing (conditional on $Y=y$) does not change the probability of ``X``
  

##

#### For _multiple_ independent random variables,

```math
\Large P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i)
```
"""

# ╔═╡ 5c0a6bf7-7ada-4122-ae09-1963969e4aa9
# let

# 	layout = @layout [a            _
# 	                  b{0.8w,0.8h} c]
# 	plot(layout = layout)
# 	plot(marg_py, joint_plt, marg_px, layout = layout)
# 	# plot!(joint_plt, subplot = 2, framestyle = :box)
# 	# plot!(marg_px, subplot = 1, framestyle = :box)
# 	# plot!(joint_plt)
# 	# plot!(marg_px)
# 	# plot!(marg_py)

# end

# ╔═╡ 2ed5995f-26dd-4f3e-92fb-a04b334cfb99
md"""

## Independence -- ``P(X|Y) =P(X)``



##### If ``X, Y`` are *independent*, 

> ##### then $P(X|Y=y)$ does NOT depend on ``Y``
"""

# ╔═╡ 61106a36-e880-499b-a442-f42d3c196e6b
md"Toggle independence: $(@bind indep CheckBox(default=false))"

# ╔═╡ 4c6d3d79-22d1-4aae-b538-87aac73787d9
begin
    # joint_P = [0 0 0 1 2 4 5 3 1; 2 5 6 6 4 2 0 0 0]
    if indep
        # joint_P = [0 0 0 1 2 4 5 3 1; 2 5 6 6 4 2 0 0 0]
        joint_P = [0.1 0.2 0.3 1 2 4 5 3 1; 2.8 * [0.1 0.2 0.3 1 2 4 5 3 1]]
    else
        joint_P = [0 0 0 1 2 4 5 3 1; 2 5 6 6 4 2 0 0 0]
    end
    joint_P = joint_P ./ sum(joint_P)
end;

# ╔═╡ 05cadada-1ace-4c57-8364-5f2cfe4abb42
joint_plt = let
    gr()
    plt = plot(xlim=[1, 10], ylim=[1, 2], yticks=[1, 2], xticks=1:9, zlim=[0, maximum(joint_P)], zaxis=false, framestyle=:origin, camera=(15, 10), size=(800, 600), xlabel=L"y", ylabel=L"x", zlabel="", title=L"P(X,Y)")
    for i in 1:size(joint_P)[1]
        for j in 1:size(joint_P)[2]
            quiver!([j], [i], [0], quiver=([0], [0], [joint_P[i, j]]), lw=5)
        end
    end
    plt
end;

# ╔═╡ b2c691a2-903a-45c4-a174-215d6fd2a220
marg_py = let
    bar(sum(joint_P, dims=1)[:], xticks=1:9, title=L"P(Y)", label="")
end;

# ╔═╡ fc585bb1-73c0-4077-a2fe-ac9c86cc8fca
marg_px = let
    bar(sum(joint_P, dims=2)[:], yticks=1:2, orientation=:h, yflip=true, label="", title=L"P(X)")
end;

# ╔═╡ 544c0d4e-97eb-4d4e-bc40-05a5c1e56bff
md"``Y=``$(@bind yⱼ Slider(1:9; default=4, show_value=true))"

# ╔═╡ 646ae2a5-6e16-4202-b175-b0b1616b2d65
let
    jindx = yⱼ
    gr()

    layout = @layout [a{0.7w,1.0h} b]
    plt = plot(xlim=[1, 10], ylim=[1, 2], yticks=[1, 2], xticks=1:9, zlim=[0, maximum(joint_P)], zaxis=false, framestyle=:origin, camera=(5, 8), xlabel=L"y", ylabel=L"x", zlabel="", title=L"P(X,Y)")
    for i in 1:size(joint_P)[1]
        for j in 1:size(joint_P)[2]
            cl = j == jindx ? :lightsalmon : Symbol(gray)
            lw_ = j == jindx ? 4 : 2
            al_ = j == jindx ? 1 : 0.5
            quiver!([j], [i], [0], quiver=([0], [0], [joint_P[i, j]]), lw=lw_, color=cl, alpha=al_)
        end
    end

    plt2 = bar(joint_P[:, jindx] / sum(joint_P[:, jindx]), xlabel=L"X", title=L"P(X|Y=%$(jindx))", label="", xticks=1:2, r=0.5)
    plot(plt, plt2, layout=layout)
end

# ╔═╡ 776649a0-d8ef-4fcb-a9ea-69e956dbd637
md"""
## Independence: genuine coin toss

> #### Which squence is a genuine **independent** coin tosses?

#### **Sequence 1**

>	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

#### **Sequence 2**

>	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

* #### both of them have ``N_h = 148`` and ``N_t= 152``
"""

# ╔═╡ 0c78067b-e2cc-4866-b12e-abebe09cf564
md"""

## Independence: genuine coin toss (cont.)


#### Recall **independence**'s definition

```math
\large
P(X_{t+1}|X_{t}) = P(X_{t+1})
```

* ##### ``X_{t}``: current toss at ``t``
* ##### ``X_{t+1}``: next toss


#### Therefore, if **independent**, then

```math
\large
P(X_{t+1}=\texttt{h}|X_{t}=\texttt{h}) = P(X_{t+1}=\texttt{h}|X_{t}=\texttt{t}) =P(X_{t+1}=\texttt{h}) = 0.5
```

"""

# ╔═╡ ef748d34-071b-4e59-a96c-2d606d483ee9
md"""

##


#### **Sequence 1**

>	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

##### The joint frequency table is

```math
\large
\begin{equation}  \begin{array}{c|cc} 
n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
\hline
X_t =\texttt h & 46 & 102 \\ 

X_t= \texttt t & 102 & 49 \\ 

\end{array} \end{equation} 

```

* ``P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{46}{46+102} \approx 0.311 \ll 0.5``
* ``P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{102}{102+49} \approx 0.675 \gg 0.5``

"""

# ╔═╡ 3caf5284-3c82-4fc9-a705-1215b9881df9
md"""

##

#### **Sequence 2**

>	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

```math

\begin{equation}  \begin{array}{c|cc} 
n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
\hline
X_t =\texttt h & 71 & 77 \\ 

X_t= \texttt t & 76 & 75 \\ 

\end{array} \end{equation} 

```

* ``\hat P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{71}{71+77} \approx 0.48 \approx 0.5``
* ``\hat P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{76}{76+75} \approx 0.503 \approx 0.5``
"""

# ╔═╡ 8afac55e-44f9-4127-b16a-969560307f90
md"""

## Conditional independence

#### ``X`` is conditionally independent of ``Y`` given ``Z``, if and only if 


$$\Large \forall x,y,z: P(x,y|z) = P(x|z)p(y|z)$$

#### or equivalently (**more useful**), if and only if 

$$\Large\forall x,y,z: P(x|z, y) = P(x|z)$$

* #### intuition: knowing ``z``, ``x`` is no longer influenced by ``y`` (therefore independence): ``p(x|z, \cancel{y})``

#### Denoted as ``X\perp Y\vert Z``





## Conditional and Unconditional independence

#### *Marginal independence* is a specific case of conditional independence (the condition is an empty set)
* ##### *i.e.* ``X\perp Y \vert \emptyset``
* the independence relationship is unconditional



"""

# ╔═╡ 5bcfc1a8-99d4-4b48-b45c-4c8d0c5acae4
md"""

## Example

#### Consider three r.v.s: **Traffic, Umbrella, Raining**


#### Based on *chain rule* (or product rule)

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Traffic}, \text{Rain})$$


"""

# ╔═╡ 6dc34d66-366f-4e1a-9b73-a74c07e58feb
md"""

## Example

#### Consider 3 r.v.s: **Traffic, Umbrella, Raining**


#### Based on *chain rule* (or product rule)

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Traffic}, \text{Rain})$$


#### The last can be simplified based on **conditional independence**

$$\large P(\text{Umbrella}|\cancel{\text{Traffic}}, {\text{Rain}}) = P(\text{Umbrella}|\text{Rain})$$

* ##### knowing it is raining, Umbrella no longer depends on Traffic

* or in CI notation: ``T \perp U|R``



"""

# ╔═╡ 12751639-7cdc-4e39-ba72-30ee2fe2cd49
# md"""

# ## Example

# Consider the coin switching problem


# Based on *chain rule* (or product rule)

# $$P(\text{S},\mathcal{D}) = P(S)  P(\mathcal{D}|S)$$

# * ``\mathcal{D} = [d_1, d_2, \ldots, d_n]``
# Then applies **conditional independence assumption**

# $$P(\mathcal{D}|S) = P(d_1|S)P(d_2|S) \ldots P(d_n|S)$$

# * conditioned on the switching time ``S``, the tosses are independent 


# Conditional independence greatly simplies probability models
# * we only need to spell out ``P(S)`` and ``P(d_i|S)`` rather than the conditional joint ``P(d_1, d_2, \ldots, d_n|S)``
# * also note that ``P(d_i|S)`` are the same for all ``i``

# """

# ╔═╡ 7ac35147-4c3d-421e-ba42-6e6e1325584f
md"""

## More example -- coin guess
"""

# ╔═╡ c31706a2-cb86-4fe2-9d64-a1d09490792c
md"""


> ##### Two coins (with biases ``0.5`` and ``0.99`` respectively) in an urn
> * ######  randomly pick one and toss it 3 times 



"""

# ╔═╡ 393e9b36-d85e-440b-a113-2d748c3ac03b
md"""

The tossing results $\{Y_1, Y_2, Y_3\}$ are conditionally independent given coin choice $C$


```math
\Large
P(Y_1, Y_2, Y_3|C) = P(Y_1|C)P(Y_2|C)P(Y_3|C)
```
"""

# ╔═╡ 532a9f66-b04a-446a-becf-d74759bcf750
md"""
## Probabilistic graphical models

#### What is Probabilistic graphical models (PGMs)
> ##### Graphical representation of a probabilistic model

* or joint distribution

###### For example, the cough example, the PGM represents

```math
	P(H, \texttt{Cough}) = P(H)P(\texttt{Cough}|H)
```

"""

# ╔═╡ 22ad1522-9c15-4f8a-ad05-715f7ce59c36
@htl("<center><img src=$(figure_url * "cough_bn.png") height='200' /></center>")

# ╔═╡ 9da0bf21-3d2f-4df0-a8b3-3891dd665c7f
md"""

## Digress: directed graphs

#### A **directed** graph is a a graph with **directed** edges
  * direction matters: $(X, Y)$ are not the same as $(Y, X)$
  * good for **asymmetric relationship**: 
    * e.g. parent-to-child relationship (the reverse is not true)
    

#### ``\textbf{parent}(\cdot)`` returns the set of parent nodes, e.g. 

$$\Large\text{parent}(Y) = \{X, Z\}$$
"""

# ╔═╡ a6032050-dc11-4774-a29f-da36944e8bfc
@htl("<center><img src=$(figure_url * "figs4CS5010/directedGraph.png") width='600' /></center>")

# ╔═╡ 41255527-96d2-41bf-92b4-8b4a9bebab24
md"""
## Digress: directed acyclic graph (DAG)

#### A **d**irected **a**cyclic **g**raph (**DAG**) is a directed graph **without** cycles 
  * ##### a cycle: directed path starts and ends at the same node
"""

# ╔═╡ 00a0c582-34b2-413d-bef2-7a7572f48340
md"""

#### *For example*, this is **NOT** a DAG
* ##### cycles are NOT allowed: $X\Rightarrow Y_1\Rightarrow X$

"""

# ╔═╡ 25810ebe-3f07-434c-9d14-22c906285e90
@htl("<center><img src=$(figure_url * "figs4CS5010/diceExampleDCG.png") width='500' /></center>")

# ╔═╡ a578a1a7-0f42-492c-8226-b8e09e92506f
md" 
#### However, multiple paths are allowed, *e.g.*
  * ##### two possible paths from $X$ to $Y_2$: $X\Rightarrow Y_2$ and $X \Rightarrow Y_1 \Rightarrow Y_2$
  * ##### still **acyclic**

"

# ╔═╡ 6567599f-6222-4754-b7c4-971dd1acddb3
@htl("<center><img src=$(figure_url * "figs4CS5010/diceExampleDAG.png") width='500' /></center>")

# ╔═╡ 20c95319-86a7-477d-93c8-adf0dbdbb4c2
md"""

"""

# ╔═╡ 275c155d-c4c5-472a-8613-c549706815b0
md"""

## Node types

"""

# ╔═╡ 2f5af892-249f-40c8-a2a2-0bc7acc7f000
md"""

* ##### **unshaded**: unobserved **random variable** (the hidden cause)
* ##### **shaded**: observed **random variable** (coughing)
* ##### **dot**: fixed parameters/input, non-random variables
  * *e.g.* the input predictors ``\mathbf{x}`` for linear regression
  * or fixed hyperparameters 
"""

# ╔═╡ cd899b61-88cf-4128-b4ba-ae8a7e18dff1
@htl("<center><img src=$(figure_url * "notations.png") width='200' /></center>")

# ╔═╡ 41ee5127-4390-4fb8-b41f-9856346877cf
md"""

## Probabilistic graphical model 
##### a DAG with local distributions 



> #### **(DAG)** + **local prob. distributions** $P(X_i|\text{parents}(X_i))$
  
* ##### **DAG**: each node is a random variable


* ##### **Local distributions**: $P(X_i|\text{parent}(X_i))$, one $P$ for each r.v. $X_i$


"""

# ╔═╡ 6699af84-5adb-44ce-89c3-03cde2fac07b
TwoColumn(md"""
\
\

#### For example, cough example

* ``H`` is with ``P(H)``
* ``\texttt{Cough}``: ``P(\texttt{Cough}|H=h)``
""",
    @htl("<center><img src=$(figure_url * "cough_bn.png") width='100' /></center>")
)

# ╔═╡ be4b66ed-dc94-4450-97dd-a57175913b20
md"""
## Probabilistic graphical model semantics


!!! important "Factoring property & chain rule"
	#### Joint distribution factorises as the product of CPTs: 
	
	$\Large P(X_1, X_2,\ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))$

"""

# ╔═╡ c7217173-3b9e-439a-ad59-6fdea81bb12c
md"""

## Example - Traffic, Rain, Umbrella



#### The full joint 

$$\large P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Rain})$$

#### The corresponding graphical model is

* a graphical representation of the joint distribution
"""

# ╔═╡ 3f8d5b4d-a009-4879-a132-b24f95ea8162
@htl("<center><img src=$(figure_url * "bayes/rtu_bn.png") width='300' /></center>")

# ╔═╡ 33a8d7e1-cb9b-48c8-9b63-7a817c0497b2
md"""

## Example -- coin guess

"""

# ╔═╡ 01794f0a-0c40-43fc-9621-32b46099d6cf
md"""


> ##### Two coins (with biases ``0.5`` and ``0.99`` respectively) in an urn
> * ######  randomly pick one and toss it 3 times 



"""

# ╔═╡ 15f2ebfd-cced-48a3-a13c-5ed166111022
TwoColumn(
    @htl("<center><img src=$(figure_url * "coin_choice.png") width='400' /></center>"),
    md"""

Coin Choice: $P(C)$

|``C``   | ``P(C=c)`` |
| --- | ---  | 
| 1   | 0.5 | 
| 2   | 0.5 | 

Observation ``Y_i``: $P(Y_i|C)$

|``C``   | ``\texttt{head}`` | ``\texttt{tail}`` |
| ---| ---   | ---     |
| 1  | 0.5 | 0.5 |
| 2  | 0.01 | 0.99 |

* each row is a valid conditional distributions 


"""
)

# ╔═╡ b96cc13e-5fd9-4fa1-a9bc-4ea7b22c07a3
md"""

## Plate notation for repeated nodes

#### Note that $P(Y_i|C)$ for $i =1,2,3$ are repeated


"""

# ╔═╡ f1a48d13-bf96-4ef5-835f-acfb1b7791c3
@htl("<center><img src=$(figure_url * "bayes/plate_coin.png") width='300' /></center>")

# ╔═╡ 17c47cc4-63ab-40ec-86af-b17d0b2d6b5f
md"""

## Some graphical model patterns

#### Three patterns in graphical models

* #####  common cause pattern
* ##### chain pattern
* ##### collider pattern

> ##### All probabilistic graphical models are **composition** of the three patterns



## Common cause pattern (*a.k.a.* tail to tail)

"""

# ╔═╡ fb5bd9c6-6e5c-49d9-9118-9c6de78fa6f7
TwoColumn(md"""


#### ``Y`` is the common cause of ``X`` and ``Z``


* also known as tail to tail (from ``Y``'s perspective)
* ##### *e.g.* the unknown bias is the common cause of the two tossing realisations

""", @htl("<center><img src=$(figure_url * "figs4CS5010/dag_common.png") width='300' /></center>"))

# ╔═╡ 2bb41562-6011-46a8-a753-b91315911577
md"""
## Common cause pattern (*a.k.a.* tail to tail)
"""

# ╔═╡ 6f4acf7a-4978-465f-a50f-25476cacd54b
TwoColumn(
    md"""

    #### It can be shown that 

    ```math
    \large X \perp Z |Y\;\; \text{or}\;\;P(X,Z|Y) =P(X|Y)P(Z|Y)
    ```


    #### But *marginally* not independent 

    ```math
    \large X \not\perp Z|\emptyset\;\; \text{or}\;\;P(X,Z) \neq P(X)P(Z)
    ```

    """,
    @htl("<center><img src=$(figure_url * "figs4CS5010/dag_common.png") width='500' /></center>")
)

# ╔═╡ 49a22f7e-b7a2-45dd-a44b-c176da48fd28
TwoColumn(
    md"""

    ##### *Example*: "Vaccine" effectiveness 
    * common cause: whether a vaccine is effective (**Y**)
    * ``X, Z`` nodes: two patients' treatment outcomes after receiving the vaccine (**X**, **Z**)
    """,
    @htl("<center><img src=$(figure_url * "CS5914/vaccine.svg") width='250' /></center>")
)

# ╔═╡ 7f201df4-9ddb-4565-8b22-aa34ef8a9950
md"""
## Chain pattern (*a.k.a.* tip to tail)

> #### ``X`` impacts ``Z`` _via_ ``Y``

"""

# ╔═╡ d621a315-a69a-44c7-974c-f5a1d14d0c97

@htl("<center><img src=$(figure_url * "figs4CS5010/dag_seq.png") width='400' /></center>")

# ╔═╡ 108d0def-f048-4010-aba9-3f4385875540
md"""

## Chain pattern (*a.k.a.* tip to tail)
"""

# ╔═╡ e7f3feae-9c29-4e9f-81b9-28895f562d59
TwoColumn(
    md"""

    #### It can be shown 


    ```math
    \large X \perp Z|Y \;\; \text{or}\;\;P(X,Z|Y) = P(X|Y)P(Z|Y)

    ```

    #### but marginally **not** independent 

    ```math
    X \not\perp Z|\emptyset \;\; \text{or}\;\;P(X,Z) \neq P(X)P(Z)

    ```

    """,
    @htl("<center><img src=$(figure_url * "figs4CS5010/dag_seq.png") width='500' /></center>")
)

# ╔═╡ e5f08691-e099-4cee-9bbb-b893b512f4bc
md"""

#### *Example*: `Rain`, `Floor_is_Wet`, `Slip`


```math
\large \begin{align}
{\text{Rain}} \rightarrow {\text{Floor is Wet}}  \rightarrow {\text{Slip}} 
\end{align}
```

* ##### add *Injured* ?
"""

# ╔═╡ c5b2a9f7-9261-444c-812a-64efa3c500ef
md"""

## A classic example -- chain case

##### *Voting for Trump* does not *directly* make you more likely to *die*!
* but there is a hidden intermediate cause, *i.e.* vaccination rate
"""

# ╔═╡ a3fe5cfe-6f41-432f-bb60-f08446231fbf
md"""

#### A suitable model

```math
\large \text{Vote For Trump} \rightarrow \text{Vaccinated} \rightarrow \text{Die Out of COVID}

```

"""

# ╔═╡ 95319916-264a-4893-854a-4d02543fc6c1
TwoColumn(html"""<center><img src="https://static01.nyt.com/images/2021/09/27/multimedia/27-MORNING-sub3-STATE-DEATH-VOTE-PLOT/27-MORNING-sub3-STATE-DEATH-VOTE-PLOT-superJumbo.png?quality=75&auto=webp" width = "320"/></center>
""", html"""<center><img src="https://static01.nyt.com/images/2021/09/27/multimedia/27-MORNING-sub2-STATE-VAX-VOTE-PLOT/27-MORNING-sub2-STATE-VAX-VOTE-PLOT-superJumbo-v2.png?quality=75&auto=webp" width = "320"/></center>
""")

# ╔═╡ 4f90a063-7aaf-4287-8616-301a7636665d
md"""
## Common effect (*a.k.a.*  collider, or tip to tip)

"""

# ╔═╡ 7f30f317-82b9-45f2-a6a7-6bcd785b825b
TwoColumn(md"""



\
\

#### ``Y`` is the joint **effect** of
* ##### combination of *independent* causes ``X`` and ``Z``

""",
    @htl("<center><img src=$(figure_url * "figs4CS5010/dag_collider.png") width='500' /></center>")
)

# ╔═╡ 67117028-356f-4ccb-86cd-25b791e76722
md"""
## Common effect (*a.k.a.*  collider, or tip to tip)

"""

# ╔═╡ b67b29a9-aa19-4bc2-ae60-26ca2453a905
TwoColumn(md"""

#### *Quite the opposite* to the previous two cases, 
* ``X, Z`` are marginally independent 
* but not conditionally independent (explain away)


```math
X\perp Z|\emptyset\;\; \text{or} \;\; P(X, Z)= P(X)P(Z)

```

```math
X\not \perp Z|Y\;\; \text{or} \;\; P(X, Z|Y)\neq P(X|Y)P(Z|Y)
```
```math
\text{or } P(X|Z, Y)\neq P(X|Y) 
```

""",
    html"""<center><img src="https://sites.cs.st-andrews.ac.uk/people/lf28/figs/figs4CS5010/dag_collider.png" width = "500"/></center>""")

# ╔═╡ 82bd4b52-34a6-44ea-8a18-2e4186cf2210
TwoColumn(md"""

#### *Example*: Exam performance
* ``X``: your knowledge of the subject
* ``Z``: exam's difficulty 
* ``Y``: exam grade 

##### *Marginally,* 
##### `Knowledge` ``\perp`` `Exam_Difficulty` 



""", html"""<center><img src="https://sites.cs.st-andrews.ac.uk/people/lf28/figs/CS5914/examdiff.svg" width = "300"/></center>""")

# ╔═╡ dd5a7d7b-e59a-4899-8517-d8168a4639ca
md"""
##### *Conditional* on the effect `Grade`, they become dependent 

```math
\begin{align}
P(\texttt{Knowledge}=good|\texttt{Grade}=fail,\; \texttt{Exam\_diff}=easy)\; {\Large\neq}\; \\ P(\texttt{Knowledge}=good|\texttt{Grade}=fail, \texttt{Exam\_diff}=hard)
\end{align}
```


* exam is easy ``\Rightarrow`` `knowledge` is bad
* exam is difficult ``\Rightarrow`` `knowledge` is probably not that bad
"""

# ╔═╡ 608eac2e-544b-44e0-8162-c777123d3e04
md"""

## More collider examples



##### Colliders are very common and they help explain a lot 
\



!!! example "Example (location vs food)"
	Why food served at restaurants at **touristy locations** are usually **bad**?



"""

# ╔═╡ c8c61f65-49f3-489d-aff8-b58f498335ca
Foldable(
    "",
    md"""

Restaurants' **Success** depends on *Good Food* and/or *Good Location*

```math
\large
\text{Good Location}\, \longrightarrow \text{Restaurant's Survival} \longleftarrow \text{Good Food}
```


* both good food and good location are desired but relatively rare; even rarer to have both

* one **explains away** the other
"""
)

# ╔═╡ 0b374958-5ee4-4c34-b653-28a115f07e6a
TwoColumn(md"""

\

!!! example "Example (skills vs attractiveness)"
	Why **attractive** actors/actress are usually **not very skilled** ?

	or the opposite: **skilled** actors/actress usually not necessarily **attractive**
""", html"""<center><img src="https://64.media.tumblr.com/3cd8403e2a6de029e2fdc36aafb1abb5/d40ea27e49d8dcd9-ee/s400x600/a6131fa863fa20887f4c1c2282737020ab2d64d7.gif" width = "300"/></center>""")

# ╔═╡ 3b488fba-d5df-4c8e-9d68-03c958a8244c
Foldable(
    "",
    md"""

An actor's **Success** depends on *Good Looking* and/or *Good Skills*

```math
\large
\text{Good Looking}\, \longrightarrow \text{Actor's Success} \longleftarrow \text{Good Skills}
```


* both attractiveness and good skills are desired but relatively rare; even rarer to have both

* one **explains away** the other
"""
)

# ╔═╡ 61e3f627-5752-48e3-9100-3c96575f1805
md"""

## Appendix
"""

# ╔═╡ c0e7844b-111f-469a-9d6e-33751a441adb
begin
    function true_f(x)
        -5 * tanh(0.5*x) * (1 - tanh(0.5*x)^2)
    end
    Random.seed!(100)
    x_input = [range(-6, -1.5, 10); range(1.5, 6, 8)]
    y_output = true_f.(x_input) .+ sqrt(0.05) * randn(length(x_input))
end;

# ╔═╡ 8f2452a3-79cc-4f22-90dc-82f88adc46be
begin
    scatter(x_input, y_output, label="Observations", xlim=[-10, 10], ylim=[-3, 3], xlabel=L"X", ylabel=L"Y")
    # plot!(true_f, lw=2, xlim=[-10, 10], framestyle=:default,  lc=:gray, ls=:dash,label="true function", xlabel=L"x", ylabel=L"y")
end

# ╔═╡ 945acfa0-9fde-4635-b87e-7f7f026da8bb
plt1 = let
    scatter(x_input, y_output, label="", xlim=[-13, 13], ylim=[-5, 5], xlabel=L"x", ylabel=L"y")
    plot!(true_f, lw=2, framestyle=:default, lc=:red, ls=:solid, label=L"f(x)", xlabel=L"x", ylabel=L"y")
end;

# ╔═╡ 3da83a5f-be41-47a1-bad1-781e59658f95
begin
    function poly_expand(x; order=2) # expand the design matrix to the pth order
        n = length(x)
        return hcat([x .^ p for p in 0:order]...)
    end
end

# ╔═╡ 0cae3fc5-19a3-4143-aebc-e9484f6253ba
begin
    Random.seed!(123)
    w = randn(length(x_input)) ./ 2
    b = range(-18, 22, length(x_input)) |> collect
    Φ = tanh.(x_input .- w' .* b')
end;

# ╔═╡ 18c3484f-c219-4a98-95c7-81c87f5c5ff0
begin
    poly_order = 6
    # Φ = poly_expand(x_input; order = poly_order)
    freq_ols_model = lm(Φ, y_output)
    # apply the same expansion on the testing dataset
    x_test = -10:0.1:10
    Φₜₑₛₜ = tanh.(x_test .- w' .* b')
    tₜₑₛₜ = true_f.(x_test)
    # predict on the test dataset
    βₘₗ = coef(freq_ols_model)
    pred_y_ols = Φₜₑₛₜ * βₘₗ
end;

# ╔═╡ 43f36576-b470-4fcf-935a-dca80c2f80f8
plt2 = let
    # plot(x_test, tₜₑₛₜ, linecolor=:black, ylim= [-3, 3], lw=2, linestyle=:dash, lc=:gray,framestyle=:default, label="true signal")
    scatter(x_input, y_output, label="", xlim=[-13, 13], ylim=[-5, 5], xlabel=L"x", ylabel=L"y")
    plot!(x_test, pred_y_ols, linestyle=:solid, lc=:blue, lw=2, xlabel=L"x", ylabel=L"y", legend=:topright, label=L"f(x)")
end;

# ╔═╡ 94ef0651-e697-4806-868a-bce137d8437e
plt=plot(plt1, plt2, layout=(1, 2), size=(600, 300));

# ╔═╡ 8eec2a38-c5fb-48c7-8801-81d6d4afb796
md"""
##


!!! note "Question"
	#### Given the data, which model below is more reasonable?
	
	* ##### the red one: $\textcolor{red}{f(x)}$
	* ##### the blue one: $\textcolor{blue}{f(x)}$

$(begin 
	plt
end)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.132"
GLM = "~1.9.5"
HypertextLiteral = "~1.0.0"
LaTeXStrings = "~1.4.1"
Latexify = "~0.16.12"
LogExpFunctions = "~1.0.3"
Plots = "~1.41.7"
PlutoTeachingTools = "~0.5.0"
PlutoUI = "~0.7.83"
StatsBase = "~0.34.13"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "8b5ac298dea50386ca2a190539af722fce7e08d8"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "dee8388a9bec89c8713ad176f96960ae2f98598a"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.4"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
registries = "General"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "7f54761502ff149a9d492e4acefe9805898e29b3"
registries = "General"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.2+0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7b841680738c19948120f6e4cf8d0200518bd564"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "8ae5ee44c8c4cbd943f6b4f21747df2bb3041513"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.2"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
registries = "General"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a7cf827a79b378a381e31e5c7feca853ec19cb94"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.4"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "a633156e74096f90b3b5e833529292b651b5671a"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.3"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
registries = "General"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "2ddbc341251ed713d39e503b504593a5d6789551"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.132"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "7bd13840b4148949e290748071bbe6826b3eabcd"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "417289ac840d201005d9ed16789a46defccbe36d"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.1"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "LogExpFunctions", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsModels"]
git-tree-sha1 = "c963639ae5b9aab54f543bdc7504f42f59880bec"
registries = "General"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
registries = "General"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "9939eb6f799a963b3392ac4a1d49d3fc0646eb80"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.11.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "8bcdc90d5c361a73635ad1c19d9c39a52bf3e0a3"
registries = "General"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.5"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NearestNeighbors]]
deps = ["AbstractTrees", "Distances", "StaticArrays"]
git-tree-sha1 = "576eb4656529c12e77a46b17c23103dfba9fa570"
registries = "General"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.29"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
registries = "General"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "5d103af2482cccf2d9737c365a2d53af0597df95"
registries = "General"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.5.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
registries = "General"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
registries = "General"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "e6ffa7bd4f8d0b8744d18f219304738d952f9a37"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.11"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
registries = "General"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "de349d5585559b692f518b6d3fc7c179d64b2364"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.23"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "0db41c4e0d9f3fa195395a6401a8290752c9cd3d"
registries = "General"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.10"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
registries = "General"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ad435e30854ed0f50f4b7f67edd71664afecc5d8"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.10.0"

    [deps.StructUtils.extensions]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
registries = "General"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "0392e5e3c5327b90a487b0ff3e818c4aecd3660b"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "cab29ed70d9355a28b95b1798007ff3a00eefc91"
registries = "General"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.8"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
registries = "General"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "9a5e6e8f1a7c40941465c04840145e527f0d6b4b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.5.0+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "df77b22eb77051b3d0ef30aab979684f7482c66f"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.48.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.2+0"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1210ba774d3427387d307bf1f416d699b7c39417"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─e56e1a50-2663-4220-8bf0-8df93633d31a
# ╟─f7ad60c5-aaa1-46b0-9847-5e693503c5ca
# ╟─6b33f22b-02c4-485a-a8e0-8329e31f49ea
# ╟─263003d7-8f36-467c-b19d-1cd92235107d
# ╟─e03e585c-23ab-4f1b-855f-4dcebee6e63c
# ╟─fdc69b6d-247c-49f4-8642-0345281c2370
# ╟─45b8e008-cfdc-11ed-24de-49f79ef9c19b
# ╟─88621eb4-0d63-4cb8-808c-78494cdfb0ad
# ╟─473eadf8-e1f7-4bdc-b937-da69aac35be5
# ╟─1371e5cd-cfd9-40ee-ad96-b97abdd30525
# ╟─fd787b38-6945-49cf-becf-1d37847a88fe
# ╟─0e80b64a-22ad-41d2-b6be-5d9b88af06c9
# ╟─692cb743-d345-4b1b-b71f-3e26d1c7c80a
# ╟─8bf95702-90ac-4b3c-a027-e1b673359a60
# ╟─3a80d31d-9f32-48b1-b5fd-788b588d96cb
# ╟─dede5ea1-c67d-4344-b6d0-c4a9b8d8dbeb
# ╟─e7242619-19fc-461b-a5e0-fc50361d2565
# ╟─63b92973-1031-439c-aa68-60f00552ab18
# ╟─8f2452a3-79cc-4f22-90dc-82f88adc46be
# ╟─8eec2a38-c5fb-48c7-8801-81d6d4afb796
# ╟─926be45c-84a4-4bb5-8cc0-56bdf74bc581
# ╟─9066781d-b542-4c95-8a5b-d244df217650
# ╟─c9b2fa82-841d-47cf-b7b4-23f8a86e60a7
# ╟─2bc778d0-6974-4a97-8769-3cb66100fd80
# ╟─2fba13a9-a0ab-44a6-ba4d-b828ebe96274
# ╟─db1f6632-f894-4ce7-a5b4-493158a13860
# ╟─f05dcb39-84c9-42e5-95f6-67238c574609
# ╟─0cc7386d-4584-4a34-bbf7-546bf9ac1134
# ╟─b52911ff-886a-4113-85a5-af508dc0cb8e
# ╟─34a4fdfa-df3f-426c-acbd-d18056246714
# ╟─910399dd-ebe1-447f-967f-7a38e4913b56
# ╟─e11b9b91-a912-45a1-913c-aeb3e78e60fc
# ╟─10beeb49-277a-489a-969d-140d51fcfde3
# ╟─17f73ee7-e842-4f10-9118-47399e67f047
# ╟─64b0c1fb-4758-4536-913b-3816ca431608
# ╟─b1ff85ec-6a23-4276-a7e8-6303e294f301
# ╟─6a3217fb-47c3-400c-9fbd-5c524a1e54fb
# ╟─5fd8cfe7-258d-4c8d-a482-c439d7755a72
# ╟─5a73ebc0-2bc1-425a-b3e6-18afd2247cb6
# ╟─b680c9da-2cb5-432f-8032-e2a37fc3e738
# ╟─1e4db58a-e57d-457e-9bf4-20961a4775b1
# ╟─42466480-2d4d-4a62-be06-ede102d818f0
# ╟─d1e63621-5e4a-4f43-aa4f-a41f12fe632a
# ╟─7750287e-96e3-4920-9e3e-0dd461e2f2ea
# ╟─7416236c-5ff0-4eb8-b448-e50acb0c016b
# ╟─c2f8d8e7-16d5-46c5-a66d-664dbaeb4c91
# ╟─be70c96c-685d-4136-8828-c8a35a5cbe0b
# ╟─48484810-1d47-473d-beb3-972ca8b934d5
# ╟─559b0936-a051-4f2e-b0ca-81becd631bce
# ╟─1a930c29-0b6e-405b-9ada-08468c69173c
# ╟─ddc281f3-6a90-476a-a9b9-34d159cca80a
# ╟─05e3c6d0-f59b-4e1e-b631-11eb8f8b0945
# ╟─32bd86a4-0396-47f5-9025-6df83268850e
# ╟─e12f0e44-2b7d-4075-9986-6322fb1498c4
# ╟─e4044dd5-f1a3-4ec6-b0e6-2e4a3cfaf0ce
# ╟─f60b1407-df78-41b5-959c-742910364fa6
# ╟─b8677457-b952-4dad-9f11-ae197c126287
# ╟─01231683-61fc-4f36-b004-c005723c53e0
# ╟─dabd0bef-40dc-49d8-bfef-0ecfd2efcae3
# ╟─cf969c4c-909d-4dc7-ad95-74352f987c1f
# ╟─571a31d9-77fe-4457-827a-6a746a529a4a
# ╟─a602e848-8438-4a41-ab76-ec3f6497de3e
# ╟─6c0333fd-5b6f-4a5c-b809-781614a7fe63
# ╟─fea3395d-250d-406c-9a8c-646f5c48ab12
# ╟─602bfca9-b90e-48cc-949f-2909aacf3546
# ╟─dd011a05-7cea-4793-b5f0-770e20f3a503
# ╟─b7973f17-7f7f-4cb2-bfd2-9defc4b5910b
# ╟─52df31d8-f439-4b44-a5f3-0b2c27db7522
# ╟─265bdf7b-2d31-4ff0-8af7-a5e9c5e881bd
# ╟─b5ed7206-7d64-4845-a56d-2bbe35da9a38
# ╟─2ca8a673-9eeb-4295-91ed-27c173a91db7
# ╟─9c2086db-b108-4b15-87c8-8f38e53767fb
# ╟─c23e3616-82fc-4316-a561-93decb98cc9c
# ╟─302a0551-dea3-49e9-bcca-409126571105
# ╟─65d9e2ed-1702-43aa-aafd-e2b38d0b403a
# ╟─5920cd07-30b9-44a7-8f5c-999c1f8ef043
# ╟─5acefb1f-209e-489f-afb9-dce517ad6225
# ╟─866cd5fd-9d9d-42da-b05e-b4b3ad958ecb
# ╟─904fc5dc-17b1-4500-9cb8-d7e472c90212
# ╟─6593bd28-f483-4168-ac69-eae469fbcb27
# ╟─9e0953ee-3451-418a-afe2-b2177a2a8898
# ╟─9b925cf9-fb2f-403e-8208-6f1358a598d9
# ╟─72fb8fa0-8263-4729-af19-ff6047865a61
# ╟─baf8e5b2-4473-4e22-bb4d-2c1fd8cccfaa
# ╟─691c897a-a282-4023-8b7a-d7df665bf53c
# ╟─4d398cf7-c3dd-49cf-b0c3-3731e50c1c37
# ╠═2ca2d909-7499-45fc-864e-cf22486896dc
# ╟─9d9e9537-8194-4716-8f1f-fd1013c7e040
# ╟─facc40e6-87cd-4884-b20e-28d84e4337c6
# ╟─7652aa5c-47c7-4263-a673-00f073f91018
# ╟─0198a401-ed47-4f5c-8aca-441cccff472b
# ╟─6db0a9fa-0137-413e-b122-24a51c4319e0
# ╟─05cadada-1ace-4c57-8364-5f2cfe4abb42
# ╟─b2c691a2-903a-45c4-a174-215d6fd2a220
# ╟─fc585bb1-73c0-4077-a2fe-ac9c86cc8fca
# ╟─4c6d3d79-22d1-4aae-b538-87aac73787d9
# ╟─5c0a6bf7-7ada-4122-ae09-1963969e4aa9
# ╟─2ed5995f-26dd-4f3e-92fb-a04b334cfb99
# ╟─61106a36-e880-499b-a442-f42d3c196e6b
# ╟─544c0d4e-97eb-4d4e-bc40-05a5c1e56bff
# ╟─646ae2a5-6e16-4202-b175-b0b1616b2d65
# ╟─776649a0-d8ef-4fcb-a9ea-69e956dbd637
# ╟─0c78067b-e2cc-4866-b12e-abebe09cf564
# ╟─ef748d34-071b-4e59-a96c-2d606d483ee9
# ╟─3caf5284-3c82-4fc9-a705-1215b9881df9
# ╟─8afac55e-44f9-4127-b16a-969560307f90
# ╟─5bcfc1a8-99d4-4b48-b45c-4c8d0c5acae4
# ╟─6dc34d66-366f-4e1a-9b73-a74c07e58feb
# ╟─12751639-7cdc-4e39-ba72-30ee2fe2cd49
# ╟─7ac35147-4c3d-421e-ba42-6e6e1325584f
# ╟─c31706a2-cb86-4fe2-9d64-a1d09490792c
# ╟─393e9b36-d85e-440b-a113-2d748c3ac03b
# ╟─532a9f66-b04a-446a-becf-d74759bcf750
# ╟─22ad1522-9c15-4f8a-ad05-715f7ce59c36
# ╟─9da0bf21-3d2f-4df0-a8b3-3891dd665c7f
# ╟─a6032050-dc11-4774-a29f-da36944e8bfc
# ╟─41255527-96d2-41bf-92b4-8b4a9bebab24
# ╟─00a0c582-34b2-413d-bef2-7a7572f48340
# ╟─25810ebe-3f07-434c-9d14-22c906285e90
# ╟─a578a1a7-0f42-492c-8226-b8e09e92506f
# ╟─6567599f-6222-4754-b7c4-971dd1acddb3
# ╟─20c95319-86a7-477d-93c8-adf0dbdbb4c2
# ╟─275c155d-c4c5-472a-8613-c549706815b0
# ╟─2f5af892-249f-40c8-a2a2-0bc7acc7f000
# ╟─cd899b61-88cf-4128-b4ba-ae8a7e18dff1
# ╟─41ee5127-4390-4fb8-b41f-9856346877cf
# ╟─6699af84-5adb-44ce-89c3-03cde2fac07b
# ╟─be4b66ed-dc94-4450-97dd-a57175913b20
# ╟─c7217173-3b9e-439a-ad59-6fdea81bb12c
# ╟─3f8d5b4d-a009-4879-a132-b24f95ea8162
# ╟─33a8d7e1-cb9b-48c8-9b63-7a817c0497b2
# ╟─01794f0a-0c40-43fc-9621-32b46099d6cf
# ╟─15f2ebfd-cced-48a3-a13c-5ed166111022
# ╟─b96cc13e-5fd9-4fa1-a9bc-4ea7b22c07a3
# ╟─f1a48d13-bf96-4ef5-835f-acfb1b7791c3
# ╟─17c47cc4-63ab-40ec-86af-b17d0b2d6b5f
# ╟─fb5bd9c6-6e5c-49d9-9118-9c6de78fa6f7
# ╟─2bb41562-6011-46a8-a753-b91315911577
# ╟─6f4acf7a-4978-465f-a50f-25476cacd54b
# ╟─49a22f7e-b7a2-45dd-a44b-c176da48fd28
# ╟─7f201df4-9ddb-4565-8b22-aa34ef8a9950
# ╟─d621a315-a69a-44c7-974c-f5a1d14d0c97
# ╟─108d0def-f048-4010-aba9-3f4385875540
# ╟─e7f3feae-9c29-4e9f-81b9-28895f562d59
# ╟─e5f08691-e099-4cee-9bbb-b893b512f4bc
# ╟─c5b2a9f7-9261-444c-812a-64efa3c500ef
# ╟─a3fe5cfe-6f41-432f-bb60-f08446231fbf
# ╟─95319916-264a-4893-854a-4d02543fc6c1
# ╟─4f90a063-7aaf-4287-8616-301a7636665d
# ╟─7f30f317-82b9-45f2-a6a7-6bcd785b825b
# ╟─67117028-356f-4ccb-86cd-25b791e76722
# ╟─b67b29a9-aa19-4bc2-ae60-26ca2453a905
# ╟─82bd4b52-34a6-44ea-8a18-2e4186cf2210
# ╟─dd5a7d7b-e59a-4899-8517-d8168a4639ca
# ╟─608eac2e-544b-44e0-8162-c777123d3e04
# ╟─c8c61f65-49f3-489d-aff8-b58f498335ca
# ╟─0b374958-5ee4-4c34-b653-28a115f07e6a
# ╟─3b488fba-d5df-4c8e-9d68-03c958a8244c
# ╟─61e3f627-5752-48e3-9100-3c96575f1805
# ╠═c0e7844b-111f-469a-9d6e-33751a441adb
# ╠═94ef0651-e697-4806-868a-bce137d8437e
# ╠═945acfa0-9fde-4635-b87e-7f7f026da8bb
# ╠═43f36576-b470-4fcf-935a-dca80c2f80f8
# ╠═3da83a5f-be41-47a1-bad1-781e59658f95
# ╠═0cae3fc5-19a3-4143-aebc-e9484f6253ba
# ╠═18c3484f-c219-4a98-95c7-81c87f5c5ff0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
