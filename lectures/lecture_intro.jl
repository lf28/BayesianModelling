### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f7ad60c5-aaa1-46b0-9847-5e693503c5ca
begin
	using PlutoTeachingTools
	using PlutoUI
	using Plots
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

# ╔═╡ 6b33f22b-02c4-485a-a8e0-8329e31f49ea
TableOfContents()

# ╔═╡ 263003d7-8f36-467c-b19d-1cd92235107d
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ e03e585c-23ab-4f1b-855f-4dcebee6e63c
bayes_figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/bayes/";

# ╔═╡ fdc69b6d-247c-49f4-8642-0345281c2370
present_button()

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


* **name**: *Lei Fang*
  * Lecturer with the School of Computer Science, University of St Andrews
  * Research interests: Bayesian machine learning

* **Email**: lf28 @ st-andrews.ac.uk
"""

# ╔═╡ 473eadf8-e1f7-4bdc-b937-da69aac35be5
md"""
## Target audience


**Target audience**

Researchers who are interested in learning Bayesian inference and applying statistical inference in their work


**Pre-requisite**

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



* Lecture notes/slides available from Moodle


* All source code available in `Github` [https://github.com/lf28/BayesianModelling](https://github.com/lf28/BayesianModelling)



"""

# ╔═╡ 0e80b64a-22ad-41d2-b6be-5d9b88af06c9
md"""

## Agenda


#### First day:

* **Lecture 1** Introduction and overview
  * probability theory review
  * Bayesian inference workflow
  * Graphical models


* **Lecture 2** Bayesian inference with i.i.d. model 
  * Priors; conjugate prior
  * Bernoulli and Gaussian examples
  * Model checking: predictive analysis
  * Beyond i.i.d. model

* **Lecture 3** MCMC 
  * Metropolis-Hastings (MH) and Gibbs sampler
  * Hamiltonian and NUTS sampler

* **Lecture 4** Practical Bayesian inference with Turing.jl
  * Quick introduction to Julia and *Turing.jl*

#### Second day:

* **Lecture 5** Linear regression and Logistic regression
  * and their extensions and variants

* **Lecture 6** Non-linear models
  * Bayesian fixed basis models (sparsity)
  * Bayesian neural network

* **Lecture 7** More topics (if time allowed)
  * Bayesian unsupervised learning
    * Bayesian finite mixture models
    * Probabilistic PCA



## Afternoon sessions:

A mix of applied and theoretical problems (all applied)

* learn how to implement what we have learnt in the morning
  * involves a bit programming
* how to use `Turing.jl`
* I will be around to help 
"""

# ╔═╡ 692cb743-d345-4b1b-b71f-3e26d1c7c80a
md"""

## Some motivating examples - 1
#### Example 1. Coughing


!!! note "Coughing example"
	*Someone **coughed** in our class. What is the chance he/she has got **cancer**?*

## Some motivating examples - 1 (cont.)
#### Example 1. Coughing


!!! note "Coughing example"
	*Someone **coughed** in our class. What is the chance he/she has got **cancer**?*

* the chance is probably low
* if agree, you might have already *implicitly* applied Bayesian inference
"""

# ╔═╡ 8bf95702-90ac-4b3c-a027-e1b673359a60
md"""

## Some motivating examples - 2 
#### Example 2. Amazon sellers

!!! note "Amazon seller"
	Which seller to choose?
	* the first with 400k reviews
	* the second with 2k reviews
"""

# ╔═╡ 3a80d31d-9f32-48b1-b5fd-788b588d96cb
Resource(figure_url * "/figs4CS5010/amazonreviews.png", :width=>700, :align=>"left")

# ╔═╡ dede5ea1-c67d-4344-b6d0-c4a9b8d8dbeb
md"""

## Some motivating examples - 3
#### Example 3. Regression



**Regression problem**: find the relationship between input $X$ and $Y$ (when $Y$ is continuous)

$Y = f(X) + \varepsilon$

"""

# ╔═╡ 93b5c93c-682a-49b4-8191-a82a22804a0d
md"""
For example, **linear regression**

```math
f(X) = \beta_0 + \beta_1 X 
```
"""

# ╔═╡ 8a8eabbd-a0cd-47f1-a2b7-1e6c10f29c2b
html"""<center><img src="https://otexts.com/fpp3/fpp_files/figure-html/SLRpop1-1.png" width = "500"/></center>
"""

# ╔═╡ 63b92973-1031-439c-aa68-60f00552ab18
md"""

Now consider a non-linear dataset:
* the relationship is non-linear
* pay attention to the gaps in the middle
"""

# ╔═╡ 926be45c-84a4-4bb5-8cc0-56bdf74bc581
md"""

## Some motivating examples - 4
#### Example 4. Classification



**Classification problem**: find the relationship between input $X$ and $Y$ (when $Y$ is categorical):

$P(Y = \text{class 1} | X=x) = \sigma(x)$

* what is the probability that an instance belongs to class 1/2
"""

# ╔═╡ 24f06a6c-f64f-488b-9fd0-dbbf24007212
md"""

!!! note "Question"
	How would you classify data at the following input locations?

	```math
	\sigma(x) = ?
	```
"""

# ╔═╡ 4f0a3ee7-d2e9-4945-9491-e48475421574
Resource(bayes_figure_url * "logis_data.png", :width=>600)

# ╔═╡ c9b2fa82-841d-47cf-b7b4-23f8a86e60a7
md"""

## Some motivating examples - 4 (cont.)

> Which one do you prefer?
"""

# ╔═╡ 2bc778d0-6974-4a97-8769-3cb66100fd80
TwoColumn(Resource(bayes_figure_url * "freq_logis.png"), Resource(bayes_figure_url * "bayes_logis.png"))

# ╔═╡ 2fba13a9-a0ab-44a6-ba4d-b828ebe96274
md"""

## We are going to see 



> How **Bayesian inference** handles the above problems in a **uniform** procedure
> * **reasonable** inference results emerge automatically without ad hoc remedies


!!! tip "Bayesian procedure"
	In a nutshell, the Bayesian inference procedure is to
	> * (**modelling**) Tell a story about your data (how your data is generated)
	> * (**computation/inference**) Leave the inference to probability rules


"""

# ╔═╡ db1f6632-f894-4ce7-a5b4-493158a13860
md"""

# Probability Theory 

"""

# ╔═╡ f05dcb39-84c9-42e5-95f6-67238c574609
md"""

## Random variable 



**Random Variables**: variables' value is *random* rather than *certain*

* *e.g.* $X$: rolling of a fair 6-faced die 🎲
  * ``X`` can take 1,2,3,...,6 possible values 
  *  with probability **distribution**

```math

\begin{equation}  \begin{array}{c|cccccc} 
 & X = 1 & X = 2 & X = 3 & X = 4 & X = 5 & X = 6 \\
\hline
P(X) & 1/6 & 1/6 & 1/6 & 1/6 & 1/6 & 1/6

\end{array} \end{equation} 

```
"""

# ╔═╡ b52911ff-886a-4113-85a5-af508dc0cb8e
md"""
## Discrete *vs* continuous r.v.s




Depending on ``\mathcal A``, there are two kinds of random variables


**Discrete random variable**: ``\mathcal A`` is discrete, *e.g.* ``\mathcal A=\{\texttt{true},\texttt{false}\}, \mathcal A=\{1,2,3,\ldots, \infty\}``
  * ``P:`` *probability mass function (p.m.f)* satisfies 
    * ``0\leq P(x) \leq 1``
    * ``\sum_{x\in \mathcal A} P(x) = 1``
  * *e.g.* Bernoulli, Binomial, Poisson and so on

**Continuous random variable**: ``\mathcal A`` is continuous, *e.g.* ``\mathcal A = [0,1], \mathcal A= (-\infty , +\infty)``
  * ``p:`` *probability density function (p.d.f)* satisfies (smaller case ``p``)
    * ``p(x) > 0``
    * ``\int_{x\in \mathcal A} p(x) \mathrm{d}x=1``
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


``X``: a random variable taking binary values

* the domain ``\mathcal{A} = \{1, 0\}`` (for example, coin tossing or court case)

* the probability distribution (probability mass function) is 

```math
P(X ) =\begin{cases}\theta & X= 1 \\ 1-\theta & X=0 \end{cases}
```

* ``0\leq \theta \leq 1`` is the parameter of the distribution: called *bias*



"""

# ╔═╡ 10beeb49-277a-489a-969d-140d51fcfde3
@bind θ Slider(0:0.1:1, default=0.5)

# ╔═╡ e11b9b91-a912-45a1-913c-aeb3e78e60fc
md""" ``θ`` = $(θ)"""

# ╔═╡ 17f73ee7-e842-4f10-9118-47399e67f047
let
	bar(Bernoulli(θ), xticks=[0,1], xlabel=L"X", ylabel=L"P(X)", label="", ylim=[0,.8])
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

## Examples: discrete r.v. -- Binomial (conti.)


**Binomial** r.v.: the number of successes for ``n`` independent Bernoulli trials (with bias ``\theta``)

$$P(X=x) = \binom{n}{x} (1-\theta)^{n-x} \theta^{x},\;\; \text{for } x= 0,1,\ldots, n$$
  * ``\binom{n}{x}``: *binomial coefficient*, e.g. $\binom{3}{2} = 3$ 


A binomial r.v. with ``n``= $(nb), ``\theta`` = $(θ_bi) is plotted below

* the number of heads: tossing a coin (with bias $(θ_bi)) $(nb) times

"""

# ╔═╡ b680c9da-2cb5-432f-8032-e2a37fc3e738
let
	binom = Binomial(nb, θ_bi)
	bar(binom, label="", xlabel=L"X", xticks = 0:nb, ylabel=L"P(X)", title=L"n=%$(nb),\;\;\theta = %$(θ_bi)", legend=:topleft)

	# p_more_heads = exp(logsumexp(logpdf.(binom, floor(nb/2)+1:1:nb)))
	# vspan!([floor(nb/2)+0.5, nb+0.5], alpha=0.5, label=L"P(X \geq \lfloor n/2 \rfloor +1)\approx%$(round(p_more_heads; digits=2))")
end

# ╔═╡ 1e4db58a-e57d-457e-9bf4-20961a4775b1
md"""

## 	Examples: continuous r.v. -- Uniform distribution


``X`` is a **uniform distribution** over interval ``[a, b]``

```math
p(x) = \begin{cases}\frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise}\end{cases}
```


As an example, a uniform distribution ``x\in [0,1]``
```math
p(x) = \begin{cases}1 & 0 \leq x \leq 1 \\ 0 & \text{otherwise}\end{cases}
```

"""

# ╔═╡ 42466480-2d4d-4a62-be06-ede102d818f0
let
	a, b = 0, 1
	plot(Uniform(0,1), fill=true, alpha= 0.5, lw=2, ylim=[0, 2], xlabel=L"X", ylabel=L"p(X)", label="", title=L"p(X) = \texttt{Uniform}(0,1)")
end

# ╔═╡ d1e63621-5e4a-4f43-aa4f-a41f12fe632a
md"""
## Examples: continuous r.v. -- Gaussian distribution



Gaussian random variable $X$ has a distribution 

$$p(X=x) = \mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x-\mu}{\sigma} \right)^2}$$

* ``\mu``: mean parameter
* ``\sigma^2``: variance

"""

# ╔═╡ 16754b05-44c5-47b2-9b2e-89f74d5f59ea
let

	μs = [-3, 0, 3]
	σ²s = [1 , 2 , 5]
	plt_gaussian = Plots.plot(title="Gaussian distributions", xlabel=L"X", ylabel=L"p(X)")
	for i in 1:3 
		plot!(plt_gaussian, Normal(μs[i], sqrt(σ²s[i])), fill=true, alpha=0.5, label=L"\mathcal{N}(μ=%$(μs[i]), σ^2=%$(σ²s[i]))")
		vline!([μs[i]], color=i, label="", linewidth=2)
	end
	plt_gaussian
end

# ╔═╡ 7416236c-5ff0-4eb8-b448-e50acb0c016b
md"""
## Joint distributions


A **joint distribution** over a set of random variables: ``X_1, X_2, \ldots, X_n`` 
* specifies a real number for each assignment (or outcome): 


```math
\begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n) = P(x_1, x_2, \ldots, x_n) \end{equation} 
```

* Must still statisfy

$P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 


Example: joint distribution of temperature (``T``) and weather (``W``): ``P(T,W)``
```math
\begin{equation}
\begin{array}{c c |c} 
T & W & P\\
\hline
hot & sun & 0.4 \\
hot & rain & 0.1 \\
cold & sun  & 0.2\\
cold & rain & 0.3\end{array} \end{equation} 
```



"""

# ╔═╡ 48484810-1d47-473d-beb3-972ca8b934d5
md"""
## Two probability rules


There are only two probability rules 

* **sum rule** 

* **product rule** 




"""

# ╔═╡ 559b0936-a051-4f2e-b0ca-81becd631bce
md"""
## Probability rule 1: sum rule



$$P(X_1) = \sum_{x_2}P(X_1, X_2=x_2);\;\; P(X_2) = \sum_{x_1}P(X_1=x_1, X_2),$$

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
## Probability rule 2: chain rule


Chain rule (*a.k.a* **product rule**)

$$P(X, Y) = P(X)P(Y|X);\;\; P(X, Y) = P(Y)P(X|Y)$$

  * the chain order doesn't matter
  * joint distribution factorised as a product
    * between marginal and conditional

"""

# ╔═╡ 32bd86a4-0396-47f5-9025-6df83268850e
md"""

## Product rule: generalisation


Chain rule for higher-dimensional random variables

* for three
$P(x_1, x_2, x_3) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)$

* for any ``n``:

$P(x_1, x_2, \ldots, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2) \ldots=\prod_i P(x_i|x_1, \ldots, x_{i-1})$

**Chain rule**: chain distributions together
"""

# ╔═╡ e12f0e44-2b7d-4075-9986-6322fb1498c4
md"""

## Bayes' theorem



**Bayes' rule** is a direct result of the two rules (based on two r.v.s)

* ``H``: hypothesis 
* ``\mathcal{D}``: observation


Recall the definition of conditional probability

```math
p(H| \mathcal{D}) =\frac{p(H, \mathcal{D})}{p(\mathcal{D})}=\frac{p(H)p(\mathcal{D}|H)}{\sum_{h'} p(H=h', \mathcal{D})}
```



"""

# ╔═╡ 2229e882-e2b6-4de3-897f-7a75d69c24ec
md"""

## Bayes' theorem


!!! infor "Bayes' rule"
	$$p(H|\mathcal D) =\frac{\overbrace{p(H)}^{\text{prior}} \overbrace{p(\mathcal D|H)}^{\text{likelihood}}}{\underbrace{p(\mathcal D)}_{\text{evidence}}}$$




	* ``p(h)`` **prior**: prior belief of ``\theta`` before we observe the data 
	* ``p(\mathcal D|h)`` **likelihood**: conditional probability of observing the data given a particular $\theta$
	* ``p(\mathcal D)`` **evidence** (also known as *marginal likelihood*)
"""

# ╔═╡ a602e848-8438-4a41-ab76-ec3f6497de3e
md"""

## Why chain rule (or Bayes' rule) is useful


Easier for us to specify the joint distribution forwardly via **chain rule** 

$$H \xRightarrow[\text{probability}]{\text{forward}} \mathcal{D}$$ 
$$\text{or}$$
$$P(H,\mathcal{D}) = P(H)P(\mathcal{D}|H)$$

   * ``h`` is the possible hypothesis (which is usually unknown), with some prior 

$$\text{prior: }P(H)$$
* ``\mathcal{D}`` is the observation; given the cause, how likely you observe the data: 
  
$$P(\mathcal D|H):\;\; \text{likelihood function}.$$


## Example: Cough example
### first Bayesian inference problem

* ``H``: the unknown cause ``h \in \mathcal{A}_H=\{\texttt{cold}, \texttt{cancer}, \texttt{healthy}\}`` 

* ``\mathcal{D}``: the observed data is ``\mathcal{D} = \texttt{Cough}``
 

The chain rule allows us to specify the full joint:

```math
P(H, \mathcal{D}) = P(H) P(\mathcal{D}|H)
```

* ``\mathcal{D}`` depends on ``H``: the cough observation depends on the hidden cause

* Bayes' rule: compute the reverse probability: ``P(H|\mathcal{D})``

"""

# ╔═╡ eca7d42f-a038-40c9-b584-a6a020ffa9da
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/cough_bn.png" width = "100"/></center>
"""

# ╔═╡ b7973f17-7f7f-4cb2-bfd2-9defc4b5910b
md"""
## Example -- Cough example (cont.)

**Posterior**: ``P(H|\mathcal{D})``

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
	x_= 1
	post_p = prior_p .* liks
	marg_lik = sum(post_p)
	post_p = post_p/marg_lik
end;

# ╔═╡ fea3395d-250d-406c-9a8c-646f5c48ab12
md"""

## Example -- Cough example (cont.)


**Prior:** ``P(H)``

* three possible causes ``h \in \{\texttt{cold}, \texttt{cancer}, \texttt{healthy}\}`` with a prior


$$P(h) = \begin{cases} 0.8 & h=\texttt{healthy} \\
0.18 & h=\texttt{cold}\\ 
0.02 & h=\texttt{cancer}\end{cases}$$

$(begin
prior_plt=plot(prior_p, fill=true, st=:bar, xticks=(1:3, Hs), markershape=:circ, ylim =[0, 1.0], label="", title="Prior: "*L"P(h)")
end)


"""

# ╔═╡ 265bdf7b-2d31-4ff0-8af7-a5e9c5e881bd
like_plt = plot(liks, fill=true, st=:bar, xticks=(1:3, Hs), markershape=:circ, ylim =[0, 1.0], label="", title="Likelihood: "*L"P(\texttt{cough}|h)", c= 2,  lw=2);

# ╔═╡ 602bfca9-b90e-48cc-949f-2909aacf3546
md"""
## Example -- Cough example (cont.)
 
**Likelihood**: ``P(\mathcal{D}|H)``


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

**Maximum likelihood estimation (MLE)** is a popular frequentist method

```math

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
	post_plt = Plots.plot(post_p, seriestype=:bar, markershape=:circle, label="", color=3,  xticks=(1:3, Hs), ylim=[0,1], ylabel="", title="Posterior: "*L"P(h|\texttt{cough})" ,legend=:outerleft)
	# plt_cough = Plots.plot(prior_plt, like_plt, post_plt, layout=l)
end

# ╔═╡ 2ca8a673-9eeb-4295-91ed-27c173a91db7
md"""
## Bayesian inference as Belief update


Bayesian inference is a belief update procedure 

```math

\text{prior } P(H) \Rightarrow \text{posterior } P(H|\mathcal{D})
```

It **update** *prior* to *posterior* based on the observed ``\mathcal{D}``

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

## Another example -- change point detection


> Your friend has two coins: 
> * one fair ``p_1= 0.5`` 
> * the other is bent with bias $p_2= 0.2$ 
> He always uses the fair coin at the beginning then switches to the bent coin after some unknown switch

> Given tosses ``\mathcal{D}= [t,h,h,t,h,t,t]``
> When did he switch?
"""

# ╔═╡ 5920cd07-30b9-44a7-8f5c-999c1f8ef043
md"""
## Forward generative modelling

Let's identify the random variables first

* hypothesis: ``S\in \{1,2,\ldots, n-1\}``, the unknown switching point

* data (observation): ``\mathcal{D}=\{d_1, d_2,\ldots\}``

Let's think **forwardly**, or **generatively**

```math
S \Rightarrow {d_1, d_2, \ldots, d_n}
```

* assume knowing ``S\in \{1,2,\ldots, n-1\}`` 

* the likelihood ``P(\mathcal{D}|S)`` is ? 




"""

# ╔═╡ 5acefb1f-209e-489f-afb9-dce517ad6225
md"""
## Forward generative modelling (cont.)


$$P(\mathcal D|S) = \underbrace{\prod_{i=1}^S P(d_i|p_1 =0.5)}_{\text{fair coin}} \underbrace{\prod_{j=S+1}^N P(d_j|p_2 = 0.2)}_{\text{bent coin}};$$ 



  * *e.g.* if ``S=1``

$$P(\{0,1,1,0,1,0,0\}|S=1) = \underbrace{0.5}_{\text{before switch: }\{0\}} \times \underbrace{0.2\cdot 0.2\cdot(1-0.2)\cdot 0.2\cdot (1-0.2)^2}_{\text{after switch: }\{1,1,0,1,0,0\}}$$

  * *e.g.* if ``S=2``

$$P(\{0,1,1,0,1,0,0\}|S=2) = \underbrace{0.5\cdot 0.5}_{\text{before switch: }\{0,1\}} \times \underbrace{0.2\cdot(1-0.2)\cdot 0.2\cdot (1-0.2)^2}_{\text{after switch: }\{1,0,1,0,0\}}$$

* then we can calculate the likelihood for ``S=1,2,\ldots, n-1`` 

"""

# ╔═╡ 866cd5fd-9d9d-42da-b05e-b4b3ad958ecb
𝒟 = [0,1,1,0,1,0,0];

# ╔═╡ 6593bd28-f483-4168-ac69-eae469fbcb27
md"""
## Forward generative modelling: Prior for P(S)


To reflect our ignorance, we use a uniform flat prior

$$P(S=s) = \frac{1}{n-1}; \; \text{for }s \in \{1,2,\ldots, n-1\}$$

"""

# ╔═╡ 9e0953ee-3451-418a-afe2-b2177a2a8898
begin
	 # to type 𝒟, type "\scrD"+ "tab"
	n1 = length(𝒟)-1
	Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1,marker=:circle, label="", ylim=[0,1.0], xlabel=L"S", title=L"P(S)")
	Plots.plot!(1:n1, 1/n1 * ones(n1),  st=:path, fill=true, color=1, alpha= 0.3,  label="Prior")
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
	Plots.plot(1:length(𝒟)-1, ℓ_switch(𝒟)[1], xlabel=L"S", ylabel=L"p(\mathcal{D}|S)", title="Likelihood "*L"p(\mathcal{D}|S)", st=:sticks, marker=:circle, label="")
end

# ╔═╡ 72fb8fa0-8263-4729-af19-ff6047865a61
md"""
## Make the inference 
The rest is very routine: **apply Bayes' rule** mechanically

$$P(S|\mathcal D) \propto P(S) P(\mathcal D|S)$$
  * then normalise by dividing by the evidence: ``P(\mathcal D) = \sum_s  P(S)P(\mathcal D|S)``



"""

# ╔═╡ baf8e5b2-4473-4e22-bb4d-2c1fd8cccfaa
begin
	post_p_switch = ℓ_switch(𝒟)[1]./ℓ_switch(𝒟)[2]
end;

# ╔═╡ 691c897a-a282-4023-8b7a-d7df665bf53c
begin
	# n1 = length(𝒟)-1
	Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1,marker=:circle, label="", legend=:outerbottom, size=(600, 500))
	Plots.plot!(1:n1, 1/n1 * ones(n1),  st=:path, fill=true, color=1, alpha= 0.3,  label="Prior")
	Plots.plot!(1:n1, post_p_switch, xlabel=L"S", ylabel=L"p(S|\mathcal{D})", title="Posterior after update: "*L"P(S|\mathcal{D})", st=:sticks, marker=:circle, color=2, label="")
	Plots.plot!(1:n1, post_p_switch, st=:path, fill=true, alpha=0.3, color=2, label="Posterior")
end

# ╔═╡ 4d398cf7-c3dd-49cf-b0c3-3731e50c1c37
md"""


## Another dataset ?

"""

# ╔═╡ 2ca2d909-7499-45fc-864e-cf22486896dc
𝒟₂ = [0,1,1,0,1,0,0,0,0,0,0,0]

# ╔═╡ 9d9e9537-8194-4716-8f1f-fd1013c7e040
begin
	
	plot(findall(𝒟₂ .== 0), 𝒟₂[𝒟₂ .== 0], seriestype=:scatter, label=L"\texttt{tail}", legend=:outerbottom, xlabel="time")
	plot!(findall(𝒟₂ .== 1), 𝒟₂[𝒟₂ .== 1], seriestype=:scatter, label=L"\texttt{head}")
end;

# ╔═╡ facc40e6-87cd-4884-b20e-28d84e4337c6
begin
	post_p₂ = ℓ_switch(𝒟₂)[1]./ℓ_switch(𝒟₂)[2]
end;

# ╔═╡ 7652aa5c-47c7-4263-a673-00f073f91018
let
	post_p = post_p₂
	n1 = length(𝒟₂)-1
	Plots.plot(1:n1, 1/n1 * ones(n1), st=:sticks, color=1,marker=:circle, label="", legend=:outerbottom, size=(600, 500))
	Plots.plot!(1:n1, 1/n1 * ones(n1),  st=:path, fill=true, color=1, alpha= 0.3,  label="Prior")
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

Random variable $X,Y$ are independent, if 

$$\forall x,y : P(x,y) = P(x)P(y)$$

* the joint is formed by the product of the marginals

Another equivalent definition

$$\forall x,y: P(x|y) = P(x)$$
  * intuition: knowing (conditional on $Y=y$) does not change the probability of ``X``

"""

# ╔═╡ 8afac55e-44f9-4127-b16a-969560307f90
md"""

## Conditional independence

``X`` is conditionally independent of ``Y`` given ``Z``, if and only if 


$$\forall x,y,z: P(x,y|z) = P(x|z)p(y|z)$$

or equivalently (**more useful**), if and only if 

$$\forall x,y,z: P(x|z, y) = P(x|z)$$

* intuition: knowing ``z``, ``x`` is no longer influenced by ``y`` (therefore independence): ``p(x|z, \cancel{y})``

Denoted as ``X\perp Y\vert Z``





## Conditional and Unconditional independence

Note that (*marginal or unconditional independence*) is a specific case of conditional independence (the condition is an empty set)
* *i.e.* ``X\perp Y \vert \emptyset``
* the independence relationship is unconditional



"""

# ╔═╡ 5bcfc1a8-99d4-4b48-b45c-4c8d0c5acae4
md"""

## Example

Consider three r.v.s: **Traffic, Umbrella, Raining**


Based on *chain rule* (or product rule)

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Traffic}, \text{Rain})$$


"""

# ╔═╡ 6dc34d66-366f-4e1a-9b73-a74c07e58feb
md"""

## Example

Consider 3 r.v.s: **Traffic, Umbrella, Raining**


Based on *chain rule* (or product rule)

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Traffic}, \text{Rain})$$


The last term can be simplified based on **conditional independence assumption**

$$P(\text{Umbrella}|\cancel{\text{Traffic}}, {\text{Rain}}) = P(\text{Umbrella}|\text{Rain})$$

* knowing it is raining, Umbrella no longer depends on Traffic

* or in CI notation: ``T \perp U|R``



"""

# ╔═╡ 12751639-7cdc-4e39-ba72-30ee2fe2cd49
md"""

## Example

Consider the coin switching problem


Based on *chain rule* (or product rule)

$$P(\text{S},\mathcal{D}) = P(S)  P(\mathcal{D}|S)$$

* ``\mathcal{D} = [d_1, d_2, \ldots, d_n]``
Then applies **conditional independence assumption**

$$P(\mathcal{D}|S) = P(d_1|S)P(d_2|S) \ldots P(d_n|S)$$

* conditioned on the switching time ``S``, the tosses are independent 


Conditional independence greatly simplies probability models
* we only need to spell out ``P(S)`` and ``P(d_i|S)`` rather than the conditional joint ``P(d_1, d_2, \ldots, d_n|S)``
* also note that ``P(d_i|S)`` are the same for all ``i``

"""

# ╔═╡ 532a9f66-b04a-446a-becf-d74759bcf750
md"""
## Probabilistic graphical models


Probabilistic graphical models are very useful probabilistic modelling tool
* to model conditional independence graphically


We consider a specific kind: **directed graphical model** 
* also known as Bayes' network
* a directed acyclic graph (DAG)
  * **node**: random variables 
  * **edges**: relationship between random variable


For example, the cough example

"""

# ╔═╡ 79f88e1e-9eca-49ed-84e1-3dac5ef1819c
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/cough_bn.png" width = "100"/></center>
"""

# ╔═╡ 9da0bf21-3d2f-4df0-a8b3-3891dd665c7f
md"""

## Digress: directed graphs

A **directed** graph is a a graph with **directed** edges
  * direction matters: $(X, Y)$ are not the same as $(Y, X)$
  * **asymmetric relationship**: 
    * e.g. parent-to-child relationship (the reverse is not true)
    

``\textbf{parent}(\cdot)`` returns the set of parent nodes, e.g. $$\text{parent}(Y) = \{X, Z\}$$
"""

# ╔═╡ 24f91696-3c70-472c-89db-fc4732c28677
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/directedGraph.png" width = "600"/></center>"""

# ╔═╡ 41255527-96d2-41bf-92b4-8b4a9bebab24
md"""
## Digress: directed acyclic graph (DAG)

A **d**irected **a**cyclic **g**raph (**DAG**) is a directed graph **without** cycles 
  * a cycle: directed path starts and ends at the same node
"""

# ╔═╡ 00a0c582-34b2-413d-bef2-7a7572f48340
md"""

**For example**, the following is **NOT** a DAG
* cycles are NOT allowed: $X\Rightarrow Y_1\Rightarrow X$

"""

# ╔═╡ 144eb24c-1b4a-44d2-bd25-33b68798c184
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/diceExampleDCG.png" width = "500"/></center>"""

# ╔═╡ a578a1a7-0f42-492c-8226-b8e09e92506f
md" However, multiple paths are allowed, *e.g.*"

# ╔═╡ 288849aa-35c5-426f-89aa-76d11a5801e9
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/diceExampleDAG.png" width = "500"/></center>"""

# ╔═╡ 20c95319-86a7-477d-93c8-adf0dbdbb4c2
md"""
  * two possible paths from $X$ to $Y_2$: $X\Rightarrow Y_2$ and $X \Rightarrow Y_1 \Rightarrow Y_2$
  * still **acyclic** though
"""

# ╔═╡ 275c155d-c4c5-472a-8613-c549706815b0
md"""

## Node types

"""

# ╔═╡ 2f5af892-249f-40c8-a2a2-0bc7acc7f000
md"""

We introduce three types of nodes

* **unshaded**: unobserved **random variable** (the hidden cause)
* **shaded**: observed **random variable** (coughing)
* **dot**: fixed parameters/input, non-random variables
  * *e.g.* the input predictors ``\mathbf{x}`` for linear regression
  * or fixed hyperparameters 
"""

# ╔═╡ 7b9577e1-e1ce-44a9-9737-a60061ad0dc7
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/notations.png' width = '200' /></center>"

# ╔═╡ 41ee5127-4390-4fb8-b41f-9856346877cf
md"""

## Directed graphical model: a DAG with local distributions 


A Directed graphical model is 

> **Directed Acyclic Graph (DAG)** + local prob. distributions, $P(X_i|\text{parents}(X_i))$
  
* **DAG**, ``G``:
    * ``V=\{X_1, X_2, \ldots, X_n\}``, one random variable per vertex (also called node) 
    * ``E``, directed edges encode (conditional) dependences between r.v.s


* **Local distributions**: $P(X_i|\text{parent}(X_i))$, one $P$ for each r.v. $X_i$


"""

# ╔═╡ eda02250-b88d-45a1-9430-114e3d60f72a
md"""

For example, cough example

* ``H`` is with ``P(H)``
* ``\texttt{Cough}``: ``P(\texttt{Cough}|H=h)``
"""

# ╔═╡ 7aea40a5-be9f-4ac0-94e7-b692985ea35d
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/cough_bn.png" width = "100"/></center>
"""

# ╔═╡ be4b66ed-dc94-4450-97dd-a57175913b20
md"""
## Graphical model semantics

Graphical model 

* a graphical way to specify the **chain rule**

!!! important "Factoring property & chain rule"
	Joint distribution factorises as the product of CPTs: 
	
	$P(X_1, X_2,\ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))$

"""

# ╔═╡ c7217173-3b9e-439a-ad59-6fdea81bb12c
md"""

## Example - Traffic, Rain, Umbrella

Recall the **Traffic, Umbrella, Raining** example


The full joint (based on the **conditional independence assumption**) becomes 

$$P(\text{Rain},\text{Umbrella}, \text{Traffic}) = P(\text{Rain})  P(\text{Traffic}|\text{Rain}) P(\text{Umbrella}|\text{Rain})$$

The corresponding graphical model is

* a graphical representation of the chain rule
"""

# ╔═╡ 4c4f8078-b3ed-479f-9c0b-35c9e81fe44d
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/rtu_bn.png" width = "300"/></center>
"""

# ╔═╡ 33a8d7e1-cb9b-48c8-9b63-7a817c0497b2
md"""

## Example -- coin guess


> Two coins (with biases ``0.5`` and ``0.99`` respectively) in an urn
> * you randomly pick one and toss it 3 times 



"""

# ╔═╡ 10ad19a7-f1ba-4f3c-ada7-8ce1391ef151
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/coin_choice.png" width = "400"/></center>"""

# ╔═╡ 884b5699-08fb-4aba-abf8-2628d71818ce
md"""

For node Coin Choice: $P(C)$:

|``C``   | ``P(C=c)`` |
| --- | ---  | 
| 1   | 0.5 | 
| 2   | 0.5 | 

"""

# ╔═╡ 71cdf6a5-90bc-4a4d-80f7-214406f22019
md"""
For node $Y_i$, the local conditional distributions $P(Y_i|C)$:

|``C``   | ``\texttt{head}`` | ``\texttt{tail}`` |
| ---| ---   | ---     |
| 1  | 0.5 | 0.5 |
| 2  | 0.01 | 0.99 |

* each row is a valid conditional distributions 

"""

# ╔═╡ b96cc13e-5fd9-4fa1-a9bc-4ea7b22c07a3
md"""

## Plate notation for repeated nodes

Note that $P(Y_i|C)$ for $i =1,2,3$ are repeated


"""

# ╔═╡ a94dac61-af9b-4820-bc2a-812a282a5603
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/plate_coin.png" width = "300"/></center>"""

# ╔═╡ 8044aa77-b299-4874-b90b-ee36fb8d374e
md"""

## Some graphical model patterns

There are three patterns in graphical models

* common cause pattern
* chain pattern
* collider pattern

All other general graphical models are **composition** of the three patterns



## Common cause pattern (*a.k.a.* tail to tail)

``Y`` is the common cause of ``X`` and ``Z``


* also known as tail to tail (from ``Y``'s perspective)
* *e.g.* the unknown bias is the common cause of the two tossing realisations
"""

# ╔═╡ 63c2da6b-4f7a-40dd-95fe-17aead4134a6
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_common.png" width = "500"/></center>"""

# ╔═╡ cc8e87da-216c-40bf-9fc1-0e8c929ad7a1
md"""

It can be shown that 

```math
X \perp Z |Y\;\; \text{or}\;\;P(X,Z|Y) =P(X|Y)P(Z|Y)
```


But not **marginally** independent 

```math
X \not\perp Z\;\; \text{or}\;\;P(X,Z) \neq P(X)P(Z)
```

"""

# ╔═╡ 15cf6021-b172-41ae-9392-0838c39d551b
md"""

**Example**: "Vaccine" effectiveness 
* common cause: whether a vaccine is effective (**Y**)
* child nodes: two independent patients' treatment outcome after receiving the vaccine (**X**, **Z**)
"""

# ╔═╡ 588eec3d-fc0b-4384-ac9b-bb8d8816b184
md"""
## Chain pattern (*a.k.a.* tip to tail)

``X`` impacts ``Z`` via ``Y``

* indirect impact 

"""

# ╔═╡ 450de0ac-e2a9-4e0a-b542-72534c751fa2
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_seq.png" width = "500"/></center>"""

# ╔═╡ b96539d8-5ca9-4a04-84bf-660db2a4f609
md"""

It can be shown 


```math
X \perp Z|Y \;\; \text{or}\;\;P(X,Z|Y) = P(X|Y)P(Z|Y)

```

but not marginally independent 

```math
X \perp Z \;\; \text{or}\;\;P(X,Z) \neq P(X)P(Z)

```

"""

# ╔═╡ c6653de5-1542-44a6-9fcb-2e9bd091f2bb
md"""

**Example**: Rain, Floor is Wet, Slip


```math
\begin{align}
{\text{Rain}} \rightarrow {\text{Floor is Wet}}  \rightarrow {\text{Slip}} 
\end{align}
```

* add *Injured* ?

* explain the CI assumptions for this case
"""

# ╔═╡ 9fafd041-2d42-4bf1-a17a-c31deb23d97d
md"""

## A classic example -- chain case

**Voting for Trump** does not **directly** make you **die**!
"""

# ╔═╡ 615b57fb-0ad5-4175-bc71-000aaae89b65
html"""<center><img src="https://static01.nyt.com/images/2021/09/27/multimedia/27-MORNING-sub3-STATE-DEATH-VOTE-PLOT/27-MORNING-sub3-STATE-DEATH-VOTE-PLOT-superJumbo.png?quality=75&auto=webp" width = "400"/></center>
"""

# ╔═╡ 83e1f7fa-c711-4db6-a2be-6914a4e7e8a2
md"""


But via **Vaccination**, everything makes sense

"""

# ╔═╡ d0868210-baa4-4bdc-8d99-f318b6b8f6df
html"""<center><img src="https://static01.nyt.com/images/2021/09/27/multimedia/27-MORNING-sub2-STATE-VAX-VOTE-PLOT/27-MORNING-sub2-STATE-VAX-VOTE-PLOT-superJumbo-v2.png?quality=75&auto=webp" width = "400"/></center>
"""

# ╔═╡ 3db2f0f1-d8b1-4483-af5e-71889be766d9
md"""

A suitable model

```math
\text{Vote For Trump} \rightarrow \text{Vaccinated} \rightarrow \text{Die Out of COVID}

```

"""

# ╔═╡ 94767935-df48-4466-9454-aa000217ddda
md"""
## Common effect (*a.k.a.*  collider, or tip to tip)

``Y`` is the effect of some joint causes of ``X`` and ``Z``
* also known as point to point (from ``Y``'s perspective)
"""

# ╔═╡ 41f94bdc-5a92-4bc7-accb-f927b02e4032
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/dag_collider.png" width = "500"/></center>"""

# ╔═╡ fbe1bc73-f1c8-484a-8c5f-02781b3f871f
md"""

**Quite the opposite** to the previous two cases, 
* ``X, Z`` are marginally independent 
* but not conditionally independent (explain away)


```math
X\perp Z|\emptyset\;\; \text{or} \;\; P(X, Z)= P(X)P(Z)

```

```math
X\not \perp Z|Y\;\; \text{or} \;\; P(X, Z|Y)\neq P(X|Y)P(Z|Y)
```

"""

# ╔═╡ 99b465f0-63d6-4daf-bcfd-4704888d9e72
md"""

**Example**: Exam performance
* ``X``: your knowledge of the subject
* ``Z``: exam's difficulty 
* ``Y``: exam grade 
"""

# ╔═╡ 583812b3-7517-4177-b1c7-7cd5b2f1908b
md"""

## Exercise & discussion



!!! exercise "Coin swicth's graphical model"
	Recall the coin switching problem
	* construct the corresponding graphical model representation
	* which pattern the model involves
    * consider the three node types: observed, unobserved, fixed
	  * which type of nodes are used 
	  * how to add *fixed* nodes?
"""

# ╔═╡ 2e2f0b68-b8b1-41fc-981d-46e43037ac3a
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/notations.png' width = '200' /></center>"

# ╔═╡ 7d48dff8-9b53-462a-bbf4-4a9cd4d51be9
md"""

## Summary


* Two probability rules: sum and chain rule


* To form a probabilistic model, it is easier to think generatively via chain rule

$$P(H,\mathcal{D}) = P(H) P(\mathcal{D}|H)$$


* Bayes' rule provides a way to do reverse inference 


* Probabilistic graphical model is a useful modelling tool


"""

# ╔═╡ 61e3f627-5752-48e3-9100-3c96575f1805
md"""

## Appendix
"""

# ╔═╡ c0e7844b-111f-469a-9d6e-33751a441adb
begin
	function true_f(x)
		-5*tanh(0.5*x) * (1- tanh(0.5*x)^2)
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
	plot!(true_f, lw=2, framestyle=:default,  lc=:red, ls=:solid,label=L"f(x)", xlabel=L"x", ylabel=L"y")
end;

# ╔═╡ 3da83a5f-be41-47a1-bad1-781e59658f95
begin
	function poly_expand(x; order = 2) # expand the design matrix to the pth order
		n = length(x)
		return hcat([x.^p for p in 0:order]...)
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
	freq_ols_model = lm(Φ, y_output);
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
	plot!(x_test, pred_y_ols, linestyle=:solid, lc =:blue, lw=2, xlabel=L"x", ylabel=L"y", legend=:topright, label=L"f(x)")
end;

# ╔═╡ 94ef0651-e697-4806-868a-bce137d8437e
plt=plot(plt1, plt2, layout=(1,2), size=(600,300));

# ╔═╡ 8eec2a38-c5fb-48c7-8801-81d6d4afb796
md"""


!!! note "Question"
	Given the data, which model below is more reasonable?
	
	* the red one: $\textcolor{red}{f(x)}$
	* the blue one: $\textcolor{blue}{f(x)}$

$(begin 
	plt
end)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
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
Distributions = "~0.25.86"
GLM = "~1.8.2"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.18"
LogExpFunctions = "~0.3.23"
Plots = "~1.38.8"
PlutoTeachingTools = "~0.2.8"
PlutoUI = "~0.7.50"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "29f81fc77ee3dcb2567d52848377a05812891cd3"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "7ebbd653f74504447f1c33b91cd706a69a1b189f"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "d57c99cc7e637165c81b30eb268eabe156a45c49"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.2.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "da9e1a9058f8d3eec3a8c9fe4faacfb89180066b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.86"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f9818144ce7c8c41edf5c4c179c684d92aa4d9fe"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.6.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7072f1e3e5a8be51d525d64f63d3ec1287ff2790"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.11"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "cd3e314957dc11c4c905d54d1f5a65c979e4748a"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "4423d87dc2d3201f3f1768a29e807ddc8cc867ef"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3657eb348d44575cc5560c80d7e55b812ff6ffe1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "d926e9c297ef4607866e8ef5df41cde1a642917f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.14"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6a125e6a4cb391e0b9adbd1afa9e771c2179f8ef"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.23"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "91a48569383df24f0fd2baf789df2aade3d0ad80"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "f49a45a239e13333b8b936120fe6d793fe58a972"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.8"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "b970826468465da71f839cdacc403e99842c18ea"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.8"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "90cb983381a9dc7d3dff5fb2d1ee52cd59877412"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "b8d897fe7fa688e93aef573711cb207c08c9e11e"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.19"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "06a230063087c11910e9bbd17ccbf5af792a27a4"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.0"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e0d5bc26226ab1b7648278169858adcfbd861780"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
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
# ╟─93b5c93c-682a-49b4-8191-a82a22804a0d
# ╟─8a8eabbd-a0cd-47f1-a2b7-1e6c10f29c2b
# ╟─63b92973-1031-439c-aa68-60f00552ab18
# ╟─8f2452a3-79cc-4f22-90dc-82f88adc46be
# ╟─8eec2a38-c5fb-48c7-8801-81d6d4afb796
# ╟─926be45c-84a4-4bb5-8cc0-56bdf74bc581
# ╟─24f06a6c-f64f-488b-9fd0-dbbf24007212
# ╟─4f0a3ee7-d2e9-4945-9491-e48475421574
# ╟─c9b2fa82-841d-47cf-b7b4-23f8a86e60a7
# ╟─2bc778d0-6974-4a97-8769-3cb66100fd80
# ╟─2fba13a9-a0ab-44a6-ba4d-b828ebe96274
# ╟─db1f6632-f894-4ce7-a5b4-493158a13860
# ╟─f05dcb39-84c9-42e5-95f6-67238c574609
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
# ╟─16754b05-44c5-47b2-9b2e-89f74d5f59ea
# ╟─7416236c-5ff0-4eb8-b448-e50acb0c016b
# ╟─48484810-1d47-473d-beb3-972ca8b934d5
# ╟─559b0936-a051-4f2e-b0ca-81becd631bce
# ╟─1a930c29-0b6e-405b-9ada-08468c69173c
# ╟─ddc281f3-6a90-476a-a9b9-34d159cca80a
# ╟─05e3c6d0-f59b-4e1e-b631-11eb8f8b0945
# ╟─32bd86a4-0396-47f5-9025-6df83268850e
# ╟─e12f0e44-2b7d-4075-9986-6322fb1498c4
# ╟─2229e882-e2b6-4de3-897f-7a75d69c24ec
# ╟─a602e848-8438-4a41-ab76-ec3f6497de3e
# ╟─eca7d42f-a038-40c9-b584-a6a020ffa9da
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
# ╟─8afac55e-44f9-4127-b16a-969560307f90
# ╟─5bcfc1a8-99d4-4b48-b45c-4c8d0c5acae4
# ╟─6dc34d66-366f-4e1a-9b73-a74c07e58feb
# ╟─12751639-7cdc-4e39-ba72-30ee2fe2cd49
# ╟─532a9f66-b04a-446a-becf-d74759bcf750
# ╟─79f88e1e-9eca-49ed-84e1-3dac5ef1819c
# ╟─9da0bf21-3d2f-4df0-a8b3-3891dd665c7f
# ╟─24f91696-3c70-472c-89db-fc4732c28677
# ╟─41255527-96d2-41bf-92b4-8b4a9bebab24
# ╟─00a0c582-34b2-413d-bef2-7a7572f48340
# ╟─144eb24c-1b4a-44d2-bd25-33b68798c184
# ╟─a578a1a7-0f42-492c-8226-b8e09e92506f
# ╟─288849aa-35c5-426f-89aa-76d11a5801e9
# ╟─20c95319-86a7-477d-93c8-adf0dbdbb4c2
# ╟─275c155d-c4c5-472a-8613-c549706815b0
# ╟─2f5af892-249f-40c8-a2a2-0bc7acc7f000
# ╟─7b9577e1-e1ce-44a9-9737-a60061ad0dc7
# ╟─41ee5127-4390-4fb8-b41f-9856346877cf
# ╟─eda02250-b88d-45a1-9430-114e3d60f72a
# ╟─7aea40a5-be9f-4ac0-94e7-b692985ea35d
# ╟─be4b66ed-dc94-4450-97dd-a57175913b20
# ╟─c7217173-3b9e-439a-ad59-6fdea81bb12c
# ╟─4c4f8078-b3ed-479f-9c0b-35c9e81fe44d
# ╟─33a8d7e1-cb9b-48c8-9b63-7a817c0497b2
# ╟─10ad19a7-f1ba-4f3c-ada7-8ce1391ef151
# ╟─884b5699-08fb-4aba-abf8-2628d71818ce
# ╟─71cdf6a5-90bc-4a4d-80f7-214406f22019
# ╟─b96cc13e-5fd9-4fa1-a9bc-4ea7b22c07a3
# ╟─a94dac61-af9b-4820-bc2a-812a282a5603
# ╟─8044aa77-b299-4874-b90b-ee36fb8d374e
# ╟─63c2da6b-4f7a-40dd-95fe-17aead4134a6
# ╟─cc8e87da-216c-40bf-9fc1-0e8c929ad7a1
# ╟─15cf6021-b172-41ae-9392-0838c39d551b
# ╟─588eec3d-fc0b-4384-ac9b-bb8d8816b184
# ╟─450de0ac-e2a9-4e0a-b542-72534c751fa2
# ╟─b96539d8-5ca9-4a04-84bf-660db2a4f609
# ╟─c6653de5-1542-44a6-9fcb-2e9bd091f2bb
# ╟─9fafd041-2d42-4bf1-a17a-c31deb23d97d
# ╟─615b57fb-0ad5-4175-bc71-000aaae89b65
# ╟─83e1f7fa-c711-4db6-a2be-6914a4e7e8a2
# ╟─d0868210-baa4-4bdc-8d99-f318b6b8f6df
# ╟─3db2f0f1-d8b1-4483-af5e-71889be766d9
# ╟─94767935-df48-4466-9454-aa000217ddda
# ╟─41f94bdc-5a92-4bc7-accb-f927b02e4032
# ╟─fbe1bc73-f1c8-484a-8c5f-02781b3f871f
# ╟─99b465f0-63d6-4daf-bcfd-4704888d9e72
# ╟─583812b3-7517-4177-b1c7-7cd5b2f1908b
# ╟─2e2f0b68-b8b1-41fc-981d-46e43037ac3a
# ╟─7d48dff8-9b53-462a-bbf4-4a9cd4d51be9
# ╟─61e3f627-5752-48e3-9100-3c96575f1805
# ╟─c0e7844b-111f-469a-9d6e-33751a441adb
# ╟─94ef0651-e697-4806-868a-bce137d8437e
# ╟─945acfa0-9fde-4635-b87e-7f7f026da8bb
# ╟─43f36576-b470-4fcf-935a-dca80c2f80f8
# ╟─3da83a5f-be41-47a1-bad1-781e59658f95
# ╟─0cae3fc5-19a3-4143-aebc-e9484f6253ba
# ╟─18c3484f-c219-4a98-95c7-81c87f5c5ff0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
