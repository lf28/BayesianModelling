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

# ╔═╡ dbdb0950-4a33-11ed-39a0-6b2ea3d774e6
begin
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using StatsPlots
	using PlutoTeachingTools
	using PlutoUI
	using MCMCChains
	using Random
	using StatsFuns:xlogy, logsumexp
	using Distributions
	using LaTeXStrings, Latexify
	using Logging; 
	Logging.disable_logging(Logging.Warn);
end;

# ╔═╡ 89f53893-93f1-49d3-b643-2aee7e6f4610
TableOfContents()

# ╔═╡ b5e1f496-02e1-4352-9912-ff7bf4ea622c
ChooseDisplayMode()

# ╔═╡ d1b809fb-e381-49ac-bfa4-77c12f3ccf17
md"""

# Bayesian modelling


**Lecture 4. Markov Chain Monte Carlo 1** 


$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang (lf28@st-andrews.ac.uk)

*School of Computer Science*

*University of St Andrews, UK*

*March 2023*

"""

# ╔═╡ 2f5f8005-fda3-4a07-9c39-969ca4417f2c
md"""


## Bayesian computation is hard


##### *Bayes' theorem* is conceptually simple

$$\Large\text{posterior}=\frac{\text{prior} \times \text{likelihood}}{\text{evidence}}\;\;\text{or}\;\;p(\theta|\mathcal D) =\frac{p(\theta) p(\mathcal D|\theta)}{p(\mathcal D)}$$


##### The evidence term:

$$\large p(\mathcal D) = \int p(\theta) p(\mathcal D|\theta) \mathrm{d}\theta$$ 


* ##### an integration: often high-dimensional, therefore *intractable*. 



"""

# ╔═╡ 0e163261-577e-4f0d-8a39-a3748a9623b5
# md"""






# ## Bayesian computation is hard (cont.)

# #### To summarise, the difficulty are two-folds


# ##### 1. the posterior's normalising constant is intractable


# ##### 2. posterior computation involves integrations


# """

# ╔═╡ 405a783c-1b82-4a3d-b984-070e78f75c13
# md"""
# ## Bayesian computation is hard (cont.)

# ##### *What's worse*, even when ``p(\mathcal D)`` is known

# * the computation can still be hard for general problems
# * we still need to evaluate the numerator's integration (and find the ratio)


# **For example**: given a $5$-dimensional Gaussian $\theta \in \mathbb{R}^5$:

# $$p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}; \boldsymbol{\mu}, \boldsymbol{\Sigma})= \underbrace{\frac{1}{(2\pi)^{d/2}} |\boldsymbol{\Sigma}|^{-1/2}}_{\text{normalising constant}} \exp \left \{ -\frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\boldsymbol{\theta} - \boldsymbol{\mu})\right\}$$

# * where $\boldsymbol{\theta} = [\theta_1, \theta_2,\ldots, \theta_5]^\top$ 
# * find the probability $P(\theta_1<\theta_2<\theta_3<\theta_4<\theta_5)$ ?

# One still needs to find it by integrating the non constant part:

# $$\begin{align}
# &P(\theta_1<\ldots<\theta_5) = \int_{\theta_1}\int_{\theta_2}\ldots\int_{\theta_5}I(\theta_1<\theta_2<\ldots< \theta_5)\mathcal N(\boldsymbol{\theta}; \boldsymbol{\mu}, \boldsymbol{\Sigma})d{\theta_1}d{\theta_2}\ldots d{\theta_5}\\
# &=\frac{1}{(2\pi)^{d/2}} |\boldsymbol{\Sigma}|^{-1/2} \int_{\theta_1}\ldots\int_{\theta_5}I(\theta_1<\ldots< \theta_5)\exp \left \{ -\frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\boldsymbol{\theta} - \boldsymbol{\mu})\right\}d{\theta_1}\ldots d{\theta_5}
# \end{align}$$

# * which is in general impossible
# """

# ╔═╡ 1a45b61e-d133-45eb-a166-9b687fddd265
md"""

## Two schools of method



#### MCMC: sampling based 

* a bit **slow** but no compromise on accuracy
\


#### Variational inference: optimisation based 


```math
\large
q^\ast(\theta) \leftarrow \arg\min_{q} \text{KL}(q(\theta)\, \Vert\, p(\theta| \mathcal{D}))
```

* **fast** but compromise on accuracy
  * simplified assumption are made on $q$

* more popular in Machine Learning community

"""

# ╔═╡ 9e1f6c0c-fd36-4290-9971-7cf91b4230f9
# md"""

# ## Variational inference*
# """

# ╔═╡ 0688cb0f-422a-413f-87d6-aaecc04f1914
md"""

## Monte Carlo method


#### *Monte Carlo*: approximate a distribution by *empirical samples*

$$\large \theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(R)} \sim p(\theta|\mathcal D),$$

* if the sample size, ``R``, is large enough:

$$\large {\mathbb{P}}(\theta \in A|\mathcal D) \approx \frac{\#\{\theta^{(r)} \in A\}}{R},$$ 
* where ``\#\{\cdot\}`` counts how many samples falls in set ``A``.


## Monte Carlo method

#### More generally, 

$$\large \mathbb E[t(\theta)|\mathcal D] = \int t(\theta) p(\theta|\mathcal D) \mathrm{d}\theta = \frac{\int t(\theta) p(\theta) p(\mathcal D|\theta) \mathrm{d}\theta}{p(\mathcal D)}$$ 

* can also be approximated by its Monte Carlo estimation

$${\mathbb{E}}[t(\theta)|\mathcal D] \approx \frac{1}{R} \sum_{r} t(\theta^{(r)}),$$

* which is the sample average evaluated at the drawn samples



"""

# ╔═╡ e9ca80e4-4b93-4c8e-9cc7-36b5077ac7ff
md"""

## Example
"""

# ╔═╡ 461e94ea-0b29-4b60-b536-dd0bfd318070
md"""


Monte Carlo method essentially uses the *histogram* (on the right) to **replace** the true distribution



"""

# ╔═╡ 12f25671-850c-40c8-972e-5face10cf105
# md"""
# As an example, to compute the expectation of ``\theta``, the Monte Carlo estimator becomes 

# $\mathbb E[\theta] \approx \frac{1}{R} \sum_r \theta^{(r)}$

# """

# ╔═╡ ce7ea4e9-8286-49a6-b8b2-353ae7571088
begin
	approx_samples = 2000
	dist = MixtureModel([Normal(2, sqrt(2)), Normal(9, sqrt(19))], [0.3, 0.7])
	
	Random.seed!(100)
	x = rand(dist, approx_samples)
	ids = (x .< 15) .& (x .> 0)
	prob_mc=sum(ids)/approx_samples
end;

# ╔═╡ 4ad30be3-b220-4e9e-9fcf-bd1d760a2c35
let
    plt = Plots.plot(
        dist;
        xlabel=raw"$\theta$",
        ylabel=raw"$p(\theta|\mathcal{D})$",
        title="true posterior",
        fill=true,
        alpha=0.3,
        xlims=(-10, 25),
        label="",
        components=false,
    )
    Plots.vline!(plt, [mean(dist)]; label="true mean", linewidth=3)

    
	plt2 = Plots.plot(
        dist;
        xlabel=raw"$\theta$",
        fill=false,
        alpha=1,
		linewidth =3,
        xlims=(-10, 25),
        label="",
        components=false,
    )
		
		
	histogram!(plt2, x, bins=110, xlims=(-10, 25), norm=true, label="", color=1, xlabel=raw"$\theta$", title="Monte Carlo Approx.")
    Plots.vline!(plt2, [mean(x)]; label="MC mean", linewidth=3, color=2)

    Plots.plot(plt, plt2)
end

# ╔═╡ 086123c6-2088-4f9c-bea3-cb05cdd515ec
md"""


## Example (cont.)
Similarly, calculating probability, such as ``\mathbb P(0\leq \theta \leq 15)``, reduces to frequency counting: 

$$\large
\hat{\mathbb{P}}(0\leq \theta \leq 15) = \frac{\#\{0\leq \theta^{(r)}\leq 15\}}{2000} =0.905,$$  

* count the proportion of samples that falls in the area of interest
"""

# ╔═╡ 795c5695-52a4-429e-8780-a57c9767d94d
let
	plt2 = Plots.plot(
        dist;
        xlabel=raw"$\theta$",
        fill=false,
        alpha=1,
		linewidth =3,
        xlims=(-10, 25),
        label="",
        components=false,
		size=(300,450)
    )
		
		
	histogram!(plt2, x, bins = -10:0.3:25 , xlims=(-10, 25), norm=false, label="", color=1, xlabel=raw"$\theta$", title="Monte Carlo est.")

	Plots.plot!(plt2, 0:0.5:15,
        dist;
        xlabel=raw"$\theta$",
        fill=true,
		color = :orange,
    	alpha=0.5,
		linewidth =3,
        xlims=(-10, 25),
        label=L"0 \leq θ \leq 15",
        components=false,
    )

	histogram!(plt2, x[ids], bins = -10:0.3:25, xlims=(-10, 25), norm=false, label="", color=:orange,  xlabel=raw"$\theta$")

end

# ╔═╡ 194a18d2-e01a-4ff3-89ed-d0495072b4a4
md"""
## How to sample ? -- MCMC
"""

# ╔═╡ c88039d7-9ceb-496f-8d9a-363e4a2fb5ad
md"""

Bayesian inference problem's posterior can only be evaluated up to some unknown constant:

$$\Large p(\theta|\mathcal D) \propto p(\theta) p(\mathcal D|\theta),$$

* where the scalar ``1/p(\mathcal D)`` involves a nasty integration

The question we face now is how to sample

> **Problem**: how to generate samples ``\{\theta^{(r)}\}_{r=1}^R`` from a un-normalised distribution:
>
> $$\large p(\theta|\mathcal D) \propto p(\theta)p(\mathcal D|\theta)?$$

"""

# ╔═╡ 01b9fcbb-9b96-44ce-bab4-ee3ca1a40c30
md"""

*Note.* *If we apply log on the posterior distribution, the log density becomes a sum of log prior and log-likelihood:
$$\ln p(\theta|\mathcal D) = \ln p(\theta) + \ln p(\mathcal D|\theta),$$*
"""

# ╔═╡ 29bd15a0-8c28-4590-8d9f-b377d0406ece
md"""

## MCMC -- Basic idea

#### *Markov Chain Monte Carlo* (MCMC) 

\

##### *Markov Chain* 

$\large p(\theta^{(r)}|\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(r-1)}) = p(\theta^{(r)}|\theta^{(r-1)})$
  * samples are generated by a Markov Chains
  * current sample depends on the previous sample

##### *Monte Carlo*

* estimation by the Markov chain samples as illustrated in the previous section
"""

# ╔═╡ b0c9fd77-bcac-475a-adbb-a22458b79b06
md"""


## Metropolis  sampling 


**Metropolis** sampling is 
* an MCMC algorithm and 
* considered a [**top 10** algorithm of the 20th century](https://www.computer.org/csdl/magazine/cs/2000/01/c1022/13rRUxBJhBm)


## Metropolis  sampling 


##### _Objective_:

!!! infor "Objective"
	$\large \theta^{(i)} \sim p(\theta|\mathcal{D})$


##### _How_:
!!! infor "How it works?"
	It samples by simulating a **Markov chain** 
    * it needs a symmetric proposal distribution ``q(\theta|\theta')`` (a conditional distribution but I have used another letter ``q``)
      * *symmetric:* ``q(\theta|\theta') = q(\theta'|\theta)``
	* propose + reject/accept

##### _Assumption_:

!!! infor "Assumption on p(θ|𝒟)"
	The target distribution is known up to some unknown constant ``1/p(\mathcal{D})``:

	$p(\theta|\mathcal{D}) \propto p(\theta, \mathcal{D}) =p(\theta)p(\mathcal{D}|\theta)$


"""

# ╔═╡ 4803df98-1392-44bb-98d3-c832d4043086
md"""
## Metropolis sampling - the algorithm

##### Metropolis algorithm only involves _two steps_!
\


##### *Assume* the *current sample*: ``\theta_{\text{current}}``



"""

# ╔═╡ f59c929a-e610-4bbe-910b-a1f082a9fabe
md"""

----
* **proposes** a candidate ``\theta_{\text{proposed}}`` according to a proposal distribution: 

$$\theta_{\text{proposed}} \sim q(\theta|\theta_{\text{current}})$$
* **reject/accept** the proposal proportional to ratio 
  
  $a =\frac{p(\theta_{\text{proposed}}|\mathcal D)}{p(\theta_{\text{current}}|\mathcal D)}$
  * intuitively, if the proposed is *good*, *i.e.* ``a > 1`` or ``a \approx 1``
    * move to ``\theta_{\text{proposed}}``, ``\theta_{\text{current}} = \theta_{\text{proposed}}``
    * otherwise stay-put: ``\theta_{\text{current}} = \theta_{\text{current}}``
  * technically, toss a coin with bias ``a`` to decide *accept* or *reject* (``a`` is capped at 1)
* repeat the above two steps
-----


"""

# ╔═╡ a22dbb21-fce0-4ecf-8810-f553297d447d
md"""
## Remarks

The next state only depends on ``\theta_{\text{current}}`` but not before
  * therefore: a **Markov chain**!

**Key of the algorithm**: it **does not** need ``p(\mathcal{D})``

$$\frac{p(\theta_{\text{proposed}}|\mathcal D)}{p(\theta_{\text{current}}|\mathcal D)} = \frac{\frac{p(\theta_{\text{proposed}})p(\mathcal D|\theta_{\text{proposed}})}{\cancel{p(\mathcal D)}}}{\frac{p(\theta_{\text{current}})p(\mathcal D|\theta_{\text{current}})}{\cancel{p(\mathcal D)}}} = \frac{p(\theta_{\text{proposed}})p(\mathcal D|\theta_{\text{proposed}})}{p(\theta_{\text{current}})p(\mathcal D|\theta_{\text{current}})} \triangleq \frac{p^\ast(\theta_{\text{proposed}})}{p^\ast(\theta_{\text{current}})}.$$

* only need to evaluate the un-normalised posterior distribution ``p^\ast``.

```math
p^\ast(\theta) = p(\theta)p(\mathcal{D}|\theta)
```

* this is our forward generative probability

## Metropolis sampling -- pseudo code

**Input:** un-normalised target distribution

```math
\large
p^\ast(\theta) = p(\theta)p(\mathcal{D}|\theta)
```


"""

# ╔═╡ b203b068-930f-458b-a123-f0dac257ea2b
md"""

!!! infor "Metropolis algorithm"
	0. Initialise ``\theta^{(0)}`` arbitrary
	1. For ``t = 1,2,\ldots``:
	   * sample a candidate from ``q``: 
	   $$\theta' \sim q(\theta|\theta^{(t)})$$
	   * evaluate ratio ``a``, where
	   $$a = \text{min}\left \{\frac{p^\ast(\theta')}{p^\ast(\theta^{(t)}) }, 1\right \}$$
	   * set
	   $$\theta^{(t+1)} = \begin{cases} \theta' & \text{with probability }  a\\ \theta^{(t)} & \text{with probability } 1-a\end{cases}$$
"""

# ╔═╡ 159c2632-e820-4a0d-8729-c1907796e8aa
md"""

##### _Finally_, discard some initial samples as *burn-in*

"""

# ╔═╡ 07b85536-72bf-47d5-9596-24c54b5a73f5
md"""

## Question & Discussion

!!! danger "Question: sample a biased 🎲 by Metropolis"
	We want to obtain samples from a completely *biased* die that lands with threes (⚂) all the time. 

	| ``x`` | 1 | 2 | 3 | 4 | 5 | 6 | 
	| -----------| ---|---|---|--- | ---|---|
	| ``p(x)``    | 0| 0| 1| 0| 0| 0 | 

	Apply the Metropolis algorithm to generate samples from the die and use a fair die as the proposal 

	| ``x`` | 1 | 2 | 3 | 4 | 5 | 6 | 
	| -----------| ---|---|---|--- | ---|---|
	| ``q(x)``    | 1/6| 1/6| 1/6| 1/6| 1/6| 1/6 | 
	* Is the proposal symmetric?
	* What would the MCMC chains look like? (assume ``\frac{0}{0} =1``)
	* Will the chain ever converge to the target distribution? 
	* And do we need to discard any as burn-in?    

"""

# ╔═╡ 35a49bf3-753b-4341-9d65-bff4b7883248
md"""

## Example: a walk through of MH

Consider the two seller's posterior

```math
\large
p(\theta_A, \theta_B| N⁺_A, N⁺_B) \propto \underbrace{p(\theta_A, \theta_B)}_{\text{prior}} \underbrace{p(N⁺_A, N⁺_B|\theta_A, \theta_B)}_{\text{likelihood}}
```

* ``\boldsymbol{\theta} =[\theta_A, \theta_B]`` together as a random variable (vector)
* prior is ``p(\theta_A, \theta_B) \propto 1`` for ``\theta \in [0,1]^2``
* likelihood is ``p(\mathcal{D}|\theta_A, \theta_B)=\binom{N_A}{N_A^+}\theta_A^{N_A^+}(1-\theta_A)^{N_A- N_A^+}\times \binom{N_B}{N_B^+}\theta_B^{N_B^+}(1-\theta_B)^{N_B- N_B^+}``



"""

# ╔═╡ a4b0ddf5-399b-415c-bf43-73acbd82c389
md"""

##

##### The *objective* is to compute the posterior


```math
\large
p(\theta_A, \theta_B| N⁺_A, N⁺_B) = \frac{p(\theta_A, \theta_B)p(N⁺_A, N⁺_B|\theta_A, \theta_B)}{\underbrace{\int \int p(\theta_A, \theta_B)p(N⁺_A, N⁺_B|\theta_A, \theta_B)  \mathrm{d}\theta_A \mathrm{d}\theta_B}_{\text{can be nasty to compute!}}}
```


**BUT** the key thing to note is the posterior can be evaluated *proportionally*

```math
\large
p(\theta_A, \theta_B| N⁺_A, N⁺_B) \propto p(\theta_A, \theta_B)p(N⁺_A, N⁺_B|\theta_A, \theta_B)
```

"""

# ╔═╡ 12965122-0ee0-4ca8-bcab-c3a93ab05f38
# md"And the contour plot
# * each coloured line is a *level set* of the mountain (of the same height)
# "

# ╔═╡ 58d6dce4-0fc1-4f8f-adae-0ab59c8032e3

md"""
## What we have done last time*


##### We have used _discretisation_



* ``(\theta_A, \theta_B) \in \{0, 0.01, 0.02, \ldots, 1\}\times \{0, 0.01, 0.02, \ldots, 1\}.`` 
  * it means the pair can take ``101^2`` possible combinations: ``(0.0, 0.0), (0.0, 0.01), (0.0, 0.02)`` and so on.



"""

# ╔═╡ 28ba62e5-e3af-4c83-9a9b-7f7ff688b339
begin
	dis_size = 101
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
end;

# ╔═╡ f1c6942c-3acb-4eda-a8f1-db603ff72ebf
let	
	gr()
	dis_size = 11
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
 	plt = plot(xlim=[0,1], ylim=[0, 1], yticks=θ₂s, xticks=θ₁s, zlim = [0, 0.1], zaxis=false, framestyle=:origin, camera=(15,15), size=(800, 600), xlabel=L"\theta_A", ylabel=L"\theta_B", zlabel="", title="Prior by discretisation: "*L"P(\theta_A,\theta_B)")
	for (i, θ₁) in enumerate(θ₁s)
		for (j, θ₂) in enumerate(θ₁s)
			quiver!([θ₂], [θ₁], [0], quiver=([0], [0], [1/dis_size^2]), lw=3, lc=:orange, alpha=0.9)
		end
	end
	plt
end

# ╔═╡ 03abb7ac-b8b2-41bf-b6da-21af9459d3a3
function ℓ(θ, n, n⁺; logprob=false)
	# logL = n⁺ * log(θ) + (n-n⁺) * log(1-θ)
	# use xlogy(x, y) to deal with the boundary case 0*log(0) case gracefully, i.e. θ=0, n⁺ = 0
	logL = xlogy(n⁺, θ) + xlogy(n-n⁺, 1-θ)
	logprob ? logL : exp(logL)
end;

# ╔═╡ d193042c-9882-4633-8367-4698fcbfe9f8
md"""
```math
P(\theta_A, \theta_B| N⁺_A, N⁺_B) \approx \frac{P(\theta_A, \theta_B)P(N⁺_A, N⁺_B|\theta_A, \theta_B)}{\underbrace{\sum_{\theta_A\in\{0, \ldots 1.0\}} \sum_{\theta_B\in\{0, \ldots 1.0\}} P(\theta_A, \theta_B)P(N⁺_A, N⁺_B|\theta_A, \theta_B)  }_{\text{can still be nasty to compute if there are a lot!}}}
```

!!! danger "problem"
	**100** seller problem ? *e.g.* becomes impossible to calculate



"""

# ╔═╡ af87bea3-319d-48dc-80e9-e392b8d53e4e
begin
	N₁, N₂ = 10, 100
	Nₕ₁, Nₕ₂ = 8, 79
end

# ╔═╡ ddf45f75-ac6e-4f7e-ba22-a98d570b9fbd
begin
	plotly()
	true_p(θ1, θ2) = pdf(Beta(1+Nₕ₁, 1+N₁-Nₕ₁), θ1) * pdf(Beta(1+Nₕ₂, 1+N₂-Nₕ₂), θ2)

	plot(0:0.02:1, 0:0.02:1, (x,y) -> true_p(x,y), xlabel="θA", ylabel="θB", st=:surface, c=:jet, cbar=false)
end

# ╔═╡ 2006210d-96a5-410f-97e4-b2869aa23c25
begin
	gr()
	plot(0:0.02:1, 0:0.02:1, (x,y) -> true_p(x,y), xlabel=L"\theta_A", ylabel=L"\theta_B", st=:contour, c=:jet, cbar=false, ratio=1, xlim=[0,1])
end

# ╔═╡ 07842aa1-8478-4095-af45-add87f354cce
function ℓ_two_coins(θ₁, θ₂; N₁=10, N₂=100, Nh₁=8, Nh₂=79, logLik=false)
	if (θ₁ >= 0 && θ₁ <= 1.0) && (θ₂ >= 0 && θ₂ <= 1.0) 
		logℓ = ℓ(θ₁, N₁, Nh₁;  logprob=true) + ℓ(θ₂, N₂, Nh₂; logprob=true)
	else
		logℓ = -Inf
	end
	logLik ? logℓ : exp(logℓ)
end

# ╔═╡ e4060d58-82d1-4551-ac40-f1cecb3ae9fc
let	
	gr()
	dis_size = 21
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
	ℓ_twos = [ℓ_two_coins(θᵢ, θⱼ; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=true) for θᵢ in θ₁s, θⱼ in θ₂s]
	ℓ_twos_vc = exp.(ℓ_twos .- logsumexp(ℓ_twos))
	ℓ_twos_vc = ℓ_twos_vc/maximum(ℓ_twos_vc)
	cv=cgrad(:jet, rev = true)
 	plt = plot(xlim=[0,1], ylim=[0, 1], yticks=θ₂s[1:2:end], xticks=θ₁s[1:2:end],  zaxis=false, framestyle=:origin, camera=(30,15), size=(800, 600), zticks =false, xlabel=L"\theta_A", ylabel=L"\theta_B", zlabel="", title="Posterior by discretisation: " *L"P(\mathcal{D}|\theta_A,\theta_B)")
	for (i, θ₁) in enumerate(θ₁s)
		for (j, θ₂) in enumerate(θ₁s)
			quiver!([θ₂], [θ₁], [0], quiver=([0], [0], [ℓ_twos_vc[j,i]]), lw=3,lc=cv[ℓ_twos_vc[j,i]],  alpha=0.9)
		end
	end
	plt
end

# ╔═╡ bb92615b-0217-4441-bbb5-2db27c04dfcd
begin
	ℓ_twos = [ℓ_two_coins(θᵢ, θⱼ; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=true) for θᵢ in θ₁s, θⱼ in θ₂s]
end;

# ╔═╡ 178420f2-b88b-4d33-a546-cde6d6fff6d0
ps = exp.(ℓ_twos .- logsumexp(ℓ_twos));

# ╔═╡ 6b36ae1f-e967-4870-9874-37cda8101e5e
md"""

## The algorithm - step by step


##### A uniform proposal distribution

```math
q(\boldsymbol \theta|\boldsymbol \theta^{(t)}) = \texttt{Uniform}(0,1) \times \texttt{Uniform}(0,1)

```

* ``\boldsymbol \theta =[\theta_A, \theta_B]``
* independent proposal (does not depend on the current state ``\boldsymbol{\theta}^{(t)}``)
* in Julia: `θ = rand(2)`

"""

# ╔═╡ 6d8123a7-9284-4a4f-9f7b-17eaf5afd4f2
# the target distribution
posterior(θ) = ℓ_two_coins(θ[1], θ[2]; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=false)

# ╔═╡ ab99ce0c-d454-4469-9ddd-d0ca63577abd
# initial guess
θ₀ = [0.5, 0.5]

# ╔═╡ d75176a7-9e78-4cc2-ad6d-6f9dc41266cf
# make a proposal
θₚᵣₒₚ = rand(2)

# ╔═╡ 1c3755cf-7f70-47db-a3c0-c67f3bbe3421
aside(tip(md"In real implementation, we evaluate log probabilities and take the difference instead:

```math
\ln a = \ln p^\ast(\theta_{\text{proposed}}) - \ln p^\ast(\theta_{\text{current}})
```
Then transform log ratio back ``a = \exp(\ln a)``;

And remember the (log) target distribution is ``\ln p^\ast(\theta) =\ln p (\theta) + \ln p(\mathcal{D}|\theta)``
"
))

# ╔═╡ 2757834e-468c-4b89-aba6-7313b70c171c
# evaluate the proposal
a = posterior(θₚᵣₒₚ) / posterior(θ₀)

# ╔═╡ e9778f1a-f83c-4d49-8843-edacef1624fe
# decide to accept it or not: proportional to a
accept = rand() < a

# ╔═╡ 03c56812-843b-404d-a516-0b5ed45b2fbb
θₙₑₓₜ = accept ? θₚᵣₒₚ : θ₀

# ╔═╡ f497726e-6189-4068-804b-f410d65abe3b
md"""
## The algorithm - step by step

"""

# ╔═╡ 1966f960-bb6c-44a8-bd8c-e206aa58e721
@bind init_ms Button("MS restart")

# ╔═╡ 8a853105-80cc-4326-b847-1bd01d95138d
begin
	gr()
	init_ms
	θ_current = [0.2, 0.2]
	pθ_current = posterior(θ_current)
	m_samples = []
	m_plot2 =Plots.plot(θ₁s, θ₂s,  ps', st=:contour, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], colorbar=false, c=:thermal, framestyle=:origin, alpha=0.5, nlevels=8, legend=:outerright)
	scatter!([θ_current[1]],[θ_current[2]], markersize= 5, mc=:orange, markershape=:xcross,  markerstrokewidth =5, label="initial")
	scatter!([θ_current[1]],[θ_current[2]], markersize= 3.5, mc=:green, markeralpha=0.8,  label="current")
	uniform_step_samples = []
end;

# ╔═╡ 207ff041-82bb-4fe0-b9bf-90ea2165d57c
@bind go Button("MS step")

# ╔═╡ 517727d4-b271-4bb5-9b9e-9900146f8128
let
	go
	if !isempty(uniform_step_samples)
		θₚ_step = rand(2)
		a_step = posterior(θₚ_step) / posterior(θ_current)
		if rand() < a_step
			scatter!(m_plot2, [θₚ_step[1]],[θₚ_step[2]], markersize= 3.5, mc=:green, markeralpha=0.8, label="")
			plot!(m_plot2, [θ_current[1], θₚ_step[1]], [θ_current[2],θₚ_step[2]], st=:path, lc=:green, la=0.5, label=false, title="")
			θ_current .= θₚ_step
			# pθ_current = pθₚ
		else
			scatter!(m_plot2, [θₚ_step[1]],[θₚ_step[2]], markersize= 3, mc=:red, markeralpha=0.3, label="")
			plot!(m_plot2, [θ_current[1], θₚ_step[1]], [θ_current[2], θₚ_step[2]], st=:path, lc=:red, linestyle=:dot, la=0.1, label=false, title="")
		end

	else

	end
	push!(uniform_step_samples, θ_current)
	m_plot2
end

# ╔═╡ 4dfb3123-0872-436a-8db4-772c500e6b6d
function metropolis_sampler_two_sellers(ℓπ, mc=500; dim=2, proposal = :uniform, x0=[0.5, 0.5], σs = [0.01, 0.01])
	samples = zeros(dim, mc)
	proposed = zeros(dim, mc-1)
	acceptance = Array{Bool, 1}(undef, mc-1)
	fill!(acceptance, false)
	samples[:, 1] = x0 # store the initial for presentation purpose!
	ℓx0 = ℓπ(x0) 
	accepted = 0
	for i in 2:mc
		if proposal == :uniform
			xstar = rand(dim)
		else
			xstar = x0 + σs .* randn(dim)
		end	
		proposed[:, i-1] = xstar
		ℓxstar = ℓπ(xstar)
		r = exp(ℓxstar - ℓx0) 
		# if accepted
		if rand() < r
			x0 = xstar
			ℓx0 = ℓxstar
			accepted += 1
			acceptance[i-1] = true
		end
		@inbounds samples[:, i] = x0
	end
	accpt_rate = accepted / (mc-1)
	return samples, proposed, acceptance, accpt_rate
end

# ╔═╡ 83b02029-9001-465b-a048-90b41d7ab065
begin
	ℓπ(θ) = ℓ_two_coins(θ[1], θ[2]; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=true)
end

# ╔═╡ e7dd3d77-1d62-4a6e-987a-1e06267bd793
md"""
## Demonstration

"""

# ╔═╡ a1c64a2e-8ae1-41bd-8661-b0c782f97080
spls, _, acc, accr = metropolis_sampler_two_sellers(ℓπ, 20000);

# ╔═╡ 85e25392-4216-43c2-a065-c56ba4d06b57
# discard the first 1000 as burn-in 
naive_chain = Chains(spls[:, 1000:1:end]', [:θA, :θB]);

# ╔═╡ f162477e-0463-49d3-8acf-e06cefd7553a
plot(naive_chain[1:500, :, :])

# ╔═╡ a074f9da-e2dd-4e29-a6d0-2939fe244746
md"""

A lot of rejections (acceptance rate less than 0.05!)
* once rejected, the current value is used again and again until a new state accepted
* the chain is very "**sticky**"


"""

# ╔═╡ 4f4fbde5-dbe4-4269-971a-56adf3cc0923
md"""

## Demonstration (conti.)
"""

# ╔═╡ a484bb4d-0dd9-4177-a123-ea2a54c3aa3c
corner(naive_chain, xlim=[0,1], ylim=[0,1])

# ╔═╡ 5345144e-7dab-4b6e-b591-6fac79ae93d4
md"Marginal distributions ``P(\theta_A|\mathcal{D}), P(\theta_B|\mathcal{D})``

* the dashed lines are the exact posterior density
* the solid lines are the MCMC's estimation

* not very good but close
"

# ╔═╡ d9d113fb-8930-47e8-872e-0aec83c20468
md"""
Nevertheless, not ideal but it kinda works 
* there are remedies, say thinning the chain or run multiple chains in parallel
* but sampling efficiency can be improved!


"""

# ╔═╡ 87461e6b-9759-4a8c-b10f-d56caa8cc576
let
	density(naive_chain[:,1,:], lw=2, label="MCMC " *L"P(\theta_A|\mathcal{D})", legend=:topleft)
	density!(naive_chain[:,2,:], lw=2, label="MCMC " *L"P(\theta_A|\mathcal{D})")
	
	plot!(Beta(Nₕ₁+1, N₁ - Nₕ₁+1), lw=1.5, lc=1, ls=:dash, label=L"P(\theta_A|\mathcal{D})")
	plot!(Beta(Nₕ₂+1, N₂ - Nₕ₂+1), lw=1.5, lc=2, ls=:dash, label=L"P(\theta_B|\mathcal{D})")
end

# ╔═╡ e92c05b7-4282-417e-99d9-30809bebe9fb
md"""

## Improvement


We can try other proposal distributions, *e.g.* Gaussian


```math
q(\boldsymbol \theta|\boldsymbol \theta^{(t)}) = q({\theta}_A|\theta_{A}^{(t)}) q(\theta_B|\theta_B^{(t)})=\mathcal{N}(\theta_A; \theta_A^{(t)}, \sigma^2_A) \mathcal{N}(\theta_B; \theta_B^{(t)}, \sigma^2_B)
```
* Gaussian proposals depends on the current location ``\boldsymbol \theta^{(t)} =[\theta_A^{(t)}, \theta_B^{(t)}]``
* with fixed variances ``\sigma^2_A, \sigma^2_B``
  * smaller ``\sigma_{A}^2, \sigma^2_B`` means the proposed will be closer to ``\boldsymbol \theta_{\text{current}}``
"""

# ╔═╡ b047ac1e-8283-47db-b0ae-54635b523e4b
md"""
## Demonstration
"""

# ╔═╡ 40492daa-594d-43cb-b6f7-5749c6771d06
md"""

Note the the proposed are all very close to the current state!
* rather than emerging everywhere within the unit box (when uniform proposal is used)
* more **systematic exploration**

* as a rule of thumb, acceptance rate should be between 0.2 to 0.6
"""

# ╔═╡ 47a4d597-b719-47d2-b66f-66e96acda02b
md"Starting location:"

# ╔═╡ 899104d2-a2c0-4d10-a178-d5edda7cb6b4
md"The proposals' standard deviation, *i.e.* ``\sqrt{\sigma^2}``"

# ╔═╡ eb6f9708-cf48-4189-9b2d-251eabd8f536
σa, σb = 0.1, 0.1

# ╔═╡ 5cb14b9c-7681-4f7d-a0c1-19730511f9de
@bind init_ms_gaussian Button("MS with Gaussian proposal restart")

# ╔═╡ 50cd5cb4-2a35-4a41-966e-0280537cfd21
begin
	init_ms_gaussian
	θ_init_gaussian = [0.2, 0.2]
end

# ╔═╡ 2e6e5618-7316-4b1c-a2be-7eed76913287
begin
	init_ms_gaussian
	θ_current_gaussian = θ_init_gaussian
	pθ_current_gaussian = posterior(θ_current_gaussian)
	# m_samples_gaussian = []
	m_plot2_gaussian = Plots.plot(θ₁s, θ₂s,  ps', st=:contour, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], colorbar=false, c=:thermal, framestyle=:origin, alpha=0.5, nlevels=8, legend=:outerright)
	scatter!([θ_current_gaussian[1]],[θ_current_gaussian[2]], markersize= 5, mc=:orange, markershape=:xcross,  markerstrokewidth =5, label="initial")
	scatter!([θ_current_gaussian[1]],[θ_current_gaussian[2]], markersize= 3.5, mc=:green, markeralpha=0.8,  label="current")
	ms_gaussian_samples = []
end;

# ╔═╡ d1967e69-b4cb-4fcc-816b-81bbe3e5a1e0
@bind go_gaussian Button("MS step with Gaussian proposal")

# ╔═╡ bc832ff2-6930-4903-b5eb-41e01deac3f7
let
	go_gaussian
	if !isempty(ms_gaussian_samples)
		θₚ_gaussian = θ_current_gaussian + [σa * randn(), σb * randn()]
		a_g_step = posterior(θₚ_gaussian) / posterior(θ_current_gaussian)
		if rand() < a_g_step
			scatter!(m_plot2_gaussian, [θₚ_gaussian[1]],[θₚ_gaussian[2]], markersize= 3.5, mc=:green, markeralpha=0.8, label="")
			plot!(m_plot2_gaussian, [θ_current_gaussian[1], θₚ_gaussian[1]], [θ_current_gaussian[2],θₚ_gaussian[2]], st=:path, lc=:green, la=0.5, label=false, title="")
			θ_current_gaussian .= θₚ_gaussian
		else
			scatter!(m_plot2_gaussian, [θₚ_gaussian[1]],[θₚ_gaussian[2]], markersize= 3, mc=:red, markeralpha=0.3, label="")
			plot!(m_plot2_gaussian, [θ_current_gaussian[1], θₚ_gaussian[1]], [θ_current_gaussian[2], θₚ_gaussian[2]], st=:path, lc=:red, linestyle=:dot, la=0.1, label=false, title="")
		end
	else
		
	end
	push!(ms_gaussian_samples, θ_current_gaussian)
	m_plot2_gaussian
end

# ╔═╡ 30d9c1ca-e2b9-4e0a-8c32-908ed26cf13c
begin
	Random.seed!(123)	
	spls_gaussian, _, _, _ = metropolis_sampler_two_sellers(ℓπ, 20000; proposal = :gaussian, x0 = [0.5, 0.5], σs = [0.08, 0.05])
end;

# ╔═╡ bb5440bd-b931-420d-91b1-16e54d00de4e
gaussian_chain = Chains(spls_gaussian[:, 100:5:end]', [L"θ_A", L"θ_B"]);

# ╔═╡ 77865cbd-8ebd-440d-83fd-26f2f91fa76e
plot(gaussian_chain)

# ╔═╡ 78f331f8-2607-4ed3-8fac-6ded7b575251
let
	# density(gaussian_chain[:,1,:], label="seller A", title="Gaussian proposal")
	# density!(gaussian_chain[:,2,:], label="seller B")

	density(gaussian_chain[:,1,:], lw=2, label="MCMC with Gaussian proposal " *L"P(\theta_A|\mathcal{D})", legend=:topleft)
	density!(gaussian_chain[:,2,:], lw=2, label="MCMC with Gaussian proposal " *L"P(\theta_A|\mathcal{D})")
	
	plot!(Beta(Nₕ₁+1, N₁ - Nₕ₁+1), lw=1.5, lc=1, ls=:dash, label="Exact "*L"P(\theta_A|\mathcal{D})")
	plot!(Beta(Nₕ₂+1, N₂ - Nₕ₂+1), lw=1.5, lc=2, ls=:dash, label="Exact "*L"P(\theta_B|\mathcal{D})")
end

# ╔═╡ c65841ca-bf38-48fe-a2b4-0f6e9b5fd874
let
	gr()
	scatter([gaussian_chain[1:5:end,1,:]], gaussian_chain[1:5:end,2,:], xlim=[0,1], ylim=[0,1],  label="", ratio=1, alpha=0.3, markersize=3, xlabel=L"\theta_A", ylabel=L"\theta_B")
	plot!((x) -> x, lw=1, lc=:gray, label="")
	equalline = Shape([(0., 0.0), (1,1), (1, 0)])
	plot!(equalline, fillcolor = plot_color(:gray, 0.2), label=L"\theta_A>\theta_B", legend=:bottomright)
end

# ╔═╡ bf130cdd-773c-4737-9868-1516311221d9
md"""


## Metropolis Hastings

An extension of Metropolis sampling

* asymmetric proposal distribution can be used

!!! infor "Metropolis-Hastings algorithm"
	0. Initialise ``\theta^{(0)}`` arbitrary
	1. For ``r = 1,2,\ldots``:
	   * sample a candidate value from ``q``: 
	   $$\theta' \sim q(\theta|\theta^{(r)})$$
	   * evaluate ``a``, where
	   $$a = \text{min}\left \{\frac{p^\ast(\theta')q(\theta^{(t)}|\theta')}{p^\ast(\theta^{(t)}) q(\theta'|\theta^{(t)})}, 1\right \}$$
	   * set
	   $$\theta^{(t+1)} = \begin{cases} \theta' & \text{with probability }  a\\ \theta^{(t)} & \text{with probability } 1-a\end{cases}$$
"""

# ╔═╡ ea2f9db8-e93e-48c2-aa50-95310972502b
md"""


*Remark*: when the proposal distribution is symmetric, i.e.

$$q(\theta^{(t)}|\theta')= q(\theta'|\theta^{(t)}),$$ the acceptance probablity reduces to

$$a = \text{min}\left \{\frac{p^\ast(\theta')}{p^\ast(\theta^{(t)}) }, 1\right \}$$ 


* the algorithm is called **Metropolis** algorthm

"""

# ╔═╡ da7a35af-1f07-40a4-8257-a078545e23e6
md"""
## Appendix

"""

# ╔═╡ df9a7d0c-ad9f-4829-8239-f8cc652771e7
begin

function produce_anim(ℓπ, chain, proposed, acpt, rate; mc = 200, xlim=[-4, 4], ylim = [-4, 4], title="Metropolis demonstration; ", with_accpt_rate=true)
	if with_accpt_rate
		title = title * "acceptance rate=$(rate)"
	end
	plt_ = plot(range(xlim[1], xlim[2], length= 100), range(ylim[1], ylim[2], length= 100), (x, y)-> exp(ℓπ([x,y])), legend = :none, st=:contour, linewidth=0.5, la=0.5, levels=5, title=title)

	anim = @animate for i  in 1:mc
		scatter!(plt_, (chain[1, i], chain[2, i]),
			 label=false, mc=:green, ma=0.5)
		if !acpt[i]
			scatter!(plt_, (proposed[1, i], proposed[2, i]), label=false, mc =:red, ma=0.4)
			plot!([chain[1, i], proposed[1, i]], [chain[2, i], proposed[2, i]], st=:path, lc=:red, linestyle=:dot, la=0.5, label=false)
		end
		plot!(plt_, chain[1, i:i+1], chain[2, i:i+1], st=:path, lc=:green, la=0.5, label=false)
	end
	
	return anim
end

end

# ╔═╡ cd0ab47e-4b0f-4240-93ae-81dad54209f3
begin

function demo_mh_gif(ℓπ; proposal =:gaussian, σs = [0.01, 0.01], mc = 200, x₀= [0.5, 0.5], xlim=[0, 1], ylim = [0, 1])
	Random.seed!(100)
	chain, proposed, acpt, rate = metropolis_sampler_two_sellers(ℓπ, mc+1; proposal =proposal, x0 = x₀, σs = σs)
	anim = produce_anim(ℓπ, chain, proposed, acpt, rate; mc = mc, xlim=xlim, ylim = ylim)
	return anim
end

	
end;

# ╔═╡ 11280ac3-7289-4677-8a88-45130ce2a4d0
anim_ms_uniform = demo_mh_gif(ℓπ; proposal =:uniform,  mc = 200, x₀= [0.2, 0.2]);

# ╔═╡ 95768d9e-e81a-417f-8b1b-846e9fd6e5e3
gif(anim_ms_uniform, fps=10)

# ╔═╡ 793de098-feb6-4b1f-bd01-a20c0bbe3813
begin
	ms_gaussian = demo_mh_gif(ℓπ; proposal =:gaussian, σs = [0.08, 0.05], mc = 200, x₀= [0.2, 0.2]);
end

# ╔═╡ d1df764c-38f9-43e6-bb64-e31415456d28
gif(ms_gaussian, fps=10)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.132"
LaTeXStrings = "~1.4.1"
Latexify = "~0.16.12"
MCMCChains = "~7.8.1"
Plots = "~1.41.7"
PlutoTeachingTools = "~0.5.0"
PlutoUI = "~0.7.83"
StatsFuns = "~2.2.1"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "c0688e1ee4817dce50e950fb8e6b47c6d49bcce1"

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

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Dates", "Distributed", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "UUIDs"]
git-tree-sha1 = "fbdfe3aff511a3a655073e89ac063f1703a06e1b"
registries = "General"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.17.0"

    [deps.AbstractMCMC.extensions]
    AbstractMCMCOnlineStatsExt = "OnlineStats"
    AbstractMCMCTensorBoardLoggerExt = "TensorBoardLogger"

    [deps.AbstractMCMC.weakdeps]
    OnlineStats = "a15396b6-48d5-5d58-9928-6d29437db91e"
    TensorBoardLogger = "899adc3e-224a-11e9-021f-63837185c80f"

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

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
registries = "General"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

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

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
registries = "General"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
registries = "General"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

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

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
registries = "General"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
registries = "General"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

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

[[deps.IntervalSets]]
git-tree-sha1 = "0db6aea5b64caa1c47e9fdd27394f0296f05e8bf"
registries = "General"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.15"

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

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

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "d4816abce26971e3237b46a47b99991f306e4832"
registries = "General"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.3.0"

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

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "d9625f27ded4ad726ceca7819394a4cc77ed25b3"
registries = "General"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.2.0"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
registries = "General"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "DataAPI", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "72a2ab7fdfd25a789da2d2d1448f99e6728aaa9a"
registries = "General"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "7.8.1"

    [deps.MCMCChains.extensions]
    MCMCChainsCSVExt = "CSV"
    MCMCChainsJSONExt = "JSON"

    [deps.MCMCChains.weakdeps]
    CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "345bedeaf7d650f673fefa042dd8384c63de6c68"
registries = "General"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.19"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLJModelInterface]]
deps = ["InteractiveUtils", "REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c275fae2e693206b4527dd9d2382aa15359ef3ed"
registries = "General"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.12.1"

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

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
registries = "General"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "99fd8f32ab6728ff205cddd24e2d9ea6cbb498d3"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

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

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
registries = "General"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
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

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
registries = "General"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

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

[[deps.ScientificTypesBase]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "e785eaa35a0f5518a388f9010e66fda64ea95ede"
registries = "General"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.1.0"

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

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "d3ddfea0085bf97e01fbe8dadf83df7d60a0aa43"
registries = "General"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.6.0"

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

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
registries = "General"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

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

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "81c9b4137edfe56a56efcdcb35d721b2ce3e2416"
registries = "General"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.8"

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
# ╟─dbdb0950-4a33-11ed-39a0-6b2ea3d774e6
# ╟─89f53893-93f1-49d3-b643-2aee7e6f4610
# ╟─b5e1f496-02e1-4352-9912-ff7bf4ea622c
# ╟─d1b809fb-e381-49ac-bfa4-77c12f3ccf17
# ╟─2f5f8005-fda3-4a07-9c39-969ca4417f2c
# ╟─0e163261-577e-4f0d-8a39-a3748a9623b5
# ╟─405a783c-1b82-4a3d-b984-070e78f75c13
# ╟─1a45b61e-d133-45eb-a166-9b687fddd265
# ╟─9e1f6c0c-fd36-4290-9971-7cf91b4230f9
# ╟─0688cb0f-422a-413f-87d6-aaecc04f1914
# ╟─e9ca80e4-4b93-4c8e-9cc7-36b5077ac7ff
# ╟─461e94ea-0b29-4b60-b536-dd0bfd318070
# ╟─4ad30be3-b220-4e9e-9fcf-bd1d760a2c35
# ╟─12f25671-850c-40c8-972e-5face10cf105
# ╟─ce7ea4e9-8286-49a6-b8b2-353ae7571088
# ╟─086123c6-2088-4f9c-bea3-cb05cdd515ec
# ╟─795c5695-52a4-429e-8780-a57c9767d94d
# ╟─194a18d2-e01a-4ff3-89ed-d0495072b4a4
# ╟─c88039d7-9ceb-496f-8d9a-363e4a2fb5ad
# ╟─01b9fcbb-9b96-44ce-bab4-ee3ca1a40c30
# ╟─29bd15a0-8c28-4590-8d9f-b377d0406ece
# ╟─b0c9fd77-bcac-475a-adbb-a22458b79b06
# ╟─4803df98-1392-44bb-98d3-c832d4043086
# ╟─f59c929a-e610-4bbe-910b-a1f082a9fabe
# ╟─a22dbb21-fce0-4ecf-8810-f553297d447d
# ╟─b203b068-930f-458b-a123-f0dac257ea2b
# ╟─159c2632-e820-4a0d-8729-c1907796e8aa
# ╟─07b85536-72bf-47d5-9596-24c54b5a73f5
# ╟─35a49bf3-753b-4341-9d65-bff4b7883248
# ╟─a4b0ddf5-399b-415c-bf43-73acbd82c389
# ╟─ddf45f75-ac6e-4f7e-ba22-a98d570b9fbd
# ╟─2006210d-96a5-410f-97e4-b2869aa23c25
# ╟─12965122-0ee0-4ca8-bcab-c3a93ab05f38
# ╟─58d6dce4-0fc1-4f8f-adae-0ab59c8032e3
# ╟─28ba62e5-e3af-4c83-9a9b-7f7ff688b339
# ╟─f1c6942c-3acb-4eda-a8f1-db603ff72ebf
# ╟─e4060d58-82d1-4551-ac40-f1cecb3ae9fc
# ╟─03abb7ac-b8b2-41bf-b6da-21af9459d3a3
# ╟─d193042c-9882-4633-8367-4698fcbfe9f8
# ╟─af87bea3-319d-48dc-80e9-e392b8d53e4e
# ╟─07842aa1-8478-4095-af45-add87f354cce
# ╟─bb92615b-0217-4441-bbb5-2db27c04dfcd
# ╟─178420f2-b88b-4d33-a546-cde6d6fff6d0
# ╟─6b36ae1f-e967-4870-9874-37cda8101e5e
# ╟─6d8123a7-9284-4a4f-9f7b-17eaf5afd4f2
# ╠═ab99ce0c-d454-4469-9ddd-d0ca63577abd
# ╠═d75176a7-9e78-4cc2-ad6d-6f9dc41266cf
# ╟─1c3755cf-7f70-47db-a3c0-c67f3bbe3421
# ╠═2757834e-468c-4b89-aba6-7313b70c171c
# ╠═e9778f1a-f83c-4d49-8843-edacef1624fe
# ╠═03c56812-843b-404d-a516-0b5ed45b2fbb
# ╟─f497726e-6189-4068-804b-f410d65abe3b
# ╟─8a853105-80cc-4326-b847-1bd01d95138d
# ╟─1966f960-bb6c-44a8-bd8c-e206aa58e721
# ╟─207ff041-82bb-4fe0-b9bf-90ea2165d57c
# ╟─517727d4-b271-4bb5-9b9e-9900146f8128
# ╟─4dfb3123-0872-436a-8db4-772c500e6b6d
# ╟─83b02029-9001-465b-a048-90b41d7ab065
# ╟─e7dd3d77-1d62-4a6e-987a-1e06267bd793
# ╟─11280ac3-7289-4677-8a88-45130ce2a4d0
# ╟─95768d9e-e81a-417f-8b1b-846e9fd6e5e3
# ╠═a1c64a2e-8ae1-41bd-8661-b0c782f97080
# ╠═85e25392-4216-43c2-a065-c56ba4d06b57
# ╟─f162477e-0463-49d3-8acf-e06cefd7553a
# ╟─a074f9da-e2dd-4e29-a6d0-2939fe244746
# ╟─4f4fbde5-dbe4-4269-971a-56adf3cc0923
# ╟─a484bb4d-0dd9-4177-a123-ea2a54c3aa3c
# ╟─5345144e-7dab-4b6e-b591-6fac79ae93d4
# ╟─d9d113fb-8930-47e8-872e-0aec83c20468
# ╟─87461e6b-9759-4a8c-b10f-d56caa8cc576
# ╟─e92c05b7-4282-417e-99d9-30809bebe9fb
# ╟─b047ac1e-8283-47db-b0ae-54635b523e4b
# ╟─40492daa-594d-43cb-b6f7-5749c6771d06
# ╟─2e6e5618-7316-4b1c-a2be-7eed76913287
# ╟─47a4d597-b719-47d2-b66f-66e96acda02b
# ╟─50cd5cb4-2a35-4a41-966e-0280537cfd21
# ╟─899104d2-a2c0-4d10-a178-d5edda7cb6b4
# ╟─eb6f9708-cf48-4189-9b2d-251eabd8f536
# ╟─5cb14b9c-7681-4f7d-a0c1-19730511f9de
# ╟─d1967e69-b4cb-4fcc-816b-81bbe3e5a1e0
# ╟─bc832ff2-6930-4903-b5eb-41e01deac3f7
# ╠═793de098-feb6-4b1f-bd01-a20c0bbe3813
# ╟─d1df764c-38f9-43e6-bb64-e31415456d28
# ╠═30d9c1ca-e2b9-4e0a-8c32-908ed26cf13c
# ╠═bb5440bd-b931-420d-91b1-16e54d00de4e
# ╟─77865cbd-8ebd-440d-83fd-26f2f91fa76e
# ╟─78f331f8-2607-4ed3-8fac-6ded7b575251
# ╟─c65841ca-bf38-48fe-a2b4-0f6e9b5fd874
# ╟─bf130cdd-773c-4737-9868-1516311221d9
# ╟─ea2f9db8-e93e-48c2-aa50-95310972502b
# ╟─da7a35af-1f07-40a4-8257-a078545e23e6
# ╠═cd0ab47e-4b0f-4240-93ae-81dad54209f3
# ╠═df9a7d0c-ad9f-4829-8239-f8cc652771e7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
