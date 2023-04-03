### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
begin
	using PlutoTeachingTools
	# using PlutoUI
	using Plots
	using Distributions, LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	using SpecialFunctions
	# using Statistics
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	# using Dagitty
	using PlutoUI
end

# ╔═╡ 45f045d9-cdb7-4ade-a8e6-1f1a984cc58a
present_button()

# ╔═╡ be719382-bf1f-4442-8419-bddcda782525
TableOfContents()

# ╔═╡ 77535564-648a-4e17-83a0-c562cc5318ec
md"""

# Bayesian modelling


**Lecture 3. more Bayesian inference** 


$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang (lf28@st-andrews.ac.uk)

*School of Computer Science*

*University of St Andrews, UK*

*March 2023*

"""

# ╔═╡ 2057c799-18b5-4a0f-b2c7-66537a3fbe79
md"""

## In this lecture

* Prior distribution choices
  * conjugate priors
  * some common distributions for priors


* Predictive distributions and their checks 
  * prior predictive check
  * posterior predictive check


  







"""

# ╔═╡ c677a2d2-f4be-483a-a53b-7b76efb1b80f
md"""

## Priors are useful

Recall the coughing example

* ``H``: the unknown cause ``h \in \mathcal{A}_H=\{\texttt{cold}, \texttt{cancer}, \texttt{healthy}\}`` 

* ``\mathcal{D}``: the observed data is ``\mathcal{D} = \texttt{Cough}``
 
and we have used the **Prior:** ``P(H)``

$$P(h) = \begin{cases} 0.8 & h=\texttt{healthy} \\
0.18 & h=\texttt{cold}\\ 
0.02 & h=\texttt{cancer}\end{cases}$$


This is an **informative** prior

* instead of a non-informative (say uniform distribution)
* without the prior, the inference results would be wrong (it would be proportional to the likelihood)

"""

# ╔═╡ 56f2bb33-637c-4450-9769-f338937594da
md"""

## Priors are useful


Priors serve useful functions in Bayesian inference
* to formally incorporate expert knowledge into the statistical inference
* regularisation effects 
  * prevents overfitting
  * priors contains information from the unseen data 

"""

# ╔═╡ 47359bc3-cfc2-4777-82f9-308e56dca491
md"""

## How to choose priors



The question is how one should choose priors for the unknown parameters


The first and foremost principle is 

!!! note ""
	**Priors with matching support**

* that is, the prior distribution has the correct *support* (or domain) of the unknown parameters. 

**Example** For the coin-flipping example

* the unknown bias ``\theta`` has a support: ``\theta \in [0,1].`` 

* therefore, a standard Gaussian distribution is not suitable


## Suitable priors for ``\theta \in [0,1]``
Some suitable priors are 

* Uniform distribution between 0 and 1, 
* truncated Gaussian (truncated between 0 and 1) 
* or Beta distribution. The probability density functions are listed below together with their plots.

```math

\begin{align}
p(\theta) &= \texttt{TruncNormal}(\mu, \sigma^2) \propto \begin{cases} \mathcal N(\mu, \sigma^2) & {0\leq \theta \leq 1} \\ 0 & \text{otherwise}  \end{cases}  \\
p(\theta) &= \texttt{Uniform}(0,1) = \begin{cases} 1 & {0\leq \theta \leq 1} \\ 0 & \text{otherwise}  \end{cases} 

\end{align}
```
"""

# ╔═╡ 35311403-187a-4e23-9e52-7eb0a212452d
let
	σ = sqrt(0.1)
	Plots.plot(TruncatedNormal(0.5, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}" *"(0.5, $(round((σ^2);digits=2)))", legend=:outerright, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"θ\in [0,1]")
	Plots.plot!(TruncatedNormal(0.25, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.25, $(round((σ^2);digits=2))))")
	Plots.plot!(TruncatedNormal(0.75, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.75, $(round((σ^2);digits=2))))")
	Plots.plot!(Uniform(0,1), lw=2, label=L"\texttt{Uniform}(0, 1)")
end

# ╔═╡ 9fa2c7bf-3fa2-4869-8515-fa0048fa3cd6
md"""

## Suitable priors for ``\theta > 0``


Suppose we want to infer the variance ``\sigma^2`` of a Gaussian sample

As the ``\sigma^2`` is a positive number: ``\sigma^2 >0``. 


Some possible priors on the positive real line are 

* e.g. Exponential distribution
* Gamma
* or Half-Cauchy

"""

# ╔═╡ ae234663-90b4-432d-bacb-188dd1ee1034
let
	xpltlim1= -1
	xpltlim2 = 6
	Plots.plot(xpltlim1:0.01:xpltlim2, Exponential(1),lw=1.5,  label=L"\texttt{Exponential}(1.0)", legend=:best, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"σ^2\in (0,∞)")
	Plots.plot!(xpltlim1:0.01:xpltlim2, Exponential(2),lw=1.5,  label=L"\texttt{Exponential}(2.0)")
	Plots.plot!(xpltlim1:0.01:xpltlim2, Exponential(5), lw=1.5, label=L"\texttt{Exponential}(5.0)")
	Plots.plot!(xpltlim1:0.01:xpltlim2, truncated(Cauchy(0, 1), lower= 0), lw=1.5, label=L"\texttt{HalfCauchy}(0, 1)")
	# Plots.plot!(0:0.1:10, truncated(Cauchy(0, 2), lower= 0), label="HalfCauchy(0, 2)")
	# Plots.plot!(0:0.1:10, truncated(Cauchy(0, 5), lower= 0), label="HalfCauchy(0, 5)")
end

# ╔═╡ d764a435-7e11-4736-a0d0-0f2d5265b7af
md"""

## Conjugate prior


Conjugate priors are priors 

* such that posterior distribution is of the functional form
  * we will see an example soon
* computational cheap and simple 
  * as the posterior will be of the closed form (usually)
  * the inference/computation is just updating prior parameters



Unfortunately, conjugate priors only exist in some very simple Bayesian models
* not that important for the practical sense
* but provides us insights into the purpose of priors


## Example: Beta-Bernoulli model


For the coin flipping example, the conjugate prior of the bias ``\theta`` is 

```math
p(\theta) = \texttt{Beta}(\theta; a_0, b_0) = \frac{1}{\text{B}(a_0, b_0)} \theta^{a_0-1}(1-\theta)^{b_0-1},
```
* ``a_0,b_0 >0`` are the prior's parameter 
* and ``B(a_0,b_0)``, the beta function, is a normalising constant for the Beta distribution: 

* *i.e.* ``\mathrm{B}(a_0, b_0) = \int \theta^{a_0-1}(1-\theta)^{b_0-1}\mathrm{d}\theta``. 


"""

# ╔═╡ 5aaf5221-e77b-4339-95c5-dc21bd815fc3
md"""

## Beta distribution


A few Beta distributions with different parameterisations are plotted below. 

* ``a_0=b_0=1``, the prior reduces to a uniform distribution. 
* ``a_0> b_0``, e.g. ``\texttt{Beta}(5,2)``, the prior belief has its mode > 0.5,

  * which implies the prior believes the coin is biased towards the head; 




* ``a_0< b_0`` and vice versa.
"""

# ╔═╡ 047a8e19-54d4-454a-8635-8becd4b93061
let
	plot(Beta(1,1), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=1,b_0=1)", linewidth=2, legend=:outerright, size=(600,300))	
	plot!(Beta(0.5,0.5), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=.5,b_0=.5)", linewidth=2)
	plot!(Beta(5,5), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=2,b_0=2)", linewidth=2)
	plot!(Beta(5,2), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=5,b_0=2)", linewidth=2)
	plot!(Beta(2,5), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=2,b_0=5)", linewidth=2)
end

# ╔═╡ 43cedf4c-65c4-4b2b-afb3-21f5827f2af6
md"""
## Example: Beta-Bernoulli model (cont.)


A classic statistical inference problem

> A coin 🪙 is tossed 10 times. 
> * the tossing results are recorded:  $\mathcal D=\{1, 1, 1, 0, 1, 0, 1, 1, 1, 0\}$; 7 out of 10 are heads
> * is the coin **fair**?


"""

# ╔═╡ 4e82a77d-af72-4653-aaa9-f7303e33c3ea
md"""

## Example: Beta-Bernoulli model (cont.)


Recall the likelihood is 


```math
p(\mathcal{D}|\theta) = \prod_i p(d_i|\theta) =\theta^{\sum_n d_n} (1-\theta)^{N-\sum_n d_n} = \theta^{N_h} (1-\theta)^{N-N_h}
```

* assume ``n`` independent Bernoulli trial
* ``N_h = \sum_i d_i`` is the total number of heads observed in the ``N`` tosses.



Apply Baye's rule, we will find the posterior is still of a Beta form (therefore, the **conjugacy**)

```math
p(\theta|\mathcal D) = \texttt{Beta}(\theta; a_N, b_N) = \frac{1}{\text{B}(a_N, b_N)} \theta^{a_N-1}(1-\theta)^{b_N-1},
```

* where ``a_N= a_0 + N_h`` and ``b_N = b_0 + N - N_h``; and 


"""

# ╔═╡ e2673877-6536-4670-b5b7-b84c36503512
Foldable("Details about Beta-Binomial model's conjugacy", md"
Apply Baye's rule, we have 

```math
\begin{align}
p(\theta|\mathcal D) &\propto p(\theta) p(\mathcal D|\theta) \\
&= \frac{1}{\text{B}(a_0, b_0)} \theta^{a_0-1}(1-\theta)^{b_0-1} \theta^{\sum_n d_n} (1-\theta)^{N-\sum_n d_n}\\
&= \theta^{a_0+ \sum_{n} d_n -1}(1-\theta)^{b_0+N-\sum_n d_n -1}\\
&= \theta^{a_N -1}(1-\theta)^{b_N -1},
\end{align} 
```
where ``a_N= a_0 + N_h`` and ``b_N = b_0 + N - N_h``.
Next we needs to normalise the non-normalised posterior to find the exact posterior distribution. The normalising constant is:

$$\int_{0}^1 \theta^{a_N -1}(1-\theta)^{b_N -1} d\theta$$

We recognise the unnormalised distribution is a Beta distribution with the updated parameters ``a_N= a_0 + N_h`` and ``b_N = b_0 + N - N_h``; the normalising constant must be ``B(a_N, b_N)``. Therefore, 

$$p(\theta|\mathcal D) =\text{Beta}(a_N, b_N).$$
")

# ╔═╡ 33714d50-a04e-41ad-844c-293ed9671163
md"""

## Example (cont.)


For our example, 

* we have observed ``N_h = 7`` number of heads in the 10 tosses. 


* if a uniform prior is used ``p(\theta)= \texttt{Beta}(1, 1),``

The updated posterior is then

$$p(\theta|\mathcal D)= \texttt{Beta}(1+7, 1+3).$$



"""

# ╔═╡ 9308405d-36d0-41a1-8c73-e93b8d699320
begin
	𝒟 = [1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
	# 𝒟 = [0, 0]
end

# ╔═╡ c32714b7-fa1d-443c-a5a2-4a511fc701bd
let
	nh, nt = sum(𝒟), length(𝒟) - sum(𝒟)
	plot(Beta(1,1), xlims=[0,1], label=L"p(\theta)= \texttt{Beta}(1,1)", linewidth=1, xlabel=L"\theta", ylabel="density" ,fill= true, lw=2, alpha=0.2, legend=:outerright, color=1, title="Conjugate posterior update")	
	vline!([mean(Beta(1,1))], label="prior mean", lw=2, lc=1, ls=:dash)
	plot!(Beta(1+nh,1+nt), xlims=[0,1], fill= true, lw=2, alpha=0.2, color=2, label=L"p(\theta|\mathcal{D})= \texttt{Beta}(8,4)", linewidth=2)
	vline!([mean(Beta(1+nh,1+nt))], label="posterior mean", lw=2, lc=2, ls=:dash)
end

# ╔═╡ c74a488d-0e1a-4ddc-9d9b-fcb5e813d587
md"""
## Remarks


Conjugate priors usually lead to very convenient posterior computation:

$$a_N= a_0+ N_h\;\; b_N= b_0+N_t.$$ 

* if we choose a truncated Gaussian prior, then the posterior distribution is much harder to compute
  * we need to resort to advanced methods like Markov Chain Monte Carlo (MCMC).



``a_N`` and ``b_N`` are the posterior counts of the heads and tails

* we can interpret the prior parameters ``a_0, b_0`` as some pseudo observations contained in the prior distribution. 

* for example, ``a_0=b_0=1`` implies the prior contains one pseudo count of head and tail each (therefore, the prior is flat). 


"""

# ╔═╡ fe47a810-cf29-4adf-a390-6f595b8f3ec9
md"""
## Informative vs Non-informative

All priors can be largely classified into two groups: 

* **informative** prior and 
* **non-informative** prior.

**Non-informative** prior, as the name suggests, contains no information in the prior and *let the data speak to itself*

* for our coin-flipping case, a possible choice is a flat uniform prior: *i.e.* ``p(\theta) \propto 1`` when ``\theta\in[0,1]``. 


**Informative** prior, on the other hand, contains the modeller's subjective prior judgement. 

* if we believe our coin should be fair, we can impose an informative prior such as ``p(\theta) =\texttt{Beta}(n,n)``. e.g. ``n=5``: a stronger prior belief in the coin being fair.


$(begin
plot(Beta(1,1), lw=2, xlabel=L"\theta", ylabel="density", label=L"\texttt{Unif}(0,1)", title="Priors with different level of information", legend=:topleft)
plot!(Beta(2,2), lw=2,label= L"\texttt{Beta}(2,2)")
plot!(Beta(3,3), lw=2,label= L"\texttt{Beta}(3,3)")
plot!(Beta(5,5), lw=2,label= L"\texttt{Beta}(5,5)")
# vline!([0.5], lw=2, ls=:dash, label= "")
end)


### The posteriors

After we observe ``N_h=7, N_t=3``, the posteriors can be calculated by updating the pseudo counts: 

$$\texttt{Beta}(n+7, n+3)$$

The corresponding posteriors 

* together with their posterior means (thick dashed lines) are plotted below.

$(begin
nh, nt= 7, 3
plot(Beta(1+7,1+3), lw=2,xlabel=L"\theta", ylabel="density", label=L"\texttt{Beta}(1+7,1+3)", title="Posteriors with different priors",legend=:topleft)
plot!(Beta(2+7,2+3), lw=2,label= L"\texttt{Beta}(2+7,2+3)")
plot!(Beta(3+7,3+3), lw=2,label= L"\texttt{Beta}(3+7,3+3)")
plot!(Beta(5+7,5+3), lw=2,label= L"\texttt{Beta}(5+3,5+3)")
vline!([mean(Beta(5+7, 5+3))], color=4, ls=:dash, lw=2, label="")
vline!([mean(Beta(1+7, 1+3))], color=1, ls=:dash, lw=2, label="")
vline!([mean(Beta(2+7, 2+3))], color=2, ls=:dash, lw=2, label="")
vline!([mean(Beta(3+7, 3+3))], color=3, ls=:dash, lw=2, label="")
end)
"""

# ╔═╡ 9638d082-5b61-4051-912b-fc263abeb239
begin
	# simulate 200 tosses of a fair coin
	N_tosses = 200
	true_θ = 0.5
	Random.seed!(100)
	coin_flipping_data = rand(N_tosses) .< true_θ
	Nh = sum(coin_flipping_data)
	Nt = N_tosses- Nh
end;

# ╔═╡ 30ece808-4852-4cbd-84c5-7e8087753ad5
md"""
## Sequential update.


Conjugate prior also provides us with a simple procedure to do **sequential** inference


* *sequential online learning:* update the posterior incrementally as data arrives
* based on the conditional independence assumption

```math
\begin{align}
p(\theta|\{d_1, d_2, \ldots, d_N\})&\propto  p(\theta) \prod_{n=1}^N p(d_n|\theta) p(d_N|\theta)\\
&= \underbrace{p(\theta) \prod_{n=1}^{N-1} p(d_n|\theta)}_{p(\theta|\mathcal D_{N-1})}\cdot p(d_N|\theta)\\
&= \underbrace{p(\theta|\mathcal D_{N-1})}_{\text{new prior}} p(d_N|\theta).
\end{align}
```

* *yesterday's posterior becomes today's prior*


To be more specific, the sequential update algorithm is:

---
Initialise with a prior ``p(\theta|\emptyset)= \texttt{Beta}(a_0, b_0)``

For ``n = 1,2,\ldots, N``:
* update 

$$a_n = a_{n-1} + \mathbf{1}(d_n=\texttt{head}), \;\; b_n = b_{n-1} +  \mathbf{1}(d_n=\texttt{tail})$$
---

Note that the function ``\mathbf{1}(\cdot)`` returns 1 if the test result of the argument is true and 0 otherwise. 

## Demonstration

A demon with $(N_tosses) tosses of 
* a fair coin 
"""

# ╔═╡ 997d45bf-cdbf-4881-b823-ebb7d9ec19db
let
	a₀, b₀ = 1, 1
	prior_θ = Beta(a₀, b₀)
	plot(prior_θ, xlim = [0, 1], lw=2, fill=(0, 0.1), label="Prior "* L"p(\theta)", xlabel=L"\theta", ylabel="density", title="Sequential update N=0", legend=:topleft)
	plot!(Beta(a₀ + Nh, b₀+ Nt), lw=2, fill=(0, 0.1), label="Posterior "* L"p(\theta|\mathcal{D})")
	vline!([true_θ], label="true "*L"θ", lw=4, lc=3)
	an = a₀
	bn = b₀
	anim=@animate for n in 1:N_tosses
		# if the toss is head, update an
		if coin_flipping_data[n]
			an += 1
		# otherwise
		else	
			bn += 1
		end
		poster_n = Beta(an, bn)
		# plot every 5-th frame
		if (n % 5) == 0
			plot!(poster_n, lw=1, label="", title="Sequential update with observation N=$(n)")
		end
	end
	gif(anim, fps=15)
end

# ╔═╡ 9bbfb923-1448-45c9-8091-bfd86c4d54bb
md"""

## More conjugate example: Gamma-Gaussian

Consider the problem
> Given i.i.d observations from a zero mean Gaussian 
> $\mathcal D=\{d_i\}_{i=1}^n$;
>
> $d_i \sim \mathcal{N}(\mu, \sigma^2)$
> * infer the variance ``\sigma^2`` of the Gaussian
> * assume ``\mu`` is known


"""

# ╔═╡ 63f39880-d150-44fe-8c02-d3dc9b4deb1f
md"""

* it is more convenient to model the precision 

$\phi\triangleq 1/\sigma^2$ 

* a conjugate prior for ``\phi`` is Gamma distribution
  * note that Gamma distributions have support on the positive real line which is required here


## Aside: MLE for ``\sigma^2`` and ``\phi``

It can be shown that the MLE for ``\hat{\sigma}^2`` is

```math
\hat{\sigma}^2 = \frac{\sum_{n} (d_n- \mu)^2}{N}
```


* sum of squared errors divided by ``N``

Therefore, the MLE for ``\phi`` is 

```math
\hat{\phi} = \frac{N}{\sum_{n} (d_n- \mu)^2}
```




## Aside: Gamma distribution

A Gamma distribution, parameterised with a shape  ``a_0>0``,  and a rate parameter ``b_0>0``

```math
p(\phi; a_0, b_0) = \texttt{Gamma}(\phi; a_0, b_0)=\frac{b_0^{a_0}}{\Gamma(b_0)} \phi^{a_0-1} e^{-b_0\phi}.
```

* we will interpret ``a_0``, ``b_0`` in a bit


!!! info ""
	The mean of a Gamma distribution is 

	$$\mathbb{E}[\phi] = \frac{a_0}{b_0}$$
"""

# ╔═╡ 898a0670-d5bb-42f9-8afd-c3a51c426065
let
	as_ = [1,2,3,5,9,7.5,0.5]
	bs_ = [1/2, 1/2, 1/2, 1, 1/0.5, 1, 1]
	plt= plot(xlim =[0,20], ylim = [0, 0.8], xlabel=L"\phi", ylabel="density")
	for i in 1:length(as_)
		plot!(Gamma(as_[i], 1/bs_[i]), fill=(0, .1), lw=1.5, label=L"\texttt{Gamma}"*"($(as_[i]),$(bs_[i]))")
	end
	plt
end

# ╔═╡ cf7b36c9-1c78-4ef6-bd06-2590b67c3703
md"""
## Conjugacy
**Conjugacy.** The data follow a Gaussian distribution. That is for ``n = 1,2,\ldots, N``:
```math
d_n \sim \mathcal N(\mu, 1/\phi);
```
* the likelihood, therefore, is a product of Gaussian likelihoods:

```math
p(\mathcal D|\mu, \phi) = \prod_{n=1}^N p(d_n|\mu, \phi) = \prod_{n=1}^N \mathcal N(d_n; \mu, 1/\phi) .
```

By Baye's rule, the posterior

```math

p(\phi|\mathcal D, \mu) \propto p(\phi) p(\mathcal D|\mu, \phi)

```
* the posterior is still of a Gamma form (therefore conjugate): 

```math
p(\phi|\mathcal D, \mu) = \texttt{Gamma}(a_N, b_N),
```
* where 

$a_N= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.$



* The Bayesian computation reduces to **hyperparameter update** again 


## Intepretation

The posterior is: 

```math
p(\phi|\mathcal D, \mu) = \texttt{Gamma}(a_N, b_N),
```
* where 

$a_N= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.$


Therefore, it is not hard to see
* ``a_N``: the updated observation counts (divided by 2)
* ``b_N``: the updated sum of squared errors (divided by 2)

Also note the posterior mean is 

$$\mathbb{E}[\phi|\mathcal D] = \frac{a_N}{b_N} = \frac{a_0 + N/2}{b_0 + {\sum_n (d_n -\mu)^2}/{2}}.$$

* if we assume ``a_0= b_0=0`` (an improper  prior), the maximum likelihood estimator for ``{\sigma^2} is recovered``

$$\hat \sigma^2 =1/\hat{\phi} = \frac{\sum_n (d_n-\mu)^2}{N}$$

"""

# ╔═╡ e1bb37e8-d3b5-416f-b90a-fb611864b402
Foldable("Derivation details on the Gamma-Gaussian conjugacy.", md"
```math
\begin{align}
 p(\phi|\mathcal D)&\propto p(\phi) p(\mathcal D|\mu, \phi)\\
&= \underbrace{\frac{b_0^{a_0}}{\Gamma(a_0)} \phi^{a_0-1} e^{-b_0 \phi}}_{p(\phi)} \underbrace{\frac{1}{ (\sqrt{2\pi})^N}\phi^{\frac{N}{2}}e^{\frac{-\phi\cdot \sum_{n} (d_n-\mu)^2}{2}}}_{p(\mathcal D|\phi, \mu)}\\
&\propto \phi^{a_0+\frac{N}{2}-1} \exp\left \{-\left (b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}\right )\phi\right \} \\
&= \phi^{a_N-1} e^{- b_N \phi}, 
\end{align}
```
where ``a_n= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.`` Note this is a unnormalised Gamma distribution (whose normalising constant can be read off directly from a Gamma distribution), therefore 


$$p(\phi|\mathcal D)= \text{Gamma}(a_N, b_N).$$

")

# ╔═╡ 4f4882c5-bd3f-4fbe-8a59-cafc5d365d99
md"""

## Demonstration


**Demonstration:** We first simulate ``N=100`` Gaussian observations with 

* unknown ``\mu=0`` and 
* ``\phi= 1/\sigma^2 = 2``.

"""

# ╔═╡ 78bd6dea-8bcb-408f-9dc3-8a34907a4566
begin
	σ² = 0.5
	true_ϕ = 1/σ²
	N = 100
	Random.seed!(100)
	# Gaussian's density in Distributions.jl is implemented with standard deviation σ rather than σ²
	gaussian_data = rand(Normal(0, sqrt(σ²)), N)
	plot(Normal(0, sqrt(σ²)), xlabel=L"d", ylabel="density", label=L"\mathcal{N}(0, 1/2)", title="Simulated Gaussian observations")
	scatter!(gaussian_data, 0.01 .* ones(N), markershape=:vline, c=1, label=L"d_n")
end

# ╔═╡ 42603c08-01de-4036-928a-d6a9c2dffbb3
gaussian_data

# ╔═╡ 83586a99-02e5-42e3-83b2-2d90d3d3e396
md"""
* a vague Gamma prior with ``a_0=b_0=0.5``; 

* and the posterior is ``\texttt{Gamma}(a_0+ 100/2, b_0+ \texttt{sse}/2)`` 
"""

# ╔═╡ 644cbe41-2027-4e3f-a31f-c656a1158466
begin
	a₀ = 0.5
	b₀ = 0.5
	# Gamma in Distributions.jl is implemented with shape and scale parameters where the second parameter is 1/b 
	prior_ϕ = Gamma(a₀, 1/b₀)
	posterior_ϕ = Gamma(a₀+ N/2, 1/(b₀+ sum(gaussian_data.^2)/2))
end;

# ╔═╡ 1a483bed-1639-41b4-ad7f-40b009bd45a9
let
	plot(prior_ϕ, xlim = [0, 5], ylim=[0, 1.5], lw=2, fill=(0, 0.2), label="Prior "* L"p(\phi)", xlabel=L"\phi", ylabel="density", title="Conjugate inference of a Gaussian's precision")

	plot!(posterior_ϕ, lw=2,fill=(0, 0.2), label="Posterior "* L"p(\phi|\mathcal{D})")
	vline!([true_ϕ], label="true "*L"\phi", lw=4)
	vline!([mean(prior_ϕ)], label="prior mean", lw=2, lc=1, ls=:dash)
	vline!([mean(posterior_ϕ)], label="posterior mean", lw=2, lc=2, ls=:dash)
	# vline!([1/σ²], label="true "*L"λ", lw=2)
	# end
end

# ╔═╡ 6b460036-890d-4364-aac2-2c61dc44ed75
md"""

## Prediction 



In many applications, we also want to make predictions based on observations ``\mathcal{D}``


**Bayesian prediction** 
* routinely applies probability rules
* aims at computing the following distribution

```math
p(\mathcal{D}_{pred}|\mathcal{D})
```


**Frequentist prediction**
* usually employs the *plug-in* principle
* plug in the estimated parameter to the likelihood
```math
p(\mathcal{D}_{pred}|\theta_{ML})
```

"""

# ╔═╡ df3c6bd8-e81f-4ccb-b0d1-98832e41537f
md"""

## Prediction -- example

Let's use the coughing case as an example again

* the observation: ``\mathcal{D} = \texttt{Cough}``

**Prediction:** he is going to **cough** **again**

```math
p(\texttt{Cough}_{again}|\texttt{Cough})
```


* for prediction, we can introduce an unobserved node to our graph
"""

# ╔═╡ 5440ceb3-b791-4935-8b59-339010546090
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/cough_pred_bn.png' width = '250' /></center>"

# ╔═╡ 8c763ff2-cfda-49a6-b0ce-d5530f665971
md"""

Bayesian routinely applies probability rules to compute

```math
\begin{align}
p(\texttt{cough}_{\text{again}} |\texttt{caugh}) &\propto p(\texttt{cough}_{\text{again}} , \texttt{caugh}) \\
&= \sum_{h} p(h, \texttt{cough}_{\text{again}} , \texttt{caugh}) \\
&= \sum_{h} p(\texttt{cough}_{\text{again}}|h) \underbrace{p(\texttt{cough}|h) p(h)}_{= p(h, \texttt{cough})\propto \; p(h|\texttt{cough})} \\
&= \sum_{h} p(\texttt{cough}_{\text{again}}|h) p(h|\texttt{cough})
\end{align}
```


## Remarks

**Bayesian** prediction is

```math
\begin{align}
p(\texttt{cough}_{\text{again}} |\texttt{caugh}) 
&= \sum_{h} p(\texttt{cough}_{\text{again}}|h) p(h|\texttt{cough})
\end{align}
```


* the prediction is an ensemble prediction, *i.e.* balanced and **weighted average**

* each unknown hypothesis ``h\in \{\texttt{cancer}, \texttt{healthy}, \texttt{cold}\}`` is used to make a prediction for another ``\texttt{cough}_{\text{again}}``

* each prediction is weighted by its corresponding posterior ``p(h|\texttt{cough})``


In comparison, the **frequentist** plug-in method's predictions are 

```math
p(\texttt{cough}_{\text{again}} | h = \hat{h}_{\text{MLE}}= \texttt{cancer})
```

* prediction by a single *point* hypothesis
"""

# ╔═╡ b13d76bf-f559-4842-bdfa-78953d74d69a
md"""

## Why Bayesian prediction is better?


Consider the coin flipping example, supposed we have observed data

> A coin 🪙 is tossed 2 times. 
> * the tossing results are recorded:  $\mathcal D=\{0,0\}$; 0 out of 2 are heads
> * **predict** the next toss?
"""

# ╔═╡ 37a564d5-8c46-462e-8d3d-45b93ff1d3e4
md"""
**Frequentist's prediction** is based on plug-in principle

The MLE is

```math
	\theta_{ML} = \frac{N^+}{N} = \frac{0}{2} = 0
```
"""

# ╔═╡ 03181c93-6a31-46d6-aa64-1521234e2341
md"""




$(let
	gr()
	𝒟 = [0,0]
	θs = 0:0.01:1
	like_plt_seller = plot(θs, θ -> exp(loglikelihood(Bernoulli(θ), [0,0] )), seriestype=:line, color=1, xlabel=L"θ", ylabel=L"P(\{0,0\}|θ)", label="", title="Likelihood: "*L"P(\{0,0\}|\theta)")

end)

If you treat $$\hat{\theta}_{\text{ML}} = 0$$ as gospel for future prediction: *i.e.*

$$P(Y_{N+1}|\theta = \hat{\theta}_{\text{ML}})=\begin{cases} 0 & Y_{N+1} =1 \\ 1 & Y_{N+1} =0\end{cases}$$
* you predict you will **never** see a `Head`/1 again: **Overfitting**


* the frequentist MLE based prediction is pathologically bad when 
  * there is not enough data
"""

# ╔═╡ 80400b4b-b130-4719-957a-4c92b556916a
md"""

## Why Bayesian is better?


The Bayesian instead computes the integration


```math
p(y_{N+1} | \mathcal{D}) = \int_\theta p(y_{N+1}|\theta) p(\theta|\mathcal{D})d\theta
```


* which is a weighted average overall ``\theta``

* the integration can be computed in closed form if we use conjugate prior

**Bayesian prediction**
```math
p(y_{N+1}=1 | \mathcal{D}) = \frac{a_N}{a_N + b_N} =\frac{a_0 + N_h}{a_0+b_0 + N}
```

* very simple, add *one pseudo* counts to ``N_h`` and ``N_t``


For our case, 

```math
p(y_{N+1}=1 | \mathcal{D}) = \frac{1 + 0}{2 + 2} = \frac{1}{4}
```

* makes better sense! we have only observed two zeros after all!
"""

# ╔═╡ f2912738-1b91-4227-a94f-c61a33a63819
md"""

## Generalisation


Bayesian predictive distribution, in general, can be found by applying the sum rule,

```math

p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred}, \theta|\mathcal D, \mathcal M) \mathrm{d}\theta=  \int p(\mathcal D_{pred} |\theta, \mathcal D, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

```

* ``\mathcal{M}``: the model assumption
* in which the unknown parameters ``\theta`` are integrated out

* assuming that past and future observations are conditionally independent given  ``\theta``, *i.e.* 

$p(\mathcal D_{pred} |\theta, \mathcal D, \mathcal M) = p(\mathcal D_{pred} |\theta, \mathcal M)$

The above equation can be written as:

```math

p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred} |\theta, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

```

* Bayesian prediction is an ensemble method by nature 
"""

# ╔═╡ b572e185-d88e-47d5-a3f3-1e25f2a6ca10
md"""

## Prior and Posterior predictive distribution

**Posterior predictive distribution:**

```math

p(\mathcal D_{pred}|\mathcal D, \mathcal M) =  \int p(\mathcal D_{pred} |\theta, \mathcal M) \underbrace{p(\theta|\mathcal D, \mathcal M) }_{\text{posterior}}\mathrm{d}\theta

```
* predictive distribution after observing the data


A related concept is **Prior predictive distribution**


```math

p(\mathcal D_{pred}|\mathcal M) =  \int p(\mathcal D_{pred} |\theta, \mathcal M) \underbrace{p(\theta|\mathcal M) }_{\text{prior}}\mathrm{d}\theta

```
* predictive distribution before observing the data
* can be useful to check your prior assumptions
"""

# ╔═╡ f2507e03-76e6-4f57-9a38-9323610efa33
md"""

## Computing predictive distribution

The prediction distribution involves the integration 

```math

p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred} |\theta, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

```

* which in general is intractable. 



However, we can use simulation-based (Monte Carlo) methods to approximate the integration


!!! information ""
	Repeat the following many times:
	
	1. Draw one sample from the posterior (or the prior for prior predictive):

	$$\tilde \theta \sim p(\theta|\mathcal D)$$
	2. Conditioning on ``\tilde \theta``, simulate pseudo observations: 

	$$\tilde{\mathcal D}\sim p(\mathcal D|\tilde{\theta})$$


* the sampled ``\{\tilde{\mathcal D} \}`` are the empirical predictive distribution
"""

# ╔═╡ 23e88265-929b-4616-95c0-61e1844c8ae5
md"""

## Example


For the coughing example


!!! warn ""
	Repeat the following many times:
	
	1. Draw one sample from the posterior (or the prior for prior predictive):

	$$\tilde h \sim p(h|\text{cough})$$
	2. Conditioning on ``\tilde h``, simulate pseudo observations: 

	$$\texttt{cough}\sim p(\texttt{cough}_{again}|\tilde{h})$$

"""

# ╔═╡ 18d66c48-be01-4f64-8a89-6b8e8c8444c8
md"""
## Example: coin-flipping model

For the conjugate coin-flipping model, the simulation is straightforward 


* both the prior and posterior distributions ``p(\theta|\mathcal M)`` (or ``p(\theta|\mathcal M)``) are Beta 

* it is also easy to simulate coin flip observations conditional on the bias ``\theta`` 



For example, one can simulate a tuple ``(\theta, y_{new})`` in Julia by

```julia
# draw a coin's bias from a Beta distribution
θ = rand(Beta(a, b))
# draw a pseudo coin-flipping sample based on θ
y_new = rand(Bernoulli(θ))
```

* one should repeat the above two steps ``R`` (e.g. 2000) times to obtain an approximation of the predictive distribution of future data
"""

# ╔═╡ 12fe751d-8ef4-4cdf-b5e5-a901a1556924
md"""


## Demonstration
"""

# ╔═╡ 8cf29145-aed3-47e6-a209-369ac393e0e3
begin
	R = 5000
	a0, b0 = 1, 1
	# nh, nt = sum(𝒟), length(𝒟) - sum(𝒟)
	an, bn = a0 + nh, b0 + nt

	y_pred = zeros(Bool, R)
	for r in 1:R
		θ = rand(Beta(an, bn))
		y_pred[r] = rand(Bernoulli(θ))
	end
	y_pred
end

# ╔═╡ 0ad979b7-8ecb-44c7-958f-be9a78a22553
md"""

There is a closed-form solution for the Beta-Bernoulli case
* this is one of the rare examples that we know how to compute the integration


```math
p(y_{N+1}|\mathcal{D}) = \begin{cases}
\frac{a_N}{a_N + b_N} & y_{N+1} = \texttt{true}\\
\frac{b_N}{a_N + b_N} & y_{N+1} = \texttt{false}
\end{cases}
```
"""

# ╔═╡ 3cb9bb88-d560-4403-b67e-7469f6eddcac
md"""


The Monte Carlo method should be very close to the ground truth
"""

# ╔═╡ 40c740a4-5385-4211-ac86-3c9f0c17b7ff
mean(y_pred), an/(an+bn)

# ╔═╡ cd01a246-5e97-4231-a8dc-b2931d151b2f
md"""

## Model checks via predictive distributions



Bayesian inference provides a great amount of modelling freedom to its user


* one can choose prior distribution to incorporate his knowledge


* and also choose a suitable likelihood that best matches the data generation process. 


* however, greater flexibility comes with a price: the modeller also needs to take full responsibility for the modelling decisions 


For example, whether 

* the chosen prior (with its parameters) makes sense;
* the generative model as a whole (*i.e.* prior plus likelihood) match the observed data? 


In other words, we need to **validate the model** 

* **Predictive checks** are a great way to empirically validate a model's assumptions.

"""

# ╔═╡ a7fcf38b-05c3-4db8-8615-44b93d6d43aa
md"""

## Predictive checks


The idea of predictive checks is to 



**First**, generate future *pseudo observations* based on the assumed model's (prior or posterior) **prediction distribution**:

$$\mathcal{D}^{(r)} \sim p(\mathcal D_{\textit{pred}}|\mathcal D, \mathcal M), \;\; \text{for }r= 1\ldots, R$$ 
  * where ``\mathcal D`` is the observed data and ``\mathcal M`` denotes the Bayesian model.  

* note that ``\mathcal{D}^{(r)}`` should be of **the same size** as ``\mathcal{D}``

and 


**Then** If the model assumptions are reasonable, 

* we should expect the generated pseudo data "agree with" the observed. 




**In practice**, we compute the predictive distribution of some summary statistics,     
* say mean, variance, median, or any meaningful statistic instead
* and visually check whether the observed statistic falls within the predictions' possible ranges. 

## Predictive checks

Based on the Monte Carlo principle, after simulating ``R`` pseudo samples,

$$\tilde{\mathcal D}^{(1)}, \tilde{\mathcal D}^{(2)}, \tilde{\mathcal D}^{(3)},\ldots, \tilde{\mathcal D}^{(R)} \sim p(\mathcal D_{pred}|\mathcal D, \mathcal M),$$ 


The predictive distribution of a summary statistic ``t(\cdot)``: 

$$p(t(D_{pred})|\mathcal D, \mathcal M)$$ 
* can be approximated by 

$\{t(\tilde{\mathcal D}^{(1)}), t(\tilde{\mathcal D}^{(2)}), \ldots, t(\tilde{\mathcal D}^{(R)})\}$ 

* one can visually check whether the observed statistic ``t(\mathcal{D})`` falls within a credible region of the empirical distribution 


## Prior and Posterior predictive checks

If one mainly wants to check the prior assumption, use a **prior predictive distribution** to simulate the *future* data: *i.e.*

$$\mathcal{D}^{(r)} \sim p(\mathcal D_{pred}|\mathcal M), \;\; \text{for }r= 1\ldots, R$$


* and the visual check based on the sample 
* the check is called **prior predictive check



**Posterior predictive checks** 

* checks both prior and likelihood model together
* a hollistic check

"""

# ╔═╡ 1975923e-66fb-4aa4-a3d3-370b38d1df34
md"""
## Example: coin-flipping model

For predictive checks, we need to simulate ``\mathcal{D}_{pred}`` of **the same length** as ``\mathcal{D}``


* we need to simulate **10** coin tosses in the second step
* instead of just one


```julia
# draw a coin's bias from a Beta distribution
θ = rand(Beta(a, b))
# draw a pseudo-coin-flipping sample of 10 tosses
D = rand(Bernoulli(θ), 10)
```

"""

# ╔═╡ 7ca55b5a-086a-4982-bd2a-6be2c7e0299a
function predictive_simulate(a, b; N=10 , R=2000)
	θ = rand(Beta(a,b), R)
	D = rand(N, R) .< θ'
	return D
end;

# ╔═╡ d009d1e0-833a-40ea-882d-1b7fdbccf5cc
md"""

Here we choose to compare **the statistic** of the sum *i.e.* the count of heads
```math
t(\mathcal{D}) = N_h= \sum_i d_i
```

"""

# ╔═╡ 3de87316-496e-4413-b7a1-3bd961dc9cc0
md"""

**Predictive checks** with the model


* ``a_0, b_0 =1``
"""

# ╔═╡ 48a972dc-c866-43b2-b74a-b10b0ae343a3
let
	Random.seed!(100)
	D = predictive_simulate(1, 1; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], bins = 20, xticks=0:10, normed=true, label="Prior predictive on Nₕ", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Prior predictive check with a₀=b₀=1")
	vline!([7], lw=4, lc=2, label="Observed Nₕ")
end

# ╔═╡ 1d296694-2ad8-4e30-a1e6-2741afaae2d3
let
	Random.seed!(100)
	D = predictive_simulate(8, 4; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], normed=true, xticks = 0:10, label="Posterior predictive on Nₕ", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Posterior predictive check with a₀=b₀=1")
	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
	vline!([7], lw=4, lc=2, label="Observed Nₕ")
end

# ╔═╡ d3ec8762-fd5c-4614-9271-6cd3de4185c3
md"""

## Example -- coin-flipping model check (cont.)
**A model with misspecified prior** 

$$\theta \sim \text{Beta}(a_0=50, b_0=1)$$

* the prior is skewed and very informative

* as a result, the posterior is dominated by the prior: 

$$\theta \sim \text{Beta}(50+7, 1+3).$$

The prior and posterior plots are shown below for reference


"""

# ╔═╡ e20c3ef6-e7f4-4fd5-be74-ef46970911d1
let
	nh, nt = 7, 3
	a₀, b₀ = 50, 1
	plot(Beta(a₀,b₀), xlims=[0.6,1], label=L"p(\theta)= \mathcal{Beta}(50,1)", linewidth=1, xlabel=L"\theta", ylabel=L"p(\theta|\cdot)" ,fill= true, lw=2, alpha=0.2, legend=:outerright, color=1, title="A mis-specified model")	
	vline!([mean(Beta(a₀,b₀))], label="prior mean", lw=2, lc=1, ls=:dash)
	plot!(Beta(a₀+nh,b₀+nt), fill= true, lw=2, alpha=0.2, color=2, label=L"p(\theta|\mathcal{D})= \mathcal{Beta}(57,4)", linewidth=2)
	vline!([mean(Beta(a₀+nh,b₀+nt))], label="posterior mean", lw=2, lc=2, ls=:dash)
end

# ╔═╡ c41fef4e-36eb-4d44-8838-1047f77c7ac2
md"""


Let's see whether **predictive checks** can spot the problem!
"""

# ╔═╡ 1c9997ad-5573-445c-8029-a57398aa442d
let
	Random.seed!(100)
	D = predictive_simulate(50, 1; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10, normed=false, label="Prior predictive on " *L"N_h", legend=:outerbottom, xlabel="number of heads", title="Prior predictive check with "*L"a_0=50, b_0=1", ylabel=L"\#"* " of counts")
	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
end

# ╔═╡ 3c18a8e6-7a9f-4e22-b1c9-fc8dad8a822d
let
	Random.seed!(100)
	D = predictive_simulate(50+7, 1+3; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10,normed=false, label="Posterior predictive on "*L"N_h", legend=:outerbottom, xlabel="number of heads", title="Posterior predictive check with "*L"a_0=50, b_0=1", ylabel=L"\#"* " of counts")
	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.75"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.18"
Plots = "~1.34.3"
PlutoTeachingTools = "~0.2.3"
PlutoUI = "~0.7.43"
SpecialFunctions = "~2.1.7"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "f5399ea34f31ff83d99042ef4679c64fbbe579a8"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

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
git-tree-sha1 = "84259bb6172806304b9101094a7cc4bc6f56dbc6"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.5"

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
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "1833bda4a027f4b2a1c984baddcf755d77266818"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.1.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "1106fa7e1256b402a86a8e7b15c00c85036fef49"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.11.0"

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
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0d7d213133d948c56e8c2d9f4eab0293491d8e4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.75"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

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
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "87519eb762f85534445f5cda35be12e32759ee14"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.4"

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

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "2c5ab2c1e683d991300b125b9b365cb0a0035d88"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.69.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

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
git-tree-sha1 = "4abede886fcba15cd5fd041fef776b230d004cee"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.4.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

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
git-tree-sha1 = "f67b55b6447d36733596aea445a9f119e83498b6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.5"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0f960b1404abb0b244c1ece579a0ec78d056a5d1"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.15"

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
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

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
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

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
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "41d162ae9c868218b1f3fe78cba878aa348c2d26"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.1.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "6872f9594ff273da6d13c7c1a1545d5a8c7d0c1c"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "efe9c8ecab7a6311d4b91568bd6c88897822fabe"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

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
git-tree-sha1 = "f561403726db82fe98c0963a382b1b839e9287b1"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.1.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

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
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

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
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "fae3b66e343703f8f89b854a4da40bce0f84da22"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.34.3"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "0e8bcc235ec8367a8e9648d48325ff00e4b0a545"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.5"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "d8be3432505c2febcea02f44e5f4396fae017503"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "2777a5c2c91b3145f5aa75b61bb4c2eb38797136"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.43"

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
git-tree-sha1 = "3c009334f45dfd546a16a57960a821a1a023d241"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.5.0"

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
git-tree-sha1 = "612a4d76ad98e9722c8ba387614539155a59e30c"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

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
git-tree-sha1 = "dad726963ecea2d8a81e26286f625aee09a91b7c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.4.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "c0f56940fc967f3d5efed58ba829747af5f8b586"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.15"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2189eb2c1f25cb3f43e5807f26aa864052e50c17"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.8"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3e59e005c5caeb1a57a90b17f582cbfc2c8da8f7"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.3"

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
git-tree-sha1 = "2d7164f7b8a066bcfa6224e67736ce0eb54aef5b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.9.0"

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
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

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
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

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
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

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
# ╟─c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
# ╟─45f045d9-cdb7-4ade-a8e6-1f1a984cc58a
# ╟─be719382-bf1f-4442-8419-bddcda782525
# ╟─77535564-648a-4e17-83a0-c562cc5318ec
# ╟─2057c799-18b5-4a0f-b2c7-66537a3fbe79
# ╟─c677a2d2-f4be-483a-a53b-7b76efb1b80f
# ╟─56f2bb33-637c-4450-9769-f338937594da
# ╟─47359bc3-cfc2-4777-82f9-308e56dca491
# ╟─35311403-187a-4e23-9e52-7eb0a212452d
# ╟─9fa2c7bf-3fa2-4869-8515-fa0048fa3cd6
# ╟─ae234663-90b4-432d-bacb-188dd1ee1034
# ╟─d764a435-7e11-4736-a0d0-0f2d5265b7af
# ╟─5aaf5221-e77b-4339-95c5-dc21bd815fc3
# ╟─047a8e19-54d4-454a-8635-8becd4b93061
# ╟─43cedf4c-65c4-4b2b-afb3-21f5827f2af6
# ╟─4e82a77d-af72-4653-aaa9-f7303e33c3ea
# ╟─e2673877-6536-4670-b5b7-b84c36503512
# ╟─33714d50-a04e-41ad-844c-293ed9671163
# ╟─9308405d-36d0-41a1-8c73-e93b8d699320
# ╟─c32714b7-fa1d-443c-a5a2-4a511fc701bd
# ╟─c74a488d-0e1a-4ddc-9d9b-fcb5e813d587
# ╟─fe47a810-cf29-4adf-a390-6f595b8f3ec9
# ╟─30ece808-4852-4cbd-84c5-7e8087753ad5
# ╟─9638d082-5b61-4051-912b-fc263abeb239
# ╟─997d45bf-cdbf-4881-b823-ebb7d9ec19db
# ╟─9bbfb923-1448-45c9-8091-bfd86c4d54bb
# ╟─63f39880-d150-44fe-8c02-d3dc9b4deb1f
# ╟─898a0670-d5bb-42f9-8afd-c3a51c426065
# ╟─cf7b36c9-1c78-4ef6-bd06-2590b67c3703
# ╟─e1bb37e8-d3b5-416f-b90a-fb611864b402
# ╟─4f4882c5-bd3f-4fbe-8a59-cafc5d365d99
# ╟─78bd6dea-8bcb-408f-9dc3-8a34907a4566
# ╠═42603c08-01de-4036-928a-d6a9c2dffbb3
# ╟─83586a99-02e5-42e3-83b2-2d90d3d3e396
# ╟─644cbe41-2027-4e3f-a31f-c656a1158466
# ╟─1a483bed-1639-41b4-ad7f-40b009bd45a9
# ╟─6b460036-890d-4364-aac2-2c61dc44ed75
# ╟─df3c6bd8-e81f-4ccb-b0d1-98832e41537f
# ╟─5440ceb3-b791-4935-8b59-339010546090
# ╟─8c763ff2-cfda-49a6-b0ce-d5530f665971
# ╟─b13d76bf-f559-4842-bdfa-78953d74d69a
# ╟─37a564d5-8c46-462e-8d3d-45b93ff1d3e4
# ╟─03181c93-6a31-46d6-aa64-1521234e2341
# ╟─80400b4b-b130-4719-957a-4c92b556916a
# ╟─f2912738-1b91-4227-a94f-c61a33a63819
# ╟─b572e185-d88e-47d5-a3f3-1e25f2a6ca10
# ╟─f2507e03-76e6-4f57-9a38-9323610efa33
# ╟─23e88265-929b-4616-95c0-61e1844c8ae5
# ╟─18d66c48-be01-4f64-8a89-6b8e8c8444c8
# ╟─12fe751d-8ef4-4cdf-b5e5-a901a1556924
# ╟─8cf29145-aed3-47e6-a209-369ac393e0e3
# ╟─0ad979b7-8ecb-44c7-958f-be9a78a22553
# ╟─3cb9bb88-d560-4403-b67e-7469f6eddcac
# ╠═40c740a4-5385-4211-ac86-3c9f0c17b7ff
# ╟─cd01a246-5e97-4231-a8dc-b2931d151b2f
# ╟─a7fcf38b-05c3-4db8-8615-44b93d6d43aa
# ╟─1975923e-66fb-4aa4-a3d3-370b38d1df34
# ╟─7ca55b5a-086a-4982-bd2a-6be2c7e0299a
# ╟─d009d1e0-833a-40ea-882d-1b7fdbccf5cc
# ╟─3de87316-496e-4413-b7a1-3bd961dc9cc0
# ╟─48a972dc-c866-43b2-b74a-b10b0ae343a3
# ╟─1d296694-2ad8-4e30-a1e6-2741afaae2d3
# ╟─d3ec8762-fd5c-4614-9271-6cd3de4185c3
# ╟─e20c3ef6-e7f4-4fd5-be74-ef46970911d1
# ╟─c41fef4e-36eb-4d44-8838-1047f77c7ac2
# ╟─1c9997ad-5573-445c-8029-a57398aa442d
# ╟─3c18a8e6-7a9f-4e22-b1c9-fc8dad8a822d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
