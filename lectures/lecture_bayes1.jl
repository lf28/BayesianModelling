### A Pluto.jl notebook ###
# v0.19.46

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

# ╔═╡ c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
begin
	using PlutoTeachingTools
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
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

# ╔═╡ f3c62849-620d-47b3-b698-79c925897d54
TableOfContents()

# ╔═╡ 30b4bf3c-82dc-46f7-bd69-5fdeada9ebad
ChooseDisplayMode()

# ╔═╡ 77535564-648a-4e17-83a0-c562cc5318ec
md"""

# Bayesian modelling


**Lecture 2. A first look at Bayesian inference** 


$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang (lf28@st-andrews.ac.uk)

*School of Computer Science*

*University of St Andrews, UK*

*March 2023*

"""

# ╔═╡ 2057c799-18b5-4a0f-b2c7-66537a3fbe79
md"""

## In this lecture

* More Bayesian inference 
  * the workflow: modelling + computation


  * Estimate the bias of a coin (Bernoulli likelihood)
    * a classic statistical inference problem
    * compare with the **frequentist** method


* Two Amazon seller problem as an example
  
  







"""

# ╔═╡ 43cedf4c-65c4-4b2b-afb3-21f5827f2af6
md"""
## Coin flipping example


A classic statistical inference problem

> A coin 🪙 is tossed 10 times. 
> * the tossing results are recorded:  $\mathcal D=\{1, 1, 1, 0, 1, 0, 1, 1, 1, 0\}$; 7 out of 10 are heads
> * is the coin **fair**?


"""

# ╔═╡ 351694f5-1b83-426f-a190-c87c69152ef0
md"""

## Forward modelling

#### The *random variables* are 

* ##### *The positive rate*: ``\theta \in [0,1]``

  * the bias for Bernoulli r.v.


* ##### *The observed tosses*: ``\mathcal{D} =\{Y_1, \ldots, Y_n\}``


\
\

##### How the random variables shall be linked?

```math
\Large \theta \Rightarrow \mathcal{D}
```
* the tossing results depend on $\theta$
"""

# ╔═╡ 66926dbf-964d-4048-b8ad-459f36236c14
md"""

## The graphical model




```math
\large
P(\theta, Y_1, Y_2, \ldots, Y_{10}) = \underbrace{P(\theta)}_{\text{prior}} \underbrace{P(Y_1, Y_2, \ldots, Y_{10}|\theta)}_{\text{likelihood}} = P(\theta)\prod_i P(Y_i|\theta)
```


"""

# ╔═╡ 1b4c860a-7245-4793-8acf-a1f01700b499
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/oneselleredges.png" width = "900"/></center>
"""

# ╔═╡ 3319498b-db49-4516-b699-b99da093c9cb
md"""


The most common pattern for statistical inference problems
* **i.i.d** case: (independently and identically distributed) 
"""

# ╔═╡ 0f8b4a5b-94b8-4f2b-975d-3579eb9c62b1
md"In plate notation"

# ╔═╡ 1f524d92-e589-48f9-b3cc-bcef0bae4bca
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/onesellerplate.png" width = "200"/></center>
"""

# ╔═╡ ce1d28e1-3c77-461e-822a-6e5783bf2744
md"""

## Prior and Likelihood Specification


#### *The prior*: ``\theta\in [0,1]``

* discretize it to ``[0, 0.1, 0.2, \ldots, 1.0]`` 11 grid values
* and a uniform prior

```math
P(\theta) = \begin{cases} \frac{1}{11} & \theta \in \{0, 0.1, \ldots, 1.0\} \\
0 & \text{otherwise}\end{cases}
```


#### *The likelihood*: Bernoulli distribution

```math
P(Y_i|\theta) = \begin{cases} 1-\theta & Y_i = 0 \\ \theta & Y_i = 1\end{cases}
```

"""

# ╔═╡ dafe8d1f-6c99-46a4-a2d1-2e34e12ba4a7
md"""


To be more specific, ``\large\mathcal{D} =\{1,0,0,1,1,1,1,1,1,0\}``


```math
\large P(\mathcal{D}|\theta) = \theta (1-\theta)(1-\theta) \theta \ldots (1-\theta) = \theta^{7}(1-\theta)^{3}
```

More generally,

```math
\large
P(\mathcal{D}|\theta) = \theta^{N^+}(1-\theta)^{N-N^+}
```
* ``N^+ = \sum_{i=1}^N Y_i``: the total number of heads 
* ``N``: total tosses

"""

# ╔═╡ 1bc6b970-f1fa-48c9-b17e-cef4d01aa47a
md"""

## Maximum likelihood estimation

"""

# ╔═╡ 15571edc-32d7-4ac5-9c53-d49f595f84b1
md"""
#### _Likelihood function_:
```math
\large
P(\mathcal{D}|\theta) = \theta^{N^+}(1-\theta)^{N-N^+}
```
> What is the maximum likelihood estimator for ``\theta``?

"""

# ╔═╡ 1d835b12-f1d7-4ce4-a62e-2cfa7a7b3a21
md"""

The MLE can be shown to be  

```math
\large
	\theta_{ML} = \frac{N^+}{N} = \frac{7}{10}
```

"""

# ╔═╡ f06264f4-f020-48f6-9ff3-cac308518eac
md"""

## Posterior computation
"""

# ╔═╡ 9308405d-36d0-41a1-8c73-e93b8d699320
begin
	𝒟 = [1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
	# 𝒟 = [0, 0]
end

# ╔═╡ 0da00057-6bf1-407d-a15f-af24ee549182
@bind step Slider(0.01:0.005:0.1, default = 0.1)

# ╔═╡ be0fe9aa-32e6-4c18-8bad-2213e05bca1c
md"We can of course change the step size to discretise ``\theta``

Step to discretise ``\theta``: $(step)"

# ╔═╡ f593d1c9-7382-4d5c-9c99-5709123af43d
θs = 0:step:1

# ╔═╡ 8f9d1066-91d5-43af-9834-b3dc2dbc6400
begin
	prior_plt_seller=Plots.plot(θs, 1/length(θs) * ones(length(θs)), ylim= [0,1], seriestype=[:path, :sticks], color= 1, fill=true, alpha=0.3, markershape=:circle, xlabel=L"θ", ylabel=L"P(θ)", label="", title="Prior")
end;

# ╔═╡ 947a0e0f-b620-4e0e-97bd-74749ed87042
function ℓ(θ, 𝒟; logprob=false)
	N = length(𝒟)
	N⁺ = sum(𝒟)
	ℓ(θ, N, N⁺; logprob=logprob)
end

# ╔═╡ 1e267303-160d-43b9-866f-2d7afc2c4c3f
function ℓ(θ, n, n⁺; logprob=false)
	# logL = n⁺ * log(θ) + (n-n⁺) * log(1-θ)
	# use xlogy(x, y) to deal with the boundary case 0*log(0) case gracefully, i.e. θ=0, n⁺ = 0
	logL = xlogy(n⁺, θ) + xlogy(n-n⁺, 1-θ)
	logprob ? logL : exp(logL)
end

# ╔═╡ 5cbd933e-db0e-49fb-b9d0-267dc93efdb7
md"""


## Aside: Maximum likelihood estimator (conti.)

#### _What if:_ ``\mathcal{D}=\{0,0\}`` ? 
* *i.e.* ``N⁺ =0, N=2``



$(let
	gr()
	𝒟 = [0,0]
	like_plt_seller = plot(θs, θ -> ℓ(θ, 𝒟), seriestype=:sticks, color=1, markershape=:circle, xlabel=L"θ", ylabel=L"P(\{0,0\}|θ)", label="", title="Likelihood: "*L"P(\{0,0\}|\theta)")

end)
The MLE is 

$$\large \hat{\theta}_{\text{ML}} = \frac{0}{2} =0$$

"""

# ╔═╡ 70c00cf9-65ac-44d7-b25c-153614d24240
begin
	
	
	function posterior_robust(𝒟, θs)
		likes = [ℓ(θ, 𝒟; logprob=true) for θ in θs]
		logsum = logsumexp(likes)
		posterior_dis = exp.(likes .- logsum)
		return posterior_dis
	end
	
end

# ╔═╡ 4bae9df7-36a2-4e8a-bdd8-5fdec68cddba
md"""
## Frequentist approach


#### To draw a comparison with the _frequentist method_

* how **frequentist** would approach the problem
* we will use the **confidence interval** method 
  * not the best frequentist method
  * for comparison purposes


"""

# ╔═╡ 2338bade-9fb2-4a75-a39c-3e056f2d00d5
md"""

## Frequentist approach

> #### **Frequentist** does **NOT** treat the unknown parameter ``\theta`` a random variable **but** something fixed but unknown

* ##### Implication: frequentist has no prior ``P(\theta)`` nor posterior  ``P(\theta|\mathcal{D})``

## Frequentist approach

> #### **Frequentist** does **NOT** treat the unknown parameter ``\theta`` a random variable **but** something fixed but unknown

* ##### Implication: frequentist has no prior ``P(\theta)`` nor posterior  ``P(\theta|\mathcal{D})``

* ##### The only random variables are the data ``\mathcal{D}`` 
  * frequentists only need the likelihood ``P(\mathcal{D}|\theta)``

## Frequentist approach


A **confidence interval** can be calculated as 

$$\Large (\hat{\theta} - z_{\alpha/2} \hat{\texttt{se}}, \hat{\theta} + z_{\alpha/2} \hat{\texttt{se}}), \text{where}$$ 

$$\large\hat{\texttt{se}}= \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{N}}.$$

* ##### ``\hat{\theta}=N^+/N,\; \hat{\texttt{se}}`` are random
* ##### the intervals themselves are random but $\theta$ is fixed
"""

# ╔═╡ b3bcf03b-f4f8-4ddb-ab59-fa5f780ed563
md"""

## Confidence interval


#### For our problem, a 90% frequentist **confidence interval** is

$$\large (0.46, 0.94)$$


> #### But how shall we interpret this interval?

* ##### ``\theta`` is not a random variable
  * not an uncertainty statement  about ``\theta`` in the Bayesian sense

* ##### but the intervals $$(\hat{\theta} - z_{\alpha/2} \hat{\texttt{se}}, \hat{\theta} + z_{\alpha/2} \hat{\texttt{se}})$$ are _random_!

## Confidence interval
!!! danger ""
	
	
	A ``\alpha=10\%`` **confidence interval** interpretation: 

	*Repeat the following two steps, say 10,000 times:*

	1. *toss the coin another 10 times*
	2. *calculate another confidence interval*: ``(\hat{\theta} - 1.645 \hat{\texttt{se}}, \hat{\theta} + 1.645 \hat{\texttt{se}})`` *as above*

	Over the 10,000 realised random intervals, approximately 90% of them trap the true unknown parameter.

"""

# ╔═╡ ff01e390-a72d-41b0-9a7c-1d63060491f9
md"""
##

The animation illustrates the idea 
* conditional on the hypothesis that the coin is fair, *i.e.* ``\theta =0.5`` (it works for any ``\theta\in [0,1]``)
* the above two steps were repeated 100 times, *i.e.* tossing the coin ten times and then forming a confidence interval
* the red ones are the 10% CIs that **do not** trap the true bias. 
"""

# ╔═╡ c1edf5ff-f25f-4a71-95fb-53515aae9d97
md"""

## Frequentist sampling theory method

#### If you think Frequentist's confidence interval is _confusing_

* #### I agree with you!

##

##### But the frequentist method is very suitable for controlled experiment settings 

* we are interested in the long term frequency pattern under repeated experiments

##### But Bayesian inference give you a more **direct answer** to the question

> "*In light of the observed data, what ``\theta`` is more credible, i.e. what is ``p(\theta|\mathcal D)``?*''

##### And Bayesian inference is procedurally simple and uniform

* forward modelling + inference computation
* frequentist methods need different tests for different purposes
"""

# ╔═╡ 84f66218-1309-4cfd-8250-11bd75f78eb4
md"""

## Bayesian vs Frequentist
"""

# ╔═╡ 98e84190-393a-4eb9-8f1c-fa90bf8973d2
TwoColumn(md"""

### Bayesian


##### ``\theta`` is random
* many ``\theta`` flausible



##### ``\mathcal{D}`` random but observed (fixed)
* only one ``\mathcal{D}`` assumed 


##### One algorithm rules all 
* *i.e.* Bayes' rule

""", md"""
### Frequentist

##### ``\theta`` is not random (fixed)
* only one ``\theta`` (although unknown) 



##### ``\mathcal{D}`` random and repeated (varies)
* many possible ``\mathcal{D}`` assumed 


##### One algorithm per problem
* recall those `tests` you have been taught
""")

# ╔═╡ 6131ad55-eee7-44ec-b965-ce17ef3f8f84
begin
	cint_normal(n, k; θ=k/n, z=1.96) = max(k/n - z * sqrt(θ*(1-θ)/n),0), min(k/n + z * sqrt(θ*(1-θ)/n),1.0)
	within_int(intv, θ=0.5) = θ<intv[2] && θ>intv[1]
end;

# ╔═╡ a152a7bd-062d-4fd9-be38-c217fdaca20f
begin
	Random.seed!(100)
	θ_null = 0.5
	n_exp = 100
	trials = 10
	outcomes = rand(Binomial(trials, θ_null), n_exp)
	intvs = cint_normal.(trials, outcomes; z= 1.645)
	true_intvs = within_int.(intvs)
	ϵ⁻ = outcomes/trials .- [intvs[i][1] for i in 1:length(intvs)]
	ϵ⁺ = [intvs[i][2] for i in 1:length(intvs)] .- outcomes/trials
end;

# ╔═╡ 5462f241-e284-45b8-8e61-3c0317f6b08b
let
	p = hline([0.5], label=L"\mathrm{true}\;θ=0.5", color= 3, linestyle=:dash, linewidth=2,  xlabel="Experiments", ylim =[0,1])
	anim = @animate for i in 1:n_exp
		k_ = outcomes[i]
		# intv = cint_normal(trials, k_; z= 1.645)
		# intv = intvs[i]
		in_out = true_intvs[i]
		col = in_out ? 1 : 2
		θ̂ = k_/trials
		Plots.scatter!([i], [θ̂],  label="", yerror= ([ϵ⁻[i]], [ϵ⁺[i]]), markerstrokecolor=col, color=col)
	end

	gif(anim, fps=3)
end

# ╔═╡ 6d136ab7-c7ea-43a7-a16d-0af53b123b59
begin
	first_20_intvs = true_intvs[1:100]
	Plots.scatter(findall(first_20_intvs), outcomes[findall(first_20_intvs)]/trials, ylim= [0,1], yerror =(ϵ⁻[findall(first_20_intvs)], ϵ⁺[findall(first_20_intvs)]), label="true", markerstrokecolor=:auto, legend=:outerbottom,legendtitle = "true " *L"\theta"*" within CI ?")
	Plots.scatter!(findall(.!first_20_intvs), outcomes[findall(.!first_20_intvs)]/trials, ylim= [0,1], yerror =(ϵ⁻[findall(.!first_20_intvs)],ϵ⁺[findall(.!first_20_intvs)]), label="false", markerstrokecolor=:auto)
	hline!([0.5], label=L"\mathrm{true}\;θ=0.5", linewidth =2, linestyle=:dash, xlabel="Experiments")
end

# ╔═╡ a2642e09-3202-4c73-914a-bdd6c0b374c2
md"""

## Amazon seller problem



!!! note "Amazon reseller"
	Two resellers on Amazon with different review counts 

	| Seller | A | B |
	|:---:|---|---|
	|#Positive ratings  |8  |799 |
	|#Total ratings  |10|1000|
	Which seller provides **better** service?


* we will consider one seller first 
* and then extend it to two seller

## One seller problem
"""

# ╔═╡ 89f0aa81-1da3-41a0-8d72-72637d8052aa
md"""

> Seller *A* is "tossed" 10 times. 
> * 8 out of the 10 experiments are positive. 
> * what is the unknown positive rating ?


The difference: the observations are ``N^+`` directly

* the likelihood is Binomial 
```math
P(N⁺ |\theta, N) = \text{Binom}(N⁺; N, \theta) = \binom{N}{N⁺} \theta^{N⁺} (1-\theta)^{N-N⁺}
```  
"""

# ╔═╡ 4f6232be-dbd1-4e1a-a28c-5fc60964bb9f
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/oneseller.png" width = "150"/></center>
"""

# ╔═╡ d76d0594-5a53-4641-818e-496ce3a52c6b
function ℓ_binom(N⁺, θ, N; logprob=false)
	# log(binomial(N, N⁺) * (1-θ)^(N-N⁺) * θ^N⁺)
	logL = logabsbinomial(N, N⁺)[1] + xlogy(N-N⁺, 1-θ)  + xlogy(N⁺, θ)
	logprob ? logL : exp(logL)
end

# ╔═╡ 19892f66-98e1-405e-abae-ffda46a546b1
begin
	Nᴬ = 10
	N⁺ = 7
	like_plt_seller = Plots.plot(θs, θ -> ℓ_binom(N⁺, θ, Nᴬ), seriestype=:sticks, color=1, markershape=:circle, xlabel=L"θ", ylabel=L"p(𝒟|θ)", label="", title="Likelihood: "*L"P(\mathcal{D}|\theta)")
end

# ╔═╡ 31c064a3-e074-409a-9431-48a573a83884
let
	gr()
	l = @layout [a b; c]
	N⁺ = sum(𝒟)
	Nᴬ = length(𝒟)
	likes = ℓ_binom.(N⁺, θs, Nᴬ; logprob=false)
	posterior_dis = likes/ sum(likes)
	post_plt = plot(θs, posterior_dis, seriestype=:sticks, markershape=:circle, label="", color=2, title="Posterior", xlabel=L"θ", ylabel=L"p(θ|𝒟)", legend=:outerright, size=(600,500))
	plot!(post_plt, θs, posterior_dis, color=2, label ="Posterior", fill=true, alpha=0.5)
	plot!(post_plt, θs, 1/length(θs) * ones(length(θs)), seriestype=:sticks, markershape=:circle, color =1, alpha=0.2, label="")
	plot!(post_plt, θs, 1/length(θs) * ones(length(θs)), color=1, label ="Prior", fill=true, alpha=0.2)
	like_plt = plot(θs, θ -> ℓ(θ, 𝒟), seriestype=:sticks, color=1, markershape=:circle, xlabel=L"θ", ylabel=L"P(\{0,0\}|θ)", label="", title="Likelihood: "*L"P(\{0,0\}|\theta)")
	plot(prior_plt_seller, like_plt, post_plt, layout=l)
end

# ╔═╡ fde1b917-9d81-4a29-b553-caad3c736682
md"""

## Two resellers ?

* The two unknowns: ``\theta_A,\theta_B``. 
* The likelihood are the counts of "heads" (or positive reviews): ``\mathcal{D} = \{N_{A}^+, N_{B}^+\}``.




"""

# ╔═╡ b4709e13-24ad-4c7e-bd9f-e7113b5089f5
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/twoseller.png" width = "350"/></center>
"""

# ╔═╡ 278bb5cb-6734-499c-b954-0590f559af1f
md"""

We have assumed here

* the two sellers' biases are independent 
* the reviews counts are also independent 
"""

# ╔═╡ 92e2e50b-bbce-4b4d-aad4-02e4f374b286
begin
	dis_size = 101
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
end

# ╔═╡ 316db89e-0b73-4993-a778-5d3ff890e807
md"""
## Two reseller: likelihood ``p(\mathcal{D}|\theta_A, \theta_B)``

#### The two resellers' performance are assumed independent

```math
\begin{align}
\large
p(\mathcal{D}|\theta_A, \theta_B) &= p(N_A^+|\theta_A, N_A=10) p(N_B^+|\theta_B, N_B=1000)\\

&=\binom{N_A}{N_A^+}\theta_A^{N_A^+}(1-\theta_A)^{N_A- N_A^+}\times \binom{N_B}{N_B^+}\theta_B^{N_B^+}(1-\theta_B)^{N_B- N_B^+}
\end{align}
```

* two independent Binomial likelihoods

"""

# ╔═╡ 2c6b9d29-1ab8-4b59-a867-1aecf27dc679
md"""

## Two reseller: posterior

Find the posterior


$$\begin{align}
\Large
p(\theta_A, \theta_B|\mathcal D) &\propto p(\theta_A, \theta_B)\cdot p(\mathcal D|\theta_A, \theta_B)\\
&\propto \theta_A^{N_A^+}(1-\theta_A)^{N_A- N_A^+}\times \theta_B^{N_B^+}(1-\theta_B)^{N_B- N_B^+}
\end{align}$$

Note that
* ``1/101^2`` is a constant *w.r.t.* ``\theta_A, \theta_B``
* ``\binom{N_B}{N_B^+}, \binom{N_A}{N_A^+}`` are also constants *w.r.t.* ``\theta_A,\theta_B``

And the normalising constant $p(\mathcal D)$ is

$$p(\mathcal D) = \sum_{\theta_A\in \{0.0, \ldots,1.0\}}\sum_{\theta_B\in \{0.0,\ldots, 1.0\}} \theta_A^{N_A^+}(1-\theta_A)^{N_A- N_A^+}\times \theta_B^{N_B^+}(1-\theta_B)^{N_B- N_B^+}$$ 


"""

# ╔═╡ cb09680f-ae53-45b1-aa61-e73f81a4bcce
md"""

## Posterior
"""

# ╔═╡ 022c5b56-e683-4f66-aba9-8b2e89f11e24
md"""
* ##### the posterior peaks around (0.8, 0.79); 
* ##### however, the ``\theta_A``'s marginal distribution is wider than the other, which makes perfect sense. 

"""

# ╔═╡ 56893f7f-e59c-434d-b8cf-2dd553111707
begin
	N₁, N₂ = 10, 1000
	Nₕ₁, Nₕ₂ = 8, 799
end;

# ╔═╡ 5a09f7ba-20df-4b28-a457-7e790efee888
begin
	function ℓ_two_coins(θ₁, θ₂; N₁=10, N₂=100, Nh₁=7, Nh₂=69, logLik=false)
		logℓ = ℓ_binom(Nh₁, θ₁, N₁;  logprob=true) + ℓ_binom(Nh₂, θ₂, N₂; logprob=true)
		logLik ? logℓ : exp(logℓ)
	end

	ℓ_twos = [ℓ_two_coins(θᵢ, θⱼ; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=true) for θᵢ in θ₁s, θⱼ in θ₂s]
	p𝒟 = exp(logsumexp(ℓ_twos))
	ps = exp.(ℓ_twos .- logsumexp(ℓ_twos))
end;

# ╔═╡ 32c06874-204d-4319-9879-8af6039d5cae
md"""


## Two resellers: Prior ``P(\theta_A, \theta_B)``


A uniform prior 

$$\large p(\theta_A, \theta_B) = \begin{cases} 1/101^2, & \theta_A,\theta_B \in \{0, 0.01, \ldots, 1.0\}^2 \\
0, & \text{otherwise}; \end{cases}$$


$(begin
gr()
p1 = Plots.plot(θ₁s, θ₂s,  (x,y) -> 1/(length(θ₁s)^2), st=:surface, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.0005, maximum(ps)], colorbar=false, c=:plasma, alpha =0.5, zlabel="density", title="Prior")
end)
"""

# ╔═╡ 5649bcda-2ab5-4bb7-b0dc-c132c5fb0a0d

let
	# using PlotlyKaleido
	plotly()
	p2 = Plots.plot(θ₁s, θ₂s,  ps', st=:surface, xlabel= "θA", ylabel="θB", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.0005, maximum(ps)], colorbar=false, c=:jet, zlabel="", alpha=0.9, title="Posterior")
end


# ╔═╡ 9c49b056-7329-49af-838b-3e8b28826157
md"""

$(begin
gr()
Plots.plot(θ₁s, θ₂s,  ps', st=:heatmap, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.003, maximum(ps)], colorbar=false, c=:plasma, zlabel="density", alpha=0.7, title="Posterior's density heapmap")

end)
"""

# ╔═╡ 0cd24282-74a4-4261-aa33-019d3afac248
md"""

## Lastly: answer the question

#### #The question boils down to a posterior probability

```math
\large P(\theta_A > \theta_B|\mathcal{D}).
```
* *In light of the data, how likely coin ``A`` has a higher bias than coin ``B``?*, 
* geometrically, calculate the probability below the shaded area


"""

# ╔═╡ 1cf90ba3-2f9a-4d58-bdd3-7b57e43fe307
begin
	gr()
	post_p_amazon =Plots.plot(θ₁s, θ₂s,  ps', st=:contour, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], colorbar=false, c=:thermal, framestyle=:origin)
	plot!((x) -> x, lw=1, lc=:gray, label="")
	equalline = StatsPlots.Shape([(0., 0.0), (1,1), (1, 0)])
	plot!(equalline, fillcolor = plot_color(:gray, 0.2), label=L"\theta_A>\theta_B", legend=:bottomright)
end

# ╔═╡ 5a58f305-4af0-40ae-830a-a318576bf85d
post_AmoreB = let
	post_AmoreB = 0
	for (i, θᵢ) in enumerate(θ₁s)
		for (j, θⱼ) in enumerate(θ₂s)
			if θᵢ > 1.0 * θⱼ
				post_AmoreB += ps[i,j]
			end
		end
	end
	post_AmoreB
end;

# ╔═╡ fdfa9423-e09e-42b6-bce3-ebbc681f3c93
md"

##### For our problem,"

# ╔═╡ bca20d87-e453-4106-a5cf-bd99d25021ab
L"\large p(\theta_A > \theta_B|\mathcal{D}) \approx %$(round(post_AmoreB, digits=2))"

# ╔═╡ da59ebaa-e366-4753-8329-27119be19d9a
md"""
* it is only about 37% chance that seller A is truly better than seller B
* 63% chance seller B is better!
"""

# ╔═╡ 3caf8d96-7cdd-45f4-a54e-08b7a1bc2251
md"""

## Question

> What's the chance that seller A's positive rate is twice better than seller B?

"""

# ╔═╡ 12f92e46-9581-4c04-be11-3f3f9b316f26
md"""

!!! hint "Answer"

	You can even calculate 

	```math
	p(\theta_A > 2\cdot \theta_B|\mathcal{D}) = \sum_{\theta_A >2 \theta_B:\;\theta_A,\theta_B\in \{0.0, \ldots 1.0\}^2} p(\theta_A, \theta_B|\mathcal{D})
	```
	
	* just a different shaded area!


"""

# ╔═╡ a30ed481-a09a-4d9c-a960-2ca3b4898e30
@bind k Slider(0.1:0.05:3; default =1, show_value=true)

# ╔═╡ d6550d70-a1c1-48fd-b99d-c041b40efad2
post_AmoreKB = let
	post_AmoreKB = 0
	for (i, θᵢ) in enumerate(θ₁s)
		for (j, θⱼ) in enumerate(θ₂s)
			if θᵢ > k*θⱼ
				post_AmoreKB += ps[i,j]
			end
		end
	end
	post_AmoreKB
end;

# ╔═╡ d7a4bd52-8cf1-4495-bc14-2fe19fc9a037
let
	gr()
	post_p_amazon =Plots.plot(θ₁s, θ₂s,  ps', st=:contour, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], colorbar=false, c=:thermal, framestyle=:origin)
	plot!((x) -> x, lw=1, lc=:gray, label="")
	equalline = StatsPlots.Shape([(0., 0.0), (1,1), (1, 0)])
	plot!(equalline, fillcolor = plot_color(:gray, 0.2), label=L"\theta_A>\theta_B", legend=:bottomright)
	kline = StatsPlots.Shape([(0., 0.0), (1,1/k), (1, 0)])
	plot!(post_p_amazon, kline, fillcolor = plot_color(:gray, 0.5), label=L"\theta_A>%$(k)\cdot \theta_B", legend=:bottomright, title =L"P(\theta_A > %$(k) \theta_B|\mathcal{D})\approx %$(round(post_AmoreKB, digits=2))")
end

# ╔═╡ 98672fd3-f85c-40e2-8f71-3a46b8632bf6
# only works for uni-modal
function find_hpdi(ps, α = 0.95)
	cum_p, idx = findmax(ps)
	l = idx - 1
	u = idx + 1
	while cum_p <= α
		if l >= 1 
			if u > length(ps) || ps[l] > ps[u]
				cum_p += ps[l]
				l = max(l - 1, 0) 
				continue
			end
		end
		
		if u <= length(ps) 
			if l == 0 || ps[l] < ps[u]
				cum_p += ps[u]
				u = min(u + 1, length(ps))
			end
		end
	end
	return l+1, u-1, cum_p
end;

# ╔═╡ e4af93c4-19f9-4d94-8ed2-729852ca9432
begin
	θs_refined = 0:0.01:1.0
	likes = ℓ_binom.(N⁺, θs_refined, Nᴬ; logprob=false)
	posterior_dis_refined = likes/ sum(likes)
	post_plt3 = Plots.plot(θs_refined, posterior_dis_refined, seriestype=:sticks, markershape=:circle, markersize=2, label="", title=L"\mathrm{Posterior}\, \, p(\theta|\mathcal{D})", xlabel=L"θ", ylabel=L"p(θ|𝒟)")
	l2, u2, _ = find_hpdi(posterior_dis_refined, 0.9)
	Plots.plot!(post_plt3, θs_refined[l2:u2], posterior_dis_refined[l2:u2], seriestype=:sticks, markershape=:circle, markersize=2, label="", legend=:topleft)
	Plots.plot!(post_plt3, θs_refined[l2:u2], posterior_dis_refined[l2:u2], seriestype=:path, fill=true, alpha=0.3, color=2, label="90% credible interval", legend=:topleft)
	Plots.annotate!((0.7, maximum(posterior_dis_refined)/2, ("90 % HDI", 15, :red, :center, "courier")))

end

# ╔═╡ 80cb7bab-4eb0-4ec1-a067-9d661f2c4da3
md"""
## Report the posterior

##### The posterior distribution ``P(\theta|\mathcal{D})`` provides all the information

\
\


Sometimes, we still want to **summarise the distribution**, one possibility is **highest probability density interval (HPDI)** 

* Bayesian **credible interval** (c.f. confidence interval)

* an interval that the encloses probability *e.g.* 90% of the posterior

$$\large p(l \leq \theta \leq u|\mathcal D) = 90 \%.$$

* however, the wide tail suggests it is not very certain; we only have observed 10 tosses after all
  * ##### the 90% region for ``\theta`` is ($(θs_refined[l2]), $(θs_refined[u2]))
"""

# ╔═╡ f7456b7a-3d05-468a-a7ee-50a83affba72
md"""

## Frequentist method ?


One may choose a test say χ²-test with testing hypothesis

> ``\mathcal{H}_0:`` A and B have the same positive review rate
>
>  and 
>
> ``\mathcal{H}_1:``  A and B have different positive review rate

* one then proceeds to compute the required statistic (with no obvious reasons why)

* and the result might be reported like

> the two sellers offer **significantly different** levels of service (``p``), say 0.05


How to interpret the statement?


It **does not** tell you

!!! danger ""
	there is a ``(1-p)`` chance that the sellers differ in positive ratings


but rather 


!!! correct ""
	if we did this experiment many times, and the two resellers had equal ratings, then p of the time you would find a value of χ2 more extreme than the one that calculated above.


And the response or interpretation does not really directly answer the original question we are interested in, *i.e.*

!!! warning ""
	Is A better B?

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
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
Distributions = "~0.25.103"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
Plots = "~1.40.8"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.54"
SpecialFunctions = "~2.4.0"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.0"
manifest_format = "2.0"
project_hash = "1aee49cc05200fd5b1875035594f03c20c7085f2"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

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
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d7477ecdafb813ddee2ae727afa94e9dcb5f3fb0"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.112"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee28ddcd5517d54e417182fec3886e7412d3926f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f31929b9e67066bee48eec8b03c0df47d31a74b3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "7c4195be1649ae622304031ed46a2f4df989f1eb"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.24"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

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
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

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
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "260dc274c1bc2cb839e758588c63d9c8b5e639d1"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "3cebfc94a0754cc329ebc3bab1e6c89621e791ad"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "650a022b2ce86c7dcfbdecf00f78afeeb20e5655"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.2"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "45470145863035bb124ca51b320ed35d071cc6c2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.8"

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
git-tree-sha1 = "5d9ab1a4faf25a62bb9d07ef0003396ac258ef1c"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.15"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "7f4228017b83c66bd6aa4fddeb170ce487e53bc7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ff11acffdb082493657550959d4feb4b6149e73a"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.5"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "1165b0443d0eca63ac1e32b8c0eb69ed2f4f8127"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
# ╟─f3c62849-620d-47b3-b698-79c925897d54
# ╟─30b4bf3c-82dc-46f7-bd69-5fdeada9ebad
# ╟─77535564-648a-4e17-83a0-c562cc5318ec
# ╟─2057c799-18b5-4a0f-b2c7-66537a3fbe79
# ╟─43cedf4c-65c4-4b2b-afb3-21f5827f2af6
# ╟─351694f5-1b83-426f-a190-c87c69152ef0
# ╟─66926dbf-964d-4048-b8ad-459f36236c14
# ╟─1b4c860a-7245-4793-8acf-a1f01700b499
# ╟─3319498b-db49-4516-b699-b99da093c9cb
# ╟─0f8b4a5b-94b8-4f2b-975d-3579eb9c62b1
# ╟─1f524d92-e589-48f9-b3cc-bcef0bae4bca
# ╟─ce1d28e1-3c77-461e-822a-6e5783bf2744
# ╟─dafe8d1f-6c99-46a4-a2d1-2e34e12ba4a7
# ╟─1bc6b970-f1fa-48c9-b17e-cef4d01aa47a
# ╟─15571edc-32d7-4ac5-9c53-d49f595f84b1
# ╟─19892f66-98e1-405e-abae-ffda46a546b1
# ╟─1d835b12-f1d7-4ce4-a62e-2cfa7a7b3a21
# ╟─5cbd933e-db0e-49fb-b9d0-267dc93efdb7
# ╟─f06264f4-f020-48f6-9ff3-cac308518eac
# ╟─9308405d-36d0-41a1-8c73-e93b8d699320
# ╟─be0fe9aa-32e6-4c18-8bad-2213e05bca1c
# ╟─0da00057-6bf1-407d-a15f-af24ee549182
# ╟─f593d1c9-7382-4d5c-9c99-5709123af43d
# ╟─31c064a3-e074-409a-9431-48a573a83884
# ╟─80cb7bab-4eb0-4ec1-a067-9d661f2c4da3
# ╟─e4af93c4-19f9-4d94-8ed2-729852ca9432
# ╟─8f9d1066-91d5-43af-9834-b3dc2dbc6400
# ╟─70c00cf9-65ac-44d7-b25c-153614d24240
# ╟─947a0e0f-b620-4e0e-97bd-74749ed87042
# ╟─1e267303-160d-43b9-866f-2d7afc2c4c3f
# ╟─4bae9df7-36a2-4e8a-bdd8-5fdec68cddba
# ╟─2338bade-9fb2-4a75-a39c-3e056f2d00d5
# ╟─b3bcf03b-f4f8-4ddb-ab59-fa5f780ed563
# ╟─5462f241-e284-45b8-8e61-3c0317f6b08b
# ╟─ff01e390-a72d-41b0-9a7c-1d63060491f9
# ╟─6d136ab7-c7ea-43a7-a16d-0af53b123b59
# ╟─c1edf5ff-f25f-4a71-95fb-53515aae9d97
# ╟─84f66218-1309-4cfd-8250-11bd75f78eb4
# ╟─98e84190-393a-4eb9-8f1c-fa90bf8973d2
# ╟─a152a7bd-062d-4fd9-be38-c217fdaca20f
# ╟─6131ad55-eee7-44ec-b965-ce17ef3f8f84
# ╟─a2642e09-3202-4c73-914a-bdd6c0b374c2
# ╟─89f0aa81-1da3-41a0-8d72-72637d8052aa
# ╟─4f6232be-dbd1-4e1a-a28c-5fc60964bb9f
# ╟─d76d0594-5a53-4641-818e-496ce3a52c6b
# ╟─5a09f7ba-20df-4b28-a457-7e790efee888
# ╟─fde1b917-9d81-4a29-b553-caad3c736682
# ╟─b4709e13-24ad-4c7e-bd9f-e7113b5089f5
# ╟─278bb5cb-6734-499c-b954-0590f559af1f
# ╟─32c06874-204d-4319-9879-8af6039d5cae
# ╟─92e2e50b-bbce-4b4d-aad4-02e4f374b286
# ╟─316db89e-0b73-4993-a778-5d3ff890e807
# ╟─2c6b9d29-1ab8-4b59-a867-1aecf27dc679
# ╟─cb09680f-ae53-45b1-aa61-e73f81a4bcce
# ╟─022c5b56-e683-4f66-aba9-8b2e89f11e24
# ╠═5649bcda-2ab5-4bb7-b0dc-c132c5fb0a0d
# ╠═56893f7f-e59c-434d-b8cf-2dd553111707
# ╟─9c49b056-7329-49af-838b-3e8b28826157
# ╟─0cd24282-74a4-4261-aa33-019d3afac248
# ╟─1cf90ba3-2f9a-4d58-bdd3-7b57e43fe307
# ╟─5a58f305-4af0-40ae-830a-a318576bf85d
# ╟─fdfa9423-e09e-42b6-bce3-ebbc681f3c93
# ╟─bca20d87-e453-4106-a5cf-bd99d25021ab
# ╟─da59ebaa-e366-4753-8329-27119be19d9a
# ╟─3caf8d96-7cdd-45f4-a54e-08b7a1bc2251
# ╟─12f92e46-9581-4c04-be11-3f3f9b316f26
# ╟─a30ed481-a09a-4d9c-a960-2ca3b4898e30
# ╟─d7a4bd52-8cf1-4495-bc14-2fe19fc9a037
# ╟─d6550d70-a1c1-48fd-b99d-c041b40efad2
# ╟─98672fd3-f85c-40e2-8f71-3a46b8632bf6
# ╟─f7456b7a-3d05-468a-a7ee-50a83affba72
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
