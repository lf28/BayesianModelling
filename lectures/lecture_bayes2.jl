### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
begin
    using PlutoTeachingTools
    using PlutoUI
    using Plots
    default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
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

# ╔═╡ be719382-bf1f-4442-8419-bddcda782525
TableOfContents()

# ╔═╡ 45f045d9-cdb7-4ade-a8e6-1f1a984cc58a
ChooseDisplayMode()

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

* #### Prior distribution choices
  * some common distributions for priors




* #### Conjugate inference 









"""

# ╔═╡ c677a2d2-f4be-483a-a53b-7b76efb1b80f
md"""

## Priors are useful

#### Recall the coughing example

* ``H``: the unknown cause ``h \in \mathcal{A}_H=\{\texttt{cold}, \texttt{cancer}, \texttt{healthy}\}`` 

* ``\mathcal{D}``: the observed data is ``\mathcal{D} = \texttt{Cough}``
 
#### and we have used the **Prior:** ``P(H)``

$$P(h) = \begin{cases} 0.8 & h=\texttt{healthy} \\
0.18 & h=\texttt{cold}\\ 
0.02 & h=\texttt{cancer}\end{cases}$$


#### This is an *informative* prior

* instead of a non-informative (say uniform distribution)
* ###### without the prior, the inference results would be wrong (it would be proportional to the likelihood)

"""

# ╔═╡ 56f2bb33-637c-4450-9769-f338937594da
md"""

## Priors are useful


#### Priors serve very useful purposes in Bayesian inference
* ##### to formally incorporate expert knowledge into the statistical inference


* ##### known as regularisation effects in machine learning
  * prevents overfitting
  * priors contains information from the unseen data 

"""

# ╔═╡ 47359bc3-cfc2-4777-82f9-308e56dca491
md"""

## How to choose priors



#### The first and foremost principle is 

!!! note ""
	#### *Priors with matching support*


#### *Example* for the coin-flipping example

* ##### the unknown bias: ``\theta \in [0,1].`` 

* ##### therefore, a Gaussian distribution is not suitable


## Suitable priors for ``\theta \in [0,1]``
#### Some suitable priors are 

* ##### Uniform distribution between 0 and 1, 
* ##### truncated Gaussian (truncated between 0 and 1) 
* ##### or Beta distribution

```math

\begin{align}
\large p(\theta) &= \texttt{TruncNormal}(\mu, \sigma^2) \propto \begin{cases} \mathcal N(\mu, \sigma^2) & {0\leq \theta \leq 1} \\ 0 & \text{otherwise}  \end{cases}  \\
p(\theta) &= \texttt{Uniform}(0,1) = \begin{cases} 1 & {0\leq \theta \leq 1} \\ 0 & \text{otherwise}  \end{cases} 

\end{align}
```
"""

# ╔═╡ 35311403-187a-4e23-9e52-7eb0a212452d
let
    σ = sqrt(0.1)
    Plots.plot(truncated(Normal(0.5, σ), 0., 1), lw=2, label=L"\texttt{TrunNormal}" * "(0.5, $(round((σ^2);digits=2)))", legend=:outerright, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"θ\in [0,1]")
    Plots.plot!(truncated(Normal(0.25, σ), 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.25, $(round((σ^2);digits=2))))")
    Plots.plot!(truncated(Normal(0.75, σ), 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.75, $(round((σ^2);digits=2))))")
    Plots.plot!(Uniform(0, 1), lw=2, label=L"\texttt{Uniform}(0, 1)")
end

# ╔═╡ 9fa2c7bf-3fa2-4869-8515-fa0048fa3cd6
md"""

## Suitable priors for ``\theta > 0``


#### To infer the variance ``\sigma^2`` of a Gaussian

* ##### note that:  ``\sigma^2 >0``. 


#### Some possible priors on the positive real line are 

* ##### e.g. Exponential distribution
* ##### Gamma, Inverse Gamma
* ##### or Half-Cauchy

"""

# ╔═╡ ae234663-90b4-432d-bacb-188dd1ee1034
let
    xpltlim1 = -1
    xpltlim2 = 6
    Plots.plot(xpltlim1:0.01:xpltlim2, Exponential(1), lw=1.5, label=L"\texttt{Exponential}(1.0)", legend=:best, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"σ^2\in (0,∞)")
    Plots.plot!(xpltlim1:0.01:xpltlim2, Exponential(2), lw=1.5, label=L"\texttt{Exponential}(2.0)")
    Plots.plot!(xpltlim1:0.01:xpltlim2, Exponential(5), lw=1.5, label=L"\texttt{Exponential}(5.0)")
    Plots.plot!(xpltlim1:0.01:xpltlim2, truncated(Cauchy(0, 1), lower=0), lw=1.5, label=L"\texttt{HalfCauchy}(0, 1)")
    # Plots.plot!(0:0.1:10, truncated(Cauchy(0, 2), lower= 0), label="HalfCauchy(0, 2)")
    # Plots.plot!(0:0.1:10, truncated(Cauchy(0, 5), lower= 0), label="HalfCauchy(0, 5)")
end

# ╔═╡ d764a435-7e11-4736-a0d0-0f2d5265b7af
md"""

## Conjugate prior


#### Conjugate priors are

* ##### posterior distribution is of the same form as prior
  * we will see an example soon
* ##### *benefits:* computational cheap and simple 
  * as the posterior will be of the closed form (usually)
  * the inference/computation is just updating prior parameters



## Example: Beta-Bernoulli model


##### For the coin flipping example, the conjugate prior of the bias ``\theta`` is 

```math
\Large p(\theta) = \texttt{Beta}(\theta; a_0, b_0) = \frac{1}{\text{B}(a_0, b_0)} \theta^{a_0-1}(1-\theta)^{b_0-1},
```
* ``a_0,b_0 >0`` are the prior's parameter 
* and ``B(a_0,b_0)``, the beta function, is a normalising constant for the Beta distribution: 

* *i.e.* ``\mathrm{B}(a_0, b_0) = \int \theta^{a_0-1}(1-\theta)^{b_0-1}\mathrm{d}\theta``. 


"""

# ╔═╡ f53420ae-6007-451f-9eb7-3937abf114ef
aside(tip(md"""

Unfortunately, conjugate priors only exist in some very simple Bayesian models
* not that important for the practical sense
* but provides us insights into the purpose of priors
"""))

# ╔═╡ 5aaf5221-e77b-4339-95c5-dc21bd815fc3
md"""

## Beta distribution


#### A few Beta distributions  are plotted below

* ``a_0=b_0=1``, the prior reduces to a uniform distribution. 
* ``a_0> b_0``, e.g. ``\texttt{Beta}(5,2)``, the prior belief has its mode > 0.5,

  * which implies the prior believes the coin is biased towards the head; 




* ``a_0< b_0`` and vice versa.
"""

# ╔═╡ 047a8e19-54d4-454a-8635-8becd4b93061
let
    plot(Beta(1, 1), xlims=[0, 1], ylim=[0, 3], xlabel=L"\theta", ylabel=L"p(\theta)", label=L"\texttt{Beta}(a_0=1,b_0=1)", linewidth=2, legend=:outerright, size=(600, 300))
    plot!(Beta(0.5, 0.5), xlims=[0, 1], ylim=[0, 3], label=L"\texttt{Beta}(a_0=.5,b_0=.5)", linewidth=2)
    plot!(Beta(5, 5), xlims=[0, 1], ylim=[0, 3], label=L"\texttt{Beta}(a_0=2,b_0=2)", linewidth=2)
    plot!(Beta(5, 2), xlims=[0, 1], ylim=[0, 3], label=L"\texttt{Beta}(a_0=5,b_0=2)", linewidth=2)
    plot!(Beta(2, 5), xlims=[0, 1], ylim=[0, 3], label=L"\texttt{Beta}(a_0=2,b_0=5)", linewidth=2)
end

# ╔═╡ 43cedf4c-65c4-4b2b-afb3-21f5827f2af6
md"""
## Example: Beta-Bernoulli model (cont.)


> #### A coin 🪙 is tossed 10 times. 
> * ##### the tossing results are recorded:  $\mathcal D=\{1, 1, 1, 0, 1, 0, 1, 1, 1, 0\}$; 7 out of 10 are heads


"""

# ╔═╡ 4e82a77d-af72-4653-aaa9-f7303e33c3ea
md"""

## Example: Beta-Bernoulli model (cont.)


#### Recall the likelihood is 


```math
\large
p(\mathcal{D}|\theta) = \prod_i p(d_i|\theta) =\theta^{\sum_n d_n} (1-\theta)^{N-\sum_n d_n} = \theta^{N_h} (1-\theta)^{N-N_h}
```

* assume ``n`` independent Bernoulli trial
* ``N_h = \sum_i d_i`` is the total number of heads observed in the ``N`` tosses.



#### Apply Baye's rule, the posterior is still of a Beta form (therefore, the **conjugacy**)

```math
\large
p(\theta|\mathcal D) = \texttt{Beta}(\theta; a_N, b_N) = \frac{1}{\text{B}(a_N, b_N)} \theta^{a_N-1}(1-\theta)^{b_N-1},
```

* where ``a_N= a_0 + N_h`` and ``b_N = b_0 + N - N_h``; and 


"""

# ╔═╡ e2673877-6536-4670-b5b7-b84c36503512
Foldable(
    "Details about Beta-Binomial model's conjugacy",
    md"
Apply Baye's rule, we have 

```math
\large\begin{align}
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
"
)

# ╔═╡ 33714d50-a04e-41ad-844c-293ed9671163
md"""

## Example (cont.)


#### For our example, 

* ``N_h = 7`` and ``N_t=3`` 


* ##### with prior ``p(\theta)= \texttt{Beta}(1, 1),``

#### The updated posterior is

$$\Large p(\theta|\mathcal D)= \texttt{Beta}(1+7, 1+3).$$



"""

# ╔═╡ 9308405d-36d0-41a1-8c73-e93b8d699320
begin
    𝒟 = [1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    # 𝒟 = [0, 0]

    # 𝒟 = rand(100) .< 0.5

end

# ╔═╡ c32714b7-fa1d-443c-a5a2-4a511fc701bd
let
    nh, nt = sum(𝒟), length(𝒟) - sum(𝒟)
    plot(Beta(1, 1), xlims=[0, 1], label=L"p(\theta)= \texttt{Beta}(1,1)", linewidth=1, xlabel=L"\theta", ylabel="density", fill=true, lw=2, alpha=0.2, legend=:outerright, color=1, title="Conjugate posterior update")
    vline!([mean(Beta(1, 1))], label="prior mean", lw=2, lc=1, ls=:dash)
    plot!(Beta(1+nh, 1+nt), xlims=[0, 1], fill=true, lw=2, alpha=0.2, color=2, label=L"p(\theta|\mathcal{D})= \texttt{Beta}(8,4)", linewidth=2)
    vline!([mean(Beta(1+nh, 1+nt))], label="posterior mean", lw=2, lc=2, ls=:dash)
end

# ╔═╡ c74a488d-0e1a-4ddc-9d9b-fcb5e813d587
md"""
## Remarks


#### Conjugate priors makes the computation simple:

$$\large a_N= a_0+ N_h\;\; b_N= b_0+N_t.$$ 

* if we choose a truncated Gaussian prior, then the posterior distribution is much harder to compute
  * we need to resort to advanced methods like Markov Chain Monte Carlo (MCMC).



#### Conjugate priors makes interpretation easy:

* ##### ``a_N`` and ``b_N`` are the posterior counts of the heads and tails

* ##### the prior parameters ``a_0, b_0``: some **pseudo observations** contained in the prior

  * e.g., ``a_0=b_0=1`` implies the prior contains one pseudo count of head and tail each (therefore, the prior is flat). 


"""

# ╔═╡ fe47a810-cf29-4adf-a390-6f595b8f3ec9
md"""
## Informative vs Non-informative

#### All priors can be largely classified into two groups: 

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

# ╔═╡ 30ece808-4852-4cbd-84c5-7e8087753ad5
# md"""
# ## Sequential update*


# Conjugate prior also provides us with a simple procedure to do **sequential** inference


# * *sequential online learning:* update the posterior incrementally as data arrives

# ```math
# \begin{align}
# p(\theta|\{d_1, d_2, \ldots, d_N\})&\propto  p(\theta) \prod_{n=1}^N p(d_n|\theta) \\
# &= \underbrace{p(\theta) \prod_{n=1}^{N-1} p(d_n|\theta)}_{p(\theta|\mathcal D_{N-1})}\cdot p(d_N|\theta)\\
# &= \underbrace{p(\theta|\mathcal D_{N-1})}_{\text{new prior}} p(d_N|\theta).
# \end{align}
# ```

# * *yesterday's posterior becomes today's prior*

# ## Sequential update*

# To be more specific, the sequential update algorithm is:

# ---
# Initialise with a prior ``p(\theta|\emptyset)= \texttt{Beta}(a_0, b_0)``

# For ``n = 1,2,\ldots, N``:
# * update 

# $$a_n = a_{n-1} + \mathbf{1}(d_n=\texttt{head}), \;\; b_n = b_{n-1} +  \mathbf{1}(d_n=\texttt{tail})$$
# ---

# Note that the function ``\mathbf{1}(\cdot)`` returns 1 if the test result of the argument is true and 0 otherwise. 

# ## Demonstration: 200 coin tosses

# """

# ╔═╡ 9638d082-5b61-4051-912b-fc263abeb239
begin
    # simulate 200 tosses of a fair coin
    N_tosses = 200
    true_θ = 0.5
    Random.seed!(100)
    coin_flipping_data = rand(N_tosses) .< true_θ
    Nh = sum(coin_flipping_data)
    Nt = N_tosses - Nh
end;

# ╔═╡ 997d45bf-cdbf-4881-b823-ebb7d9ec19db
# let
# 	a₀, b₀ = 1, 1
# 	prior_θ = Beta(a₀, b₀)
# 	plot(prior_θ, xlim = [0, 1], lw=2, fill=(0, 0.1), label="Prior "* L"p(\theta)", xlabel=L"\theta", ylabel="density", title="Sequential update "*L"N=0", legend=:topleft)
# 	plot!(Beta(a₀ + Nh, b₀+ Nt), lw=2, fill=(0, 0.1), label="Posterior "* L"p(\theta|\mathcal{D})")
# 	vline!([true_θ], label="true "*L"θ", lw=4, lc=3)
# 	an = a₀
# 	bn = b₀
# 	anim=@animate for n in 1:N_tosses
# 		# if the toss is head, update an
# 		if coin_flipping_data[n]
# 			an += 1
# 		# otherwise
# 		else	
# 			bn += 1
# 		end
# 		poster_n = Beta(an, bn)
# 		# plot every 5-th frame
# 		if (n % 5) == 0
# 			plot!(poster_n, lw=1, label="", title="Sequential update with observation "*L"N=%$(n)")
# 		end
# 	end
# 	gif(anim, fps=15)
# end

# ╔═╡ 9bbfb923-1448-45c9-8091-bfd86c4d54bb
md"""

## More conjugate example: Gamma-Gaussian*

#### Consider the problem
> #### Given i.i.d observations from a zero mean Gaussian 
> $\large \mathcal D=\{d_i\}_{i=1}^n$
> $\large d_i \sim \mathcal{N}(\mu, \sigma^2)$
> * ##### infer the variance ``\sigma^2`` of the Gaussian
> * ##### assume ``\mu`` is known


"""

# ╔═╡ 63f39880-d150-44fe-8c02-d3dc9b4deb1f
md"""
#### 
* ##### it is more convenient to model the precision 

$\large\phi\triangleq 1/\sigma^2$ 

* ##### a conjugate prior for ``\phi`` is Gamma distribution
  * note that Gamma distributions have support on the positive real line which is required here


## Aside: MLE for ``\sigma^2`` and ``\phi``

#### It can be shown that the MLE for ``\hat{\sigma}^2`` is

```math
\large \hat{\sigma}^2 = \frac{\sum_{n} (d_n- \mu)^2}{N}
```


* ##### sum of squared errors divided by ``N``

#### Therefore, the MLE for ``\phi`` is 

```math
\large \hat{\phi} = \frac{N}{\sum_{n} (d_n- \mu)^2}
```




## Aside: Gamma distribution

#### A Gamma distribution, parameterised with a shape  ``a_0>0``,  and a rate parameter ``b_0>0``

```math
\large
p(\phi; a_0, b_0) = \texttt{Gamma}(\phi; a_0, b_0)=\frac{b_0^{a_0}}{\Gamma(b_0)} \phi^{a_0-1} e^{-b_0\phi}.
```

* we will interpret ``a_0``, ``b_0`` in a bit


!!! info ""
	The mean of a Gamma distribution is 

	$$\mathbb{E}[\phi] = \frac{a_0}{b_0}$$
"""

# ╔═╡ 898a0670-d5bb-42f9-8afd-c3a51c426065
let
    as_ = [1, 2, 3, 5, 9, 7.5, 0.5]
    bs_ = [1/2, 1/2, 1/2, 1, 1/0.5, 1, 1]
    plt = plot(xlim=[0, 20], ylim=[0, 0.8], xlabel=L"\phi", ylabel="density")
    for i in 1:length(as_)
        plot!(Gamma(as_[i], 1/bs_[i]), fill=(0, 0.1), lw=1.5, label=L"\texttt{Gamma}"*"($(as_[i]),$(bs_[i]))")
    end
    plt
end

# ╔═╡ 363da006-2e5c-4bb0-8849-a730ffd6c745
md"""

## Conjugacy


#### The posterior of the precision $\phi$



```math
\Large
p(\phi|\mathcal D, \mu) = \texttt{Gamma}(a_N, b_N),
```
* #### where 

$\large a_N= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.$




#### Interpretation
* ##### ``a_N``: the number of observation counts (divided by 2)
* ##### ``b_N``: the sum of squared errors (divided by 2)
"""


# ╔═╡ cf7b36c9-1c78-4ef6-bd06-2590b67c3703
md"""
## Details*
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



The posterior is: 

```math
p(\phi|\mathcal D, \mu) = \texttt{Gamma}(a_N, b_N),
```
* where 

$a_N= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.$


Therefore, it is not hard to see
* ``a_N``: the updated observation counts (divided by 2)
* ``b_N``: the updated sum of squared errors (divided by 2)


##

Also note the posterior mean is 

$$\mathbb{E}[\phi|\mathcal D] = \frac{a_N}{b_N} = \frac{a_0 + N/2}{b_0 + {\sum_n (d_n -\mu)^2}/{2}}.$$

* if we assume ``a_0= b_0=0`` (an improper  prior), the maximum likelihood estimator for ``{\sigma^2}`` is recovered

$$\hat \sigma^2 =1/\hat{\phi} = \frac{\sum_n (d_n-\mu)^2}{N}$$

"""

# ╔═╡ e1bb37e8-d3b5-416f-b90a-fb611864b402
Foldable(
    "Derivation details on the Gamma-Gaussian conjugacy.",
    md"
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

"
)

# ╔═╡ 4f4882c5-bd3f-4fbe-8a59-cafc5d365d99
md"""

## Demonstration


#### **Demonstration:** We first simulate ``N=100`` Gaussian observations with 

* ##### mean ``\mu=0`` and 
* ##### ``\phi= 1/\sigma^2 = 2``.

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
    posterior_ϕ = Gamma(a₀ + N/2, 1/(b₀ + sum(gaussian_data .^ 2)/2))
end;

# ╔═╡ 1a483bed-1639-41b4-ad7f-40b009bd45a9
let
    plot(prior_ϕ, xlim=[0, 5], ylim=[0, 1.5], lw=2, fill=(0, 0.2), label="Prior " * L"p(\phi)", xlabel=L"\phi", ylabel="density", title="Conjugate inference of a Gaussian's precision")

    plot!(posterior_ϕ, lw=2, fill=(0, 0.2), label="Posterior " * L"p(\phi|\mathcal{D})")
    vline!([true_ϕ], label="true "*L"\phi", lw=4)
    vline!([mean(prior_ϕ)], label="prior mean", lw=2, lc=1, ls=:dash)
    vline!([mean(posterior_ϕ)], label="posterior mean", lw=2, lc=2, ls=:dash)
    # vline!([1/σ²], label="true "*L"λ", lw=2)
    # end
end

# ╔═╡ 29a49584-3a52-410e-8f15-1d9293955826
# md"""

# # Bayesian predictive checks

# ## This time

# * Predictive distributions and their checks 
#   * prior predictive check
#   * posterior predictive check




# """

# ╔═╡ 6b460036-890d-4364-aac2-2c61dc44ed75
# md"""

# ## Prediction 



# In many applications, we also want to make predictions based on observations ``\mathcal{D}``


# **Bayesian prediction** 
# * routinely applies probability rules
# * aims at computing the following distribution

# ```math
# p(\mathcal{D}_{pred}|\mathcal{D})
# ```


# **Frequentist prediction**
# * usually employs the *plug-in* principle
# * plug in the estimated parameter to the likelihood
# ```math
# p(\mathcal{D}_{pred}|\theta_{ML})
# ```

# """

# ╔═╡ df3c6bd8-e81f-4ccb-b0d1-98832e41537f
# md"""

# ## Prediction -- example

# Let's use the coughing case as an example again

# * the observation: ``\mathcal{D} = \texttt{Cough}``

# **Prediction:** he is going to **cough** **again**

# ```math
# p(\texttt{Cough}_{again}|\texttt{Cough})
# ```


# * for prediction, we can introduce an unobserved node to our graph
# """

# ╔═╡ 5440ceb3-b791-4935-8b59-339010546090
# html"<center><img src='https://sites.cs.st-andrews.ac.uk/people/lf28/figs/cough_pred_bn.png' width = '250' /></center>"

# ╔═╡ 8c763ff2-cfda-49a6-b0ce-d5530f665971
# md"""

# Bayesian routinely applies probability rules to compute

# ```math
# \begin{align}
# p(\texttt{cough}_{\text{again}} |\texttt{caugh}) &\propto p(\texttt{cough}_{\text{again}} , \texttt{caugh}) \\
# &= \sum_{h} p(h, \texttt{cough}_{\text{again}} , \texttt{caugh}) \\
# &= \sum_{h} p(\texttt{cough}_{\text{again}}|h) \underbrace{p(\texttt{cough}|h) p(h)}_{= p(h, \texttt{cough})\propto \; p(h|\texttt{cough})} \\
# &= \sum_{h} p(\texttt{cough}_{\text{again}}|h) p(h|\texttt{cough})
# \end{align}
# ```


# ## Remarks

# **Bayesian** prediction is

# ```math
# \begin{align}
# p(\texttt{cough}_{\text{again}} |\texttt{caugh}) 
# &= \sum_{h} p(\texttt{cough}_{\text{again}}|h) p(h|\texttt{cough})
# \end{align}
# ```


# * the prediction is an ensemble prediction, *i.e.* balanced and **weighted average**

# * each unknown hypothesis ``h\in \{\texttt{cancer}, \texttt{healthy}, \texttt{cold}\}`` is used to make a prediction for another ``\texttt{cough}_{\text{again}}``

# * each prediction is weighted by its corresponding posterior ``p(h|\texttt{cough})``


# In comparison, the **frequentist** plug-in method's predictions are 

# ```math
# p(\texttt{cough}_{\text{again}} | h = \hat{h}_{\text{MLE}}= \texttt{cancer})
# ```

# * prediction by a single *point* hypothesis
# """

# ╔═╡ b13d76bf-f559-4842-bdfa-78953d74d69a
# md"""

# ## Why Bayesian prediction is better?


# Consider the coin flipping example, supposed we have observed data

# > A coin 🪙 is tossed 2 times. 
# > * the tossing results are recorded:  $\mathcal D=\{0,0\}$; 0 out of 2 are heads
# > * **predict** the next toss?
# """

# ╔═╡ 37a564d5-8c46-462e-8d3d-45b93ff1d3e4
# md"""
# **Frequentist's prediction** is based on plug-in principle

# The MLE is

# ```math
# 	\theta_{ML} = \frac{N^+}{N} = \frac{0}{2} = 0
# ```
# """

# ╔═╡ 03181c93-6a31-46d6-aa64-1521234e2341
# md"""




# $(let
# 	gr()
# 	𝒟 = [0,0]
# 	θs = 0:0.01:1
# 	like_plt_seller = plot(θs, θ -> exp(loglikelihood(Bernoulli(θ), [0,0] )), seriestype=:line, color=1, xlabel=L"θ", ylabel=L"P(\{0,0\}|θ)", label="", title="Likelihood: "*L"P(\{0,0\}|\theta)")

# end)

# If you treat $$\hat{\theta}_{\text{ML}} = 0$$ as gospel for future prediction: *i.e.*

# $$P(Y_{N+1}|\theta = \hat{\theta}_{\text{ML}})=\begin{cases} 0 & Y_{N+1} =1 \\ 1 & Y_{N+1} =0\end{cases}$$
# * you predict you will **never** see a `Head`/1 again: **Overfitting**


# * the frequentist MLE based prediction is pathologically bad when 
#   * there is not enough data
# """

# ╔═╡ 80400b4b-b130-4719-957a-4c92b556916a
# md"""

# ## Why Bayesian is better?


# The Bayesian instead computes the integration


# ```math
# p(y_{N+1} | \mathcal{D}) = \int_\theta p(y_{N+1}|\theta) p(\theta|\mathcal{D})d\theta
# ```


# * which is a weighted average overall ``\theta``

# * the integration can be computed in closed form if we use conjugate prior

# **Bayesian prediction**
# ```math
# p(y_{N+1}=1 | \mathcal{D}) = \frac{a_N}{a_N + b_N} =\frac{a_0 + N_h}{a_0+b_0 + N}
# ```

# * very simple, add *one pseudo* counts to ``N_h`` and ``N_t``


# For our case, 

# ```math
# p(y_{N+1}=1 | \mathcal{D}) = \frac{1 + 0}{2 + 2} = \frac{1}{4}
# ```

# * makes better sense! we have only observed two zeros after all!
# """

# ╔═╡ f2912738-1b91-4227-a94f-c61a33a63819
# md"""

# ## Generalisation


# Bayesian predictive distribution, in general, can be found by applying the sum rule,

# ```math

# p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred}, \theta|\mathcal D, \mathcal M) \mathrm{d}\theta=  \int p(\mathcal D_{pred} |\theta, \mathcal D, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

# ```

# * ``\mathcal{M}``: the model assumption
# * in which the unknown parameters ``\theta`` are integrated out

# * assuming that past and future observations are conditionally independent given  ``\theta``, *i.e.* 

# $p(\mathcal D_{pred} |\theta, \mathcal D, \mathcal M) = p(\mathcal D_{pred} |\theta, \mathcal M)$

# The above equation can be written as:

# ```math

# p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred} |\theta, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

# ```

# * Bayesian prediction is an ensemble method by nature 
# """

# ╔═╡ b572e185-d88e-47d5-a3f3-1e25f2a6ca10
# md"""

# ## Prior and Posterior predictive distribution

# **Posterior predictive distribution:**

# ```math

# p(\mathcal D_{pred}|\mathcal D, \mathcal M) =  \int p(\mathcal D_{pred} |\theta, \mathcal M) \underbrace{p(\theta|\mathcal D, \mathcal M) }_{\text{posterior}}\mathrm{d}\theta

# ```
# * predictive distribution after observing the data


# A related concept is **Prior predictive distribution**


# ```math

# p(\mathcal D_{pred}|\mathcal M) =  \int p(\mathcal D_{pred} |\theta, \mathcal M) \underbrace{p(\theta|\mathcal M) }_{\text{prior}}\mathrm{d}\theta

# ```
# * predictive distribution before observing the data
# * can be useful to check your prior assumptions
# """

# ╔═╡ f2507e03-76e6-4f57-9a38-9323610efa33
# md"""

# ## Computing predictive distribution

# The prediction distribution involves the integration 

# ```math

# p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred} |\theta, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

# ```

# * which in general is intractable. 



# However, we can use simulation-based (Monte Carlo) methods to approximate the integration


# !!! information ""
# 	Repeat the following many times:

# 	1. Draw one sample from the posterior (or the prior for prior predictive):

# 	$$\tilde \theta \sim p(\theta|\mathcal D)$$
# 	2. Conditioning on ``\tilde \theta``, simulate pseudo observations: 

# 	$$\tilde{\mathcal D}\sim p(\mathcal D|\tilde{\theta})$$


# * the sampled ``\{\tilde{\mathcal D} \}`` are the empirical predictive distribution
# """

# ╔═╡ 23e88265-929b-4616-95c0-61e1844c8ae5
# md"""

# ## Example


# For the coughing example


# !!! warn ""
# 	Repeat the following many times:

# 	1. Draw one sample from the posterior (or the prior for prior predictive):

# 	$$\tilde h \sim p(h|\text{cough})$$
# 	2. Conditioning on ``\tilde h``, simulate pseudo observations: 

# 	$$\texttt{cough}\sim p(\texttt{cough}_{again}|\tilde{h})$$

# """

# ╔═╡ 18d66c48-be01-4f64-8a89-6b8e8c8444c8
# md"""
# ## Example: coin-flipping model

# For the conjugate coin-flipping model, the simulation is straightforward 


# * both the prior and posterior distributions ``p(\theta|\mathcal M)`` (or ``p(\theta|\mathcal M)``) are Beta 

# * it is also easy to simulate coin flip observations conditional on the bias ``\theta`` 



# For example, one can simulate a tuple ``(\theta, y_{new})`` in Julia by

# ```julia
# # draw a coin's bias from a Beta distribution
# θ = rand(Beta(a, b))
# # draw a pseudo coin-flipping sample based on θ
# y_new = rand(Bernoulli(θ))
# ```

# * one should repeat the above two steps ``R`` (e.g. 2000) times to obtain an approximation of the predictive distribution of future data
# """

# ╔═╡ 12fe751d-8ef4-4cdf-b5e5-a901a1556924
# md"""


# ## Demonstration
# """

# ╔═╡ 8cf29145-aed3-47e6-a209-369ac393e0e3
# begin
# 	R = 5000
# 	a0, b0 = 1, 1
# 	# nh, nt = sum(𝒟), length(𝒟) - sum(𝒟)
# 	an, bn = a0 + nh, b0 + nt

# 	y_pred = zeros(Bool, R)
# 	for r in 1:R
# 		θ = rand(Beta(an, bn))
# 		y_pred[r] = rand(Bernoulli(θ))
# 	end
# 	y_pred
# end

# ╔═╡ 0ad979b7-8ecb-44c7-958f-be9a78a22553
# md"""

# There is a closed-form solution for the Beta-Bernoulli case
# * this is one of the rare examples that we know how to compute the integration


# ```math
# p(y_{N+1}|\mathcal{D}) = \begin{cases}
# \frac{a_N}{a_N + b_N} & y_{N+1} = \texttt{true}\\
# \frac{b_N}{a_N + b_N} & y_{N+1} = \texttt{false}
# \end{cases}
# ```
# """

# ╔═╡ 3cb9bb88-d560-4403-b67e-7469f6eddcac
# md"""


# The Monte Carlo method should be very close to the ground truth
# """

# ╔═╡ 40c740a4-5385-4211-ac86-3c9f0c17b7ff
# mean(y_pred), an/(an+bn)

# ╔═╡ cd01a246-5e97-4231-a8dc-b2931d151b2f
# md"""

# ## Model checks via predictive distributions



# Bayesian inference provides a great amount of modelling freedom to its user


# * one can choose prior distribution to incorporate his knowledge


# * and also choose a suitable likelihood that best matches the data generation process. 


# * however, greater flexibility comes with a price: the modeller also needs to take full responsibility for the modelling decisions 


# For example, whether 

# * the chosen prior (with its parameters) makes sense;
# * the generative model as a whole (*i.e.* prior plus likelihood) match the observed data? 


# In other words, we need to **validate the model** 

# * **Predictive checks** are a great way to empirically validate a model's assumptions.

# """

# ╔═╡ a7fcf38b-05c3-4db8-8615-44b93d6d43aa
# md"""

# ## Predictive checks


# The idea of predictive checks is to 



# **First**, generate future *pseudo observations* based on the assumed model's (prior or posterior) **prediction distribution**:

# $$\mathcal{D}^{(r)} \sim p(\mathcal D_{\textit{pred}}|\mathcal D, \mathcal M), \;\; \text{for }r= 1\ldots, R$$ 
#   * where ``\mathcal D`` is the observed data and ``\mathcal M`` denotes the Bayesian model.  

# * note that ``\mathcal{D}^{(r)}`` should be of **the same size** as ``\mathcal{D}``

# and 


# **Then** If the model assumptions are reasonable, 

# * we should expect the generated pseudo data "agree with" the observed. 




# **In practice**, we compute the predictive distribution of some summary statistics,     
# * say mean, variance, median, or any meaningful statistic instead
# * and visually check whether the observed statistic falls within the predictions' possible ranges. 

# ## Predictive checks

# Based on the Monte Carlo principle, after simulating ``R`` pseudo samples,

# $$\tilde{\mathcal D}^{(1)}, \tilde{\mathcal D}^{(2)}, \tilde{\mathcal D}^{(3)},\ldots, \tilde{\mathcal D}^{(R)} \sim p(\mathcal D_{pred}|\mathcal D, \mathcal M),$$ 


# The predictive distribution of a summary statistic ``t(\cdot)``: 

# $$p(t(D_{pred})|\mathcal D, \mathcal M)$$ 
# * can be approximated by 

# $\{t(\tilde{\mathcal D}^{(1)}), t(\tilde{\mathcal D}^{(2)}), \ldots, t(\tilde{\mathcal D}^{(R)})\}$ 

# * one can visually check whether the observed statistic ``t(\mathcal{D})`` falls within a credible region of the empirical distribution 


# ## Prior and Posterior predictive checks

# If one mainly wants to check the prior assumption, use a **prior predictive distribution** to simulate the *future* data: *i.e.*

# $$\mathcal{D}^{(r)} \sim p(\mathcal D_{pred}|\mathcal M), \;\; \text{for }r= 1\ldots, R$$


# * and the visual check based on the sample 
# * the check is called **prior predictive check



# **Posterior predictive checks** 

# * checks both prior and likelihood model together
# * a hollistic check

# """

# ╔═╡ 1975923e-66fb-4aa4-a3d3-370b38d1df34
# md"""
# ## Example: coin-flipping model

# For predictive checks, we need to simulate ``\mathcal{D}_{pred}`` of **the same length** as ``\mathcal{D}``


# * we need to simulate **10** coin tosses in the second step
# * instead of just one


# ```julia
# # draw a coin's bias from a Beta distribution
# θ = rand(Beta(a, b))
# # draw a pseudo-coin-flipping sample of 10 tosses
# D = rand(Bernoulli(θ), 10)
# ```

# """

# ╔═╡ 7ca55b5a-086a-4982-bd2a-6be2c7e0299a
# function predictive_simulate(a, b; N=10 , R=2000)
# 	θ = rand(Beta(a,b), R)
# 	D = rand(N, R) .< θ'
# 	return D
# end;

# ╔═╡ d009d1e0-833a-40ea-882d-1b7fdbccf5cc
# md"""

# Here we choose to compare **the statistic** of the sum *i.e.* the count of heads
# ```math
# t(\mathcal{D}) = N_h= \sum_i d_i
# ```

# """

# ╔═╡ 3de87316-496e-4413-b7a1-3bd961dc9cc0
# md"""

# **Predictive checks** with the model


# * ``a_0, b_0 =1``
# """

# ╔═╡ 48a972dc-c866-43b2-b74a-b10b0ae343a3
# let
# 	Random.seed!(100)
# 	D = predictive_simulate(1, 1; N=10 , R=5000)
# 	histogram(sum(D, dims=1)[:], bins = 20, xticks=0:10, normed=true, label="Prior predictive on "*L"N_h", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Prior predictive check with "*L"a_0=b_0=1")
# 	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
# end

# ╔═╡ 1d296694-2ad8-4e30-a1e6-2741afaae2d3
# let
# 	Random.seed!(100)
# 	D = predictive_simulate(8, 4; N=10 , R=5000)
# 	histogram(sum(D, dims=1)[:], normed=true, xticks = 0:10, label="Posterior predictive on "*L"N_h", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Posterior predictive check with "*L"a_0=b_0=1")
# 	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
# 	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
# end

# ╔═╡ d3ec8762-fd5c-4614-9271-6cd3de4185c3
# md"""

# ## Example -- coin-flipping model check (cont.)
# **A model with misspecified prior** 

# $$\theta \sim \text{Beta}(a_0=50, b_0=1)$$

# * the prior is skewed and very informative

# * as a result, the posterior is dominated by the prior: 

# $$\theta \sim \text{Beta}(50+7, 1+3).$$

# The prior and posterior plots are shown below for reference


# """

# ╔═╡ e20c3ef6-e7f4-4fd5-be74-ef46970911d1
# let
# 	nh, nt = 7, 3
# 	a₀, b₀ = 50, 1
# 	plot(Beta(a₀,b₀), xlims=[0.6,1], label=L"p(\theta)= \mathcal{Beta}(50,1)", linewidth=1, xlabel=L"\theta", ylabel=L"p(\theta|\cdot)" ,fill= true, lw=2, alpha=0.2, legend=:outerright, color=1, title="A mis-specified model")	
# 	vline!([mean(Beta(a₀,b₀))], label="prior mean", lw=2, lc=1, ls=:dash)
# 	plot!(Beta(a₀+nh,b₀+nt), fill= true, lw=2, alpha=0.2, color=2, label=L"p(\theta|\mathcal{D})= \mathcal{Beta}(57,4)", linewidth=2)
# 	vline!([mean(Beta(a₀+nh,b₀+nt))], label="posterior mean", lw=2, lc=2, ls=:dash)
# end

# ╔═╡ c41fef4e-36eb-4d44-8838-1047f77c7ac2
# md"""


# Let's see whether **predictive checks** can spot the problem!
# """

# ╔═╡ 1c9997ad-5573-445c-8029-a57398aa442d
# let
# 	Random.seed!(100)
# 	D = predictive_simulate(50, 1; N=10 , R=5000)
# 	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10, normed=false, label="Prior predictive on " *L"N_h", legend=:outerbottom, xlabel="number of heads", title="Prior predictive check with "*L"a_0=50, b_0=1", ylabel=L"\#"* " of counts")
# 	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
# end

# ╔═╡ 3c18a8e6-7a9f-4e22-b1c9-fc8dad8a822d
# let
# 	Random.seed!(100)
# 	D = predictive_simulate(50+7, 1+3; N=10 , R=5000)
# 	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10,normed=false, label="Posterior predictive on "*L"N_h", legend=:outerbottom, xlabel="number of heads", title="Posterior predictive check with "*L"a_0=50, b_0=1", ylabel=L"\#"* " of counts")
# 	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
# 	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
# end

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
Distributions = "~0.25.132"
LaTeXStrings = "~1.4.1"
Latexify = "~0.16.12"
LogExpFunctions = "~1.0.3"
Plots = "~1.41.7"
PlutoTeachingTools = "~0.5.0"
PlutoUI = "~0.7.83"
SpecialFunctions = "~2.9.0"
StatsBase = "~0.34.13"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "4b0fde2aa57a8e3c18fdaf1a8759ef0e928eff73"

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
# ╟─c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
# ╟─be719382-bf1f-4442-8419-bddcda782525
# ╟─45f045d9-cdb7-4ade-a8e6-1f1a984cc58a
# ╟─77535564-648a-4e17-83a0-c562cc5318ec
# ╟─2057c799-18b5-4a0f-b2c7-66537a3fbe79
# ╟─c677a2d2-f4be-483a-a53b-7b76efb1b80f
# ╟─56f2bb33-637c-4450-9769-f338937594da
# ╟─47359bc3-cfc2-4777-82f9-308e56dca491
# ╟─35311403-187a-4e23-9e52-7eb0a212452d
# ╟─9fa2c7bf-3fa2-4869-8515-fa0048fa3cd6
# ╟─ae234663-90b4-432d-bacb-188dd1ee1034
# ╟─d764a435-7e11-4736-a0d0-0f2d5265b7af
# ╟─f53420ae-6007-451f-9eb7-3937abf114ef
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
# ╟─363da006-2e5c-4bb0-8849-a730ffd6c745
# ╟─e1bb37e8-d3b5-416f-b90a-fb611864b402
# ╟─4f4882c5-bd3f-4fbe-8a59-cafc5d365d99
# ╟─78bd6dea-8bcb-408f-9dc3-8a34907a4566
# ╠═42603c08-01de-4036-928a-d6a9c2dffbb3
# ╟─83586a99-02e5-42e3-83b2-2d90d3d3e396
# ╠═644cbe41-2027-4e3f-a31f-c656a1158466
# ╟─1a483bed-1639-41b4-ad7f-40b009bd45a9
# ╟─cf7b36c9-1c78-4ef6-bd06-2590b67c3703
# ╟─29a49584-3a52-410e-8f15-1d9293955826
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
# ╟─40c740a4-5385-4211-ac86-3c9f0c17b7ff
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
