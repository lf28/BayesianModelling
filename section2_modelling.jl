### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 69b19a54-1c79-11ed-2a22-ebe65ccbfcdb
begin
    using PlutoUI
	using Distributions
	using StatsPlots
    using Random
	using LaTeXStrings
	using SpecialFunctions
	using Logging; Logging.disable_logging(Logging.Warn);
end;

# ╔═╡ 904c5f8d-e5a9-4f18-a7d7-b4561ad37655
TableOfContents()

# ╔═╡ 5b4c3b01-bd19-4085-8a83-781086c85825
md"""


## Prior choice
"""

# ╔═╡ 0c123e4e-e0ff-4c47-89d6-fdf514f9e9d0
md"""
As part of the Bayesian modelling, we need to choose suitable priors for the unknown variables. In this section, we are going to introduce some important concepts about the prior specification. 


### Priors with matching support
**First and foremost**, when choosing priors for the unknowns, the modeller needs to make sure the prior distribution has the correct *support* of the unknown parameters. 

*Example* For the coin-flipping example, the unknown bias ``\theta`` has a support: ``\theta \in [0,1].`` Therefore, a standard Gaussian distribution is not suitable: as a Gaussian sample can take any value on the real line (so not within the correct range between 0 and 1). A suitable choice for the bias ``\theta`` can be e.g. Uniform distribution between 0 and 1, truncated Gaussian (truncated between 0 and 1) or Beta distribution. The probability density functions are listed below together with their plots.

```math

\begin{align}
p(\theta) &= \texttt{TruncNormal}(\mu, \sigma^2) \propto \begin{cases} \mathcal N(\mu, \sigma^2) & {0\leq \theta \leq 1} \\ 0 & \text{otherwise}  \end{cases}  \\
p(\theta) &= \texttt{Uniform}(0,1) = \begin{cases} 1 & {0\leq \theta \leq 1} \\ 0 & \text{otherwise}  \end{cases} 

\end{align}
```

"""

# ╔═╡ f479a1f3-a008-4622-82f0-ab154a431a33
let
	σ = sqrt(0.1)
	Plots.plot(TruncatedNormal(0.5, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}" *"(0.5, $(round((σ^2);digits=2)))", legend=:outerright, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"θ\in [0,1]")
	Plots.plot!(TruncatedNormal(0.25, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.25, $(round((σ^2);digits=2))))")
	Plots.plot!(TruncatedNormal(0.75, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.75, $(round((σ^2);digits=2))))")
	Plots.plot!(Uniform(0,1), lw=2, label=L"\texttt{Uniform}(0, 1)")
end

# ╔═╡ 2dfafb7b-773a-4c51-92e7-1f192fa354ea
md"""
*Example.* A Gaussian distribution's variance ``\sigma^2`` is a positive number: ``\sigma^2 >0``. Priors on the positive real line are e.g. Exponential distribution, or Half-Cauchy.

"""

# ╔═╡ e574a570-a76b-4ab2-a395-ca40dc383e5e
let
	Plots.plot(0:0.1:10, Exponential(1),lw=2,  label=L"\texttt{Exponential}(1.0)", legend=:best, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"σ^2\in (0,∞)")
	Plots.plot!(0:0.1:10, Exponential(2),lw=2,  label=L"\texttt{Exponential}(2.0)")
	Plots.plot!(0:0.1:10, Exponential(5), lw=2, label=L"\texttt{Exponential}(5.0)")
	Plots.plot!(0:0.1:10, truncated(Cauchy(0, 1), lower= 0), lw=2, label=L"\texttt{HalfCauchy}(0, 1)")
	# Plots.plot!(0:0.1:10, truncated(Cauchy(0, 2), lower= 0), label="HalfCauchy(0, 2)")
	# Plots.plot!(0:0.1:10, truncated(Cauchy(0, 5), lower= 0), label="HalfCauchy(0, 5)")
end

# ╔═╡ 9bb06170-34d0-4528-bd9b-988ecf952089
md"""

### Conjugate prior

Conjugate prior is a class of prior distributions such that the posterior distribution is of the same distribution form. When conjugacy holds, Bayesian computation becomes very simple: one only needs to update the prior's parameters to find the posterior. Unfortunately, conjugate priors only exist in some very simple Bayesian models. And for most real-world applications, no such conjugate priors can be found. Nevertheless, conjugate models provide us with some insights into the role of prior in Bayesian computation.

It is easier to study the concept by seeing a few concrete examples. 

#### Example: Beta-Bernoulli model
For the coin flipping example, the conjugate prior is 

```math
p(\theta) = \texttt{Beta}(\theta; a_0, b_0) = \frac{1}{\text{B}(a_0, b_0)} \theta^{a_0-1}(1-\theta)^{b_0-1},
```
where ``a_0,b_0 >0`` are the prior's parameter and ``B(a_0,b_0)``, the beta function, is a normalising constant for the Beta distribution: i.e. ``\mathrm{B}(a_0, b_0) = \int \theta^{a_0-1}(1-\theta)^{b_0-1}\mathrm{d}\theta``. 


*Remarks.
A few Beta distributions with different parameterisations are plotted below. Note that when ``a_0=b_0=1``, the prior reduces to a uniform distribution. Also note that when ``a_0> b_0``, e.g. ``\texttt{Beta}(5,2)``, the prior belief has its peak, or mode, greather 0.5, which implies the prior believe the coin is biased towards the head; and vice versa.*


"""

# ╔═╡ 13cb967f-879e-49c2-b077-e9ac87569d87
let
	plot(Beta(1,1), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=1,b_0=1)", linewidth=2, legend=:outerright, size=(600,300))	
	plot!(Beta(0.5,0.5), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=.5,b_0=.5)", linewidth=2)
	plot!(Beta(5,5), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=2,b_0=2)", linewidth=2)
	plot!(Beta(5,2), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=5,b_0=2)", linewidth=2)
	plot!(Beta(2,5), xlims=[0,1], ylim=[0,3], label=L"\texttt{Beta}(a_0=2,b_0=5)", linewidth=2)
end

# ╔═╡ 0437436b-16b9-4c90-b554-f7cf5ea8b4c0
md"""

Apply Baye's rule, we can find the posterior is still of a Beta form (therefore the conjugacy holds):

```math
p(\theta|\mathcal D) = \texttt{Beta}(\theta; a_N, b_N) = \frac{1}{\text{B}(a_N, b_N)} \theta^{a_N-1}(1-\theta)^{b_N-1},
```

where ``a_N= a_0 + N_h`` and ``b_N = b_0 + N - N_h``; and ``N_h = \sum_i d_i`` is the total number of heads observed in the ``N`` tosses.


"""

# ╔═╡ 3c5b6566-6e4e-41df-9865-fff8a839a70e
md"""
For our example, we have observed ``N_h = 7`` number of heads in the 10 tosses. Therefore, if we choose a prior of 
``p(\theta)= \texttt{Beta}(1, 1),``
The updated posterior is 

$$p(\theta|\mathcal D)= \texttt{Beta}(1+7, 1+3).$$

The prior and posterior distributions are plotted below. The posterior now peaks at 0.7 which is in agreement with the observed data. Also note the uncertainty (spread of the density) around the peak. 

"""

# ╔═╡ f10d3513-77ee-426a-aa5c-d2bf887572d9
let
	nh, nt = 7, 3
	plot(Beta(1,1), xlims=[0,1], label=L"p(\theta)= \texttt{Beta}(1,1)", linewidth=1, xlabel=L"\theta", ylabel="density" ,fill= true, lw=2, alpha=0.2, legend=:outerright, color=1, title="Conjugate posterior update")	
	vline!([mean(Beta(1,1))], label="prior mean", lw=2, lc=1, ls=:dash)
	plot!(Beta(1+nh,1+nt), xlims=[0,1], fill= true, lw=2, alpha=0.2, color=2, label=L"p(\theta|\mathcal{D})= \texttt{Beta}(8,4)", linewidth=2)
	vline!([mean(Beta(1+nh,1+nt))], label="posterior mean", lw=2, lc=2, ls=:dash)
end

# ╔═╡ 8c6233e0-14fb-4202-92b9-3eddaea3e107
md"""
**Interpretation of the prior parameters.**
Conjugate priors usually lead to very convenient posterior computation. For the coin-flipping example, we only need to update the prior's parameters with the count of heads and tails respectively:

$$a_N= a_0+ N_h\;\; b_N= b_0+N_t.$$ If we choose a truncated Gaussian prior, then the posterior distribution is much harder to compute. For more general prior choices, we need to resort to advanced methods like Markov Chain Monte Carlo (MCMC).


The updated parameters ``a_N`` ``b_N`` are the posterior counts of the heads and tails. And they together determine the shape of the posterior (check the plot above). For our case, the posterior peaks at ``0.7`` which is in agreement with the oberved data. And the posterior mean is ``\frac{8}{8+4} \approx 0.67``.

As a result, we can interpret the prior parameters ``a_0, b_0`` as some pseudo observations contained in the prior distribution. For example, ``a_0=b_0=1`` implies the prior contains one pseudo count of head and tail each (therefore, the prior is flat). 


"""

# ╔═╡ 7fd2a003-bed6-45d4-8335-f8d4f149c8d8
md"""

*Remark. One benefit of Bayesian inference arises when we have observed little data. Assume we have only tossed the coin twice and observed two heads: i.e. ``N_h=2, N_t=0``. Frequentist's estimation of the bias will be ``\hat{\theta}=\frac{N_h}{N_h+N_t}=0``, the observed frequency. This is clearly an over-confident conclusion (a.k.a. over-fitting). After all, we have only observed two data points. On the other hand, the Bayesian estimation will make better sense. The posterior is ``p(\theta|N_h=2, N_t=0) = \texttt{Beta}(3, 1),`` which is plotted below. The posterior peaks at 1.0 but there is a great amount of uncertainty about the bias, which leads to a posterior mean ``E[\theta|\mathcal D] =\frac{3}{3+1} = 0.75``. One should expect 0.75 (a.k.a. a maximum a posteriori estimator) makes better sense here. However, as stated before, a true Bayesian answer is to report the posterior distribution as an answer.*


$(begin
plot(Beta(1,1), label=L"p(\theta)", fill=true, alpha=0.3, xlabel=L"\theta", ylabel=L"p(\theta|\mathcal{D})", legend=:outerright)
plot!(Beta(3,1), label=L"p(\theta|N_h=2, N_t=0)", xlabel=L"\theta", ylabel=L"p(\theta|\mathcal{D})", legend=:outerright, fill=true, alpha=0.3, lw=2, title="Bayesian inference with little data")
vline!([mean(Beta(3,1))], ylim = [0,3],label="posterior mean", lw=2, lc=2, ls=:dash)
end)
 
"""

# ╔═╡ f5dfcd1f-9d10-49d1-b68b-eafdf10baaec
begin
	# simulate 200 tosses of a fair coin
	N_tosses = 200
	true_θ = 0.5
	Random.seed!(100)
	coin_flipping_data = rand(N_tosses) .< true_θ
	Nh = sum(coin_flipping_data)
	Nt = N_tosses- Nh
end;

# ╔═╡ 754c5fe2-3899-4467-8fd4-828fb0ec5040
md"""
**Sequential update.** The conjugacy also provides us a computational cheap way to sequentially update the posterior. Due to the independence assumption, we do not need to calculate the posterior in one go, but update the posterior incrementally as data arrives.

```math
\begin{align}
p(\theta|\{d_1, d_2, \ldots, d_N\})&\propto  p(\theta) \prod_{n=1}^N p(d_n|\theta) p(d_N|\theta)\\
&= \underbrace{p(\theta) \prod_{n=1}^{N-1} p(d_n|\theta)}_{p(\theta|\mathcal D_{N-1})}\cdot p(d_N|\theta)\\
&= \underbrace{p(\theta|\mathcal D_{N-1})}_{\text{new prior}} p(d_N|\theta).
\end{align}
```
Suppose we have observed ``N-1`` coin tosses, and the posterior so far is ``p(\theta|\mathcal D_{N-1})``. The posterior now serves as a *new prior* to update the next observation ``d_N``. It can be shown that the final posterior sequentially updated this way is the same as the off-line posterior.

To be more specific, the sequential update algorithm is:

Initialise with a prior ``p(\theta|\emptyset)= \texttt{Beta}(a_0, b_0)``

For ``n = 1,2,\ldots, N``:
* update 

$$a_n = a_{n-1} + \mathbf{1}(d_n=\texttt{head}), \;\; b_n = b_{n-1} +  \mathbf{1}(d_n=\texttt{tail})$$
* report the posterior at ``n`` if needed

Note that the function ``\mathbf{1}(\cdot)`` returns 1 if the test result of the argument is true and 0 otherwise. 


*Demonstration.* An animation that demonstrates the sequential update idea is listed below. $(N_tosses) tosses of a fair coin are first simulated to be used for the demonstration. 
"""

# ╔═╡ 2ba73fd3-ae2b-4347-863f-28e6c21e7a91
md"""
We apply the sequantial learning algorithm on the simulated data. The posterior update starts from a vague flat prior. As more data is observed and absorbed into the posterior, the posterior distribution is more and more informative and finally recovers the final posterior.
"""

# ╔═╡ 56e80a69-6f84-498d-b04e-5be59c1488eb
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

# ╔═╡ 65f0dc31-c2bd-464f-983a-9f4ca2c04a35
md"""

#### Examples: Gamma-Gaussian model

> Assume we have made ``N`` independent samples from a Gaussian distribution with known ``\mu`` but unknown variance ``\sigma^2>0``. What is the observation variance ``\sigma^2`` ?

It is more convenient to model the precision ``\lambda\triangleq 1/\sigma^2``. A conjugate prior for the precision parameter is Gamma distribution. Gamma distributions have support on the positive real line which matches the precision's value range.

A Gamma distribution, parameterised with a shape  ``a_0>0``,  and a rate parameter ``b_0>0``, has a probability density function:

```math
p(\lambda; a_0, b_0) = \texttt{Gamma}(\lambda; a_0, b_0)=\frac{b_0^{a_0}}{\Gamma(b_0)} \lambda^{a_0-1} e^{-b_0\lambda}.
```
"""

# ╔═╡ d2c3bd6d-fb8c-41df-8112-c7bfbc180b0a
let
	as_ = [1,2,3,5,9,7.5,0.5]
	bs_ = [1/2, 1/2, 1/2, 1, 1/0.5, 1, 1]
	plt= plot(xlim =[0,20], ylim = [0, 0.8], xlabel=L"λ", ylabel="density")
	for i in 1:length(as_)
		plot!(Gamma(as_[i], 1/bs_[i]), fill=(0, .1), lw=1.5, label=L"\texttt{Gamma}"*"($(as_[i]),$(bs_[i]))")
	end
	plt
end

# ╔═╡ 76c641dd-c790-4f51-abc0-e157c00e3ba7
md"""

**Conjugacy.** The data follow a Gaussian distribution. That is for ``n = 1,2,\ldots, N``:
```math
d_n \sim \mathcal N(\mu, 1/\lambda);
```
The likelihood therefore is a product of Gaussian likelihoods:


```math
p(\mathcal D|\mu, \lambda) = \prod_{n=1}^N p(d_n|\mu, \lambda) = \prod_{n=1}^N \mathcal N(d_n; \mu, 1/\lambda) .
```

It can be shown that the posterior formed according to Baye's rule

```math

p(\lambda|\mathcal D, \mu) \propto p(\lambda) p(\mathcal D|\mu, \lambda)

```



is still of a Gamma form (therefore conjugacy is established): 

```math
p(\lambda|\mathcal D, \mu) = \texttt{Gamma}(a_N, b_N),
```

where 

$$a_n= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.$$

The Bayesian computation reduces to hyperparameter update again. Note the posterior mean (standard result of a Gamma distribution) is 

$$\mathbb{E}[\lambda|\mathcal D] = \frac{a_N}{b_N} = \frac{a_0 + N/2}{b_0 + {\sum_n (d_n -\mu)^2}/{2}}.$$

If we assume ``a_0= b_0=0`` (a flat prior), we recover the regular maximum likelihood estimator for ``{\sigma^2}``: 

$$\hat \sigma^2 =1/\hat{\lambda} = \frac{\sum_n (d_n-\mu)^2}{N}.$$

Based on the above result, we can intuitively interpret the hyperparameters as 
* ``a_N`` : the total count of observations contained in the posterior (both pseudo ``a_0`` and real observation ``N/2``) ;
* ``b_N`` : the rate parameter is the total sum of squares;


"""

# ╔═╡ edbf5fa3-a064-47a4-9ff4-a93dbd7b9112
md"""
**Demonstration:** We first simulate ``N=100`` Gaussian observations with unknown ``\mu=0`` and ``\lambda= 1/\sigma^2 = 2``.

"""

# ╔═╡ 958d6028-e38f-4a56-a606-47b6f8ee86f1
begin
	σ² = 0.5
	true_λ = 1/σ²
	N = 100
	Random.seed!(100)
	# Gaussian's density in Distributions.jl is implemented with standard deviation σ rather than σ²
	gaussian_data = rand(Normal(0, sqrt(σ²)), N)
	plot(Normal(0, sqrt(σ²)), xlabel=L"d", ylabel="density", label=L"\mathcal{N}(0, 1/2)", title="Simulated Gaussian observations")
	scatter!(gaussian_data, 0.01 .* ones(N), markershape=:vline, c=1, label=L"d_n")
end

# ╔═╡ 59975c2b-c537-4950-aeea-985377f22d93
md"""
We have used a relatively vague Gamma prior with ``a_0=b_0=0.5``; and the posterior can be easily calculated by updating the hyperparameters: ``\texttt{Gamma}(a_0+ 100/2, b_0+ \texttt{sse}/2).`` The update can be easily in Julia:
"""

# ╔═╡ cb9ad8b5-7975-4d54-bb94-afc9f4953a67
begin
	a₀ = 0.5
	b₀ = 0.5
	# Gamma in Distributions.jl is implemented with shape and scale parameters where the second parameter is 1/b 
	prior_λ = Gamma(a₀, 1/b₀)
	posterior_λ = Gamma(a₀+ N/2, 1/(b₀+ sum(gaussian_data.^2)/2))
end;

# ╔═╡ 03a650cb-7610-4d20-8ae7-2c6e308770f6
md"""
We can plot the prior and posterior distributions to visually check the Bayesian update's effect. To plot a random variable in Julia, one can simply use `plot()` function from `StatsPlots.jl` package. The figure of both the prior and posterior plots is shown below.
"""

# ╔═╡ cb37f8a8-83aa-44f6-89a5-43fe0e4a5fa8
let
	plot(prior_λ, xlim = [0, 5], ylim=[0, 1.5], lw=2, fill=(0, 0.2), label="Prior "* L"p(\lambda)", xlabel=L"\lambda", ylabel="density", title="Conjugate inference of a Gaussian's precision")

	plot!(posterior_λ, lw=2,fill=(0, 0.2), label="Posterior "* L"p(\lambda|\mathcal{D})")
	vline!([true_λ], label="true "*L"λ", lw=4)
	vline!([mean(prior_λ)], label="prior mean", lw=2, lc=1, ls=:dash)
	vline!([mean(posterior_λ)], label="posterior mean", lw=2, lc=2, ls=:dash)
	# vline!([1/σ²], label="true "*L"λ", lw=2)
	# end
end

# ╔═╡ 2d7792cd-5b32-4228-bfef-e1ab250724f3
md"""
**Sequential update.** The conjugacy also provides us a computational cheap way to sequentially update the posterior. Due to the independence assumption, we do not need to calculate the posterior in one go.

```math
\begin{align}
p(\lambda|\{d_1, d_2, \ldots, d_N\})&\propto  p(\lambda) \prod_{n=1}^N p(d_n|\lambda) p(d_N|\lambda)\\
&= \underbrace{p(\lambda) \prod_{n=1}^{N-1} p(d_n|\lambda)}_{p(\lambda|\mathcal D_{N-1})}\cdot p(d_N|\lambda)\\
&= \underbrace{p(\lambda|\mathcal D_{N-1})}_{\text{new prior}} p(d_N|\lambda)
\end{align}
```
With observations up to ``N-1``, we obtain a posterior ``p(\lambda|\mathcal D_{N-1})``.
The posterior now serves as the new prior to update the next observation ``d_N``. It can be shown that the final posterior sequentially updated this way is the same as the off-line posterior.

To be more specific, the sequential update algorithm for the precision example is:

Initialise with a prior ``p(\lambda|\emptyset)``;
for ``n = 1,2,\ldots, N``
* update ``a_n = a_{n-1} + 0.5 ``, ``b_n = b_{n-1} + 0.5 \cdot (d_n-\mu)^2``
* report the posterior at ``n`` if needed

To demonstrate the idea, check the following animatin. The posterior update starts from a vague prior. As more data observed and absorbed to the posterior, the posterior distribution is more and more informative and finally recover the ground truth.
"""

# ╔═╡ 7ccd4caf-ef8f-4b78-801f-b000f5aad430
let
	plot(prior_λ, xlim = [0, 5], ylim=[0, 1.5], lw=2, fill=(0, 0.1), label="Prior "* L"p(\lambda)", xlabel=L"\lambda", ylabel="density", title="Sequential update N=0")
	plot!(posterior_λ, lw=2, fill=(0, 0.1), label="Posterior "* L"p(\lambda|\mathcal{D})")
	vline!([true_λ], label="true "*L"λ", lw=2, lc=3)
	anim=@animate for n in [(1:9)..., (10:10:N)...]
		an = a₀ + n/2
		bn = b₀ + sum(gaussian_data[1:n].^2)/2
		poster_n = Gamma(an, 1/bn)
		plot!(poster_n, lw=1, label="", title="Sequential update N=$(n)")
	end
	gif(anim, fps=3)
end

# ╔═╡ 87b411d1-a62f-4462-b81b-ef0e8ac97d7e
md"""
### Informative vs Non-informative

All priors can be largely classified into two groups: **informative** prior and **non-informative** prior.

**Non-informative** prior, as the name suggests, contains no information in the prior [^1] and *let the data speak to itself*. For our coin-flipping case, a possible choice is a flat uniform prior: i.e. ``p(\theta) \propto 1`` when ``\theta\in[0,1]``. 


**Informative** prior, on the other hand, contains the modeller's subjective prior judgement. 

For example, if we believe our coin should be fair, we can impose an informative prior such as ``p(\theta) =\texttt{Beta}(n,n)``. When ``n`` gets larger, e.g. ``n=5``,  the prior becomes more concentrated around ``\theta=0.5``, which implies a stronger prior belief on the coin being fair.

$(begin
plot(Beta(1,1), lw=2, xlabel=L"\theta", ylabel="density", label=L"\texttt{Unif}(0,1)", title="Priors with different level of information", legend=:topleft)
plot!(Beta(2,2), lw=2,label= L"\texttt{Beta}(2,2)")
plot!(Beta(3,3), lw=2,label= L"\texttt{Beta}(3,3)")
plot!(Beta(5,5), lw=2,label= L"\texttt{Beta}(5,5)")
# vline!([0.5], lw=2, ls=:dash, label= "")
end)

After we observe ``N_h=7, N_t=3``, the posteriors can be calculated by updating the pseudo counts: ``\texttt{Beta}(n+7, n+3)``. The corresponding posteriors together with their posterior means (thick dashed lines) are plotted below.

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

# ╔═╡ 58d7878a-f9d6-46b4-9ac4-1944f7508f03
md"""

It can be observed that all posteriors peak at different locations now. When the prior is more informative (that the coin is fair), it shrinks the posterior closer to the centre ``\theta=0.5``. 
"""

# ╔═╡ 3cdf9875-9c42-46b3-b88c-74e0f363bc4a
md"""
Historically, **non-informative** Bayesian was the dominating choice due to its *objectiveness*. However, it should be noted that it is almost impossible to be completely non-informative. Indeed, for our coin-flipping example, the flat uniform distributed prior still shrinks the posterior mean towards 0.5, therefore not non-informative or objective.

The flexibility of introducing priors that reflect a modeller's local expert knowledge, however, is now more considered an *advantage* of the Bayesian approach. After all, modelling itself is a subjective matter. A modeller needs to take the responsibility for their modelling choices. The preferred approach is to impose subjective (possibly weak) informative priors but carefully check one's prior assumptions via methods such as posterior/prior predictive checks.
 
"""

# ╔═╡ 3ca3ee53-c744-4474-a8da-6209ec5e6904
md"""

## More on Bayesian modelling

"""

# ╔═╡ 5cbace78-8a92-43aa-9d3a-45054be48532
md"""
Bayesian modelling is more an art than science. Ideally, each problem should have its own bespoke model. We have only seen problems where the observed data is assumed to be independently and identically distributed (**i.i.d.**). For example, in the coin-flipping example, since the ten coin tosses ``\{d_1, d_2, \ldots, d_{10}\}`` are *independent* tossing realisations of the *same* coin, the i.i.d. assumption makes sense. We assume each ``d_i`` follows the same Bernoulli distribution and they are all independent.

However, there are problems in which more elaborate modelling assumptions are required. To demonstrate the idea, we consider the following inference problem.



"""

# ╔═╡ 2cb31d23-a542-4f8d-a056-025fa574f0d7
md"""

!!! question "Seven scientist problem"
	[*The question is adapted from [^1]*] Seven scientists (A, B, C, D, E, F, G) with widely-differing experimental skills measure some signal ``\mu``. You expect some of them to do accurate work (i.e. to have small observation variance ``\sigma^2``, and some of them to turn in wildly inaccurate answers (i.e. to have enormous measurement error). What is the unknown signal ``\mu``?

	The seven measurements are:

	| Scientists | A | B | C | D | E | F | G |
	| -----------| ---|---|---|--- | ---|---|---|
	| ``d_n``    | -27.020| 3.570| 8.191| 9.898| 9.603| 9.945| 10.056|
"""

# ╔═╡ 31afbad6-e447-44d2-94cf-c8f99c7fa64a
begin
	scientist_data = [-27.020, 3.57, 8.191, 9.898, 9.603, 9.945, 10.056]
	μ̄ = mean(scientist_data)
end

# ╔═╡ de5c3254-1fbe-46ec-ab09-77889405510d
md"""
The measurements are also plotted below.
"""

# ╔═╡ ff49403c-8e0e-407e-8141-6ccf178c152b
let
	ylocations = range(0, 2, length=7) .+ 0.5
	plt = plot(ylim = [0., 3.0], xminorticks =5, yticks=false, showaxis=:x, size=(600,200))
	scientists = 'A':'G'
	δ = 0.1
	for i in 1:7
		plot!([scientist_data[i]], [ylocations[i]], label="", markershape =:circle, markersize=5, markerstrokewidth=1, st=:sticks, c=1)
		annotate!([scientist_data[i]].+7*(-1)^i * δ, [ylocations[i]].+ δ, scientists[i], 8)
	end
	vline!([μ̄], lw=2, ls=:dash, label="sample mean", legend=:topleft)
	# density!(scientist_data, label="")
	plt
end

# ╔═╡ 024ce64f-29ef-49f2-a66f-87c2d4eb67a7
md"""
*Remarks. Based on the plot, scientists C,D,E,F,G all made similar measurements. Scientists A, B's experimental skills seem questionable. This is a problem in which the frequentist method should find challenging. If all 7 measurements were observed by one scientist or scientists with a similar level of experimental skill, the sample mean: 
$$\frac{\sum_n d_n}{N} \approx 3.46$$ 
would have been a good estimator. 
An ad hoc remedy is probably to treat the first two observations as outliers and take an average over the rest of 5 measurements. This remedy is lack formal justification and does not scale well with problems with a lot of measurements. One cannot check every observation individually.*

"""

# ╔═╡ 52fd1028-05aa-4b37-b57c-daabf4d77f50
md"""

### A bad Bayesian model

**Modelling**
One possible model is to ignore the sutblies and reuse our coin-flipping model's assumption. Since the observed data is real-valued, we only need to replace a Bernoulli likelihood with a Gaussian. We then assume observations ``d_n`` are i.i.d distributed with a Gaussian 

$$d_n \overset{\mathrm{i.i.d}}{\sim} \mathcal N(\mu, \sigma^2),$$

where the mean is the unknown signal ``\mu`` and a shared ``\sigma^2`` is the observation variance. The model implies each scientist's observation is the true signal ``\mu`` plus some Gaussian distributed observation noise.

To specify a Bayesian model, we need to continue to specify a prior model for the two unknowns ``\mu``, ``\sigma^2``. For computational convenience, we assume a Gaussian prior for the signal ``\mu``:

$$\mu \sim \mathcal N(m_0, v_0),$$
* ``m_0`` is our prior guess of the signal's centre
* ``v_0`` represents our prior belief strength;

To show our ignorance, we can set ``m_0=0`` (or the sample average) and ``v_0`` to a very large positive number, say 10,000. The prior then becomes a very flat vague distribution.

It is more convenient to model the observation precision ``\lambda \triangleq 1/\sigma^2`` instead of variance ``\sigma^2``. Here we assume a Gamma prior for the precision parameter:

$$\lambda \sim \texttt{Gamma}(a_0, b_0)$$

Again, to show our ignorance, we can set ``a_0, b_0`` such that the distribution is as flat and vague as possible. A possible parameterisation is ``a_0=b_0=0.5.`` Note that Gamma is a distribution on the positive real line which has matching support for the precision parameter. 

To put them together, the full Bayesian model is:
```math
\begin{align}
\text{prior}: \mu &\sim \mathcal N(m_0=0, v_0=10000)\\
\lambda &\sim \texttt{Gamma}(a_0=0.5, b_0=0.5)\\
\text{likelihood}: d_n &\overset{\mathrm{i.i.d}}{\sim} \mathcal N(\mu, 1/\lambda) \;\; \text{for } n = 1,2,\ldots, 7.
\end{align}
```
**Computation:**
After specifying the model, we need to apply Baye's rule to compute the posterior distribution:

```math
\begin{align}
p(\mu, \lambda|\mathcal D) &\propto p(\mu, \lambda) p(\mathcal D|\mu, \lambda) 
\\
&= p(\mu)p(\lambda) p(\mathcal D|\mu, \lambda) \\
&= p(\mu)p(\lambda) \prod_{n=1}^N p(d_n|\mu, \lambda);
\end{align}
```
where we have assumed the prior for ``\mu`` and ``\lambda`` are independent. Sub-in the definition of the prior and likelihood, we can plot the posterior.
"""

# ╔═╡ d6056188-9a46-4b5f-801b-e028b6eb0b7f
let
	m₀, v₀ = 0, 10000
	a₀, b₀ = 0.5, 0.5
	function ℓπ(μ, λ; data)  
		σ² = 1/λ
		logprior = logpdf(Normal(m₀, v₀), μ) + logpdf(Gamma(a₀, 1/b₀), λ) 
		logLik = sum(logpdf.(Normal(μ, sqrt(σ²)), data))
		return logprior + logLik
	end

	plot(-13:0.05:20, 0:0.001:0.019, (x, y) -> exp(ℓπ(x, y; data=scientist_data)), st=:contour,fill=true, ylim=[0.001, 0.015], xlabel=L"μ", ylabel=L"λ", title="Contour plot of "*L"p(\mu, \lambda|\mathcal{D})")
end

# ╔═╡ bbbb73c8-0254-463a-8e81-5acdd08583ac
md"""
**Marginal posterior ``p(\mu|\mathcal D)``:**
The posterior distribution shows the posterior peaks around ``\mu = 3.5``, which is roughly the same as the sample average. However, to better answer the question, we should treat ``\lambda`` as a *nuisance* parameter, and integrate it out to find the marginal posterior for ``\mu`` only. After some algebra, we find the unnormalised marignal posterior is of the form:

```math
p(\mu|\mathcal D) \propto p(\mu)\cdot \Gamma\left (a_0+ \frac{N}{2}\right )\left (b_0+\frac{\sum_n (d_n-\mu)^2}{2}\right )^{- (a_0+ \frac{N}{2})},
```
where ``N=7`` for our problem. 

"""

# ╔═╡ 2f2e3e34-76c0-4f35-8c9b-21184f86cf66
md"""

We can implement the (log) density in Julia (check `log_marginal_μ_wrong` function below). It is more common to compute log probability to avoid numerical issues. For reference, the log posterior density is:

```math

\ln p(\mu|\mathcal D) = \ln p(\mu) + \ln\Gamma\left (a_0+ \frac{N}{2}\right )- \left (a_0+ \frac{N}{2}\right )\cdot \ln \left (b_0+\frac{\sum_n (d_n-\mu)^2}{2}\right ) 
```
"""

# ╔═╡ 9732358b-f592-4c66-9d8a-fdc221249a56
function log_marginal_μ_wrong(μ; data, a₀=0.5, b₀=0.5, m₀=0, v₀=10000)  
	N = length(data)
	logprior = logpdf(Normal(m₀, sqrt(v₀)), μ) 
	logLik = loggamma(a₀ + N/2) - (a₀+ N/2)* log(b₀+ 0.5 * sum((data .- μ).^2))
	return logprior + logLik
end

# ╔═╡ 9d737cc5-c533-4e86-8654-b1adba943fc0
md"""
We plot the unnormalised marginal posterior below. It shows the most likely estimate for the signal is about 3.46, which is counter-intuitive.
"""

# ╔═╡ 1d7783bf-a8d0-47d6-812d-b31ae0c0b138
let
	μs = -30:0.01:30
	ℓs = log_marginal_μ_wrong.(μs; data= scientist_data)
	_, maxμ = findmax(ℓs)
	plot(μs, (ℓs), color=2, alpha=0.5, label="", fillrange = -100, ylim=[-39, -28],
    fillalpha = 0.5,
    fillcolor = 2, xlabel=L"μ", yaxis=false, ylabel="log density",title="Unnormalised marginal posterior "*L"\ln p(μ|\mathcal{D})")
	vline!([μs[maxμ]], lw=2, color=2, ls=:dash, label="Mode")
	xticks!([(-30:10:30)...; μs[maxμ]])
end

# ╔═╡ 73bb12a3-db59-4e71-bf05-5a4b4e018f51
md"""

### A better Bayesian model
"""

# ╔═╡ 8eb641f0-a01f-43f9-a362-e7d54c26411a
md"""

**Modelling:**
The i.i.d assumption does not reflect the sutblties of the data generation process. A better model should assume each observation follows an independent but not identical Gaussian distribution. In particular, for each scientist, we should introduce their own observation precision, ``\lambda_n \triangleq \sigma^2_n`` to reflect their "different levels of experimental skills".


The improved model now includes 7+1 unknown parameters: the unknown signal ``\mu`` and seven observation precisions ``\lambda_n`` for the seven scientists:

```math
\begin{align}
\mu &\sim \mathcal N(m_0, v_0)\\
\lambda_n &\sim \texttt{Gamma}(a_0, b_0)\;\; \text{for }n = 1,2,\ldots,7\\
d_n &\sim \mathcal N(\mu, 1/\lambda_n)\;\; \text{for }n = 1,2,\ldots,7\\
\end{align}
```
"""

# ╔═╡ a06f8785-fbbc-4973-b3d0-5a6db967b3cc
md"""
**Computation:**

Similarly, we want to find out the marginal posterior distribution. After integrating out the nuisance parameters ``\{\lambda_n\}`` from the posterior, we can find the marginal posterior is of the following form:

```math
p(\mu|\mathcal D) \propto p(\mu)\prod_{n=1}^N p(d_n|\mu) \propto p(\mu)\prod_{n=1}^N \frac{\Gamma{(a_n)}}{b_n^{a_n}},
```

where for ``n= 1,2\ldots, 7``:

$$a_n = a_0 + \frac{1}{2}, b_n = b_0+ \frac{1}{2} (d_n-\mu)^2.$$ 
"""

# ╔═╡ 5fc1d8aa-5835-4559-860b-b73031d3bfe7
md"""
We can implement the log posterior distribution in Julia and plot the density.
"""

# ╔═╡ da7f1485-a532-4d3c-96db-d1d50f2bdee6
function log_marginal_μ(μ; data, a₀=0.5, b₀=0.5, m₀=0, v₀=10000)  
	aₙ = a₀ + 0.5
	bₙ = b₀ .+ 0.5 .* (data .-μ).^2
	logpdf(Normal(m₀, sqrt(v₀)), μ) + length(data) * loggamma(a₀) - sum(aₙ * log.(bₙ))
end

# ╔═╡ ee8fce9a-0a5b-40c2-a4d9-f124b40188f4
md"""
The posterior distribution looks more complicated now but makes much better sense. 

* the mode now correctly sits around 10 (circa 9.7)
* also note a few small bumps near locations of -27 and 3.5

The posterior makes much better sense. It not only correctly identifies the consensus of the majority of the scientists (i.e. scientists C, D, E,..., G) and also takes into account scientists A and B's noisy observations.
"""

# ╔═╡ d6341bf9-7a95-4a10-8a26-9495b724b907
let

	μs = -30:0.01:30
	ℓs = log_marginal_μ.(μs; data= scientist_data)
	_, maxμ = findmax(ℓs)
	plot(μs, ℓs, xlabel=L"μ", ylabel="log density", title="Unnormalised marginal log posterior: "*L"\ln p(\mu|\mathcal{D})", lw=2, label="", lc=2)
	vline!([μs[maxμ]], lw=2, color=2, ls=:dash, label="Mode")
	# xticks!([(-30:10:30)...; μs[maxμ]])
end

# ╔═╡ 0a8a5aae-3567-41a3-9f15-d07235d07da2
md"""

## Predictive checks
"""

# ╔═╡ 8fe3ed2f-e826-4faa-ace0-2286b944991f
md"""

### Posterior predictive check
Bayesian inference provides a great amount of modelling freedom to its user. The modeller, for example, can incorporate his or her expert knowledge in the prior distribution and also choose a suitable likelihood that best matches the data generation process. However, greater flexibility comes with a price. The modeller also needs to take full responsibility for the modelling decisions. For example, for a Bayesian model, one needs to check whether
* the prior makes sense? 
* the generative model as a whole (i.e. prior plus likelihood) match the observed data? 
In other words, the user needs to **validate the model** before making any final inference decisions. **Predictive checks** are a great way to empirically validate a model.

The idea of predictive checks is to generate future *pseudo observations* based on the assumed model's (posterior) **prediction distribution**:

$$\mathcal{D}^{(r)} \sim p(\mathcal D_{pred}|\mathcal D, \mathcal M), \;\; \text{for }r= 1\ldots, R$$ 

where ``\mathcal D`` is the observed data and ``\mathcal M`` denotes the Bayesian model. Note that the posterior predictive distribution indicates what future data might look like, given the observed data and our model. If the model assumptions (both the prior and likelihood) are reasonable, we should expect the generated pseudo data agree with the observed. 

Comparing vector data is not easy (note that a sample ``\mathcal D`` is usually a vector). In practice, we compute the predictive distribution of some summary statistics, say mean, variance, median, or any meaningful statistic instead, and visually check whether the observed statistic falls within the prediction. Based on the Monte Carlo principle, after simulating ``R`` pseudo samples,

$$\tilde{\mathcal D}^{(1)}, \tilde{\mathcal D}^{(2)}, \tilde{\mathcal D}^{(3)},\ldots, \tilde{\mathcal D}^{(R)} \sim p(\mathcal D_{pred}|\mathcal D, \mathcal M),$$ the predictive distribution of a summary statistic ``t(\cdot)``: ``p(t(D_{pred})|\mathcal D, \mathcal M)``, can be approximated by the empirical distribution of ``\{t(\tilde{\mathcal D}^{(1)}), t(\tilde{\mathcal D}^{(2)}), \ldots, t(\tilde{\mathcal D}^{(R)})\}``. 


### Prior predictive check
Alternatively, one can use a **prior predictive distribution** to simulate the *future* data: i.e.

$$\mathcal{D}^{(r)} \sim p(\mathcal D_{pred}|\mathcal M), \;\; \text{for }r= 1\ldots, R$$
and the visual check based on the sample is called **prior predictive check**. The prior predictive distribution is a distribution of possible data sets given the priors and the likelihood, *before any real observations are taken into account*. Since the distribution relies on the prior information only, a prior predictive check is particularly useful to validate a prior's specification. 

Note that the two predictive distributions are very much related. By setting ``\mathcal D=\emptyset``, we recover the prior predictive distribution from the posterior predictive distribution.





"""

# ╔═╡ 79b85624-a68d-41df-856a-7a7670ff4d0e
md"""

### Approximate predictive distributions

The predictive distribution, in general, can be found by applying the sum rule,

```math

p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred}, \theta|\mathcal D, \mathcal M) \mathrm{d}\theta=  \int p(\mathcal D_{pred} |\theta, \mathcal D, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

```
in which the unknown parameters ``\theta`` are integrated out. Assuming that past and future observations are conditionally independent given  ``\theta``, i.e. ``p(\mathcal D_{pred} |\theta, \mathcal D, \mathcal M) = p(\mathcal D_{pred} |\theta, \mathcal M)``, the above equation can be written as:

```math

p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred} |\theta, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

```

The integration in general is intractable. However, we can rely on sampling-based methods to approximate the integration.


!!! information ""
	Repeat the following many times:
	
	1. Draw one sample from the posterior (or the prior for prior predictive): $$\tilde \theta \sim p(\theta|\mathcal D)$$
	2. Conditioning on ``\tilde \theta``, simulate the pseudo observations: ``\tilde{\mathcal D}\sim p(\mathcal D|\tilde{\theta}) ``

The Monte Carlo approximation method can also be viewed as an ensemble method:


$$p(\mathcal D_{pred}|\mathcal D, \mathcal M) \approx \frac{1}{R} \sum_{r} p(\mathcal D_{pred}|\theta^{(r)}).$$
We are making predictions of future data on an ensemble of ``R`` models by taking the average, and each ensemble element model is indexed by one posterior (prior) sample. 


The ensemble view highlights a key difference between the Bayesian and frequentist methods. When it comes to prediction, the frequentist usually applies the plut-in principle: use a point estimator of ``\theta`` (e.g. the maximum likelihood estimator) and that singleton model to predict:

$$p(\mathcal D_{pred}|\hat{\theta}, \mathcal M).$$
On the other hand, the Bayesian adopts a more democratic approach: an ensemble of models is consulted.


*Remarks. In simulating a pseudo sample ``\tilde{\mathcal D}``, we need to generate the same number of observations as ``\mathcal D``. For example, in the coin flipping example, for each ``\tilde\theta``, we need to simulate 10 tosses.* 



"""

# ╔═╡ e9a87ae2-f906-4be9-a969-7b681c2cff7b
md"""
### Example: coin-flipping model

For the conjugate coin-flipping model, the parameter sampling step is straightforward since both the prior and posterior distributions ``p(\theta|\mathcal M)`` (or ``p(\theta|\mathcal M)``) are both Beta distributed. 

Similarly, it is also easy to simulate coin flip observations conditional on the bias ``\theta``. For example, one can simulate a tuple ``(\theta, \mathcal D_{pred})`` in Julia by

```julia
# draw a coin's bias from a Beta distribution
θ = rand(Beta(a, b))
# draw a pseudo coin-flipping sample of 10 tosses
D = rand(Bernoulli(θ), 10)
```

One should repeat the above two steps ``R`` (e.g. 2000) times to obtain an approximation of the predictive distribution of future data.
"""

# ╔═╡ ebdb9711-20cc-458c-9474-8e3ace69797e
function predictive_simulate(a, b; N=10 , R=2000)
	θ = rand(Beta(a,b), R)
	D = rand(N, R) .< θ'
	return D
end;

# ╔═╡ 13fae558-d12e-4e27-b36f-ec59eabb65cf
let
	Random.seed!(100)
	D = predictive_simulate(1, 1; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], bins = 20, xticks=0:10, normed=true, label="Prior predictive on Nₕ", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Prior predictive check with a₀=b₀=1")
	vline!([7], lw=4, lc=2, label="Observed Nₕ")
end

# ╔═╡ 5bc82f81-1c49-47e3-b9c8-6c5b79b4e88a
let
	Random.seed!(100)
	D = predictive_simulate(8, 4; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], normed=true, xticks = 0:10, label="Posterior predictive on Nₕ", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Posterior predictive check with a₀=b₀=1")
	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
	vline!([7], lw=4, lc=2, label="Observed Nₕ")
end

# ╔═╡ 858d3a5f-053d-4282-83f7-ff33ad8b5f58
md"""
**A model with mis-specified prior.** Predictive checks can identify potential problems in a misspecified model. Suppose the modeller has mistakenly specified a very strong informative prior:

$$\theta \sim \text{Beta}(50, 1),$$

And the posterior becomes 

$$\theta \sim \text{Beta}(50+7, 1+3).$$

The prior and posterior plots are shown below. 


"""

# ╔═╡ 4c1f3725-ddfa-4a19-adaf-4b48ef47c79b
let
	nh, nt = 7, 3
	a₀, b₀ = 50, 1
	plot(Beta(a₀,b₀), xlims=[0.6,1], label=L"p(\theta)= \mathcal{Beta}(50,1)", linewidth=1, xlabel=L"\theta", ylabel=L"p(\theta|\cdot)" ,fill= true, lw=2, alpha=0.2, legend=:outerright, color=1, title="A mis-specified model")	
	vline!([mean(Beta(a₀,b₀))], label="prior mean", lw=2, lc=1, ls=:dash)
	plot!(Beta(a₀+nh,b₀+nt), fill= true, lw=2, alpha=0.2, color=2, label=L"p(\theta|\mathcal{D})= \mathcal{Beta}(57,4)", linewidth=2)
	vline!([mean(Beta(a₀+nh,b₀+nt))], label="posterior mean", lw=2, lc=2, ls=:dash)
end

# ╔═╡ 461dd52d-478e-4b1e-b0e0-55ecb98d4022
md"""

It can be observed that the prior (blue curve) has a strong belief that the coin is biased towards the head, which does not agree with the observed data (7 out of 10 tosses are head)! As a result, the posterior is heavily influenced by the prior and deviates a lot from the observed frequency.
"""

# ╔═╡ 97f2393d-d433-40b6-8189-61f0aace3760
md"""

Predictive checks can spot the problem for us. Let's first use the prior predictive check. The figure below shows the prior predictive check result. 5000 pseudo data sample were simulated from the prior predictive distribution. It can be observed that the observed count of heads ``N_h=7`` has zero probability of being generated from the prior.

"""

# ╔═╡ c8c7d2d7-4971-4852-aff9-a0c85a0881b3
let
	Random.seed!(100)
	D = predictive_simulate(50, 1; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10, normed=false, label="Prior predictive on Nₕ", legend=:outerbottom, xlabel="number of heads", title="Prior predictive check with a₀=50, b₀=1", ylabel=L"\#"* " of counts")
	vline!([7], lw=4, lc=2, label="Observed Nₕ")
end

# ╔═╡ edf0f6ab-3ace-47b0-9b6f-26154859862b
md"""

The posterior predictive check shows a similar story: the observed and what is expected from the posterior model are drastically different. Both checks signal that there are some problems in the model assumption.
"""

# ╔═╡ 960ec68f-75bc-4543-b91c-c06b27e54926
let
	Random.seed!(100)
	D = predictive_simulate(50+7, 1+3; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10,normed=false, label="Posterior predictive on Nₕ", legend=:outerbottom, xlabel="number of heads", title="Posterior predictive check with a₀=50, b₀=1", ylabel=L"\#"* " of counts")
	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
	vline!([7], lw=4, lc=2, label="Observed Nₕ")
end

# ╔═╡ 1ee811d8-36f7-42b1-be50-c68ee9cabf23
md"""

### Example: seven-scientist problem

"""

# ╔═╡ 83f3ee86-567b-4998-9216-2cca4bdaad0a
md"""

As a final example, we revisit the seven scientist problem and carry out model checks on the two models: the one with an oversimplified i.i.d assumption and also the better one. Instead of computing summary statistics on the predicted pseudo data, we can also simply check their empirical distributions. One possibility is to plot the fitted kernel density estimation (KDE) of the observed and simulated data and check their discrepancy.
"""

# ╔═╡ 9790024a-6cc8-47fe-892e-9ce32d7399a5
md"""

One can visualise KDE of a dataset by using `density(data)` command. For the seven scientist data, the observed density is:
"""

# ╔═╡ 36ba6267-db0b-4147-b376-49106a15426f
let
	density(scientist_data, lw=2, label="KDE density", xlabel=L"d", ylabel="density", legend=:topleft, size=(550,350), title="Density estimation of the seven-scientist data")
	plot!(scientist_data, 0.001 * ones(7), st=:scatter, label="observed data", markersize=5,  color=1, markerstrokewidth=2, shape=:vline)
end

# ╔═╡ dd91becf-167e-40eb-83e1-d1f004eb5491
md"""

**Predictive check on the wrong model.**
To carry out posterior predictive checks, one can plot KDE densities of each simulated pseudo data: ``\mathcal D_{pred}`` [^2]. The following figure carries out a posterior check on the over-simplified model with i.i.d assumption. 30 pseudo observations ``\mathcal D_{pred}`` are simulated from the predictive distribution first; then their empirical KDE estimated densities are estimated and plotted. It can be seen that the observed density has a hump around 10 but none of the simulated datasets shows a similar pattern, which signals there is a problem with the model assumption.
"""

# ╔═╡ 2ca82ec8-5350-4d21-9738-487d4181bf73
md"""
**Predictive check on the correct model.** We carry out the same check on the more realistic model where each scientist has his/her own observation precision parameter. The plot of both the observed and simulated posterior predictive densities is shown below. Compared with the previous model, the pseudo observation look alike the observed: which indicates the data generation scheme of the model matches better with the observed.

"""

# ╔═╡ 0864aa09-97b0-4421-b3be-2c621323503a
md"""

## What's next ?

Bayesian inference is powerful but the drawback is that it cannot be simply treated as a black box. One does need to understand the problem and come up with a  model that matches the problem's context, *i.e.* a story about the hypothetical data generation process. In this chapter, we have focused on some **general** modelling concepts and tools. More task-specific, such as linear regression and classification, details will be introduced later in the second half of the course. 

In the next chapter, we will move on to the computation part. A powerful inference algorithm called MCMC will be introduced. The algorithm can be used to infer complicated Bayesian models when conjugate models do not exist.

So far we have done Bayesian inferences by hand: that is all modelling and computation are manually implemented. It is not practical for day-to-day users. We will see how to do Bayesian inference with the help of probabilistic programming languages in chapter 4. With the help of PPL, an inference problem can be done easily with 4-5 lines of code. 

"""

# ╔═╡ 058134b1-19b0-4f19-9508-e288e613b116
md"""
## Notes

[^1]: [MacKay, D. J. C. Information Theory, Inference, and Learning Algorithms. (Cambridge University Press, 2003).](https://www.inference.org.uk/itprnn/book.pdf)

[^2]: Simulating predictive data for this case is bit more involved. The difficulty lies in simulating posterior samples from the posterior distribution: ``p(\mu, \{\lambda_n\}|\mathcal D)``. The joint distribution is not in any simple form. We need algorithms such as MCMC to draw samples from such a complicated posterior. MCMC will be introduced in the next chapter.
"""

# ╔═╡ 83bf58df-e2d0-4552-98b7-29ce0ce5fb5e
md"""
## Appendix

"""

# ╔═╡ a30e54cb-b843-428b-b2f0-f5cf84961ce0
begin
	begin
		struct Foldable{C}
		    title::String
		    content::C
		end
		
		function Base.show(io, mime::MIME"text/html", fld::Foldable)
		    write(io,"<details><summary>$(fld.title)</summary><p>")
		    show(io, mime, fld.content)
		    write(io,"</p></details>")
		end
	end
end

# ╔═╡ 13f7968f-fde3-4204-8237-ab33a2a5cfd0
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

# ╔═╡ cb3191ef-3b54-48ff-996c-d4993c876063
Foldable("Derivation details on the Gamma-Gaussian conjugacy.", md"
```math
\begin{align}
 p(\lambda|\mathcal D)&\propto p(\lambda) p(\mathcal D|\mu, \lambda)\\
&= \underbrace{\frac{b_0^{a_0}}{\Gamma(a_0)} \lambda^{a_0-1} e^{-b_0 \lambda}}_{p(\lambda)} \underbrace{\frac{1}{ (\sqrt{2\pi})^N}\lambda^{\frac{N}{2}}e^{\frac{-\lambda\cdot \sum_{n} (d_n-\mu)^2}{2}}}_{p(\mathcal D|\lambda, \mu)}\\
&\propto \lambda^{a_0+\frac{N}{2}-1} \exp\left \{-\left (b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}\right )\lambda\right \} \\
&= \lambda^{a_N-1} e^{- b_N \lambda}, 
\end{align}
```
where ``a_n= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.`` Note this is a unnormalised Gamma distribution (whose normalising constant can be read off directly from a Gamma distribution), therefore 


$$p(\lambda|\mathcal D)= \text{Gamma}(a_N, b_N).$$

")

# ╔═╡ 546cfb29-6f60-44be-9450-b4d33c8238e6
Foldable("Details on the posterior marginal distribution of μ.",  md"
```math
\begin{align}
p(\mu|\mathcal D) &= \int_0^\infty p(\mu, \lambda|\mathcal D) \mathrm{d}\lambda \\
&= \int_0^\infty p(\mu, \lambda|\mathcal D) \mathrm{d}\lambda\\
&= \frac{1}{p(\mathcal D)} \int_0^\infty p(\mu)p(\lambda) p(\mathcal D|\mu, \lambda)\mathrm{d}\lambda\\
&\propto p(\mu)\int_0^\infty p(\lambda) p(\mathcal D|\mu, \lambda)\mathrm{d}\lambda\\
&= p(\mu)\int_0^\infty \frac{b_0^{a_0}}{\Gamma(a_0)} \lambda^{a_0-1} e^{-b_0 \lambda} \frac{1}{ (\sqrt{2\pi})^N}\lambda^{\frac{N}{2}}e^{\frac{-\lambda\cdot \sum_{n} (d_n-\mu)^2}{2}}\mathrm{d}\lambda\\
&\propto p(\mu)\int_0^\infty\lambda^{a_0+\frac{N}{2}-1} \exp\left \{-\left (b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}\right )\lambda\right \}\mathrm{d}\lambda \\
&= p(\mu)\int_0^\infty\lambda^{a_N-1} e^{- b_N \lambda}\mathrm{d}\lambda \\
&= p(\mu)\frac{\Gamma(a_N)}{b_N^{a_N}},
\end{align}
```
where ``a_n= a_0 +\frac{N}{2}`` and ``b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}``. Note that we have used the normalising constant trick of Gamma distribution in the second last step, where we recognise ``\lambda^{a_N-1} e^{- b_N \lambda}`` is the unnormalised part of a Gamma distribution with ``a_N, b_N`` as parameters; Then the corresponding Gamma density must integrates to one:

```math
\int_{0}^\infty \frac{b_N^{a_N}}{\Gamma(a_N)}\lambda^{a_N-1} e^{- b_N \lambda}\mathrm{d}\lambda =1,
```
which leads to ``\int_{0}^\infty \lambda^{a_N-1} e^{- b_N \lambda}\mathrm{d}\lambda= \frac{\Gamma(a_N)}{b_N^{a_N}}.``
")

# ╔═╡ 15b09064-2c48-4013-8a08-c32e32d1f4df
Foldable("Derivation details on the marginal distribution.", md"

To find the marginal posterior for ``\mu``, we need to find the following marginal likelihood for observation ``d_n``:

$p(d_n|\mu) = \int p(\lambda_n) p(d_n|\mu, \lambda_n)\mathrm{d}\lambda_n,$

where we have assumed ``\mu, \lambda_n`` are independent, i.e. ``p(\lambda_n|\mu) = p(\lambda_n)``.

The marginal likelihood is the normalising constant of the marginal posterior ``p(\lambda_n|d_n, \mu)``, since

$$p(\lambda_n|d_n, \mu) = \frac{p(\lambda_n) p(d_n|\mu, \lambda_n)}{p(d_n|\mu)}.$$

Due to conjugacy, it can be shown that the conditional posterior of ``\lambda_n`` is of Gamma form 

$p(\lambda_n|\mu, x_n) = \text{Gamma}(a_n, b_n),$ where
$$a_n = a_0 + \frac{1}{2}, b_n = b_0+ \frac{1}{2} (d_n-\mu)^2.$$

The normalising constant of the unnormalised posterior is therefore the corresponding Gamma distribution's normalising constant:

$$p(d_n|\mu) \propto \frac{b_0^{a_0}}{\Gamma(a_0)} \frac{\Gamma{(a_n)}}{b_n^{a_n}}.$$
")

# ╔═╡ 7b4d6471-59f5-4167-bc35-a61549aaaef9
Foldable("More explanation on predictive distribution approximation.", md"
Note that the sampling procedure starts with

```math
\tilde{\theta} \sim p(\theta|\mathcal D, \mathcal M)
```

Then conditional on ``\theta``, simulate pseudo observation:

```math
\tilde{\mathcal{D}} \sim p(\mathcal D_{pred}|\tilde{\theta}, \mathcal M)
```

Due to the independence assumption, the joint can be factored as: 

$$p(\theta, \mathcal D_{pred}|\mathcal D, \mathcal M)= p( \mathcal D_{pred}|\theta, \mathcal M)p(\theta|\mathcal D, \mathcal M).$$

As a result, the tuple ``\tilde\theta, \tilde{\mathcal{D}}`` is actually drawn from the joint distribution:

$$\tilde{\theta}, \tilde{\mathcal{D}} \sim p(\theta,  \mathcal D_{pred}|\mathcal D,\mathcal M),$$

we can therefore retain the marginal sample ``\{\tilde{\mathcal{D}}\}`` to approximate the marginal distribution ``p(\mathcal{D}_{pred}|\mathcal D, \mathcal M)``, which provides another way to show the correctness of the approximation method.
")

# ╔═╡ 2747567e-dcab-4a99-885f-5bb19a33ab27
function gibbs_seven_scientists(data ; discard = 1000, mc= 1000, a₀=0.5, b₀=.5, m₀=0, v₀=1e4)
	μsample = zeros(mc)
	λsample = zeros(length(data), mc)
	μ = 0.0
	λ = ones(length(data))
	aₙ = a₀ + 0.5
	r₀ = 1/v₀
	for i in 1:(mc+discard)
		# sample λs
		bₙ = b₀ .+ 0.5 .* (data .- μ).^2
		λ = rand.(Gamma.(aₙ, 1.0 ./ bₙ))
		# sample μ
		vₙ = 1/(r₀ + sum(λ))
		mₙ = (m₀ * r₀ + sum(λ .* data)) * vₙ
		μ = rand(Normal(mₙ, sqrt(vₙ)))
		if i > discard
			μsample[i-discard] = μ
			λsample[:, i-discard] = λ
		end
	end
	return μsample, λsample
end

# ╔═╡ bcfd15ba-807f-4886-86fd-3e8e8661a87b
begin
	Random.seed!(100)
	mus, lambdas = gibbs_seven_scientists(scientist_data; mc= 2000)
end;

# ╔═╡ dae0d74e-3f6d-4f5d-8ece-390a4e1a170a
let
	R = 30
	D_pred = zeros(7, R)
	Random.seed!(321)
	for i in 1:R
		D_pred[:, i] = rand.(Normal.(mus[i], sqrt.(1 ./lambdas[:, i])))
	end
	
	plt = density(scientist_data, label="Observed", lw=2, xlim=[-32,30], ylim=[0, 0.25], xlabel=L"d", ylabel="density", title="Posterior predictive check on the correct model")

	for i in 1: R
		density!(D_pred[:, i], label="", lw=0.4)
	end
	plt
end

# ╔═╡ b640bef2-a58f-44ee-955e-586fd2c9ac11
function gibbs_seven_scientists_wrong(data ; discard = 1000, mc= 1000, a₀=0.5, b₀=.5, m₀=0, v₀=1e4)
	μsample = zeros(mc)
	λsample = zeros(mc)
	μ = 0.0
	λ = 1.0
	N = length(data)
	aₙ = a₀ + 0.5 * N
	r₀ = 1/v₀
	for i in 1:(mc+discard)
		# sample λs
		bₙ = b₀ + 0.5 * sum((data .- μ).^2)
		λ = rand(Gamma(aₙ, 1.0 / bₙ))
		# sample μ
		vₙ = 1/(r₀ + λ * N)
		mₙ = (m₀ * r₀ + sum(λ .* data)) * vₙ
		μ = rand(Normal(mₙ, sqrt(vₙ)))
		if i > discard
			μsample[i-discard] = μ
			λsample[i-discard] = λ
		end
	end
	return μsample, λsample
end

# ╔═╡ 7becd0a5-d391-4947-96ee-89b2ebed5759
begin
	Random.seed!(100)
	mus_wrong, lambda_wrong = gibbs_seven_scientists_wrong(scientist_data; mc= 2000)
end;

# ╔═╡ b035c02f-9362-4f8c-926c-71eefed7fbc5
let
	R = 30
	D_pred = zeros(7, R)
	Random.seed!(123)
	for i in 1:R
		# mus_wrong, lambda_wrong are posterior draws from the posterior by Gibbs sampling
		D_pred[:, i] = rand(Normal(mus_wrong[i], sqrt(1.0 /lambda_wrong[i])), 7)
	end
	plt = density(scientist_data, label="Observed", lw=2, xlim=[-32,30], ylim=[0, 0.25], legend =:topleft, xlabel=L"d", ylabel="density", title="Posterior predictive check on the wrong model")

	for i in 1: R
		density!(D_pred[:, i], label="", lw=0.4)
	end
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.66"
LaTeXStrings = "~1.3.0"
PlutoUI = "~0.7.39"
SpecialFunctions = "~2.1.7"
StatsPlots = "~0.15.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "9e7c33f8b66bcf1aff44f5bd106dce65ecb9c825"

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

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
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
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

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
git-tree-sha1 = "924cdca592bc16f14d2f7006754a621735280b74"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.1.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

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
git-tree-sha1 = "aafa0665e3db0d3d0890cdc8191ea03dc279b042"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.66"

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

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

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
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

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
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "037a1ca47e8a5989cc07d19729567bb71bfabd0c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "c8ab731c9127cd931c93221f65d6a1008dad7256"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.0+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

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
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f0956f8d42a92816d2bf062f8a6a6a0ad7f9b937"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.2.1"

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
git-tree-sha1 = "64f138f9453a018c8f3562e7bae54edc059af249"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

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
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

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
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

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
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

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
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "6d019f5a0465522bbfdd68ecfad7f86b535d6935"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.0"

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
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

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
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "a19652399f43938413340b2068e11e55caa46b65"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

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
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

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
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

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
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

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
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

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
git-tree-sha1 = "2d4e51cfad63d2d34acde558027acbc66700349b"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.3"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

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
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "2b35ba790f1f823872dcf378a6d3c3b520092eac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

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
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

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

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

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
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

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
# ╠═69b19a54-1c79-11ed-2a22-ebe65ccbfcdb
# ╟─904c5f8d-e5a9-4f18-a7d7-b4561ad37655
# ╟─5b4c3b01-bd19-4085-8a83-781086c85825
# ╟─0c123e4e-e0ff-4c47-89d6-fdf514f9e9d0
# ╟─f479a1f3-a008-4622-82f0-ab154a431a33
# ╟─2dfafb7b-773a-4c51-92e7-1f192fa354ea
# ╟─e574a570-a76b-4ab2-a395-ca40dc383e5e
# ╟─9bb06170-34d0-4528-bd9b-988ecf952089
# ╟─13cb967f-879e-49c2-b077-e9ac87569d87
# ╟─0437436b-16b9-4c90-b554-f7cf5ea8b4c0
# ╟─13f7968f-fde3-4204-8237-ab33a2a5cfd0
# ╟─3c5b6566-6e4e-41df-9865-fff8a839a70e
# ╟─f10d3513-77ee-426a-aa5c-d2bf887572d9
# ╟─8c6233e0-14fb-4202-92b9-3eddaea3e107
# ╟─7fd2a003-bed6-45d4-8335-f8d4f149c8d8
# ╟─754c5fe2-3899-4467-8fd4-828fb0ec5040
# ╠═f5dfcd1f-9d10-49d1-b68b-eafdf10baaec
# ╟─2ba73fd3-ae2b-4347-863f-28e6c21e7a91
# ╠═56e80a69-6f84-498d-b04e-5be59c1488eb
# ╟─65f0dc31-c2bd-464f-983a-9f4ca2c04a35
# ╟─d2c3bd6d-fb8c-41df-8112-c7bfbc180b0a
# ╟─76c641dd-c790-4f51-abc0-e157c00e3ba7
# ╟─cb3191ef-3b54-48ff-996c-d4993c876063
# ╟─edbf5fa3-a064-47a4-9ff4-a93dbd7b9112
# ╠═958d6028-e38f-4a56-a606-47b6f8ee86f1
# ╟─59975c2b-c537-4950-aeea-985377f22d93
# ╠═cb9ad8b5-7975-4d54-bb94-afc9f4953a67
# ╟─03a650cb-7610-4d20-8ae7-2c6e308770f6
# ╟─cb37f8a8-83aa-44f6-89a5-43fe0e4a5fa8
# ╟─2d7792cd-5b32-4228-bfef-e1ab250724f3
# ╟─7ccd4caf-ef8f-4b78-801f-b000f5aad430
# ╟─87b411d1-a62f-4462-b81b-ef0e8ac97d7e
# ╟─58d7878a-f9d6-46b4-9ac4-1944f7508f03
# ╟─3cdf9875-9c42-46b3-b88c-74e0f363bc4a
# ╟─3ca3ee53-c744-4474-a8da-6209ec5e6904
# ╟─5cbace78-8a92-43aa-9d3a-45054be48532
# ╟─2cb31d23-a542-4f8d-a056-025fa574f0d7
# ╠═31afbad6-e447-44d2-94cf-c8f99c7fa64a
# ╟─de5c3254-1fbe-46ec-ab09-77889405510d
# ╠═ff49403c-8e0e-407e-8141-6ccf178c152b
# ╟─024ce64f-29ef-49f2-a66f-87c2d4eb67a7
# ╟─52fd1028-05aa-4b37-b57c-daabf4d77f50
# ╟─d6056188-9a46-4b5f-801b-e028b6eb0b7f
# ╟─bbbb73c8-0254-463a-8e81-5acdd08583ac
# ╟─546cfb29-6f60-44be-9450-b4d33c8238e6
# ╟─2f2e3e34-76c0-4f35-8c9b-21184f86cf66
# ╠═9732358b-f592-4c66-9d8a-fdc221249a56
# ╟─9d737cc5-c533-4e86-8654-b1adba943fc0
# ╟─1d7783bf-a8d0-47d6-812d-b31ae0c0b138
# ╟─73bb12a3-db59-4e71-bf05-5a4b4e018f51
# ╟─8eb641f0-a01f-43f9-a362-e7d54c26411a
# ╟─a06f8785-fbbc-4973-b3d0-5a6db967b3cc
# ╟─15b09064-2c48-4013-8a08-c32e32d1f4df
# ╟─5fc1d8aa-5835-4559-860b-b73031d3bfe7
# ╠═da7f1485-a532-4d3c-96db-d1d50f2bdee6
# ╟─ee8fce9a-0a5b-40c2-a4d9-f124b40188f4
# ╟─d6341bf9-7a95-4a10-8a26-9495b724b907
# ╟─0a8a5aae-3567-41a3-9f15-d07235d07da2
# ╟─8fe3ed2f-e826-4faa-ace0-2286b944991f
# ╟─79b85624-a68d-41df-856a-7a7670ff4d0e
# ╟─7b4d6471-59f5-4167-bc35-a61549aaaef9
# ╟─e9a87ae2-f906-4be9-a969-7b681c2cff7b
# ╠═ebdb9711-20cc-458c-9474-8e3ace69797e
# ╠═13fae558-d12e-4e27-b36f-ec59eabb65cf
# ╟─5bc82f81-1c49-47e3-b9c8-6c5b79b4e88a
# ╟─858d3a5f-053d-4282-83f7-ff33ad8b5f58
# ╟─4c1f3725-ddfa-4a19-adaf-4b48ef47c79b
# ╟─461dd52d-478e-4b1e-b0e0-55ecb98d4022
# ╟─97f2393d-d433-40b6-8189-61f0aace3760
# ╟─c8c7d2d7-4971-4852-aff9-a0c85a0881b3
# ╟─edf0f6ab-3ace-47b0-9b6f-26154859862b
# ╟─960ec68f-75bc-4543-b91c-c06b27e54926
# ╟─1ee811d8-36f7-42b1-be50-c68ee9cabf23
# ╟─83f3ee86-567b-4998-9216-2cca4bdaad0a
# ╟─9790024a-6cc8-47fe-892e-9ce32d7399a5
# ╟─36ba6267-db0b-4147-b376-49106a15426f
# ╟─dd91becf-167e-40eb-83e1-d1f004eb5491
# ╟─b035c02f-9362-4f8c-926c-71eefed7fbc5
# ╟─2ca82ec8-5350-4d21-9738-487d4181bf73
# ╠═dae0d74e-3f6d-4f5d-8ece-390a4e1a170a
# ╟─0864aa09-97b0-4421-b3be-2c621323503a
# ╟─058134b1-19b0-4f19-9508-e288e613b116
# ╟─83bf58df-e2d0-4552-98b7-29ce0ce5fb5e
# ╠═a30e54cb-b843-428b-b2f0-f5cf84961ce0
# ╠═2747567e-dcab-4a99-885f-5bb19a33ab27
# ╠═bcfd15ba-807f-4886-86fd-3e8e8661a87b
# ╠═b640bef2-a58f-44ee-955e-586fd2c9ac11
# ╠═7becd0a5-d391-4947-96ee-89b2ebed5759
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
