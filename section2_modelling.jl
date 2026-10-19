### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 69b19a54-1c79-11ed-2a22-ebe65ccbfcdb
begin
    using PlutoUI
	using Distributions
	using StatsPlots
	using PlutoTeachingTools
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
    using Random
	using LaTeXStrings
	using SpecialFunctions
	using Logging; Logging.disable_logging(Logging.Warn);
end;

# ╔═╡ 904c5f8d-e5a9-4f18-a7d7-b4561ad37655
TableOfContents()

# ╔═╡ 085021d6-4565-49b2-b6f5-10e2fdfff15b
md"
[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section3_mcmc.html)
"

# ╔═╡ 613a65ce-9384-46ed-ab0d-abfda374f73c
md"""

# More on Bayesian modelling

"""

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

	
	# Plots.plot(TruncatedNormal(0.5, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}" *"(0.5, $(round((σ^2);digits=2)))", legend=:outerright, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"θ\in [0,1]")
	Plots.plot(truncated(Normal(0.5, σ), 0.0, 1), lw=2, label=L"\texttt{TrunNormal}" *"(0.5, $(round((σ^2);digits=2)))", legend=:outerright, xlabel=L"\theta", ylabel=L"p(\theta)", title="Priors for "*L"θ\in [0,1]")
	# Plots.plot!(TruncatedNormal(0.25, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.25, $(round((σ^2);digits=2))))")
	Plots.plot!(truncated(Normal(0.25, σ), 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.25, $(round((σ^2);digits=2))))")
	# Plots.plot!(TruncatedNormal(0.75, σ, 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.75, $(round((σ^2);digits=2))))")
	Plots.plot!(truncated(Normal(0.75, σ), 0., 1), lw=2, label=L"\texttt{TrunNormal}"*"(0.75, $(round((σ^2);digits=2))))")
	Plots.plot!(Uniform(0,1), lw=2, label=L"\texttt{Uniform}(0, 1)")
end

# ╔═╡ 2dfafb7b-773a-4c51-92e7-1f192fa354ea
md"""
*Example.* A Gaussian distribution's variance ``\sigma^2`` is a positive number: ``\sigma^2 >0``. Priors on the positive real line are e.g. Exponential distribution, or Half-Cauchy.

"""

# ╔═╡ e574a570-a76b-4ab2-a395-ca40dc383e5e
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
where ``a_0,b_0 >0`` are the prior's parameter and ``B(a_0,b_0)``, the beta function, is a normalising constant for the Beta distribution: *i.e.* ``\mathrm{B}(a_0, b_0) = \int_0^1 \theta^{a_0-1}(1-\theta)^{b_0-1}\mathrm{d}\theta``. 


*Remarks.
A few Beta distributions with different parameterisations are plotted below. Note that when ``a_0=b_0=1``, the prior reduces to a uniform distribution. Also note that when ``a_0> b_0``, e.g. ``\texttt{Beta}(5,2)``, the prior belief has its peak, or mode, greater than 0.5, which implies the prior believes the coin is biased towards the head; and vice versa.*


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
We apply the sequantial learning algorithm to the simulated data. The posterior update starts from a vague flat prior. As more data are observed and absorbed into the posterior, the posterior distribution is more and more informative and finally recovers the final posterior.
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

It is more convenient to model the precision ``\phi\triangleq 1/\sigma^2``. A conjugate prior for the precision parameter is Gamma distribution. Gamma distributions have support on the positive real line which matches the precision's value range.

A Gamma distribution, parameterised with a shape  ``a_0>0``,  and a rate parameter ``b_0>0``, has a probability density function:

```math
p(\phi; a_0, b_0) = \texttt{Gamma}(\phi; a_0, b_0)=\frac{b_0^{a_0}}{\Gamma(b_0)} \phi^{a_0-1} e^{-b_0\phi}.
```
"""

# ╔═╡ d2c3bd6d-fb8c-41df-8112-c7bfbc180b0a
let
	as_ = [1,2,3,5,9,7.5,0.5]
	bs_ = [1/2, 1/2, 1/2, 1, 1/0.5, 1, 1]
	plt= plot(xlim =[0,20], ylim = [0, 0.8], xlabel=L"\phi", ylabel="density")
	for i in 1:length(as_)
		plot!(Gamma(as_[i], 1/bs_[i]), fill=(0, .1), lw=1.5, label=L"\texttt{Gamma}"*"($(as_[i]),$(bs_[i]))")
	end
	plt
end

# ╔═╡ 76c641dd-c790-4f51-abc0-e157c00e3ba7
md"""

**Conjugacy.** The data follow a Gaussian distribution. That is for ``n = 1,2,\ldots, N``:
```math
d_n \sim \mathcal N(\mu, 1/\phi);
```
The likelihood, therefore, is a product of Gaussian likelihoods:


```math
p(\mathcal D|\mu, \phi) = \prod_{n=1}^N p(d_n|\mu, \phi) = \prod_{n=1}^N \mathcal N(d_n; \mu, 1/\phi) .
```

It can be shown that the posterior formed according to Baye's rule

```math

p(\phi|\mathcal D, \mu) \propto p(\phi) p(\mathcal D|\mu, \phi)

```



is still of a Gamma form (therefore conjugacy is established): 

```math
p(\phi|\mathcal D, \mu) = \texttt{Gamma}(a_N, b_N),
```

where 

$$a_N= a_0 +\frac{N}{2}, \;\;b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}.$$

The Bayesian computation reduces to hyperparameter update again. Note the posterior mean is 

$$\mathbb{E}[\phi|\mathcal D] = \frac{a_N}{b_N} = \frac{a_0 + N/2}{b_0 + {\sum_n (d_n -\mu)^2}/{2}}.$$

If we assume ``a_0= b_0=0`` (a flat prior), we recover the regular maximum likelihood estimator for ``{\sigma^2}``: 

$$\hat \sigma^2 =1/\hat{\phi} = \frac{\sum_n (d_n-\mu)^2}{N}.$$

Based on the above result, we can intuitively interpret the hyperparameters as 
* ``a_N`` : the total count of observations contained in the posterior (both pseudo ``a_0`` and real observation ``N/2``) ;
* ``b_N`` : the rate parameter is the total sum of squares;


"""

# ╔═╡ cb3191ef-3b54-48ff-996c-d4993c876063
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

# ╔═╡ edbf5fa3-a064-47a4-9ff4-a93dbd7b9112
md"""
**Demonstration:** We first simulate ``N=100`` Gaussian observations with unknown ``\mu=0`` and ``\phi= 1/\sigma^2 = 2``.

"""

# ╔═╡ 958d6028-e38f-4a56-a606-47b6f8ee86f1
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

# ╔═╡ 59975c2b-c537-4950-aeea-985377f22d93
md"""
We have used a relatively vague Gamma prior with ``a_0=b_0=0.5``; and the posterior can be easily calculated by updating the hyperparameters: ``\texttt{Gamma}(a_0+ 100/2, b_0+ \texttt{sse}/2).`` The update is implemented below in Julia:
"""

# ╔═╡ cb9ad8b5-7975-4d54-bb94-afc9f4953a67
begin
	a₀ = 0.5
	b₀ = 0.5
	# Gamma in Distributions.jl is implemented with shape and scale parameters where the second parameter is 1/b 
	prior_ϕ = Gamma(a₀, 1/b₀)
	posterior_ϕ = Gamma(a₀+ N/2, 1/(b₀+ sum(gaussian_data.^2)/2))
end;

# ╔═╡ 03a650cb-7610-4d20-8ae7-2c6e308770f6
md"""
We can plot the prior and posterior distributions to visually check the Bayesian update's effect. To plot a random variable in Julia, one can simply use `plot()` function from `StatsPlots.jl` package. The figure of both the prior and posterior plots is shown below.
"""

# ╔═╡ cb37f8a8-83aa-44f6-89a5-43fe0e4a5fa8
let
	plot(prior_ϕ, xlim = [0, 5], ylim=[0, 1.5], lw=2, fill=(0, 0.2), label="Prior "* L"p(\phi)", xlabel=L"\phi", ylabel="density", title="Conjugate inference of a Gaussian's precision")

	plot!(posterior_ϕ, lw=2,fill=(0, 0.2), label="Posterior "* L"p(\phi|\mathcal{D})")
	vline!([true_ϕ], label="true "*L"\phi", lw=4)
	vline!([mean(prior_ϕ)], label="prior mean", lw=2, lc=1, ls=:dash)
	vline!([mean(posterior_ϕ)], label="posterior mean", lw=2, lc=2, ls=:dash)
	# vline!([1/σ²], label="true "*L"λ", lw=2)
	# end
end

# ╔═╡ 2d7792cd-5b32-4228-bfef-e1ab250724f3
md"""
**Sequential update.** The conjugacy also provides us with a computationally cheap way to sequentially update the posterior. Due to the independence assumption, we do not need to calculate the posterior in one go.

```math
\begin{align}
p(\phi|\{d_1, d_2, \ldots, d_N\})&\propto  p(\phi) \prod_{n=1}^N p(d_n|\phi) p(d_N|\phi)\\
&= \underbrace{p(\phi) \prod_{n=1}^{N-1} p(d_n|\phi)}_{p(\phi|\mathcal D_{N-1})}\cdot p(d_N|\phi)\\
&= \underbrace{p(\phi|\mathcal D_{N-1})}_{\text{new prior}} p(d_N|\phi)
\end{align}
```
With observations up to ``N-1``, we obtain a posterior ``p(\phi|\mathcal D_{N-1})``.
The posterior now serves as the new prior waiting to be updated with the next observation ``d_N``. It can be shown that the final posterior sequentially updated this way is the same as the off-line posterior.

To be more specific, the sequential update algorithm for the precision example is:

Initialise with a prior ``p(\phi|\varnothing)``;
for ``n = 1,2,\ldots, N``
* update ``a_n = a_{n-1} + 0.5 ``, ``b_n = b_{n-1} + 0.5 \cdot (d_n-\mu)^2``
* report the posterior at ``n`` if needed

To demonstrate the idea, check the following animation. The posterior update starts from a vague prior. As more data is observed and absorbed into the posterior, the posterior distribution is more and more informative and finally recovers the ground truth.
"""

# ╔═╡ 7ccd4caf-ef8f-4b78-801f-b000f5aad430
let
	plot(prior_ϕ, xlim = [0, 5], ylim=[0, 1.5], lw=2, fill=(0, 0.1), label="Prior "* L"p(\phi)", xlabel=L"\phi", ylabel="density", title="Sequential update N=0")
	plot!(posterior_ϕ, lw=2, fill=(0, 0.1), label="Posterior "* L"p(\phi|\mathcal{D})")
	vline!([true_ϕ], label="true "*L"\phi", lw=2, lc=3)
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

**Non-informative** prior, as the name suggests, contains no information in the prior [^1] and *let the data speak for itself*. For our coin-flipping case, a possible choice is a flat uniform prior: *i.e.* ``p(\theta) \propto 1`` when ``\theta\in[0,1]``. 


**Informative** prior, on the other hand, contains the modeller's subjective prior judgement. 

For example, if we believe our coin should be fair, we can impose an informative prior such as ``p(\theta) =\texttt{Beta}(n,n)``. When ``n`` gets larger, e.g. ``n=5``,  the prior becomes more concentrated around ``\theta=0.5``, which implies a stronger prior belief in the coin being fair.

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

## Likelihood model variants

"""

# ╔═╡ 5cbace78-8a92-43aa-9d3a-45054be48532
md"""
Bayesian modelling is more an art than science. Ideally, each problem should have its own bespoke model. We have only seen problems where the observed data is assumed to be independently and identically distributed (**i.i.d.**). For example, in the coin-flipping example, since the ten coin tosses ``\{d_1, d_2, \ldots, d_{10}\}`` are *independent* tossing realisations of the *same* coin, the i.i.d. assumption makes sense. We assume each ``d_i`` follows the same Bernoulli distribution and they are all independent.

However, there are problems in which more elaborate modelling assumptions are required. To demonstrate the idea, we consider the following inference problem.



"""

# ╔═╡ 2cb31d23-a542-4f8d-a056-025fa574f0d7
md"""

!!! question "Seven scientist problem"
	[*The question is adapted from [^1]*] Seven scientists (A, B, C, D, E, F, G) with widely-differing experimental skills measure some signal ``\mu``. You expect some of them to do accurate work (*i.e.* to have small observation variance ``\sigma^2``, and some of them to turn in wildly inaccurate answers (*i.e.* to have enormous measurement error). What is the unknown signal ``\mu``?

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
*Remarks. Based on the plot, scientists C, D, E, F, and G all made similar measurements. Scientists A and B's experimental skills seem questionable. This is a problem that the frequentist method should find challenging. If all 7 measurements were observed by one scientist or scientists with a similar level of experimental skill, the sample mean: 
$$\frac{\sum_n d_n}{N} \approx 3.46$$ 
would have been a good estimator. 
An ad hoc remedy is probably to treat the first two observations as outliers and take an average over the rest of the 5 measurements. This remedy lacks formal justification and does not scale well with a larger dataset.*

"""

# ╔═╡ 52fd1028-05aa-4b37-b57c-daabf4d77f50
md"""

### A bad Bayesian model

**Modelling**:
One possible model is to ignore the subtleties and reuse our coin-flipping model's assumption. Since the observed data is real-valued, we only need to replace a Bernoulli likelihood with a Gaussian. We then assume observations ``d_n`` are i.i.d distributed with a Gaussian 

$$d_n \overset{\mathrm{i.i.d}}{\sim} \mathcal N(\mu, \sigma^2),$$

where the mean is the unknown signal ``\mu`` and a shared ``\sigma^2`` is the observation variance. The model implies each scientist's observation is the true signal ``\mu`` plus some Gaussian distributed observation noise.

To specify a Bayesian model, we need to continue to specify a prior model for the two unknowns ``\mu``, ``\sigma^2``. For computational convenience, we assume a Gaussian prior for the signal ``\mu``:

$$\mu \sim \mathcal N(m_0, v_0),$$
* ``m_0`` is our prior guess of the signal's centre
* ``v_0`` represents our prior belief strength;

To show our ignorance, we can set ``m_0=0`` (or the sample average) and ``v_0`` to a very large positive number, say 10,000. The prior then becomes a very flat vague distribution.

It is more convenient to model the observation precision ``\phi \triangleq 1/\sigma^2`` instead of variance ``\sigma^2``. Here we assume a Gamma prior for the precision parameter:

$$\phi \sim \texttt{Gamma}(a_0, b_0)$$

Again, to show our ignorance, we can set ``a_0, b_0`` such that the distribution is as flat and vague as possible. A possible parameterisation is ``a_0=b_0=0.5.`` Note that Gamma is a distribution on the positive real line which has matching support for the precision parameter. 

To put them together, the full Bayesian model is:
```math
\begin{align}
\text{prior}: \mu &\sim \mathcal N(m_0=0, v_0=10000)\\
\phi &\sim \texttt{Gamma}(a_0=0.5, b_0=0.5)\\
\text{likelihood}: d_n &\overset{\mathrm{i.i.d}}{\sim} \mathcal N(\mu, 1/\phi) \;\; \text{for } n = 1,2,\ldots, 7.
\end{align}
```
**Computation:**
After specifying the model, we need to apply Baye's rule to compute the posterior distribution:

```math
\begin{align}
p(\mu, \phi|\mathcal D) &\propto p(\mu, \phi) p(\mathcal D|\mu, \phi) 
\\
&= p(\mu)p(\phi) p(\mathcal D|\mu, \phi) \\
&= p(\mu)p(\phi) \prod_{n=1}^N p(d_n|\mu, \phi);
\end{align}
```
where we have assumed the prior for ``\mu`` and ``\phi`` are independent. Sub-in the definition of the prior and likelihood, we can plot the posterior.
"""

# ╔═╡ d6056188-9a46-4b5f-801b-e028b6eb0b7f
let
	m₀, v₀ = 0, 10000
	a₀, b₀ = 0.5, 0.5
	function ℓπ(μ, ϕ; data)  
		σ² = 1/ϕ
		logprior = logpdf(Normal(m₀, v₀), μ) + logpdf(Gamma(a₀, 1/b₀), ϕ) 
		logLik = sum(logpdf.(Normal(μ, sqrt(σ²)), data))
		return logprior + logLik
	end

	plot(-13:0.05:20, 0:0.001:0.019, (x, y) -> exp(ℓπ(x, y; data=scientist_data)), st=:contour,fill=true, ylim=[0.001, 0.015], xlabel=L"μ", ylabel=L"\phi", title="Contour plot of "*L"p(\mu, \phi|\mathcal{D})")
end

# ╔═╡ bbbb73c8-0254-463a-8e81-5acdd08583ac
md"""
**Marginal posterior ``p(\mu|\mathcal D)``:**
The posterior distribution shows the posterior peaks around ``\mu = 3.5``, which is roughly the same as the sample average. However, to better answer the question, we should treat ``\phi`` as a *nuisance* parameter, and integrate it out to find the marginal posterior for ``\mu`` only. After some algebra, we find the unnormalised marginal posterior is of the form:

```math
p(\mu|\mathcal D) \propto p(\mu)\cdot \Gamma\left (a_0+ \frac{N}{2}\right )\left (b_0+\frac{\sum_n (d_n-\mu)^2}{2}\right )^{- (a_0+ \frac{N}{2})},
```
where ``N=7`` for our problem. 

"""

# ╔═╡ 546cfb29-6f60-44be-9450-b4d33c8238e6
Foldable("Details on the posterior marginal distribution of μ.",  md"
```math
\begin{align}
p(\mu|\mathcal D) &= \int_0^\infty p(\mu, \phi|\mathcal D) \mathrm{d}\phi \\
&= \frac{1}{p(\mathcal D)} \int_0^\infty p(\mu)p(\phi) p(\mathcal D|\mu, \phi)\mathrm{d}\phi\\
&\propto p(\mu)\int_0^\infty p(\phi) p(\mathcal D|\mu, \phi)\mathrm{d}\phi\\
&= p(\mu)\int_0^\infty \frac{b_0^{a_0}}{\Gamma(a_0)} \phi^{a_0-1} e^{-b_0 \phi} \frac{1}{ (\sqrt{2\pi})^N}\phi^{\frac{N}{2}}e^{\frac{-\phi\cdot \sum_{n} (d_n-\mu)^2}{2}}\mathrm{d}\phi\\
&\propto p(\mu)\int_0^\infty\phi^{a_0+\frac{N}{2}-1} \exp\left \{-\left (b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}\right )\phi\right \}\mathrm{d}\phi \\
&= p(\mu)\int_0^\infty\phi^{a_N-1} e^{- b_N \phi}\mathrm{d}\phi \\
&= p(\mu)\frac{\Gamma(a_N)}{b_N^{a_N}},
\end{align}
```
where ``a_n= a_0 +\frac{N}{2}`` and ``b_N = b_0 + \frac{\sum_{n} (d_n- \mu)^2}{2}``. Note that we have used the normalising constant trick of Gamma distribution in the second last step, where we recognise ``\phi^{a_N-1} e^{- b_N \phi}`` is the unnormalised part of a Gamma distribution with ``a_N, b_N`` as parameters; Then the corresponding Gamma density must integrate to one:

```math
\int_{0}^\infty \frac{b_N^{a_N}}{\Gamma(a_N)}\phi^{a_N-1} e^{- b_N \phi}\mathrm{d}\phi =1,
```
which leads to ``\int_{0}^\infty \phi^{a_N-1} e^{- b_N \phi}\mathrm{d}\phi= \frac{\Gamma(a_N)}{b_N^{a_N}}.``
")

# ╔═╡ 2f2e3e34-76c0-4f35-8c9b-21184f86cf66
md"""

We can implement the (log) un-normalised density in Julia (check `log_marginal_μ_wrong` function below). It is more common to compute log probability to avoid numerical issues. For reference, the log posterior density is:

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
The unnormalised marginal posterior is plotted below. It shows the most likely estimate for the signal is about 3.46, which is counter-intuitive.
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
The i.i.d assumption does not reflect the subtleties of the data generation process. A better model should assume each observation follows an independent but not identical Gaussian distribution. In particular, for each scientist, we should introduce their own observation precision, ``\phi_n \triangleq \sigma^2_n`` to reflect their "different levels of experimental skills".


The improved model now includes 7+1 unknown parameters: the unknown signal ``\mu`` and the observation precisions ``\{\phi_n\}_{n=1}^7`` for each of the seven scientists:

```math
\begin{align}
\mu &\sim \mathcal N(m_0, v_0)\\
\phi_n &\sim \texttt{Gamma}(a_0, b_0)\;\; \text{for }n = 1,2,\ldots,7\\
d_n &\sim \mathcal N(\mu, 1/\phi_n)\;\; \text{for }n = 1,2,\ldots,7\\
\end{align}
```

Intuitively, the precision ``\phi_n`` reflects a scientist's experimental skill level. Higher precision (lower observation variance) implies better skill.
"""

# ╔═╡ a06f8785-fbbc-4973-b3d0-5a6db967b3cc
md"""
**Computation:**

Similarly, we want to find out the marginal posterior distribution. After integrating out the nuisance parameters ``\{\phi_n\}`` from the posterior, we can find the marginal posterior is of the following form:

```math
p(\mu|\mathcal D) \propto p(\mu)\prod_{n=1}^N p(d_n|\mu) \propto p(\mu)\prod_{n=1}^N \frac{\Gamma{(a_n)}}{b_n^{a_n}},
```

where for ``n= 1,2\ldots, 7``:

$$a_n = a_0 + \frac{1}{2}, b_n = b_0+ \frac{1}{2} (d_n-\mu)^2.$$ 
"""

# ╔═╡ 15b09064-2c48-4013-8a08-c32e32d1f4df
Foldable("Derivation details on the marginal distribution.", md"

To find the marginal posterior for ``\mu``, we need to find the following marginal likelihood for observation ``d_n``:

$p(d_n|\mu) = \int p(\phi_n) p(d_n|\mu, \phi_n)\mathrm{d}\phi_n,$

where we have assumed ``\mu, \phi_n`` are independent, i.e. ``p(\phi_n|\mu) = p(\phi_n)``.

The marginal likelihood is the normalising constant of the marginal posterior ``p(\phi_n|d_n, \mu)``, since

$$p(\phi_n|d_n, \mu) = \frac{p(\phi_n) p(d_n|\mu, \phi_n)}{p(d_n|\mu)}.$$

Due to conjugacy, it can be shown that the conditional posterior of ``\phi_n`` is of Gamma form 

$p(\phi_n|\mu, x_n) = \text{Gamma}(a_n, b_n),$ where
$$a_n = a_0 + \frac{1}{2}, b_n = b_0+ \frac{1}{2} (d_n-\mu)^2.$$

The normalising constant of the unnormalised posterior is therefore the corresponding Gamma distribution's normalising constant:

$$p(d_n|\mu) \propto \frac{b_0^{a_0}}{\Gamma(a_0)} \frac{\Gamma{(a_n)}}{b_n^{a_n}}.$$
")

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

The posterior makes much better sense. It not only correctly identifies the consensus of the majority of the scientists (*i.e.* scientists C, D, E,..., G) and also takes into account scientists A and B's noisy observations.
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

## Check your models: Predictive checks

Bayesian inference provides a great amount of modelling freedom to its user. The modeller, for example, can incorporate his or her expert knowledge in the prior distribution and also choose a suitable likelihood that best matches the data generation process. However, greater flexibility comes with a price. The modeller also needs to take full responsibility for the modelling decisions. To be more specific, for a Bayesian model, one at least needs to check whether
* the prior makes sense?
* the generative model as a whole (*i.e.* prior plus likelihood) match the observed data? 
In other words, the user needs to **validate the model** before making any final inference decisions. **Predictive checks** are a great way to empirically validate a model's assumptions.
"""

# ╔═╡ 8fe3ed2f-e826-4faa-ace0-2286b944991f
md"""

### Posterior predictive check


The idea of predictive checks is to generate future *pseudo observations* based on the assumed model's (posterior) **prediction distribution**:

$$\mathcal{D}^{(r)} \sim p(\mathcal D_{\textit{pred}}|\mathcal D, \mathcal M), \;\; \text{for }r= 1\ldots, R$$ 

where ``\mathcal D`` is the observed data and ``\mathcal M`` denotes the Bayesian model. Note that the posterior predictive distribution indicates what future data might look like, given the observed data and our model. If the model assumptions (both the prior and likelihood) are reasonable, we should expect the generated pseudo data "agree with" the observed. 

Comparing vector data is not easy (note that a sample ``\mathcal D`` is usually high-dimensional). In practice, we compute the predictive distribution of some summary statistics, say mean, variance, median, or any meaningful statistic instead, and visually check whether the observed statistic falls within the predictions' possible ranges. 


The possible credible ranges can be calculated based on the Monte Carlo method. Based on the Monte Carlo principle, after simulating ``R`` pseudo samples,

$$\tilde{\mathcal D}^{(1)}, \tilde{\mathcal D}^{(2)}, \tilde{\mathcal D}^{(3)},\ldots, \tilde{\mathcal D}^{(R)} \sim p(\mathcal D_{pred}|\mathcal D, \mathcal M),$$ the predictive distribution of a summary statistic ``t(\cdot)``: 

$$p(t(D_{pred})|\mathcal D, \mathcal M)$$ 

can be approximated by the empirical distribution of ``\{t(\tilde{\mathcal D}^{(1)}), t(\tilde{\mathcal D}^{(2)}), \ldots, t(\tilde{\mathcal D}^{(R)})\}``. One can check whether the observed statistic ``t(\mathcal{D})`` falls within a credible region of the empirical distribution ``\{t(\tilde{\mathcal D}^{(1)}), t(\tilde{\mathcal D}^{(2)}), \ldots, t(\tilde{\mathcal D}^{(R)})\}``. We will see some examples soon to demonstrate the idea.


### Prior predictive check
Alternatively, one can use a **prior predictive distribution** to simulate the *future* data: *i.e.*

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
in which the unknown parameters ``\theta`` are integrated out. Assuming that past and future observations are conditionally independent given  ``\theta``, *i.e.* ``p(\mathcal D_{pred} |\theta, \mathcal D, \mathcal M) = p(\mathcal D_{pred} |\theta, \mathcal M)``, the above equation can be written as:

```math

p(\mathcal D_{pred}|\mathcal D, \mathcal M) = \int p(\mathcal D_{pred} |\theta, \mathcal M) p(\theta|\mathcal D, \mathcal M) \mathrm{d}\theta

```

The integration in general is intractable. However, we can rely on sampling-based methods to approximate the integration.


!!! information ""
	Repeat the following many times:
	
	1. Draw one sample from the posterior (or the prior for prior predictive): $$\tilde \theta \sim p(\theta|\mathcal D)$$
	2. Conditioning on ``\tilde \theta``, simulate pseudo observations: ``\tilde{\mathcal D}\sim p(\mathcal D|\tilde{\theta}) ``

The Monte Carlo approximation method can also be viewed as an ensemble method:


$$p(\mathcal D_{pred}|\mathcal D, \mathcal M) \approx \frac{1}{R} \sum_{r} p(\mathcal D_{pred}|\theta^{(r)}).$$
We are making predictions of future data on an ensemble of ``R`` models by taking the average, and each ensemble element model is indexed by one posterior (prior) sample. 


The ensemble view highlights a key difference between the Bayesian and frequentist methods. When it comes to prediction, the frequentist usually applies the plug-in principle: use a point estimator of ``\theta`` (e.g. the maximum likelihood estimator) and that singleton model to predict:

$$p(\mathcal D_{pred}|\hat{\theta}, \mathcal M).$$
On the other hand, the Bayesian adopts a more democratic approach: an ensemble of models is consulted.


*Remarks. In simulating a pseudo sample ``\tilde{\mathcal D}``, we need to generate the same number of observations as ``\mathcal D``. For example, in the coin flipping example, for each ``\tilde\theta``, we need to simulate 10 tosses.* 



"""

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

$$p(\theta, \mathcal D_{pred}|\mathcal D, \mathcal M)= p( \mathcal D_{pred}|\theta, \cancel{\mathcal{D}},\mathcal M)p(\theta|\mathcal D, \mathcal M).$$

As a result, the tuple ``\tilde\theta, \tilde{\mathcal{D}}`` is actually drawn from the joint distribution:

$$\tilde{\theta}, \tilde{\mathcal{D}} \sim p(\theta,  \mathcal D_{pred}|\mathcal D,\mathcal M),$$

we can therefore retain the marginal sample ``\{\tilde{\mathcal{D}}\}`` to approximate the marginal distribution ``p(\mathcal{D}_{pred}|\mathcal D, \mathcal M)``, which provides another way to show the correctness of the approximation method.
")

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
	histogram(sum(D, dims=1)[:], bins = 20, xticks=0:10, normed=true, label="Prior predictive on "*L"N_h", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Prior predictive check with "*L"a_0=b_0=1")
	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
end

# ╔═╡ 5bc82f81-1c49-47e3-b9c8-6c5b79b4e88a
let
	Random.seed!(100)
	D = predictive_simulate(8, 4; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], normed=true, xticks = 0:10, label="Posterior predictive on "*L"N_h", legend=:outerbottom, xlabel="number of heads "*L"N_h", title="Posterior predictive check with "*L"a_0=b_0=1")
	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
end

# ╔═╡ 858d3a5f-053d-4282-83f7-ff33ad8b5f58
md"""
**A model with misspecified prior.** Predictive checks can identify potential problems in a misspecified model. Suppose the modeller has mistakenly used a very strongly informative prior

$$\theta \sim \text{Beta}(50, 1),$$

which contradicts the observation. As a result, the posterior is dominated by the prior: 

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

Predictive checks can spot the problem for us. Let's first use the prior predictive check. The figure below shows the prior predictive check result. 5000 pseudo-data samples were simulated from the prior predictive distribution. It can be observed that the observed count of heads ``N_h=7`` has a near zero probability of being generated from the prior.

"""

# ╔═╡ c8c7d2d7-4971-4852-aff9-a0c85a0881b3
let
	Random.seed!(100)
	D = predictive_simulate(50, 1; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10, normed=false, label="Prior predictive on " *L"N_h", legend=:outerbottom, xlabel="number of heads", title="Prior predictive check with "*L"a_0=50, b_0=1", ylabel=L"\#"* " of counts")
	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
end

# ╔═╡ edf0f6ab-3ace-47b0-9b6f-26154859862b
md"""

The posterior predictive check shows a similar story: the observed and what is expected from the posterior model are drastically different. Both checks signal that there are some problems in the model assumption.
"""

# ╔═╡ 960ec68f-75bc-4543-b91c-c06b27e54926
let
	Random.seed!(100)
	D = predictive_simulate(50+7, 1+3; N=10 , R=5000)
	histogram(sum(D, dims=1)[:], xlim = [0,11], xticks=0:10,normed=false, label="Posterior predictive on "*L"N_h", legend=:outerbottom, xlabel="number of heads", title="Posterior predictive check with "*L"a_0=50, b_0=1", ylabel=L"\#"* " of counts")
	# density!(sum(D, dims=1)[:], xlim=[0,10], lc=1, lw=2, label="")
	vline!([7], lw=4, lc=2, label="Observed "*L"N_h")
end

# ╔═╡ 1ee811d8-36f7-42b1-be50-c68ee9cabf23
md"""

### Example: seven-scientist problem

"""

# ╔═╡ 83f3ee86-567b-4998-9216-2cca4bdaad0a
md"""

As a final example, we revisit the seven-scientist problem and carry out model checks on the two models: the one with an oversimplified i.i.d assumption and also the better one. Instead of computing summary statistics on the predicted pseudo data, we can also simply check their empirical distributions. One possibility is to plot the fitted **k**ernel **d**ensity **e**stimation (KDE) of the observed and simulated data and check their discrepancy.
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

[^2]: Simulating predictive data for this case is bit more involved. The difficulty lies in simulating posterior samples from the posterior distribution: ``p(\mu, \{\phi_n\}|\mathcal D)``. The joint distribution is not in any simple form. We need algorithms such as MCMC to draw samples from such a complicated posterior. MCMC will be introduced in the next chapter.
"""

# ╔═╡ 567665ad-d548-4ccd-aebe-dd3ccbc2a535
md"
[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section3_mcmc.html)
"

# ╔═╡ 83bf58df-e2d0-4552-98b7-29ce0ce5fb5e
md"""
## Appendix

"""

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
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.132"
LaTeXStrings = "~1.4.1"
Plots = "~1.41.7"
PlutoTeachingTools = "~0.5.0"
PlutoUI = "~0.7.83"
SpecialFunctions = "~2.9.0"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "d5bb4957bf905b3fe23a8beb6190cf53dd2e7703"

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
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

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
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

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
git-tree-sha1 = "704274ca960c42049fd9bda089fc4be32af5cdca"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.5+0"

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
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

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
# ╟─69b19a54-1c79-11ed-2a22-ebe65ccbfcdb
# ╟─904c5f8d-e5a9-4f18-a7d7-b4561ad37655
# ╟─085021d6-4565-49b2-b6f5-10e2fdfff15b
# ╟─613a65ce-9384-46ed-ab0d-abfda374f73c
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
# ╟─56e80a69-6f84-498d-b04e-5be59c1488eb
# ╟─65f0dc31-c2bd-464f-983a-9f4ca2c04a35
# ╟─d2c3bd6d-fb8c-41df-8112-c7bfbc180b0a
# ╟─76c641dd-c790-4f51-abc0-e157c00e3ba7
# ╟─cb3191ef-3b54-48ff-996c-d4993c876063
# ╟─edbf5fa3-a064-47a4-9ff4-a93dbd7b9112
# ╟─958d6028-e38f-4a56-a606-47b6f8ee86f1
# ╟─59975c2b-c537-4950-aeea-985377f22d93
# ╟─cb9ad8b5-7975-4d54-bb94-afc9f4953a67
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
# ╟─ff49403c-8e0e-407e-8141-6ccf178c152b
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
# ╟─13fae558-d12e-4e27-b36f-ec59eabb65cf
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
# ╠═b035c02f-9362-4f8c-926c-71eefed7fbc5
# ╟─2ca82ec8-5350-4d21-9738-487d4181bf73
# ╠═dae0d74e-3f6d-4f5d-8ece-390a4e1a170a
# ╟─0864aa09-97b0-4421-b3be-2c621323503a
# ╟─058134b1-19b0-4f19-9508-e288e613b116
# ╟─567665ad-d548-4ccd-aebe-dd3ccbc2a535
# ╟─83bf58df-e2d0-4552-98b7-29ce0ce5fb5e
# ╠═2747567e-dcab-4a99-885f-5bb19a33ab27
# ╠═bcfd15ba-807f-4886-86fd-3e8e8661a87b
# ╠═b640bef2-a58f-44ee-955e-586fd2c9ac11
# ╠═7becd0a5-d391-4947-96ee-89b2ebed5759
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
