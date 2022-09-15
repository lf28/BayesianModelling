### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ ce716e92-0c3c-11ed-364f-2f8266fa10f2
begin
    using PlutoUI
	using StatsPlots
    using LinearAlgebra
	using Distributions
	using ForwardDiff
	using MCMCChains
    using DataFrames
	using Random
	using LaTeXStrings
	using HypertextLiteral
	using Logging; Logging.disable_logging(Logging.Info);
end;

# ╔═╡ 8daf0123-b22e-4cda-8eec-7cad3484c8e0
TableOfContents()

# ╔═╡ 70f5222b-aadf-4dc0-bc03-b1d10e7db8b6
md"""
# Bayesian inference with MCMC
"""

# ╔═╡ 46af5470-afc8-47a8-a6c4-07de22105e91
md"""
In this chapter, we introduce a technique called Markov Chain Monte Carlo (MCMC), which is arguably the most important algorithm for Bayesian inference. 
In a nutshell, MCMC aims at computing the posterior distribution ``p(\theta|\mathcal{D})`` efficiently. Without it, most Bayesian computations cannot be finished in a realistic timeframe. Therefore, it is safe to say the algorithm is instrumental to the very success of the modern Bayesian method. 

In the following sections, we will first explain why the algorithm is needed for Bayesian inference. Different forms of MCMC algorithms, such as Metropolis-Hastings, Gibbs sampling, and Hamiltonian sampler, are introduced afterwards. The intuitions behind the algorithms are given higher priority than their theoretical buildups. From a Bayesian practitioner's perspective, it is probably more important to know how to use the algorithm in real-world analysis. Therefore, the practical aspects of the technique, such as MCMC diagnostic tools, are also explained in detail.
"""

# ╔═╡ f9584f19-9020-481d-93b2-8d52e8c4bdfc
md"""


## Bayesian computation is hard

At a first glance, it might be hard to see why Bayesian computation can be difficult. After all, **Baye's theorem** has provided us with a very straightforward mechanism to compute the posterior:

$$\text{posterior}=\frac{\text{prior} \times \text{likelihood}}{\text{evidence}}\;\;\text{or}\;\;p(\theta|\mathcal D) =\frac{p(\theta) p(\mathcal D|\theta)}{p(\mathcal D)},$$
where 

$$p(\mathcal D) = \int p(\theta) p(\mathcal D|\theta) \mathrm{d}\theta,$$ known as *evidence* or *marginal likelihood*, is a constant with respect to the model parameter ``\theta``.

The recipe is only easy to write down, but the daunting bit actually lies in the computation of the normalising constant: ``p(\mathcal D)``. The integration is often high-dimensional, therefore usually *intractable*. 

As a result, posterior probability calculations, such as $\theta \in A$

$$\mathbb{P}(\theta \in A|\mathcal D) = \int_{\theta \in A} p(\theta |\mathcal D) \mathrm{d}\theta = \frac{ \int_{\theta \in A} p(\theta) p(\mathcal D|\theta) \mathrm{d}\theta}{p(\mathcal D)},$$ can not be evaluated exactly. For example, in the coin tossing example introduced last time, we may consider any coin with a bias ``0.5 \pm 0.05`` fair, the posterior we want to know is 

$$\mathbb{P}(0.45 \leq \theta \leq 0.55|\mathcal D).$$ 



We sidestepped the integration problem last time by discretising the parameter space $\Theta = [0,1]$ into some finite discrete choices. The method has essentially replaced a difficult integration with a **brute-force enumeration** summation

$$\mathbb{P}(0.45 \leq \theta \leq 0.55|\mathcal D) = \frac{\int_{.45}^.55 p(\theta)p(\theta |\mathcal D) \mathrm{d}\theta}{p(\mathcal D)}\approx \frac{\sum_{\theta_0'\in \{.5\}} p(\theta=\theta_0')p(\mathcal D|\theta=\theta_0')}{\sum_{\theta_0\in \{0, 0.1, \ldots, 1.0\}} p(\theta=\theta_0)p(\mathcal D|\theta=\theta_0)}.$$

Unfortunately, this brute force discretisation-based method is not scalable. When the parameter space's dimension gets larger, the algorithm becomes too slow to use. To see it, consider discretising a ``D`` dimensional parameter space, ``\Theta \in R^D``, if each parameter is discretised with 2 choices (which is a very crude discretisation), the total discretized space is of order ``2^D``, which grows exponentially. Such an exponentially growing size soon becomes problematic for all modern computers to handle. 


What's worse, the difficulty does not end here. Assuming we knew the normalising constant, *i.e.* ``p(\mathcal D)`` were known and the posterior can be evaluated exactly, we still need to evaluate numerator's integration: ``\int_{\theta \in A} p(\theta)p(\mathcal D|\theta) \mathrm{d}\theta``, which is again generally intractable.


To summarise, the difficulty of Bayesian computation are two-folds
1. the posterior distribution is only evaluable up to some unknown normalising constant;
2. posterior summary involves integrations, which are intractable.


"""

# ╔═╡ 4e93aa8d-00d8-4677-8e06-0032a0634a5a
md"""

## How to estimate ? -- Monte Carlo 

"""

# ╔═╡ e0dcfd8c-2415-4c4f-b254-5f9f52cf8ebf
md"""

Recall that, to summarise a posterior, we need to calculate intractable integrations such as

$$\mathbb{P}(\theta \in A|\mathcal D) = \int_{\theta \in A} p(\theta |\mathcal D) \mathrm{d}\theta = \frac{ \int_{\theta \in A} p(\theta) p(\mathcal D|\theta) \mathrm{d}\theta}{p(\mathcal D)}.$$ 

More generally, we want to calculate expectations of *any functions of random variable* ``\theta`` under the posterior:

"""

# ╔═╡ cda9d867-a6d4-4194-8afd-dbc437637b23
md"""

Note that when ``t(\cdot)`` is a *counting function*, e.g.

$$t(\theta) = \mathbf 1(\theta \in A)=\begin{cases} 1 & \theta \in A \\ 0 & \theta \notin A,\end{cases}$$ we recover the first question as 

$$\mathbb E[t(\theta)|\mathcal D] = \int \mathbf{1}(\theta\in A) p(\theta|\mathcal D) \mathrm{d}\theta = \int_{\theta \in A} 1\cdot p(\theta|\mathcal D) \mathrm{d}\theta = \mathbb P(\theta\in A|\mathcal D).$$ That's why the expectation problem is more "general".


Therefore, the problem we want to tackle is:

> **Problem 1**: How to estimate expectations of functions of ``\theta``, such as equation (1), under the posterior?
"""

# ╔═╡ 29c29fc1-d6c6-4a5e-879b-e24e675a335c
md"""

**Monte Carlo** methods are widely known for their capability to approximate intractable integrations. Suppose we can *sample* from the posterior distribution,

$$\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(R)} \sim p(\theta|\mathcal D),$$

if the sample size, ``R``, is large enough, due to the law of large numbers, we can approximate integration by frequency counting:

$${\mathbb{P}}(\theta \in A|\mathcal D) \approx \frac{\#\{\theta^{(r)} \in A\}}{R},$$ where ``\#\{\cdot\}`` counts how many samples falls in set ``A``.

And general expectations of the form

$$\mathbb E[t(\theta)|\mathcal D] = \int t(\theta) p(\theta|\mathcal D) \mathrm{d}\theta = \frac{\int t(\theta) p(\theta) p(\mathcal D|\theta) \mathrm{d}\theta}{p(\mathcal D)}$$ 

can also be approximated by its Monte Carlo estimation

$${\mathbb{E}}[t(\theta)|\mathcal D] \approx \hat t =\frac{1}{R} \sum_{r} t(\theta^{(r)}),$$

which is the sample average evaluated at the drawn samples. 


"""

# ╔═╡ ddf162c5-8425-439f-a7a5-b6735f6849d1
md"""


### Visual intuition

The following diagram illustrates the idea of Monte Carlo approximations. The true posterior distribution ``p(\theta|\mathcal D)`` is plotted on the left; and the histogram of ``R=2000`` random samples of the posterior is plotted on the right. Monte Carlo method essentially uses the *histogram* on the right to replace the true posterior. For example, to calculate the mean or expectation of ``\theta``, i.e. ``t(\theta) = \theta``, the Monte Carlo estimator becomes 

$$\mathbb E[\theta|\mathcal D] \approx \frac{1}{R} \sum_r \theta^{(r)},$$ the sample average of the samples (which is very close to the ground truth on the left).
"""

# ╔═╡ 27e25739-b2fd-4cee-8df9-089cfbce4321
begin
	approx_samples = 2000
	dist = MixtureModel([Normal(2, sqrt(2)), Normal(9, sqrt(19))], [0.3, 0.7])
	
	Random.seed!(100)
	x = rand(dist, approx_samples)
	ids = (x .< 15) .& (x .> 0)
	prob_mc=sum(ids)/approx_samples
end;

# ╔═╡ 306bb2c5-6c7e-40f7-b3af-738e6fb7613e
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

# ╔═╡ 80b7c755-fe82-43d9-a2f1-e37edaaab25a
md"""

Similarly, calculating probability, such as ``\mathbb P(0\leq \theta \leq 15)``, reduces to frequency counting: 

$$
\hat{\mathbb{P}}(0\leq \theta \leq \theta) = \frac{\#\{0\leq \theta^{(r)}\leq 15\}}{2000} =0.905,$$  counting the proportion of samples that falls in the area of interest. The idea is illustrated in the following diagram.
"""

# ╔═╡ 62743b4f-acc5-4370-8586-627e45a5c9ed
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

# ╔═╡ 3df816ac-f2d0-46a7-afe0-d128a1f185b1
md"""

### Monte Carlo method is scalable
The most important property of Monte Carlo method is its **scalability**, which makes it a practical solution to **Problem 1** even when ``\theta \in R^D`` is of high dimension.

!!! note "Monte Carlo method is scalable"
	The accuracy of the Monte Carlo estimate does not depend on the dimensionality of the space sampled, ``D``.  Roughly speaking, regardless of the dimensionality of ``\theta``,  the accuracy (squared error from the mean) remains the same. 

"""

# ╔═╡ 8892b42f-20c8-4dd3-be84-635d7b5f07fe
md"""
## How to sample ? -- MCMC



"""

# ╔═╡ 5e4551b3-1cd9-40f1-a2b7-ece9fbc7c082
md"""

In the previous section, we have established that Monte Carlo estimation is a scalable method *if we can obtain samples from a posterior distribution*. That is a big "*if*" to assume. Without a practical sampling method, no Monte Carlo estimators can be calculated.

We should also note that for a general Bayesian inference problem the posterior can only be evaluated up to some unknown constant:

$$p(\theta|\mathcal D) \propto p(\theta) p(\mathcal D|\theta),$$

where the scalar ``1/p(\mathcal D)`` involves a nasty integration which we do not know how to compute. 

The question we face now is a sample generation problem:

> **Problem 2**: how to generate samples ``\{\theta^{(r)}\}_{r=1}^R`` from a un-normalised distribution:
> $$p(\theta|\mathcal D) \propto p(\theta)p(\mathcal D|\theta)?$$

"""

# ╔═╡ a54dc923-ea9e-4016-8980-56f21c3d3ca6
md"""

*Note.* *If we apply log on the posterior distribution, the log density becomes a sum of log prior and log-likelihood:
$$\ln p(\theta|\mathcal D) = \ln p(\theta) + \ln p(\mathcal D|\theta),$$ which is faster and numerically stable for computers to compute. Additions are in general faster than multiplications/divisions for floating number operations.*
"""

# ╔═╡ 16e59a89-daa8-467b-82b7-e4058c99edb8
md"""

### Basic idea

A class of methods called **Markov Chain Monte Carlo** (MCMC) is a popular and successful candidate to generate samples from a non-standard distribution. Markov Chain Monte Carlo (MCMC) algorithm is formed by two concepts: 
  * **Markov Chain**: $p(\theta^{(r)}|\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(r-1)}) = p(\theta^{(r)}|\theta^{(r-1)})$
    * the current state only depends on the previous state given the whole history
    * samples are generated by some carefully crafted Markov Chains
      * current sample depends on the previous sample
  * **Monte Carlo**: estimation by the Markov chain samples as illustrated in the previous section


The idea is to produce posterior samples ``\{\theta^{(r)}\}_{r=1}^R`` **in sequence**, each one depending only on ``\theta^{(r-1)}`` and not on its more distant history of predecessors, i.e. a **Markov Chain** (which accounts for the first **MC** of the acronym **MCMC**). When the transition probability of the chain satisfies certain conditions, Markov Chain theory then says that, under quite general conditions, the empirical distribution of the simulated samples will approach the desired target distribution as we simulate the chain long enough, i.e. when ``R`` is large.

Since the sequence converges to the target distribution when ``R`` is large, we usually only retain the last chunk of the sequence as "good samples" or equivalently **discard** the initial samples as **burn-in** since they are usually not representative of the distribution to be approximated. For example, if the Markov Chain has been simulated by 4,000 steps, we only keep the last 2000 to form an empirical distribution of the target.
"""

# ╔═╡ bc071d70-6eae-4416-9fa6-af275bd74f0b
md"""
### Metropolis Hastings

"""

# ╔═╡ 60e9fc94-a3e6-4d0a-805c-6759ee94dcae
md"""

*Metropolis-Hastings* (MH) algorithm is one of the MCMC methods (and arguably the most important one). MH constructs a Markov Chain in two simple steps: 
* it *proposes* a candidate ``\theta_{\text{proposed}}`` based on the current state ``\theta_{\text{current}}`` according to a proposal distribution: ``q(\theta_{\text{proposed}}|\theta_{\text{current}})``
* it then either moves to ``\theta'`` (or *accept* the proposal) or stays put (or *reject*) based on the proposal's "quality" which involves a ratio: 
  
  $\frac{p(\theta_{\text{proposed}}|\mathcal D)}{p(\theta_{\text{current}}|\mathcal D)}^\ast$
  * intuitively, if the proposal state is good, or the ratio is well above 1, we move to ``\theta_{\text{proposed}}``, otherwise stay put 
  * ``\ast`` *note the acceptance probability ratio also involves another ratio of the proposal distributions (check the algorithm below for details)*

A key observation to note here is MH's operations do no involve the normalising constant:

$$\frac{p(\theta_{\text{proposed}}|\mathcal D)}{p(\theta_{\text{current}}|\mathcal D)} = \frac{\frac{p(\theta_{\text{proposed}})p(\mathcal D|\theta_{\text{proposed}})}{\cancel{p(\mathcal D)}}}{\frac{p(\theta_{\text{current}})p(\mathcal D|\theta_{\text{current}})}{\cancel{p(\mathcal D)}}} = \frac{p(\theta_{\text{proposed}})p(\mathcal D|\theta_{\text{proposed}})}{p(\theta_{\text{current}})p(\mathcal D|\theta_{\text{current}})} \triangleq \frac{p^\ast(\theta_{\text{proposed}})}{p^\ast(\theta_{\text{current}})}.$$

Therefore, we only need to evaluate the un-normalised posterior distribution ``p^\ast``.



The algorithm details are summarised below.
"""

# ╔═╡ 8ddc5751-be76-40dd-a029-cf5f87cdb09d
md"""

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

# ╔═╡ 918f4482-716d-42ff-a8d1-46168e9b920a
md"""*Remark*: *when the proposal distribution is symmetric, i.e.*

$$q(\theta^{(t)}|\theta')= q(\theta'|\theta^{(t)}),$$ *the acceptance probablity reduces to*

$$a = \text{min}\left \{\frac{p^\ast(\theta')}{p^\ast(\theta^{(t)}) }, 1\right \}$$ *and the algorithm is called **Metropolis** algorthm.* *A common symmetric proposal distribution is random-walk Gaussian, i.e.* ``q(\theta^{(t)}|\theta')= \mathcal N(\theta', \Sigma),`` *where ``\Sigma`` is some fixed variance matrix.*

"""

# ╔═╡ 00081011-582b-4cbe-aba8-c94c47d96c87
md"""

!!! danger "Question: sample a biased 🎲 by Metropolis Hasting"
	We want to obtain samples from a completely *biased* die that lands with threes (⚂) all the time. If one uses the Metropolis-Hastings algorithm [^1] to generate samples from the biased die and a fair die is used as the proposal of the MH algorithm. 
	* Is the proposal symmetric?
	* What would the MCMC chains look like? 
	* Will the chain ever converge to the target distribution? And do we need to discard any as burn-in?    
"""

# ╔═╡ a072f6a9-472d-40f2-bd96-4a09292dade8
md"""
!!! hint "Answer"
	Depending on the initial state, the chain can either be ``3,3,3,\ldots`` or 
	``x,x,\ldots,x,3,3,3,\ldots`` where ``x \in \{1,2,4,5,6\}``.

	The proposal is symmetric. Since a fair die is used as the proposal, the proposal probability distribution is 1/6 for all 6 facets. 

	As a result, the acceptance probability ``a`` only depends on the ratio of the target distribution. There are two possible scenarios:
	
	1. the chain starts at 3, then all following proposals other than 3 will be rejected (as ``a=\tfrac{0}{1}=0\%``). Therefore, only samples of 3 will be produced.

	2. the chain is initialised with a state other than 3, e.g. ``x=1``, then all proposals other than 3 will be rejected [^2], so the initial samples will be all ones (``1,1,1,\ldots``); until a three is proposed, then it will be accepted with ``a=\frac{1}{0}= 100\%`` chance; and only three will be produced onwards (the same argument of case 1).

	Regardless of the starting state, the chain will eventually converge and produce the correct samples, i.e. ``3,3,3,3,\ldots``. For chains starting with a state other than 3, the initial chunk (all ones in the previous case) should be discarded as **burn-in**: the chain has not yet converged. 

	Luckily, the discard initial chunk will be short. On average, we should expect the burn-in lasts for 6 iterations only, as the length is a geometric random variable with ``p=1/6``.
"""

# ╔═╡ 716920e3-2c02-4da9-aa6b-8a51ba638dfa
md"""

#### Demonstration
"""

# ╔═╡ 24efefa9-c4e3-487f-a228-571fec271886
md"""

To demonstrate the idea, consider sampling from a bivariate Gaussian distribution:

$$\begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix} \sim \mathcal N\left (\begin{bmatrix} 0 \\ 0 \end{bmatrix} , \begin{bmatrix} 1 & \rho  \\ \rho & 1 \end{bmatrix}\right ).$$

The Gaussian distribution has a mean of zero and variance of 1; ``\rho`` measures the correlation between the two dimensions. Here we use ``\rho = 0.9``. The probability density can be written as 

$$p^\ast(\boldsymbol \theta) \propto \text{exp}\left (-\frac{1}{2} \boldsymbol  \theta^\top \mathbf  \Sigma^{-1} \boldsymbol  \theta\right ),$$ where
``\mathbf \Sigma = \begin{bmatrix} 1 & 0.9  \\ 0.9 & 1 \end{bmatrix}.`` The target distribution's contour plot is shown below. 
"""

# ╔═╡ f0f07e50-45ee-4907-8ca8-50d5aaeeafb4
begin
	ρ = 0.9
	μ = [0, 0]
	Σ = [1.0 ρ; ρ 1]
	target_p = (x) -> logpdf(MvNormal(μ, Σ), x)
	plt_contour = plot(-5:0.1:5, -5:0.1:5, (x, y)-> exp(target_p([x,y])),legend = :none, st=:contour, linewidth=0.5, la=0.5, levels=5, xlabel=L"\theta_1", ylabel=L"\theta_2", title="Contour plot a highly correlated bivariate Gaussian")
end

# ╔═╡ 44c167b9-375d-4051-a4c1-825e5ec9570c
md"""

To apply an MH algorithm, we adopt a simple uncorrelated bivariate Gaussian as the proposal distribution:

$$q(\theta'|\theta^{(t)}) = \mathcal N\left (\theta^{(t)},  \sigma^2_q \times \begin{bmatrix} 1 & 0  \\ 0 &  1\end{bmatrix}\right )$$

* the proposal distribution proposes a Gaussian random walk conditional on the current state ``\theta^{(t)}``.
* ``\sigma^2_q > 0`` is a tuning parameter to set.
* note that since the proposal distribution is symmetric, i.e.

  $$q(\theta'|\theta^{(t)}) = q(\theta^{(t)}|\theta')$$ 
  the acceptance probability simplifies to

$$a = \text{min}\left \{\frac{p^\ast(\theta')}{p^\ast(\theta^{(t)})}, 1\right \}$$

The Metropolis-Hastings algorithm with the mentioned proposal distribution can be implemented in Julia as follows:

```julia
# Metropolis Hastings with isotropic Gaussian proposal
- `ℓπ`: the target log probability density
- `mc`: number of Monte Carlo samples to draw 
- `dim`: the dimension of the target space
- `Σ`: proposal's covariance matrix
- `x0`: initial starting state
function metropolis_hastings(ℓπ, mc=500; dim=2, Σ = 10. * Matrix(I, dim, dim), x0=zeros(dim))
	samples = zeros(dim, mc)
	samples[:, 1] = x0
	L = cholesky(Σ).L
	ℓx0 = ℓπ(x0) 
	accepted = 0
	for i in 2:mc
		# radomly draw from the proposal distribution 
		# faster than xstar = rand(MvNormal(x0, Σ))
		xstar = x0 + L * randn(dim)
		ℓxstar = ℓπ(xstar)
		# calculate the acceptance ratio `a`
		# note ℓπ is in log scale, need to apply exp
		a = exp(ℓxstar - ℓx0) 
		# if accepted
		if rand() < a
			x0 = xstar
			ℓx0 = ℓxstar
			accepted += 1
		end
		# store the sample 
		@inbounds samples[:, i] = x0
	end
	accpt_rate = accepted / (mc-1)
	return samples, accpt_rate
end

```

To use the algorithm draw samples from the target Gaussian distribution, we simply feed in the required inputs, e.g.

```julia
	# set the target posterior distribution to sample
	target_ℓπ = (x) -> logpdf(MvNormal(μ, Σ), x)
	# set proposal distribution's variance
	σ²q = 1.0
	mh_samples, rate=metropolis_hastings(target_p, 2000; Σ = σ²q * Matrix(I, 2, 2), x0=[-2.5, 2.5])
```
"""

# ╔═╡ 5de392a4-63d7-4313-9cbd-8eaeb4b08eea
md"""

An animation is listed below to demonstrate the MH algorithm with a proposal distribution ``\sigma^2_q = 1.0``, where
* the ``{\color{red}\text{red dots}}`` dots are the rejected proposals
* the ``{\color{green}\text{green dots}}`` dots are the MCMC samples (either accepted new proposals or stay-put old samples)
"""

# ╔═╡ 5b3f3b8a-1cfa-4b32-9a20-1e1232549e78
md"""

It can be observed that 
* the acceptance rate is about 0.32
* and the chain is **mixing well**, which means the high-density area of the target distribution has been explored well. 


2000 samples drawn from the MH algorithm after 2000 burn-in are shown below.


"""

# ╔═╡ 922a89e6-a1a0-4f07-b815-552a9b2a4fbd
md"""

### Reduce dependence of the chain
"""

# ╔═╡ eba93b03-23d4-4ccd-88d2-dcea51bb20b9
md"""

Ideally, the traditional Monte Carlo method requires samples to be *independent*. MCMC samples, however, are simulated as Markov chains: i.e. the next sample depends on the current state, therefore MCMC samples are all dependent. 


It is worth noting that Monte Carlo estimation is still valid even when the samples are dependent as long as the chain reaches equilibrium. However, an estimation's standard error, in general, deteriorates when the samples in use are more dependent. 

There are two commonly used practices to reduce the temporal correlations of an MCMC chain: *thinning* and *parallelism*. And they can be used together to further improve the sample quality.
"""

# ╔═╡ d3a88626-0d60-45ad-9725-9ed23853fc85
md"""

#### Thinning
"""

# ╔═╡ 76a0e827-927c-4a48-a4c9-98c401357211
md"""
As the dependence decreases as the lag increases, one therefore can retain MCMC samples at some fixed lag. For example, after convergence, save every 10th sample. 
"""

# ╔═╡ f56bb3ef-657d-4eff-8778-11d550e6d803
md"""
#### Parallel chains

"""

# ╔═╡ ae766de8-2d2a-4dd3-8127-4ca69a6082f1
md"""

Each MCMC chain simulates independently, to fully make use of modern computers' concurrent processing capabilities, we can run several chains in parallel. 
"""

# ╔═╡ 2628cd3b-d2dc-40b9-a0c2-1e6b79fb736b
md"""

*Why should we run parallel chains rather than a single long chain?* Note that ideally, we want the samples to be *independent*. Samples from one single chain is temporally dependent. Running multiple chains exploring the posterior landscape independently, when mixing together, produces more independent samples.  
"""

# ╔═╡ 5a7a300a-e2e0-4c99-9f71-077b43602fdd
md"""
The following animation shows the evolution of four parallel Metropolis-Hastings chains. To make the visualisation cleaner, rejected proposals are not shown in the animation. The four chains start at four initial locations (i.e. the four corners). Note that all four chains quickly move to the high-density regions regardless of their starting locations, which is a key property of MCMC.

"""

# ╔═╡ 99153a03-8954-48c6-8396-1c2b669e4ea6
md"""

### Limitations of Metropolis-Hastings
"""

# ╔═╡ 0802dc90-8312-4509-82f0-6eca735e852b
md"""
MH algorithm's performance depends on the proposal's quality. And setting a suitable proposal distribution for an MH algorithm is not easy, especially when the target distribution is high-dimensional. As a rule of thumb, we should aim at an acceptance rate between 0.2 to 0.6. Too high or too low implies the chain is struggling. 

**Acceptance rate too high** 

Usually happens when ``\sigma^2_q`` is too small, proposals are all close to each other in one high-density area ``\Rightarrow`` most of the proposals are accepted, but, as a result ``\Rightarrow`` other high-density areas are not explored sufficiently enough.


The following animation illustrates the idea: the proposal's variance is set as ``\sigma_q^2 = 0.005``; despite a high acceptance rate of 0.924, the chain only makes tiny moves and therefore does not explore the target distribution well enough.

"""

# ╔═╡ a00b9476-af30-4609-8b1e-4693246fdaef
md"""

**Acceptance rate too low** 

Usually happens when ``\sigma^2_q`` is too large, the proposals jump further but likely to propose less desired locations ``\Rightarrow`` a lot of rejections ``\Rightarrow`` very temporal correlated samples.

The chain shown below employs a more audacious proposal distribution with ``\sigma_q^2 = 10.0``. As a result, most of the proposals are rejected which leads to an inefficient sampler (most of the samples are the same).
"""

# ╔═╡ dd15f969-733e-491c-a1b7-e0fbf4532e64
md"""

To avoid the difficulty of setting good proposal distributions, more advanced variants of MH algorithms, such as the **Hamiltonian sampler** (HMC), **No-U-Turn sampler** (NUTS), have been proposed. Instead of randomly exploring the target distribution, HMC and its variants employ the gradient information of the target distribution to help explore the landscape.
"""

# ╔═╡ 85cad369-df16-4f06-96ae-de5f9c5bb6cd
md"""
## Digress: Julia's `MCMCChains.jl`

"""

# ╔═╡ 0cda2d90-a2fa-41ab-a655-c7e4550e4eb1
md"""

Julia has a wide range of packages to analyse MCMC chains. In particular, `MCMCChains.jl` package provides us tools to visualise and summarise MCMC chains. 

We can construct `Chains` object using `MCMCChains.jl` by passing a matrix of raw chain values along with optional parameter names:
"""

# ╔═╡ bdf1a4ca-a9ef-449d-82a3-07dd9db1f1ba
md"""

The initial 3 samples of the raw samples:

```julia
mh_samples_parallel[1:3, :, :]
```

"""

# ╔═╡ 3d7c71ce-6532-4cd8-a4a3-5ef2ded70caa
md"""
Create a `Chains` object by passing the raw sample matrix and names of the parameters (optional)

```julia
# use MCMCChains.jl to create a Chains object 
# the first 2000 samples are discarded as burn-in
# Note that to apply thinning=2, one can use ``mh_samples_parallel[2001:2:end, :, :]``
chain_mh = Chains(mh_samples_parallel[2001:end, :, :], [:θ₁, :θ₂])
```

"""

# ╔═╡ 358719cd-d903-45ab-bba6-7fe91697d1ee
md"""

### Visualisations
One can visualise `Chains` objects by  
* `traceplot(chain_mh)`: plots the chain samples at each iteration of the Markov Chain, i.e. time series plot
* `density(chain_mh)`: plots kernel density estimations fit on the samples
* or `plot(chain_mh)`: plots both trace and density plots

"""

# ╔═╡ 2c18acb0-9bc5-4336-a10d-727538dbd3c8
md"""
For example, the following plots show the four parallel MH chains for the bi-variate Gaussian example

* the left two plots show the trace plots of four chains
* the density of the right show fit to the four chains

It can be seen visually that all four chains converge to roughly the same equilibrium distributions.
"""

# ╔═╡ 55ed9e58-7feb-4dbc-b807-05796a02fc62
md"""
Other visualisation methods include

* `meanplot(c::Chains)`: running average of the chain
* `histogram(c::Chains)`: construct histogram of the chain
* `autocorplot(c::Chains)`: auto correlations of the chain
* `corner(c::Chains)`: make a scatter plot of the chains; a tool to view correlations between dimensions of the samples
"""

# ╔═╡ 2c06c3f1-11e6-4191-b505-080342a9b787
md"For example, the following plot uses `corner()` to show a scatter plot of the chain."

# ╔═╡ 17a4d60b-7181-4268-866c-ddbde37dc349
md"""
### Summary statistics
"""

# ╔═╡ 45eb8c7b-5f17-43ac-b412-0f3ced44a018
md"""
`MCMCChains.jl` also provides easy-to-use interfaces to tabulate important statistics of MCMC chains instances
* `summarystats`: aggregates statistics such as mean, standard deviations, and essential sample size etc 
* `describe`: apart from above, it shows 2.5% to 97.5% percentile of the samples


```julia
# summary of the chain
summarystats(chain_mh)
describe(chain_mh)
```


"""

# ╔═╡ e0e4f50c-52c0-4261-abae-3cf0399e04e0
md"""
# MCMC diagonistics
"""

# ╔═╡ 81ce6e4c-ef48-4a07-9133-4414867c2b29
md"""



Markov chain theory only provides us with a *theoretical guarantee*:

> if one simulates an MCMC chain *long enough*, *eventually* the sample empirical distribution will *converge* to the target posterior distribution. 

Unfortunately, this is only a *theoretical guarantee*. We do not have the time and resources to run a chain indefinitely long. In practice, we simulate chains at fixed lengths and inpect the chains to check **convergence**.





### MCMC metrics

Luckily, there are multiple easy-to-compute metrics to diagnose a chain. Most of the techniques apply stationary time series models to access the performance of a chain. The most commonly used metrics are **$\hat R$ statistic**, and **effective sample size** (`ess`).

**R̂ statistic (`rhat`)**  

R̂ is a metric that measures the stability of a chain. Upon convergence, any chunk of a chain, e.g. the first and last halves of the chain, or parallel chains, should be *similar to each other*. ``\hat R`` statistic, which is defined as the ratio of within-chain to between-chain variability, should be close to one if all chains have converged. 


> As a rule of thumb, a valid converging chain's ``\hat R`` statistic should be less than 1.01. 

"""

# ╔═╡ 57c26917-8300-45aa-82e7-4fd5d5925eba
md"""
**Effective sample size (`ess`)**

The Traditional Monte Carlo method requires samples to be independent. However, MCMC samples are all dependent, as the chain is simulated in a Markovian fashion: the next sample depends on the previous state. It is worth noting that Monte Carlo estimation is still valid even when the samples are dependent as long as the chain has mixed well. However, the standard error of the estimation, in general, deteriorates when the samples are more dependent. 

**`ess`** roughly estimates how many equivalent *independent* samples are in a dependent chain. Since samples in an MCMC chain are correlated, they contain less information than truly independent samples. `ess` therefore discounts the sample size by some factor that depends on the temporal correlation between the iterations.

We can also measure the *efficiency* of an MCMC algorithm by calculating 

$$\text{efficiency} = \frac{\text{ess}}{\text{iterations}},$$

which measures the information content in a chain. One needs to run a highly efficient algorithm with fewer iterations to achieve the same result.

> Larger effective sample size (ess) implies the MCMC algorithm is more efficient
"""

# ╔═╡ 97f81b83-50ff-47c0-8c2d-df95816b2ac3
md"""

**Examples** To demonstrate the ideas, we compare `ess` and `R̂` of the three MH chains with different proposal variances: ``\sigma^2_q=`` 1.0, .005 and 20. Remember, ``\sigma^2_q=1.0`` performs the best; .005 and 20 are less optimal.
"""

# ╔═╡ d2ae1f4f-e324-42f6-8eaa-c8e32ec3fc39
md"""
**Summary statistics with MH chain of ``\sigma^2_q=1.0``**
"""

# ╔═╡ 5487dd11-7c8e-428f-a727-300780dd02a7
md"**Summary statistics with MH chain of ``\sigma^2_q=.005``**"

# ╔═╡ 3a92e430-8fba-4d69-aa25-13e08a485720
md"**Summary statistics with MH chain of ``\sigma^2_q=20.0``**"

# ╔═╡ e38c1d3d-fb36-4793-94fc-665dc0eacd99
md"""
It can be observed that 
* `ess`: `ess` are around 360, 20, and 134 respectively with the three algorithms, which implies efficiencies of 0.18, 0.01, and 0.067. The second algorithm with small proposal variance, although achieving a high acceptance rate, is the least efficient.

* `rhat`: both the second and third algorithm's `rhat` > 1.01, which indicates they do not mix well.
"""

# ╔═╡ 2998ed6a-b505-4cfb-b5da-bdb3f887a646
md"""

### Visual inspection
"""

# ╔═╡ 95e71f9b-ed81-4dc1-adf9-bd4fe1fc8bbe
md"""

Simple trace plots reveal a lot of information about a chain. One can diagnose a chain by visual inspection. 

**What do bad chains look like?** The following two trace-plots show the chain traces of the two less desired MH algorithms for the bivariate Gaussian example: one with ``\sigma^2_q=0.005`` and ``\sigma^2_q=20``.

Recall that when ``\sigma^2_q`` is too small, a chain proposes small changes at each iteration. As a result, the chain does not explore HPD region sufficiently well. If one splits the chain into two halves, the two chunks (with green and red backgrounds) exhibit drastic different values. 
"""

# ╔═╡ 82dc987c-4649-4e65-83e7-e337be0e99e8
md"""
On the other hand, when ``\sigma^2_q`` is too large, the chain jumps back and force with large steps, resulting most proposals being rejected and old samples being retained. The chain becomes very *sticky*. The chain does not contain a lot of information comparing with independent samples.

"""

# ╔═╡ ac904acb-4438-45d9-8795-8ab724240da0
md"""

**What good chains should look like ?**
The figure below shows trace plots of four different chains with `ess=` 4.6, 49.7, 90.9, and 155. The top two are *bad* chains. And the bottom two chains are of relatively better quality. A good chain should show stable statistical properties across the iterations with a reasonable level of variance.
"""

# ╔═╡ 68c98e53-7ac3-4832-a7dd-97459a89d7cb
md"""
# Other MCMC samplers
"""

# ╔═╡ 5f1f12b2-5b63-416e-a459-9b5d0e37b0e8
md"""

Metropolis-Hastings is considered the mother of most modern MCMC algorithms. There are a lot of other variants of MCMC algorithms that can be considered as specific cases of MH algorithm. We will quickly introduce two other popular variants: Gibbs sampling and Hamiltonian sampler. We will focus on the intuition behind the samplers.

"""

# ╔═╡ b29597f1-3fd7-4b44-9097-7c1dc1b7629b
md"""
## Gibbs sampling

Gibbs sampling reduces a multivariate sampling problem to a series of uni-variate samplings. Assume the target distribution is bivariate with density ``p(\theta_1, \theta_2)``, and the chain is currently at state ``(\theta_1^{(r)}, \theta_2^{(r)})``, Gibbs sampling alternatively samples from their full conditionals in two steps:

* sample ``\theta_1^{(r+1)} \sim p(\theta_1|\theta_2^{(r)})``
* sample ``\theta_2^{(r+1)} \sim p(\theta_2|\theta_1^{(r+1)})``

The new sample ``(\theta_1^{(r+1)}, \theta_2^{(r+1)})`` is retained as a new sample.
"""

# ╔═╡ 6c83a79a-8581-45cb-8d31-31185dc42a0f
md"""

### Visual intuition

The following diagram demonstrates how Gibbs sampling explores the bivariate Gaussian distribution. Note the zig-zag behaviour: at each step, Gibbs sampling only changes one dimension of the sample.
"""

# ╔═╡ 9a281767-3d88-4742-8efc-e4fe764c705a
md"""

**Multivariate Gibbs sampling** The general Gibbs sampling algorithm for a multi-variate problem is listed below. The idea is the same, at each step, a series of small steps based on the full conditionals are made and chained together.

"""

# ╔═╡ 022b7068-de1b-4506-abf1-2289976f1597
md"""

!!! infor "Gibbs sampling"
	0. Initialise ``\boldsymbol \theta^{(0)}=[\theta_1^{(0)},\theta_2^{(0)}, \ldots, \theta_D^{(0)} ]`` arbitrary
	1. For ``r = 1,2,\ldots``:
	   * sample dimension ``1``: 
	   $$\theta_1^{(r)} \sim p(\theta_1|\theta_2^{(r-1)}, \ldots, \theta_D^{(r-1)})$$
	   * sample dimension ``2``: 
	   $$\theta_2^{(r)} \sim p(\theta_2|\theta_1^{(r)}, \theta_3^{(r-1)}, \ldots, \theta_D^{(r-1)})$$
	   $$\vdots$$
	   * sample dimension ``D``: 
	   $$\theta_D^{(r)} \sim p(\theta_D|\theta_1^{(r)}, \ldots, \theta_{D-1}^{(r)})$$
"""

# ╔═╡ 16be37d7-f6d5-469b-b0fd-20816b42d4e5
md"""

**Gibbs sampling is a Metropolis-Hastings**

One drawback of MH sampler is one needs to specify a proposal distribution. Gibbs sampling alleviates this burden from the user **by using the full conditionals of the target distribution as proposal distribution**. Gibbs sampling, therefore, is a specific case of MH algorithm. One can also show that when full conditionals are used, the acceptance probability is always 100%. Therefore, there is no rejection step in a Gibbs sampling.


"""

# ╔═╡ d6d332c5-769c-4fe1-8f84-fd2bbf19250a
md"""
We have already shown that a high acceptance rate does not necessarily imply high efficiency. The same applies to Gibbs sampling. The zig-zag exploration scheme of Gibbs sampling can be slow for highly correlated sample space. For example, in our bi-variate Gaussian example, Gibbs sampling has achieved around `ess = 129` (out of 2000 samples) and an efficiency around $(129/2000).
"""

# ╔═╡ 39edf0df-a9c5-4e76-af01-f730b4a619b9
md"**Summary statistics of Gibbs sampling**"

# ╔═╡ dd8095b0-8e42-4f6d-a545-8ac1db07ff79
md"""
## Hamiltonian sampler

"""

# ╔═╡ c387be0b-3683-48d6-9041-d8f987428499
md"""
We have discussed the limitation of the Metropolis-Hastings algorithm: the proposal distribution is hard to tune. Ideally, we want a proposal distribution with the following properties:
* propose the next state as far, therefore, *independent*, from the current state as possible
* at the same time, the proposed state should be of *good* quality, 
  * ideally as good (of high probability density) as the current one, leading to an acceptance rate ``a`` close to one


Simple proposals, such as Gaussians, cannot achieve both of the desired properties. The problem lies in their random-walk exploration nature. The proposals blindly propose the next steps and ignore the local geographical information of the target distribution. *Gradients*, which always point to the steepest ascent direction, are the *geographic* information that can guide the proposal to reach a further yet high probability region. 
"""

# ╔═╡ df08ac1d-bcd7-4cb7-bcd1-68f5de699aa0
md"""
### Visual intuition

"""

# ╔═╡ b15f101a-b593-4992-ae80-f32dc894c773
md"""
To understand the idea, let's consider a more complicated target distribution which is formed by super-imposing three bivariate Gaussians together. The three Gaussians are with means ``[-4,0], [4,0],`` and ``[0,0]``, and the variances ``\begin{bmatrix} 2 & 1.5 \\ 1.5& 2\end{bmatrix}``, ``\begin{bmatrix} 2 & -1.5 \\ -1.5 & 2\end{bmatrix}`` and ``\begin{bmatrix} 2 & 0 \\ 0 & 2\end{bmatrix}`` respectively. 

The contour and surface of the targe distribution is plotted below for reference. 
"""

# ╔═╡ 2fc46172-cd45-48b2-bff2-9dd5a91e21d1
begin
	d1 = MvNormal([-4, 0], [2 1.5; 1.5 2])
	d2 = MvNormal([4, 0], [2 -1.5; -1.5 2])
	d3 = MvNormal([0, 0], [2 0; 0. 2])
	d = MixtureModel([d1, d2, d3])
end;

# ╔═╡ bcca401f-b64d-45f5-b8c0-766cc2a1a50e
begin
	ℓ(x) = logpdf(d, x)
	∇ℓ(x) = ForwardDiff.gradient(ℓ, x)
end;

# ╔═╡ d3c70aee-c9ce-44f7-9b02-ca674f8c5f01
begin
	con_p = contour(-9:0.1:9, -6:0.1:6, (x,y) -> pdf(d, [x,y]), legend=false, ratio=1, xlim =[-9,9], ylim=[-6,6])
	surf_p = surface(-9:0.1:9, -6:0.1:6, (x,y) -> pdf(d, [x,y]), legend=false)
	Plots.plot(con_p, surf_p, layout=(1,2), size=(780,400))
end

# ╔═╡ d20cf19f-70fc-47dd-a031-22079bbd10b9
md"""

A vector field view of the gradient of the target density is shown below. Note that at each location, a gradient (a blue vector) always points to the steepest ascent direction. The gradient therefore provides key geographical information about the landscape.  Hamiltonian sampler proposes the next state by making use of the gradient information. 

"""

# ╔═╡ 1fbb02a8-299a-406c-a337-a74c5ad444e1
begin
	meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	∇ℓ_(x, y) = ∇ℓ([x, y])/5
	contour(-9:0.1:9, -6:0.1:6, (x,y) -> pdf(d, [x,y]), legend=false, title="Contour plot and gradient plot of the target density")
  		xs,ys = meshgrid(range(-9, stop=9, length=15), range(-6,stop=6,length=11))
	quiver!(xs, ys, quiver=∇ℓ_, c=:blue)
end

# ╔═╡ 89a6cb78-b3e2-4780-9961-679a73967f5e
md"""
#### A useful physics metaphor
A very useful metaphor to understand the Hamiltonian sampler is to **imagine a hockey pluck sliding over a surface of the negative target distribution without friction**. The surface is formed by the negative target density (i.e. flip the surface plot, and intuitively a mountain becomes a valley).

"""

# ╔═╡ bad85a4b-6a14-445a-8af6-ac5f5ebaf183
surface(-9:0.1:9, -6:0.1:6, (x,y) -> -pdf(d, [x,y]), legend=false, title="Negative of the target density")

# ╔═╡ ba2545d4-cc39-4069-b6a4-a60f15911cec
md"""
At each iteration, a random initial force (usually drawn from a Gaussian) is applied to the pluck, the pluck then slides in the landscape to reach the next state. 

A numerical algorithm called *Leapfrog* integration is usually used to simulate the movement of the pluck. In particular, the trajectory is simulated by a sequence of ``T`` steps with length ``\epsilon``. The two tuning parameters are:
* ``\epsilon``: each *Leapfrog*'s step length
* `T`: total number of steps taken to form the whole trajectory
Therefore, the total length of the pluck's movement is proportional to ``\epsilon \times T``.



The animation below demonstrates the idea, where 10 independent HMC proposals are simulated from an initial starting location at ``[0, 2.5]`` by the *Leapfrog* algorithm:
* 10 different random forces of different directions and strengths were applied to the pluck
* the plank follows the law of physics to explore the landscape (note how the plucks moves around the curvature of the valley)

"""

# ╔═╡ d820035a-9d5c-4ec0-89cd-7e112d42eb8e
md"""
It is important to observe that 

* each proposal is far away from the initial starting location 
* and the quality of the ending proposals is good compared with the starting point. 
"""

# ╔═╡ d037451e-9ece-4f20-963f-b58eaa022c0a
md"""We have only illustrated the intuition behind HMC method. Keen readers should read references such as [^3] for further details about Hamiltonian MC methods."""

# ╔═╡ 39e5d60e-dedf-4107-936f-b80158c62e4d
md"""

### A comparison between the algorithms

"""

# ╔═╡ 6e41b059-7460-4b6d-ba90-dba6acb30c18
md"""
In this section, we are going to compare the original Metropolis-Hastings (MH) with the Hamiltonian Monte Carlo sampler (HMC). The mixture density model is reused here. 

#### Animation check
Firstly, we visually inspect the algorithms' sampling process. Animations of MH and HMC algorithms are presented below to help us to gain some feeling about the algorithms. The MH algorithm has used a proposal variance ``\sigma_q^2 = 0.1``; and the HMC is simulated with the following tuning parameters
* ``\epsilon=0.1``: each Hamiltonian's step size of the proposal; the larger ``\epsilon`` is, the proposed trajectory is further
* `Trange = [10, 50]`: the number of steps to take for each proposal, a random number between 10 and 50; 
"""

# ╔═╡ de52ebac-9d24-4ad8-9686-90a11008ae26
md"""
It can be observed that HMC performs significantly better than the original MH sampler
* note that MH has only managed to explore the left two modes in the first 200 iterations.
* HMC has a much higher acceptance rate
* yet it explores the landscape better (and it does not suffer the usual drawback of the high acceptance rate of the MH sampler); 
#### Chain inspection

We can use `Julia`'s `MCMCChains.jl` to carry out standard chain diagnosis. 

**`traceplot` visualisation**

We first inspect the chains visually by using `traceplot`. The MH's trace plots show high temporal correlations between the samples (which implies the chain has not yet converged) while HMC's traces seem to mix well. 
"""

# ╔═╡ 27f14c5e-e229-44fa-a252-a8efc9d9dc4a
md"**MH sampler's trace plot:** note the high temporal correlations between the iterations"

# ╔═╡ e51beb02-0807-4316-b330-c6578c9fc315
md"**HMC sampler's trace plot**: both dimensions have mixed well"

# ╔═╡ 5a44aecd-449b-4269-a435-c6a6bae3cffd
md"""

**`ess` and efficiency**

Efficiency metrics can also computed and compared between the two algorithms. For 2000 iterations (after the first 2000 discarded as burn-in), HMC produces around 1183 independent samples while there are only less than 17 effective sample contained in the original MH sample. HMC is therefore 60 fold more efficient than the ordinaly MH algorithm.
"""

# ╔═╡ 2f2c94fd-a2f2-45f8-a284-78d73721f623
md"""

# Conclusion

In this section, we have introduced MCMC, a class practical computational method for Bayesian inference. MCMC aims at generating Monte Carlo samples from a target distribution, where the unknown normalising constant is not required. All inference questions can then be calculated by Monte Carlo estimation, which is scalable even if the parameter space is of high dimensions. 


However, sampling from a general high-dimensional distribution *efficiently* is not trivial. Traditional MCMC algorithms, such as Metropolis-Hastings and Gibbs sampling, suffer from random walk behaviour. More advanced algorithms, such as Hamiltonian sampler which employs gradient information of the target distribution, usually perform better.

Implementing MCMC algorithms by oneself clearly is not idea. Luckily, as an end user, one rarely needs to implement a MCMC algorithm. In the next chapter, we will see how probabilistic programming software such as `Turing.jl` simplifies the process. One only usually needs to specify a Bayesian model and leave the MCMC sampling task to the software. Nevertheless, the user still needs to be able to do practical convergence diagnoistics to check the quality of the sample, which arguably is the most important takeaway knowledge for an applied Bayesian inference user. 

Some important concepts/terminologies about MCMC are summarised below:
* **MCMC**: aims at generating samples from a non-normalised distribution; the **Markov chain** samples can then be used for **Monte Carlo** estimation
* **warm-up** (or **burn-in**): refers to the initial chunk of MCMC samples that are not representative of the target distribution; the burn-in samples should be discarded
* **mix well**: the chain has explored the target distribution well and reached equilibrium
* **`ess`**: effective sample size; how many independent samples are contained in a dependent chain; the larger the better
* **`R` statistic**: a statistic to diagnose a chain; as a rule of thumb, a good chain should have an `R` statistic less than 1.01




"""

# ╔═╡ d56be408-3517-45a4-88e8-56e194ce33f0
md"""

## Notes

[^1]: This is a contrived question to illustrate the idea of MH algorithm. We do not need to run any algorithm to generate samples from a deterministic die. The sample should be all threes.

[^2]: It depends on how ``\tfrac{0}{0}`` is defined. We have assumed it is zero or `NaN` here. 

[^3]: [Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of markov chain monte carlo 2.11 (2011): 2.](https://arxiv.org/pdf/1206.1901.pdf)  
"""

# ╔═╡ 6a9863cb-3067-4250-ad89-6a1c8dc1fddc
md"""
# Appendix
"""

# ╔═╡ ca2662a9-2754-45c1-9ce8-5b8599eef240
md"**Juia code used for this chapter**"

# ╔═╡ 9375f800-7c4b-4cae-b263-f17198b04011
md"""
## `demo_mh_gif`
"""

# ╔═╡ af617b31-396f-465e-b27e-2fa14b3b2423
begin
"""
	produce_anim(ℓπ, chain, proposed, acpt, rate; mc = 200, xlim=[-4, 4], ylim = [-4, 4], title="MH demonstartion; ")

Produce an animation that demonstrates how Metropolis-Hastings-liked algorithm works

### Input

- `ℓπ` -- log pdf of the target distribution
- `chain` -- the mcmc trace, should be a `dim` × `mc+1` matrix (including initial state)
- `proposed` -- proposal history
- `acpt` -- acceptance history, array of booleans
- `rate` -- acceptance rate
- `mc`  -- number of iterations to show
- `xlim, ylim` -- horizontal and vertical limits of the plot

### Output

- `anim` -- an array of figures 

### Examples

```julia
ℓπ = (x) -> logpdf(MvNormal([0,0], [1.0 0.9; 0.9 1.0]), x)
chain, proposed, acpt, rate = metropolis_hastings(ℓπ, 201)
anim = produce_anim(ℓπ, chain, proposed, acpt, rate)
gif(anim, fps= 10)
```
"""
	function produce_anim(ℓπ, chain, proposed, acpt, rate; mc = 200, xlim=[-4, 4], ylim = [-4, 4], title="MH demonstartion; ", with_accpt_rate=true)
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

# ╔═╡ 18ebc039-1656-4f74-8e9f-f03a8d39d7c4
md"""
## `metropolis_hastings`
"""

# ╔═╡ 88696ab1-2866-46f1-978e-bd032566cef7
begin

"""
	metropolis_hastings(ℓπ, mc=500; dim=2, Σ = 10. * Matrix(I, dim, dim), x0=zeros(dim))

Sample a target probability distribution by Metropolis-Hastings algorithm with random walk Gaussian proposal

### Input

- `ℓπ` -- log pdf of the target distribution
- `mc`   -- number of MC samples to simulate
- `dim`  -- dimension of the target distribution
- `Σ` -- proposal distribution's variance, should be `dim` × `dim` symmetric and P.D. matrix
- `x0` -- the initial starting point

### Output

- `samples` -- the `mc` iterations of samples, a `dim` × `mc` array
- `proposed` -- the history of the proposals, only for visualisation purpose
- `acceptance` -- whether the proposals being accepted or not
- `accpt_rate` -- acceptance rate of the chain

### Examples

```julia
ℓπ = (x) -> logpdf(MvNormal([0,0], [1.0 0.9; 0.9 1.0]), x)
metropolis_hastings(ℓπ, 500; Σ = 1.0 * Matrix(I, 2, 2))
```
"""
	function metropolis_hastings(ℓπ, mc=500; dim=2, Σ = 10. * Matrix(I, dim, dim), x0=zeros(dim))
		samples = zeros(dim, mc)
		proposed = zeros(dim, mc-1)
		acceptance = Array{Bool, 1}(undef, mc-1)
		fill!(acceptance, false)
		samples[:, 1] = x0
		L = cholesky(Σ).L
		ℓx0 = ℓπ(x0) 
		accepted = 0
		for i in 2:mc
			# xstar = rand(MvNormal(x0, Σ))
			xstar = x0 + L * randn(dim)
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

end

# ╔═╡ d8fad6e3-9adf-4ae9-9260-885f21d07fa9
begin
	Random.seed!(100)
	σ²q = 1.0
	mh_samples = metropolis_hastings(target_p, 4000; Σ = σ²q * Matrix(I,2,2), x0=[-2, 2])[1]
end;

# ╔═╡ c3169cf8-6b7c-418f-8052-4fd242a07592
begin
	plt = covellipse(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-4, 4), ylims=(-4, 4),
    alpha=0.3,
    c=:steelblue,
    label="90% HPD",
    xlabel=L"\theta_1", ylabel=L"\theta_2")

	scatter!(plt, mh_samples[1, 2001:end], mh_samples[2, 2001:end], label="MH samples", mc=:red, ma=0.3)

end

# ╔═╡ a253c403-111f-4940-b866-c1b5233f18d0
begin
	Random.seed!(100)
	# simulate 4000 MCMC steps for each chain
	mc_iters = 4000
	# pre-allocate storage for the chains
	mh_samples_parallel = zeros(mc_iters, 2, 4)
	# four starting locations
	x0s = [[-1, -1], [-1, 1], [1,1], [1, -1]] .* 3.5
	# simulate 4 chains in parallel
	Threads.@threads for t in 1:4
		mh_samples_parallel[:, :, t] = metropolis_hastings(target_p, mc_iters; Σ = σ²q * Matrix(I, 2, 2), x0=x0s[t])[1]'
	end
end

# ╔═╡ 99decdfe-6bd8-40af-a42d-f1639c98b323
let
	# create a plot of the high probability density for the target distribution
	plt_anim = covellipse(μ, Σ,
    n_std=1.64, # 5% - 95% quantiles
    xlims=(-4, 4), ylims=(-4, 4),
    alpha=0.3,
    c=:steelblue,
	legend = :outerright,
    label="90% HPD",
    xlabel=L"\theta_1", ylabel=L"\theta_2", title="Animation of 4 Parallel MH Chains")

	labels = "chain " .* string.(1:4)
	# create the animation for the first 100 steps
	mh_anim_parallel = @animate for i  in 1:99
		for ch in 1:4
			scatter!(plt_anim, (mh_samples_parallel[i, 1, ch], mh_samples_parallel[i, 2, ch]),
	             label= i == 1 ? labels[ch] : false, mc=ch, ma=0.5)
			plot!(plt_anim, mh_samples_parallel[i:i+1, 1, ch], mh_samples_parallel[i:i+1, 2, ch], st=:path, lc=ch, la=0.5, label=false)
		end
		
	end
	# show the animation at 10 frames per second
	gif(mh_anim_parallel, fps= 10)
end

# ╔═╡ a022c310-cc66-4377-b168-8dcef141afa6
mh_samples_parallel[1:3, :,:]

# ╔═╡ 52b29fcc-c534-4bba-866a-e7622c5a5e11
begin
	chain_mh = Chains(mh_samples_parallel[2001:end, :, :], [:θ₁, :θ₂])
	plot(chain_mh)
end

# ╔═╡ cca21050-aa12-4bb7-bcb1-918d94aa0bec
begin
	# for cleaner visualisation, plot with 1000 samples
	corner(sample(chain_mh,1000), size=(400,400))
end

# ╔═╡ be3c585b-425b-4995-b2bc-ae7bb33d6bad
summarystats(chain_mh)

# ╔═╡ 2d7ac989-25c9-4023-ba75-5cad9b43f44a
describe(chain_mh)

# ╔═╡ 3157973e-f1a1-4ac3-88c7-9c073dfc849c
summarystats(chain_mh)

# ╔═╡ 730f7e65-d7c9-4506-94fe-f99e244a1c74
begin
	# simulate two chains with two different proposal variances
	Random.seed!(100)
	σ²q_too_small = 0.005
	σ²q_too_big = 20.0
	mh_samples_too_small = zeros(mc_iters, 2, 4)
	mh_samples_too_big = zeros(mc_iters, 2, 4)
	mh_samples_ideal = zeros(mc_iters, 2, 4)
	Threads.@threads for t in 1:4
		mh_samples_too_small[:, :, t] = metropolis_hastings(target_p, mc_iters; Σ = σ²q_too_small * Matrix(I, 2, 2), x0=x0s[t])[1]'

		mh_samples_too_big[:, :, t] = metropolis_hastings(target_p, mc_iters; Σ = σ²q_too_big * Matrix(I, 2, 2), x0=x0s[t])[1]'

		mh_samples_ideal[:, :, t] = metropolis_hastings(target_p, mc_iters; Σ = Σ, x0=x0s[t])[1]'
	end
	chain_too_small = Chains(mh_samples_too_small[2001:end, :, :], [:θ₁, :θ₂])
	chain_too_big = Chains(mh_samples_too_big[2001:end, :, :], [:θ₁, :θ₂])
	chain_ideal = Chains(mh_samples_ideal[2001:end, :, :], [:θ₁, :θ₂])
end;

# ╔═╡ 7d4706f8-6938-4241-8bbb-f1779119422c
summarystats(chain_too_small)

# ╔═╡ 83926597-ac50-4f7b-9e6d-84bad3ac129f
summarystats(chain_too_big)

# ╔═╡ 09244ee2-a153-4fc5-bd64-ee752c132743
begin
plot(1:1000, Array(chain_too_small[1:1000,1:1, 1:1]), title="A bad chain; proposal variance " * L"\sigma^2_q = 0.005", label="", color=1, xlabel="Iteration", ylabel="Sample value", size=(600,300))
vspan!([0,1000], color = :green, alpha = 0.2, labels = "");
plot!(1001:2000, Array(chain_too_small[1001:end, 1:1, 1:1]),  label="", color=1)
vspan!([1001,2000], color = :red, alpha = 0.2, labels = "");
end

# ╔═╡ ae1ae203-95f0-4d49-ba9f-14c334af087b
plot(Array(chain_too_big[:,1:1, 1:1]), title="A bad chain; proposal variance "*L"\sigma^2_q = 20", xlabel="Iteration", ylabel="Sample value", label="", size=(600,300))

# ╔═╡ 96cecfcf-75ac-4327-90e1-ec1ac5b3f39c
let
	ess = zeros(4)
	ess[1] = MCMCChains.ess_rhat(chain_too_small[:,1:1,1:1])[:,:ess]
	ess[2] = MCMCChains.ess_rhat(chain_too_big[:,1:1,1:1])[:,:ess]
	ess[3] = MCMCChains.ess_rhat(chain_mh[:,1:1,1:1])[:,:ess]
	ess[4] = MCMCChains.ess_rhat(chain_ideal[:,1:1,1:1])[:,:ess]
	ess= round.(ess; digits=1)
	p1 = plot(chain_too_small[:, 1, 1], title="ess=$(ess[1])", label="")
	p2 = plot(chain_too_big[:,1,1], title="ess=$(ess[2])", label="")
	p3 = plot(chain_mh[:,1,1], title="ess=$(ess[3])", label="")
	p4 = plot(chain_ideal[1:end, 1, 1], title="ess=$(ess[4])", label="")
	Plots.plot(p1, p2, p3, p4)
end

# ╔═╡ 280069b8-f9fe-4cd6-866d-a20b9eab036e
begin
"""
	demo_mh_gif(ℓπ;)

Produce a gif that demonstrates how Metropolis-Hastings algorithm works

### Input

- `ℓπ` -- log pdf of the target distribution
- `qΣ` -- proposal distribution's variance, should be `dim` × `dim` symmetric and P.D. matrix
- `mc`  -- number of iterations to simulate
- `x₀` -- the initial starting point
- `xlim, ylim` -- horizontal and vertical limits of the plot

### Output

- `anim` -- an array of figures 

### Examples

```julia
ℓπ = (x) -> logpdf(MvNormal([0,0], [1.0 0.9; 0.9 1.0]), x)
anim = demo_mh_gif(ℓπ; qΣ = 0.005 * Matrix(I,2,2))
gif(anim, fps= 10)
```
"""
	function demo_mh_gif(ℓπ; qΣ = 1.0 * Matrix(I,2,2), mc = 200, x₀= zeros(2), xlim=[-4, 4], ylim = [-4, 4])
		Random.seed!(100)
		chain, proposed, acpt, rate = metropolis_hastings(ℓπ, mc+1; Σ = qΣ, x0 = x₀)
		anim = produce_anim(ℓπ, chain, proposed, acpt, rate; mc = mc, xlim=xlim, ylim = ylim)
		return anim
	end


	
end;

# ╔═╡ cf968e52-7518-40b0-af57-1f552c41dc07
begin
	anim_σ_1 = demo_mh_gif(target_p; qΣ = 1.0 * Matrix(I,2,2), x₀ = [-2, 2])
	gif(anim_σ_1, fps=5)
end

# ╔═╡ 5a6c25f0-e7d2-469d-bc19-54617e417b6e
begin
	anim_σ_001 = demo_mh_gif(target_p; qΣ = 0.005 * Matrix(I,2,2), x₀ = [-2, -2], mc = 500)
	gif(anim_σ_001, fps=5)
end

# ╔═╡ 5d704270-3a0f-4241-b622-ab8135d6b545
begin
	anim_σ_15 = demo_mh_gif(target_p; qΣ = 10 * Matrix(I,2,2), x₀ = [-2, -2], mc = 500)
	gif(anim_σ_15, fps=5)
end

# ╔═╡ fd0d20a0-5732-4afd-b2d7-b013eb624834
md"""
## `gibbs_sampling`
"""

# ╔═╡ 74deaa30-4c5c-4e40-99bf-912bd8611374
begin

"""
	gibbs_sampling(mc=500; μ= zeros(2), ρ=0.9, σ₁² = 1.0, σ₂²=1.0, x0=zeros(2))

Sample a bivariate Gaussian with mean ``\\mu`` and variance-covariance ``\\begin{bmatrix} \\sigma_1^2 & \\rho \\\\ \\rho & \\sigma_2^2 \\end{bmatrix}`` by Gibbs sampling. 


### Note
This only works for a bivariate Gaussian target distribution; it is not a general sampling algorithm

### Examples
gibbs\\_xs = gibbs\\_sampling(4000)

chain\\_gibbs = Chains(gibbs_xs')
"""
function gibbs_sampling(mc=500; μ= zeros(2), ρ=0.9, σ₁² = 1.0, σ₂²=1.0, x0=zeros(2))
	samples = zeros(2, mc)
	x₁, x₂ = x0[1], x0[2]
	σ₁, σ₂ = sqrt(σ₁²), sqrt(σ₂²) 
	k₁ = ρ * σ₂ / σ₁
	k₂ = ρ * σ₁ / σ₂
	σ₁₂ = σ₂ * sqrt(1 - ρ^2)
	σ₂₁ = σ₁ * sqrt(1 - ρ^2)
	@inbounds samples[:, 1] = x0
	for i in 2:mc
		if i % 2 == 0
			x₁ = rand(Normal(μ[1] + k₁ * (x₂ - μ[2]), σ₁₂))
		else
			x₂ = rand(Normal(μ[2] + k₂ * (x₁ - μ[1]), σ₂₁))	
		end
		@inbounds samples[:, i] = [x₁, x₂]
	end
	return samples
end
end

# ╔═╡ 1016fe42-c0f8-4823-9ef4-09fd310eaf34
begin
	Random.seed!(100)
	# run Gibbs sampling  for the bivariate Gaussian example
	# check the appendix for the details of the implementation
	# 4000 iterations in total and starting location at x0
	samples_gibbs = gibbs_sampling(4000; x0=[-2.5, 2.5])

end;

# ╔═╡ 77b70944-9771-4eee-b0fc-d842bed3b504
let
	plt_gibbs = covellipse(μ, Σ,
	    n_std=1.64, # 5% - 95% quantiles
	    xlims=(-4, 4), ylims=(-4, 4),
	    alpha=0.3,
	    c=:steelblue,
	    label="90% HPD",
	    xlabel=L"\theta_1", ylabel=L"\theta_2")
	# create the animation
	gibbs_anim = @animate for i in 1:200
	    scatter!(plt_gibbs, (samples_gibbs[1, i], samples_gibbs[2, i]),
	             label=false, mc=:red, ma=0.5)
	    plot!(samples_gibbs[1, i:i + 1], samples_gibbs[2, i:i + 1], seriestype=:path,
	          lc=:green, la=0.5, label=false)
	end
	gif(gibbs_anim, fps=10)
end

# ╔═╡ 77fe324c-bb47-4aed-9d10-27add827239d
begin
	gibbs_chain = Chains(samples_gibbs[:, 2001:end]', [:θ₁, :θ₂])
	summarystats(gibbs_chain)
end

# ╔═╡ 60afbdbf-6e83-44e2-99ed-2d64e339b0d9
md"""

## `hmc_sampling`
"""

# ╔═╡ 55cd420e-461c-4a1b-a82c-a2e04be87d74
begin

"""
	hmc_sampling(ℓπ, ∇ℓπ, mc, x₀; ϵ, Τrange =[100,200])

Sample a target probability distribution by Hamiltonian Monte Carlo sampler (HMC)

### Input

- `ℓπ` -- log pdf of the target distribution
- `∇ℓ`   -- gradient of the log pdf
- `mc`  -- the number of samples to draw
- `x₀` -- the initial starting point
- `ϵ` -- the Leepfrog's step size
- `Trange` -- the min and max Leep Frog steps

### Output

- `samples` -- the `mc` iterations of samples, a `dim` × `mc` array
- `accpt_rate` -- acceptance rate of the HMC

### Examples

```julia
ℓπ = (x) -> logpdf(MvNormal([0,0], [1.0 0.9; 0.9 1.0]), x)
∇ℓπ = (x) -> ForwardDiff.gradient(ℓπ, x)
hmc_sampling(ℓπ, ∇ℓπ, 1000, randn(2);  ϵ= 0.05, Τrange = [100,200])
```
"""
	function hmc_sampling(ℓ, ∇ℓ, mc, x₀; ϵ, Τrange =[100,200])
		dim = length(x₀)
		samples = zeros(dim, mc)
		proposed = zeros(dim, mc-1)
		acceptance = Array{Bool, 1}(undef, mc-1)
		fill!(acceptance, false)
		samples[:, 1] = x₀
		g₀ = -1 * ∇ℓ(x₀)
		E₀ = -1 * ℓ(x₀)
		n_accept = 0 
		for i in 2:mc
			# initial momentum is Normal(0, 1)
			p = randn(dim)
			# evaluate old H(x₀, p) = E(x₀) + K(p)
			# where E(x₀) = - ℓ(x₀)
			H = (p' * p)/2 + E₀

			xnew, gnew = x₀, g₀
			# make Τ leapfrog simulation steps
			Τ = rand(Τrange[1]:Τrange[2])
			# optional
			ϵ = rand([-1, 1]) * ϵ
			for τ in 1:Τ
				# make half-step in p
				p = p - ϵ * gnew /2
				# make step in x
				xnew = xnew + ϵ * p
				# find new gradient
				gnew = -1 * ∇ℓ(xnew)
				# make half-step in p
				p = p - ϵ * gnew /2
			end
			#  find new value of E and then H
			proposed[:, i-1] = xnew 
			Enew = -1 * ℓ(xnew)
			Hnew = (p' * p)/2 + Enew
			dH = Hnew - H

			if dH < 0 
				accept = true
			elseif rand() < exp(-dH)
				accept = true
			else 
				accept = false
			end
		
			if accept
				x₀, g₀, E₀ = xnew, gnew, Enew
				n_accept = n_accept + 1
			end
			samples[:, i] = x₀
			acceptance[i-1] = accept
		end
		return samples, proposed, acceptance, n_accept/(mc-1)
	end
end

# ╔═╡ 42eec440-ef63-46db-b7bb-5f0c2c5046b6
begin
	Random.seed!(100)
	spl_mh_mix, _, _, _ = metropolis_hastings(ℓ, 4000; Σ = 0.1 * Matrix(I, 2, 2))
	chain_mh_mix = Chains(spl_mh_mix[:, 2001:end]', [:θ₁, :θ₂])
	spl_hmc_mix, _, _, _ = hmc_sampling(ℓ, ∇ℓ, 4000, [0, 0]; ϵ=0.1, Τrange =[10,50])
	chain_hmc_mix = Chains(spl_hmc_mix[:, 2001:end]', [:θ₁, :θ₂])
end;

# ╔═╡ d6805966-deee-4db1-b9c4-66820e05ef91
traceplot(chain_mh_mix)

# ╔═╡ 160f5345-a517-4b52-972c-bafdf4972835
traceplot(chain_hmc_mix)

# ╔═╡ ad463d15-aa9b-4c43-8732-d23a7a920f5f
begin
	ess_stats = [mean(summarystats(chain_mh_mix)[:, :ess]), mean(summarystats(chain_hmc_mix)[:, :ess])]
	efficienty_stats = ess_stats/2000
	df = DataFrame(methods = ["MH", "HMC"], ess=ess_stats, efficiency=efficienty_stats)
end

# ╔═╡ 0138dedf-f874-4bbd-bf87-7f6bbe8ca816
begin
"""
	demo_hmc_gif(ℓπ, ∇ℓπ; ϵ = 0.05, mc = 200, x₀= zeros(2), Trange=[100,200], xlim=[-4, 4], ylim = [-4, 4])

Produce a gif that demonstrates how Metropolis-Hastings algorithm works

### Input

- `ℓπ` -- log pdf of the target distribution
- `∇ℓπ` -- gradient of the log pdf of the target distribution
- `ϵ` -- step size of the HMC's Leap Frog simulation
- `mc`  -- number of iterations to simulate
- `x₀` -- the initial starting point
- `Trange` -- a range of possible steps of the Leap Frog algorithm
- `xlim, ylim` -- horizontal and vertical limits of the plot

### Output

- `anim` -- an array of figures 

### Examples

```julia
ℓπ = (x) -> logpdf(MvNormal([0,0], [1.0 0.9; 0.9 1.0]), x)
∇ℓπ = (x) -> ForwardDiff.gradient(ℓπ, x)
anim = demo_hmc_gif(ℓπ, ∇ℓπ)
gif(anim, fps= 10)
```
"""
	function demo_hmc_gif(ℓπ, ∇ℓπ; ϵ = 0.05, mc = 200, x₀= zeros(2), Trange=[100,200], xlim=[-4, 4], ylim = [-4, 4])
		Random.seed!(100)
		chain, proposed, acpt, rate = hmc_sampling(ℓπ, ∇ℓπ, mc+1, x₀; ϵ = ϵ, Τrange =Trange)
		anim = produce_anim(ℓπ, chain, proposed, acpt, rate; mc = mc, xlim=xlim, ylim = ylim, title="HMC demonstration", with_accpt_rate=false)
		return anim
	end


	
end;

# ╔═╡ 3bf64dd1-c44e-4499-9c15-3c4725554ef1
begin
	anim_mh_mixture = demo_mh_gif(ℓ;xlim= [-9,9], ylim=[-6,6], qΣ= 0.1* Matrix(I,2,2))
	anim_hmc_mixture = demo_hmc_gif(ℓ, ∇ℓ; xlim= [-9,9], ylim=[-6,6], ϵ= 0.1, Trange=[10, 50])
end;

# ╔═╡ 53298cd1-af19-4611-9088-c9dd0cb082ef
gif(anim_mh_mixture; fps= 10 )

# ╔═╡ ce95c7c4-f10e-4a67-9f65-884a772ec37b
gif(anim_hmc_mixture; fps=10)

# ╔═╡ 42cfa97c-6298-47f1-a17a-8a5ff642e570
begin

	function hamiltonian_proposal_step(ℓπ, ∇ℓπ, x₀, ϵ, T)
		dim = length(x₀)
		g₀ = -1 * ∇ℓπ(x₀)
		E₀ = -1 * ℓπ(x₀)
		# initial momentum is Normal(0, 1)
		p = randn(dim)
		# evaluate old H(x₀, p) = E(x₀) + K(p)
		# where E(x₀) = - ℓπ(x₀)
		H = (p' * p)/2 + E₀
		xnew, ∇new = x₀, g₀
		xs = zeros(dim, T+1)
		xs[:, 1] = xnew
		# make Τ leapfrog simulation steps
		# Τ = rand(Τrange[1]:Τrange[2])
		# optional
		# ϵ = rand([-1, 1]) * ϵ
		for τ in 1:T
			# make half-step in p
			p = p - ϵ * ∇new /2
			# make a step in x based on the speed p
			xnew = xnew + ϵ * p
			# find new gradient
			∇new = -1 * ∇ℓπ(xnew)
			# make half-step in p
			p = p - ϵ * ∇new /2
			xs[:, τ+1] = xnew 
		end
		return xs
	end


end

# ╔═╡ 56f98135-da40-4ccd-a93d-8ac3e01aee5d
begin
	T_step = 50
	traj_n = 10
	x0 = [0, 2.5]
	hmc_trajs = zeros(2, T_step+1, traj_n) 
	Random.seed!(189)
	for t in 1:traj_n	
		hmc_trajs[:, :, t] = hamiltonian_proposal_step(ℓ, ∇ℓ, x0, 0.1, T_step)
	end
end

# ╔═╡ 858cbf82-14a5-4605-a9ba-61e1084c2c8c
let
	hmc_anim_plt = plot(-9:0.1:9, -6:0.1:6, (x,y) -> -pdf(d, [x,y]), st=:contour, legend=false, title="HMC's proposal animation", xlabel=L"\theta_1", ylabel=L"\theta_2")

	scatter!(hmc_anim_plt, [x0[1]], [x0[2]], color= traj_n+1)
	ts = 1:traj_n
	hmc_anim = @animate for i in 1:2:T_step
		for t in ts 
			plot!(hmc_trajs[1, i:i + 1, t], hmc_trajs[2, i:i + 1, t], seriestype=:path,
					  lc=t, la=1.0, lw=3.0, label="")
		end
	end
	gif(hmc_anim, fps=2)
end

# ╔═╡ 331710c1-0fc7-4228-b26c-1d170c505205
let
	hmc_plt = plot(-9:0.1:9, -6:0.1:6, (x,y) -> -pdf(d, [x,y]), st=:contour, legend=false, title="HMC's proposals end snapshot", xlabel=L"\theta_1", ylabel=L"\theta_2")
	
	scatter!([x0[1]], [x0[2]], color= traj_n+1)
	for t in 1:traj_n
		for i in 1:2:T_step
			plot!(hmc_trajs[1, i:i + 1, t], hmc_trajs[2, i:i + 1, t], seriestype=:path,
					  lc=t, la=1.0, lw=3.0, label="")
		end
		# scatter!([hmc_trajs[1, end, t]], [hmc_trajs[2, end, t]], color=t, label="HMC $(t)")
		x = [x0[1], hmc_trajs[1, end, t]]
		y = [x0[2], hmc_trajs[2, end, t]]
		plot!(x,y, marker=:circle, arrow=true, arrowsize=1, lw=2, lc=t, mc=t)
	end
	
	# plot!(x,y, marker=:circle, arrow=true, arrowsize=0.5)
	hmc_plt
end

# ╔═╡ 84c51030-d87a-4269-89b3-3b43aac03c61
md"""
## Others
"""

# ╔═╡ 8f1eadcb-71f2-4265-9ee1-7a8f5f252bc7
md"""
**Code for referencing equations**
"""

# ╔═╡ 9cf68cd6-1175-4660-8918-58bb50b4ecf3
js(x) = HypertextLiteral.JavaScript(x)

# ╔═╡ 16611a44-fc1d-4fa6-8bfd-4788d57432d4
"""
`texeq(code::String)`
Take an input string and renders it inside an equation environemnt (numbered) using KaTeX
Equations can be given labels by adding `"\\\\label{name}"` inside the `code` string and subsequently referenced in other cells using `eqref("name")`
### Note
Unfortunately backward slashes have to be doubled when creating the TeX code to be put inside the equation
When Pluto will support interpretation of string literal macros, this could be made into a macro
"""
function texeq(code,env="equation")
	code_escaped = code 			|>
	x -> replace(x,"\\" => "\\\\")	|>
	x -> replace(x,"\n" => " ")
	println(code_escaped)
	@htl """
	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css" integrity="sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc" crossorigin="anonymous">
	<script src="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.js" integrity="sha384-YNHdsYkH6gMx9y3mRkmcJ2mFUjTd0qNQQvY9VYZgQd7DcN7env35GzlmFaZ23JGp" crossorigin="anonymous"></script>
	
	<script>
	katex.render('\\\\begin{$(js(env))} $(js(code_escaped)) \\\\end{$(js(env))}',currentScript.parentElement,{
					displayMode: true,
					trust: context => [
						'\\\\htmlId', 
						'\\\\href'
					].includes(context.command),
					macros: {
					  "\\\\label": "\\\\htmlId{#1}{}"
					},
				})
	</script>
	"""
end

# ╔═╡ c1de6474-f057-448c-8214-b5f57358e3c4
texeq("\\mathbb E[t(\\theta)|\\mathcal D] = \\int t(\\theta) p(\\theta|\\mathcal D) \\mathrm{d}\\theta = \\frac{\\int t(\\theta) p(\\theta) p(\\mathcal D|\\theta) \\mathrm{d}\\theta}{p(\\mathcal D)} \\label{exp}")

# ╔═╡ 94d47c8a-0f84-4c67-853b-f7e4a71cc90f
"""
`eqref(label::String)`
Function that create an hyperlink pointing to a previously defined labelled equation using `texeq()`
"""
eqref(label) = @htl """
<a eq_id="$label" id="eqref_$label" href="#$label" class="eq_href">(?)</a>
"""

# ╔═╡ 8cf9b88b-e17c-445a-9ebe-449d45c9c3cc
@htl """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css" integrity="sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc" crossorigin="anonymous">
<style>
a.eq_href {
	text-decoration: none;
}
</style>
		<script src="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.js" integrity="sha384-YNHdsYkH6gMx9y3mRkmcJ2mFUjTd0qNQQvY9VYZgQd7DcN7env35GzlmFaZ23JGp" crossorigin="anonymous"></script>
<script id="katex-eqnum-script">
const a_vec = [] // This will hold the list of a tags with custom click, used for cleaning listeners up upon invalidation
const eqrefClick = (e) => {
	e.preventDefault() // This prevent normal scrolling to link
	const a = e.target
	const eq_id = a.getAttribute('eq_id')
	window.location.hash = 'eqref-' + eq_id // This is to be able to use the back function to resume previous view, 'eqref-' is added in front to avoid the viewport actually going to the equation without having control of the scroll
	const eq = document.getElementById(eq_id)
	eq.scrollIntoView({
		behavior: 'smooth',
		block: 'center',
	})
}
const checkCounter = (item, i) => {
	return item.classList.contains('enclosing')	?
	i											:
	i + 1
}
const updateCallback = () => {
a_vec.splice(0,a_vec.length) // Reset the array
const eqs = document.querySelectorAll('span.enclosing, span.eqn-num')
let i = 0;
eqs.forEach(item => {
	i = checkCounter(item,i)
	console.log('item',i,'=',item)
	if (item.classList.contains('enclosing')) {
		const id = item.id
		const a_vals = document.querySelectorAll(`[eq_id=\${id}]`)
		a_vals !== null && a_vals.forEach(a => {
			a_vec.push(a) // Add this to the vector
			a.innerText = `(\${i+1})`
			a.addEventListener('click',eqrefClick)
		})
	}
})
}
const notebook = document.querySelector("pluto-notebook")
// We have a mutationobserver for each cell:
const observers = {
	current: [],
}
const createCellObservers = () => {
	observers.current.forEach((o) => o.disconnect())
	observers.current = Array.from(notebook.querySelectorAll("pluto-cell")).map(el => {
		const o = new MutationObserver(updateCallback)
		o.observe(el, {attributeFilter: ["class"]})
		return o
	})
}
createCellObservers()
// And one for the notebook's child list, which updates our cell observers:
const notebookObserver = new MutationObserver(() => {
	updateCallback()
	createCellObservers()
})
notebookObserver.observe(notebook, {childList: true})
invalidation.then(() => {
	notebookObserver.disconnect()
	observers.current.forEach((o) => o.disconnect())
	a_vec.forEach(a => a.removeEventListener('click',eqrefClick))
})
</script>
"""

# ╔═╡ 27c4c76b-65d8-40cb-89f3-fc42c4a6848e
md"---"

# ╔═╡ 8fd498e7-0690-4947-867a-9e6495a8bcd9
md"""
**Foldable details**
"""

# ╔═╡ f0997ca0-5cfd-4bbe-b094-5fe5a7ea8b65
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

# ╔═╡ 9fca6e17-c8ee-4739-9d2e-0aaf85032a6b
Foldable("Further details about the scalability*", md"For simplicity, we assume ``t`` is a scalar-valued function. Note all expectations here are w.r.t the posterior distribution from which the samples are drawn. Firstly, it can be shown that the Monte Carlo estimator is unbiased: 

$$\mathbb E[\hat t] = \mathbb E\left [\frac{1}{R}\sum_r t(\theta^{(r)})\right ] =\frac{1}{R}\sum_r \mathbb E[t(\theta^{(r)})] = \mathbb E[t(\theta)].$$ It means, on average, the estimator converges to the true integration value. 


To measure the estimator's accuracy, we only need to find the estimator's variance as it measures the average squared error between the estimator and true value:

$$\mathbb V[\hat t] = \mathbb E[(\hat t -t)^2].$$ 

If we assume samples are independent draws from the distribution, the variance is then 

$$\mathbb V[\hat t] =\mathbb V\left [\frac{1}{R}\sum_r t(\theta^{(r)})\right ]= \frac{1}{R^2} \sum_r \mathbb{V}[t(\theta^{(r)})]=\frac{R\cdot \mathbb{V}[t(\theta^{(r)})]}{R^2}=\frac{\sigma^2}{R},$$ where

$$\sigma^2 = \mathbb V[t] = \int p(\theta|\mathcal D) (t(\theta)-\mathbb E[t])^2\mathrm{d}\theta$$ is some positive constant that only depends on the function ``t``. Therefore, as the number of samples ``R`` increases, the variance of ``\hat t``  will decrease linearly (the standard deviation, ``\sqrt{\mathbb V[\hat t]}``,  unfortunately, shrinks at a rate of ``\sqrt{R}``). Note the accuracy (variance) only depends on ``\sigma^2``, the variance of the particular statistic ``t`` rather than ``D``, the dimensionality.")

# ╔═╡ aafd802c-e529-4116-8d58-13f2ba1c4e49
Foldable("Further details on Gibbs sampling is a MH algorithm.", md"For simplicity, we only consider bivariate case. Assume the chain is currently at state ``\boldsymbol \theta^{(r)}=(\theta_1^{(r)}, \theta_2^{(r)})``, the proposal distribution proposes to move to a new state ``\boldsymbol \theta' = (\theta_1', \theta_2')`` with a proposal density

$$q(\boldsymbol \theta'|\boldsymbol \theta^{(r)}) = p(\theta_1'|\theta_2') \cdot \mathbf 1({\theta_2'= \theta_2^{(r)})},$$

where ``\mathbf 1(\cdot)`` returns 1 when the testing condition is true and 0 otherwise. The transition kernel basically states ``\theta_2^{(r)}`` is not changed.


The acceptance probability then is 

$$a \triangleq \frac{p(\boldsymbol \theta') q(\boldsymbol \theta^{(r)}|\boldsymbol \theta')}{p(\boldsymbol \theta^{(r)})q(\boldsymbol \theta'|\boldsymbol \theta^{(r)})} = \frac{p(\theta_1', \theta_2') \times p(\theta_1^{(r)}|\theta_2')\cdot \mathbf 1({\theta_2'= \theta_2^{(r)})}}{p(\theta_1^{(r)}, \theta_2^{(r)})\times p(\theta_1'|\theta_2')\cdot \mathbf 1({\theta_2'= \theta_2^{(r)})}}.$$

When ``\theta_2' = \theta_2^{(r)}``,

$$a = \frac{p(\theta_1', \theta_2^{(r)}) \times p(\theta_1^{(r)}|\theta_2^{(r)})}{p(\theta_1^{(r)}, \theta_2^{(r)})\times p(\theta_1'|\theta_2^{(r)})}=  \frac{p(\theta_1', \theta_2^{(r)}) \times \frac{p(\theta_1^{(r)}, \theta_2^{(r)})}{p(\theta_2^{(r)})}}{p(\theta_1^{(r)}, \theta_2^{(r)})\times \frac{p(\theta_1',\theta_2^{(r)})}{p(\theta_2^{(r)})}} = \frac{p(\theta_2^{(r)})}{p(\theta_2^{(r)})} =1.0,$$

when ``\theta_2'\neq \theta_2^{(r)}``, we have ``a=\tfrac{0}{0}=1``, a trivial case. Therefore, there is no need to test the proposal. The proposed state should also be accepted.
")

# ╔═╡ c9fa60c5-f216-4eb9-b052-b3472e551bdf
Foldable("Julia code for the target distribution and visualisation.", md"By using Julia's `Distributions.jl` and `ForwardDiff.jl`, we can formulate the density and its corresponding gradient function as following:

```julia
d1 = MvNormal([-4, 0], [2 1.5; 1.5 2])
d2 = MvNormal([4, 0], [2 -1.5; -1.5 2])
d3 = MvNormal([0, 0], [2 0; 0. 2])
# Superimpose the three Gaussians together to form a new density
d = MixtureModel([d1, d2, d3])
# log likelihood of the mixture distribution
ℓ(x) = logpdf(d, x)
# gradient of the log pdf
∇ℓ(x) = ForwardDiff.gradient(ℓ, x)
```

The contour and surface of the targe distribution can be plotted easily via

```julia
contour(-9:0.1:9, -6:0.1:6, (x,y) -> pdf(d, [x,y]), legend=false, ratio=1, xlim =[-9,9], ylim=[-6,6])
surface(-9:0.1:9, -6:0.1:6, (x,y) -> pdf(d, [x,y]), legend=false)
```

")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
DataFrames = "~1.3.4"
Distributions = "~0.25.66"
ForwardDiff = "~0.10.30"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
MCMCChains = "~5.3.1"
PlutoUI = "~0.7.39"
StatsPlots = "~0.15.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.1"
manifest_format = "2.0"
project_hash = "276b15c64407c890cf52786c63ecb31bab29d199"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "5c26c7759412ffcaf0dd6e3172e55d783dd7610b"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.1.3"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

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

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

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

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

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
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2f18915445b248731ec5db4e4a17e451020bf21e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.30"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

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

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

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
git-tree-sha1 = "ed47af35905b7cc8f1a522ca684b35a212269bd8"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.2.0"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "57af5939800bce15980bddd2426912c4f83012d8"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.1"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

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
git-tree-sha1 = "0a7ca818440ce8c70ebb5d42ac4ebf3205675f04"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.4"

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

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

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
git-tree-sha1 = "7c88f63f9f0eb5929f15695af9a4d7d3ed278a91"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.16"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8cb9b8fb081afd7728f5de25b9025bff97cb5c7a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.3.1"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "058d08594e91ba1d98dcc3669f9421a76824aa95"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.3"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "16fa7c2e14aa5b3854bc77ab5f1dbe2cdc488903"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.6.0"

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
git-tree-sha1 = "9f4f5a42de3300439cb8300236925670f844a555"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.1"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

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
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "0a0da27969e8b6b2ee67c112dcf7001a659049a0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

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

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

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
git-tree-sha1 = "2690681814016887462cf5ac37102b51cd9ec781"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.2"

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

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "23368a3313d12a2326ad0035f0db0c0966f438ef"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "0005d75f43ff23688914536c5e9d5ac94f8077f7"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.20"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "b422791c9db86c777cf5887d768a8bcbfffc43eb"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.0"

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

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

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

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

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
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─ce716e92-0c3c-11ed-364f-2f8266fa10f2
# ╟─8daf0123-b22e-4cda-8eec-7cad3484c8e0
# ╟─70f5222b-aadf-4dc0-bc03-b1d10e7db8b6
# ╟─46af5470-afc8-47a8-a6c4-07de22105e91
# ╟─f9584f19-9020-481d-93b2-8d52e8c4bdfc
# ╟─4e93aa8d-00d8-4677-8e06-0032a0634a5a
# ╟─e0dcfd8c-2415-4c4f-b254-5f9f52cf8ebf
# ╟─c1de6474-f057-448c-8214-b5f57358e3c4
# ╟─cda9d867-a6d4-4194-8afd-dbc437637b23
# ╟─29c29fc1-d6c6-4a5e-879b-e24e675a335c
# ╟─ddf162c5-8425-439f-a7a5-b6735f6849d1
# ╟─27e25739-b2fd-4cee-8df9-089cfbce4321
# ╟─306bb2c5-6c7e-40f7-b3af-738e6fb7613e
# ╟─80b7c755-fe82-43d9-a2f1-e37edaaab25a
# ╟─62743b4f-acc5-4370-8586-627e45a5c9ed
# ╟─3df816ac-f2d0-46a7-afe0-d128a1f185b1
# ╟─9fca6e17-c8ee-4739-9d2e-0aaf85032a6b
# ╟─8892b42f-20c8-4dd3-be84-635d7b5f07fe
# ╟─5e4551b3-1cd9-40f1-a2b7-ece9fbc7c082
# ╟─a54dc923-ea9e-4016-8980-56f21c3d3ca6
# ╟─16e59a89-daa8-467b-82b7-e4058c99edb8
# ╟─bc071d70-6eae-4416-9fa6-af275bd74f0b
# ╟─60e9fc94-a3e6-4d0a-805c-6759ee94dcae
# ╟─8ddc5751-be76-40dd-a029-cf5f87cdb09d
# ╟─918f4482-716d-42ff-a8d1-46168e9b920a
# ╟─00081011-582b-4cbe-aba8-c94c47d96c87
# ╟─a072f6a9-472d-40f2-bd96-4a09292dade8
# ╟─716920e3-2c02-4da9-aa6b-8a51ba638dfa
# ╟─24efefa9-c4e3-487f-a228-571fec271886
# ╟─f0f07e50-45ee-4907-8ca8-50d5aaeeafb4
# ╟─44c167b9-375d-4051-a4c1-825e5ec9570c
# ╟─5de392a4-63d7-4313-9cbd-8eaeb4b08eea
# ╟─cf968e52-7518-40b0-af57-1f552c41dc07
# ╟─5b3f3b8a-1cfa-4b32-9a20-1e1232549e78
# ╟─d8fad6e3-9adf-4ae9-9260-885f21d07fa9
# ╟─c3169cf8-6b7c-418f-8052-4fd242a07592
# ╟─922a89e6-a1a0-4f07-b815-552a9b2a4fbd
# ╟─eba93b03-23d4-4ccd-88d2-dcea51bb20b9
# ╟─d3a88626-0d60-45ad-9725-9ed23853fc85
# ╟─76a0e827-927c-4a48-a4c9-98c401357211
# ╟─f56bb3ef-657d-4eff-8778-11d550e6d803
# ╟─ae766de8-2d2a-4dd3-8127-4ca69a6082f1
# ╟─2628cd3b-d2dc-40b9-a0c2-1e6b79fb736b
# ╟─5a7a300a-e2e0-4c99-9f71-077b43602fdd
# ╠═a253c403-111f-4940-b866-c1b5233f18d0
# ╟─99decdfe-6bd8-40af-a42d-f1639c98b323
# ╟─99153a03-8954-48c6-8396-1c2b669e4ea6
# ╟─0802dc90-8312-4509-82f0-6eca735e852b
# ╠═5a6c25f0-e7d2-469d-bc19-54617e417b6e
# ╟─a00b9476-af30-4609-8b1e-4693246fdaef
# ╟─5d704270-3a0f-4241-b622-ab8135d6b545
# ╟─dd15f969-733e-491c-a1b7-e0fbf4532e64
# ╟─85cad369-df16-4f06-96ae-de5f9c5bb6cd
# ╟─0cda2d90-a2fa-41ab-a655-c7e4550e4eb1
# ╟─bdf1a4ca-a9ef-449d-82a3-07dd9db1f1ba
# ╟─a022c310-cc66-4377-b168-8dcef141afa6
# ╟─3d7c71ce-6532-4cd8-a4a3-5ef2ded70caa
# ╟─358719cd-d903-45ab-bba6-7fe91697d1ee
# ╟─2c18acb0-9bc5-4336-a10d-727538dbd3c8
# ╟─52b29fcc-c534-4bba-866a-e7622c5a5e11
# ╟─55ed9e58-7feb-4dbc-b807-05796a02fc62
# ╟─2c06c3f1-11e6-4191-b505-080342a9b787
# ╟─cca21050-aa12-4bb7-bcb1-918d94aa0bec
# ╟─17a4d60b-7181-4268-866c-ddbde37dc349
# ╟─45eb8c7b-5f17-43ac-b412-0f3ced44a018
# ╠═be3c585b-425b-4995-b2bc-ae7bb33d6bad
# ╠═2d7ac989-25c9-4023-ba75-5cad9b43f44a
# ╟─e0e4f50c-52c0-4261-abae-3cf0399e04e0
# ╟─81ce6e4c-ef48-4a07-9133-4414867c2b29
# ╟─57c26917-8300-45aa-82e7-4fd5d5925eba
# ╟─97f81b83-50ff-47c0-8c2d-df95816b2ac3
# ╟─d2ae1f4f-e324-42f6-8eaa-c8e32ec3fc39
# ╠═3157973e-f1a1-4ac3-88c7-9c073dfc849c
# ╟─730f7e65-d7c9-4506-94fe-f99e244a1c74
# ╟─5487dd11-7c8e-428f-a727-300780dd02a7
# ╠═7d4706f8-6938-4241-8bbb-f1779119422c
# ╟─3a92e430-8fba-4d69-aa25-13e08a485720
# ╠═83926597-ac50-4f7b-9e6d-84bad3ac129f
# ╟─e38c1d3d-fb36-4793-94fc-665dc0eacd99
# ╟─2998ed6a-b505-4cfb-b5da-bdb3f887a646
# ╟─95e71f9b-ed81-4dc1-adf9-bd4fe1fc8bbe
# ╟─09244ee2-a153-4fc5-bd64-ee752c132743
# ╟─82dc987c-4649-4e65-83e7-e337be0e99e8
# ╟─ae1ae203-95f0-4d49-ba9f-14c334af087b
# ╟─ac904acb-4438-45d9-8795-8ab724240da0
# ╟─96cecfcf-75ac-4327-90e1-ec1ac5b3f39c
# ╟─68c98e53-7ac3-4832-a7dd-97459a89d7cb
# ╟─5f1f12b2-5b63-416e-a459-9b5d0e37b0e8
# ╟─b29597f1-3fd7-4b44-9097-7c1dc1b7629b
# ╟─6c83a79a-8581-45cb-8d31-31185dc42a0f
# ╟─1016fe42-c0f8-4823-9ef4-09fd310eaf34
# ╠═77b70944-9771-4eee-b0fc-d842bed3b504
# ╟─9a281767-3d88-4742-8efc-e4fe764c705a
# ╟─022b7068-de1b-4506-abf1-2289976f1597
# ╟─16be37d7-f6d5-469b-b0fd-20816b42d4e5
# ╟─aafd802c-e529-4116-8d58-13f2ba1c4e49
# ╟─d6d332c5-769c-4fe1-8f84-fd2bbf19250a
# ╟─39edf0df-a9c5-4e76-af01-f730b4a619b9
# ╠═77fe324c-bb47-4aed-9d10-27add827239d
# ╟─dd8095b0-8e42-4f6d-a545-8ac1db07ff79
# ╟─c387be0b-3683-48d6-9041-d8f987428499
# ╟─df08ac1d-bcd7-4cb7-bcd1-68f5de699aa0
# ╟─b15f101a-b593-4992-ae80-f32dc894c773
# ╟─2fc46172-cd45-48b2-bff2-9dd5a91e21d1
# ╟─bcca401f-b64d-45f5-b8c0-766cc2a1a50e
# ╟─d3c70aee-c9ce-44f7-9b02-ca674f8c5f01
# ╟─d20cf19f-70fc-47dd-a031-22079bbd10b9
# ╟─1fbb02a8-299a-406c-a337-a74c5ad444e1
# ╟─c9fa60c5-f216-4eb9-b052-b3472e551bdf
# ╟─89a6cb78-b3e2-4780-9961-679a73967f5e
# ╟─bad85a4b-6a14-445a-8af6-ac5f5ebaf183
# ╟─ba2545d4-cc39-4069-b6a4-a60f15911cec
# ╠═56f98135-da40-4ccd-a93d-8ac3e01aee5d
# ╟─858cbf82-14a5-4605-a9ba-61e1084c2c8c
# ╟─331710c1-0fc7-4228-b26c-1d170c505205
# ╟─d820035a-9d5c-4ec0-89cd-7e112d42eb8e
# ╟─d037451e-9ece-4f20-963f-b58eaa022c0a
# ╟─39e5d60e-dedf-4107-936f-b80158c62e4d
# ╟─6e41b059-7460-4b6d-ba90-dba6acb30c18
# ╟─3bf64dd1-c44e-4499-9c15-3c4725554ef1
# ╟─53298cd1-af19-4611-9088-c9dd0cb082ef
# ╟─ce95c7c4-f10e-4a67-9f65-884a772ec37b
# ╟─de52ebac-9d24-4ad8-9686-90a11008ae26
# ╟─42eec440-ef63-46db-b7bb-5f0c2c5046b6
# ╟─27f14c5e-e229-44fa-a252-a8efc9d9dc4a
# ╟─d6805966-deee-4db1-b9c4-66820e05ef91
# ╟─e51beb02-0807-4316-b330-c6578c9fc315
# ╟─160f5345-a517-4b52-972c-bafdf4972835
# ╟─5a44aecd-449b-4269-a435-c6a6bae3cffd
# ╟─ad463d15-aa9b-4c43-8732-d23a7a920f5f
# ╟─2f2c94fd-a2f2-45f8-a284-78d73721f623
# ╟─d56be408-3517-45a4-88e8-56e194ce33f0
# ╟─6a9863cb-3067-4250-ad89-6a1c8dc1fddc
# ╟─ca2662a9-2754-45c1-9ce8-5b8599eef240
# ╟─9375f800-7c4b-4cae-b263-f17198b04011
# ╟─280069b8-f9fe-4cd6-866d-a20b9eab036e
# ╟─0138dedf-f874-4bbd-bf87-7f6bbe8ca816
# ╠═af617b31-396f-465e-b27e-2fa14b3b2423
# ╟─18ebc039-1656-4f74-8e9f-f03a8d39d7c4
# ╠═88696ab1-2866-46f1-978e-bd032566cef7
# ╟─fd0d20a0-5732-4afd-b2d7-b013eb624834
# ╠═74deaa30-4c5c-4e40-99bf-912bd8611374
# ╟─60afbdbf-6e83-44e2-99ed-2d64e339b0d9
# ╠═55cd420e-461c-4a1b-a82c-a2e04be87d74
# ╠═42cfa97c-6298-47f1-a17a-8a5ff642e570
# ╟─84c51030-d87a-4269-89b3-3b43aac03c61
# ╟─8f1eadcb-71f2-4265-9ee1-7a8f5f252bc7
# ╟─9cf68cd6-1175-4660-8918-58bb50b4ecf3
# ╟─16611a44-fc1d-4fa6-8bfd-4788d57432d4
# ╟─94d47c8a-0f84-4c67-853b-f7e4a71cc90f
# ╟─8cf9b88b-e17c-445a-9ebe-449d45c9c3cc
# ╟─27c4c76b-65d8-40cb-89f3-fc42c4a6848e
# ╟─8fd498e7-0690-4947-867a-9e6495a8bcd9
# ╠═f0997ca0-5cfd-4bbe-b094-5fe5a7ea8b65
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
