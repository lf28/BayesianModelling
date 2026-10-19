### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 078e8979-6753-411f-9c34-ba0164ea9cb2
begin
    using PlutoUI
	using Distributions
	using StatsPlots
	using PlutoTeachingTools
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
    using LogExpFunctions, Random
	using LaTeXStrings
	using Logging; Logging.disable_logging(Logging.Info);
end;

# ╔═╡ b2ad130c-5d40-4a4b-adbe-5880984a9460
TableOfContents()

# ╔═╡ 1e40f26a-6381-4bb7-a70f-b35a51f33dfd
md"[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section2_modelling.html)"

# ╔═╡ 3388e67f-a5bb-409a-843d-0958243c3b8f
md"

# An introduction to Bayesian inference
"

# ╔═╡ 97d325ca-2c41-42b4-82b8-f74659e61dc3
md"""

## Introduction

**Statistical inference**, or learning in the machine learning community, is the process of inferring properties about the population distribution from observed data. The observed data is usually assumed to be drawn from the population:

$$d_1, d_2, \ldots, d_N \sim \mathcal P,$$

where "``\sim``" means the data are *distributed according to*, or equivalently, drawn from the distribution. And we want to *infer* some properties of the population distribution ``\mathcal P`` based on the sample ``\mathcal D = \{d_n\}``. 


The most common inference case is parametric inference: that is when the population ``\mathcal P`` is some probability distribution with some finite number of parameters. For example, to test whether a coin is fair, we toss the coin ``N`` times. In this case, the population distribution is a Bernoulli distribution, *i.e.* a random variable with binary outcomes. The population distribution is fully determined by one parameter ``\theta \in [0,1]``, *i.e.* the bias of the coin:

$$\mathcal P \triangleq \text{Bernoulli}(d; \theta)= \begin{cases} \theta, & d=\texttt{head} \\ 1-\theta, & d= \texttt{tail}\end{cases}$$


and ``d_i`` are random independent samples from the distribution: for ``i \in 1, \ldots, N``

$$d_i \sim \text{Bernoulli}(\theta).$$  

Then a typical inference query is: 


> Based on the observed data ``\mathcal D=\{d_1, d_2, \ldots, d_N\},`` what the population distribution's bias ``\theta`` is?

There are two camps of statistical inferences: the **frequentist** and the **Bayesian approach**. Historically, the frequentist's approach has been more widely taught and used. However, the Bayesian approach has gained more momentum in the 21st century, after being successfully applied in a wide range of applications.

Compared with the frequentist approach, Bayesian inference is procedurally standard. All Bayesian inferences start with a generative stochastic model that tells a story of how the data is generated and end with invoking Bayes' rule to find the updated distribution of the unknowns. In other words, as long as you can tell a story about your data, you can be a proper Bayesian statistician. 

However, the same thing cannot be said for the Frequentist's method. Speaking for myself, I cannot recall the ANOVA's (or just choose any from the following: MANOVA, ``\chi^2``,  Paired/Independent T-test) test procedure. And I certainly have forgotten the derivation details behind the steps of those tests. 

Apart from flexibility, if properly specified, Bayesian inference offers more benefits: such as numerical stability, better-generalised performance, and natural interpretation. However, the benefits come with a price: Bayesian methods cannot be used like a black box (that said, neither should Frequentist methods or any scientific method). You do need to understand your data well to be able to come up with a good story (or a proper Bayesian model) first. And after the modelling, one also should be able to diagnose whether the inference algorithms are behaving well. However, modern computational tools, such as probabilistic programming language, make the hurdle easier to overcome.


In this first section, are going to have an overview of Bayesian inference. Firstly, we are going to review Bayes' theorem, which is the cornerstone of Bayesian inference. Next, the coin-tossing problem is used as an example to demonstrate how Bayesian inference is used in practice. To better understand the differences between the two inference camps, a frequentist's take is also shown. 
"""

# ╔═╡ 300929c0-3279-47ef-b165-0e936b757679
md"""

## Bayes' theorem

We first review Bayes' theorem, which forms the core idea of Bayesian inference.


!!! infor "Bayes' theorem"
	Bayes' rule provides us with a mechanism to infer an unknown quantity $\theta$ based on observed data $\mathcal D$:
	
	$$\text{posterior}=\frac{\text{prior} \times \text{likelihood}}{\text{evidence}}\;\;\text{or}\;\;p(\theta|\mathcal D) =\frac{\overbrace{p(\theta)}^{\text{prior}} \overbrace{p(\mathcal D|\theta)}^{\text{likelihood}}}{\underbrace{p(\mathcal D)}_{\text{evidence}}},$$

	which is often simplified to:

	$$p(\theta|\mathcal D) \propto {p(\theta)} p(\mathcal D|\theta),$$ 
	
	("``\propto``" denotes "proportional to") as the evidence term 

	$$p(\mathcal D) = \int p(\theta) (\mathcal D|\theta)\mathrm{d}\theta$$ is constant respect to ``\theta``.
	* ``p(\theta)`` **prior** distribution: prior belief of ``\theta`` before we observe the data 
	* ``p(\mathcal D|\theta)`` **likelihood**: conditional probability of observing the data given a particular $\theta$
	* ``p(\mathcal D)`` **evidence** (also known as *marginal likelihood*). It scales the product of prior and likelihood, ``p(\theta) \times p(\mathcal D|\theta)``, such that the posterior ``p(\theta|\mathcal D)`` is a valid probability distribution. That is the total probability mass sum to one.
"""

# ╔═╡ 41e15a4f-8ddc-47f9-8286-bf583b7d748a
md"""
### Why Bayes' theorem is useful?

Human brains are not very good at doing reverse logic reasoning directly. We are however good at thinking in the forward direction. Bayes' rule provides us with a tool to do the reverse reasoning routinely based on the forward logic.

When it comes to statistical inference, it is way more straightforward to specify the *forward probability*:

$$\theta \xRightarrow[\text{probability}]{\text{forward}} \mathcal{D}.$$

That is assuming the unknown parameter ``\theta``, how likely one observes the data, which is known as **likelihood** function and denoted as a conditional probability: 

$$p(\mathcal D|\theta):\;\; \text{likelihood function}.$$

In practice, we are more interested in the opposite direction: 

$$\mathcal D \xRightarrow[\text{probability}]{\text{inverse}} \theta.$$


That is given the observed data ``\mathcal D``, what ``\theta \in \Theta`` is more likely to have been used to generate data, which is exactly the posterior distribution:

$$p(\theta|\mathcal D):\;\; \text{Posterior distribution}.$$  


Bayes' rule provides the exact tool to do this reverse engineering. And the recipe is simple and mechanical: multiply a prior with the likelihood (forward probability):

$$p(\theta|\mathcal D) \propto p(\theta) p(\mathcal D|\theta).$$

The following example demonstrates how Bayes' rule is used to answer a non-trivial inverse probability problem.

"""

# ╔═╡ c91937d3-1a46-4dc2-9d0d-f6fda482ea36
md"""

**Example (Switch point detection)** Your friend has two coins: one fair (*i.e.* with the probability of a head turning up $p=0.5$) and one bent (with a probability of head turning up $p= 0.2$). He always uses the fair coin at the beginning and then switches to the bent coin at some unknown time.
You have observed the following tossing results: ``\mathcal{D}= [0,1, 0,0,0,0]``. We use 1 to represent the head and 0 for the tail. When did he switch the coin? 
"""

# ╔═╡ 41bd090a-ed31-4ebc-ad57-0785323378d4
function ℓ_switch(D, p₁=0.5, p₂=0.2)
	likes = zeros(length(D)-1)
	for t in 1:length(likes)
# 		Bernoulli(p) return a instance of Bernoulli r.v.
# 		pdf(Bernoulli(p), y) return the probability, either p or 1-p depends on y
# 		prod times everything together
		likes[t] = prod(pdf.(Bernoulli(p₁), D[1:t])) * prod(pdf.(Bernoulli(p₂), D[(t+1):end]))
	end
	return likes, sum(likes)
end;

# ╔═╡ 6f57947d-0ecb-4e9c-b8c7-6ca8de2630d8
md"""
**Solution** 

We identify the unknown $\theta \triangleq S$: *i.e.* the unknown switching point ``S``:
* ``S\in \{1,2,3,\ldots, 5\}``: after $S$-th toss, he switches the coin; 


First, we determine the **likelihood**, or **forward probability**: the conditional probability of observing the data ``\mathcal D`` assuming (*i.e.* conditioning on) we know the switching point S. It is worth noting that this forward probability is straightforward to specify. Knowing the switching point, the whole data becomes two segments of independent coin tosses:

$$p(\mathcal D|S) = \underbrace{\prod_{i=1}^S p(d_i|p =0.5)}_{\text{coin 1}} \underbrace{\prod_{j=S+1}^N p(d_j|p = 0.2)}_{\text{coin 2}};$$ 

For example, if ``S=2``, the likelihood is ``p(\mathcal D|S=2) = \underbrace{0.5\cdot 0.5}_{\text{before switch}} \times \underbrace{(1-0.2)^4}_{\text{after switch}}``.


$(begin
	Plots.plot(1:5, ℓ_switch([0,1,0,0,0,0])[1], xlabel=L"S", ylabel=L"p(\mathcal{D}|S)", title="Likelihood function", st=:sticks, marker=:circle, label="")
end)

To apply Bayes' rule, we need to impose a *prior* distribution over the unknown. To reflect our ignorance, a natural choice is a uniform distribution: for ``S \in \{1,2,\ldots, 5\}``: ``p(S) = 1/5.``

Lastly, we combine the prior and likelihood according to Bayes' rule:

$$p(S|\mathcal D) \propto p(S) p(\mathcal D|S).$$


$(begin
	Plots.plot(1:5, 1/5 * ones(5), st=:sticks, color=1,marker=:circle, label="")
	Plots.plot!(1:5, 1/5 * ones(5),  st=:path, fill=true, color=1, alpha= 0.3,  label="Prior")
	Plots.plot!(1:5, ℓ_switch([0,1,0,0,0,0])[1]./ℓ_switch([0,1,0,0,0,0])[2], xlabel=L"S", ylabel=L"p(S|\mathcal{D})", title="Posterior distribution of the switching point", st=:sticks, marker=:circle, color=2, label="")
	Plots.plot!(1:5, ℓ_switch([0,1,0,0,0,0])[1]./ℓ_switch([0,1,0,0,0,0])[2], st=:path, fill=true, alpha=0.3, color=2, label="Posterior")
end)

Bayes' theorem **updates** our prior belief to the posterior by combining the likelihood. And according to the posterior, it is *most likely* that the switch point is after the second toss. 

However, there is a great amount of uncertainty about this *single point estimator*: an alternative hypothesis ``S=3`` is also likely. A true Bayesian way to answer the question is to ship the posterior distribution as an answer rather than just report the most likely point estimator.
"""

# ╔═╡ dc551c95-cae5-4459-bdcd-d98aecb8b5e5
md"""


## Bayesian inference procedure

The switching point example demonstrates how Bayes' theorem is used to answer an inference question. The whole process can be summarised in two stages: **modelling**  
and **computation**.


**Bayesian *modelling* stage**

At the modelling stage, we are *telling a story* of the forward probability: *i.e.* how the data is generated hypothetically.  

Different from the frequentist approach, Bayesian inference assumes both the **unknown parameter** ``\theta`` and the **observed data** ``\mathcal D`` random.
Therefore, Bayesian's storyline includes the data generation process for ``\theta``, *i.e.* **prior** and the observed ``\mathcal D``, *i.e.* **likelihood**. 

In other words, both methods require the likelihood but specifying priors for the unknowns is unique to the Bayesian approach.

**Bayesian *computation* stage**

At the **computation** stage, we routinely apply Bayes' rule to answer the inverse inference question or the posterior distribution and finally summarise the posterior to answer specific downstream inference questions.

However, the computation step is only *conceptually* straightforward. There is no technical difficulty when the parameter space $\Theta$ is discrete and finite: *e.g.* the unknown switching point can take one of the five discrete choices. In practice, this exact enumeration method becomes unattainable when the parameter space is continuous and higher-dimensional. And we need more advanced algorithms, *e.g.* Markov Chain Monte Carlo (MCMC) or variational method, to make the computation scalable. We will discuss the advanced computation algorithm in the next chapter.

"""


# ╔═╡ 07e16486-c07c-450c-8de3-8c088c80816a
md"""
It is also worth noting that all Bayesian inference problems follow the same two procedures. Compared with the frequentist methods which are formed by a variety of techniques (to name a few: *Z-test, T-test, Χ²-test, bootstrap*), the Bayesian procedure is uniform and standard. 
"""

# ╔═╡ fcb0fab4-d2f7-411f-85f6-0db536f8ee5a
md"""

In summary, Bayesian inference is formed by two blocks of steps: modelling, and computation.

!!! note "Modelling"

	1. Specify **Prior** distribution for the unknown: ``p(\theta)``. As the name suggests, a prior represents our prior belief of the unknown variable *before* we see the data. It represents a subjective belief. 


	2. Determine **Likelihood** function for the observed: ``p(\mathcal D|\theta)``. A likelihood function is a conditional probability of observing the data $\mathcal D$ given a particular $\theta \in \Theta$. The likelihood is used both in classic Frequentist and Bayesian inference. 


!!! note "Computation"
	3. Compute **Posterior** distribution: ``p(\theta|\mathcal D)``. The third step is straightforward, at least conceptually: *i.e.* mechanically apply Bayes' rule to find the posterior.
	4. (optional) Report findings. Summarise the posterior to answer the inference question at hand. This step is optional as Bayesians report the posterior distribution from step three as the answer.
"""

# ╔═╡ 481044fe-07ea-4118-83d8-e9f287406d10
md"""

To consolidate the idea of Bayesian inference's procedure, we will see a few inference examples. When possible, both the frequentist and Bayesian methods will be applied and compared.
"""

# ╔═╡ 198e2787-4901-4478-a092-a84410ad2dd5
md"""

## An example: coin-flipping

To better understand the Bayesian inference procedure and also draw comparisons with the Frequentist's approach, we revisit the inference problem mentioned earlier: inferring the bias of a coin.

> A coin 🪙 is tossed 10 times. And the tossing results are recorded: 
> $$\mathcal D=\{1, 1, 1, 0, 1, 0, 1, 1, 1, 0\}$$; 
> *i.e.* seven out of the ten tosses are heads (ones). Is the coin **fair**?


We will first see how the frequentist approach solves the problem then the Bayesian method.
"""

# ╔═╡ 32c1a567-0b61-4927-81fc-f66773ed3e05
md"""

### Method 1. Frequentist approach*

The frequentist approach believes ``\theta`` is not a random variable but some fixed constant. Therefore, they will not use Bayes' rule, which treats ``\theta`` as a random variable (with prior and posterior distributions). Instead, the frequentists will form some estimator for ``\theta`` and try to find some long-term frequency property of the estimator. A natural estimator for the bias ``\theta`` is the observed frequency: 

$$\hat\theta = \frac{1}{N}\sum_{n=1}^N d_n = \frac{N_h}{N} =0.7,$$

where ``N_h=\sum_n d_n`` denotes the total count of heads, and ``N`` is the number of tosses. However, the true bias of the coin will not exactly be ``\hat\theta=0.7``. If we toss the coin another 10 times, the observed frequency will likely be different from ``0.7``. So how serious shall we treat this ``0.7``? The frequentist wants to know the long-term frequency pattern of the estimator ``\hat\theta`` (but **not** ``\theta``).

One of the frequentist's methods to achieve this is to form a **confidence interval**: an interval ``\theta \in (l, u)`` that traps the true unknown parameter with some good confidence or high probability.

"""

# ╔═╡ ff41d114-0bd4-41ea-9c22-b6af52b3fa21
md"""



A Gaussian-based confidence interval can be formed for the coin's unknown bias:

$$(\hat{\theta} - z_{\alpha/2} \hat{\texttt{se}}, \hat{\theta} + z_{\alpha/2} \hat{\texttt{se}}),$$

where 

$$\hat{\texttt{se}}= \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{N}}.$$

"""

# ╔═╡ ccd3bb12-02a9-4593-b53c-d7081c4e743f
Foldable("Details about the Gaussian based confidence interval.", md"
Based on central limit theory, we know all sample average, regardless of their population distribution, asymptotically converges to a Gaussian: i.e.

$$\hat\theta \sim \mathcal N\left (\theta, \frac{\theta(1-\theta)}{N}\right ).$$

*Note that ``\hat\theta`` (the estimator rather than the parameter) is assumed a random variable here as it is a function of the data ``d_i``, which are assumed to be drawn from a Bernoulli distribution. Therefore, we can find ``\hat\theta``'s mean and variance.* 

And we apply the so called *plug-in principle*: replace all ``\theta`` in the sampling distribution with its estimator ``\hat\theta``.

Note that better methods exist for the model. Ideally, the Gaussian-based confidence interval should only be used when ``N`` is large. We are only illustrating the idea of the frequentist's confidence interval here. 
")

# ╔═╡ 278d302f-6b81-4b00-bdd2-55057417809e
md"""

For our case, 
``\hat\theta = \frac{\sum_{n=1}^N d_n}{N}=0.7`` and the standard error about ``\hat \theta`` is: ``\hat{\texttt{se}} = \sqrt{\frac{\hat{\theta} (1-\hat{\theta})}{N}}\approx 0.145.``


Picking a significance level at ``\alpha = 10\%``, the corresponding $z_{\alpha/2}=1.645$. A confidence interval of ``90\%`` therefore is

$$(0.46, 0.94).$$

Since the confidence interval encloses ``\theta=0.5``, the frequentists conclude that we do not have overwhelming evidence to rule out the coin being fair.

But what does the confidence interval mean here? Frequentist's methods assume ``\theta`` is a fixed unknown quantity (and only data $\mathcal D$ is random). The confidence interval surely is **not** a probability statement about ``\theta``. But rather it is a probability statement about the interval itself (which depends on the data therefore random). The idea is probably better explained in the following way.

!!! danger ""
	
	
	A ``\alpha=10\%`` **confidence interval**: if you 

	*repeat the following two steps, say 10,000 times:*

	1. *toss the coin another 10 times*
	2. *calculate another confidence interval*: ``(\hat{\theta} - 1.645 \hat{\texttt{se}}, \hat{\theta} + 1.645 \hat{\texttt{se}})`` *as above*

	Over the 10,000 realised random intervals, approximately 90% of them trap the true parameter.


The animation below illustrates the idea of a confidence interval. Conditional on the hypothesis that the coin is fair, *i.e.* ``\theta =0.5`` (it works for any ``\theta\in [0,1]``), the above two steps were repeated 100 times, *i.e.* tossing the coin ten times and then forming a confidence interval. The red vertical intervals (there are roughly 10 of them) are the 10% CIs that **do not** trap the true bias. 
"""

# ╔═╡ 1c64b953-5a3d-4a80-b523-8b8f213f6849
md"""
The final plot of the 100 confidence intervals is also listed below for your reference.
"""

# ╔═╡ 65b17042-1e16-4097-86c9-83c516de803d
md"""
In practice, we make an inference decision based on one of the realised CI (0.46, 0.94), *i.e.* one experiment: it either traps or misses the true value. Nevertheless, it is likely (90% of chance) that the CI at hand is one of the good CIs. And based on this fact, we reach a conclusion. However, there is a small chance that the CI is wrong, *i.e.* it misses the true value. And the error is known as the **Type 1** error. 
"""

# ╔═╡ b42dc5e4-143e-4469-be7f-7b1f73ca9064
md"""

If you think the Frequentist's confidence interval is confusing, I agree with you. The thought experiment of repeated experiments does not even make sense in cases like time series analysis: how can one travel back in time to collect another sample? 

If you want a more direct answer to the inference question, we need to resort to Bayesian inference. It provides us with a probability statement about the unknown quantity directly: 
> "*In light of the observed data, what ``\theta`` is more credible, i.e. what is ``p(\theta|\mathcal D)``?*''

"""

# ╔═╡ c8c99b5a-1bd2-4762-beb2-518e0e8c9aa3
md"""

### Method 2. Bayesian approach

The Bayesian approach assumes ``\theta`` is a random variable with some prior subjective belief. And apply Bayes' rule to update the prior belief to answer the question.


Bayesian inference starts with a model: or "a story on how the data is observed". Note that, from Bayesian's perspective,  the data here include **all** data: both known (or observed) and the unknown. On the other hand, the frequentists' storyline only cares about the observed.

For the coin flipping problem, the *story* is straightforward: the unknown ``\theta`` is first drawn from the prior belief's distribution, and then ``N`` realisations ``d_n`` are subsequently drawn from the coin. 


**Step 1. set a prior for the unknown: ``p(\theta)``**

The bias, denoted as ``\theta``, is the unknown parameter. To make the posterior computation easier, let's assume $\theta$ can only take 11 discrete values: 0, 0.1, 0.2 up to 1.0:

$\theta \in [0.0, 0.1, 0.2, \ldots, 1.0],$

To show our ignorance, we can further assume a uniform prior over the 11 discrete choices, *i.e.*

$$p(\theta) = \begin{cases} 1/11, & \theta \in \{0, 0.1, \ldots, 1.0\} \\
0, & \text{otherwise}; \end{cases}$$

The prior distribution is shown below:

$(begin
	prior_plt = Plots.plot(0:0.1:1.0, 1/11 * ones(11), ylim= [0,1], seriestype=[:path, :sticks], color= 1, fill=true, alpha=0.3, markershape=:circle, xlabel=L"θ", ylabel=L"p(θ)", label="", title="Prior")
end)

"""

# ╔═╡ 65ef62da-f095-4bb2-aa0f-120827bed6e0
md"""

After spelling out the model (*i.e.* prior and likelihood), we apply Bayes' rule to find the posterior.

**Step 3. Apply Bayes' theorem**

Mechanically apply Bayes' rule: *i.e.* multiplying the prior and likelihood then normalise.

Note the prior is a constant, and the posterior is proportional to the likelihood:

$$p(\theta|\mathcal D) \propto p(\theta)\cdot p(\mathcal D|\theta) = \frac{1}{11} \cdot p(\mathcal D|\theta)\propto p(\mathcal D|\theta).$$

The normalising constant $p(\mathcal D)$ is

$$p(\mathcal D) = \sum_{\theta\in \{0.0, 0.1, \ldots, 1.0\}} p(\theta)\cdot p(\mathcal D|\theta).$$ 



"""

# ╔═╡ db7f7652-c37d-4f3c-a7c6-0fc57112c3ba
Foldable("Why it is called normalising constant?*", md"It *normalises* the unnormalised posterior (the product of prior and likelihood) such that the posterior probability mass add to one:

$$\sum_\theta p(\theta|\mathcal D) =  \sum_\theta \frac{p(\theta) \cdot p(\mathcal D|\theta)}{p(\mathcal D)} = \sum_\theta \frac{p(\theta) \cdot p(\mathcal D|\theta)}{\sum_\theta p(\theta)\cdot p(\mathcal D|\theta)} = 1.$$")

# ╔═╡ 785bdafd-4bfc-441b-bd79-f1bbced4efb9
md"""
The posterior therefore can be calculated as follows: for ``\theta \in \{0.0, 0.1, \ldots, 1.0\}``:

$$p(\theta|\mathcal D) = \frac{p(\mathcal D|\theta)}{\sum_{\theta} p(\mathcal D|\theta)}$$

The posterior update procedure is illustrated below. Note that the posterior is of the same shape as the likelihood (since a flat prior has been used). After observing the data, the posterior now centres around 0.7. But it seems 0.8 and 0.6 are also likely.
"""

# ╔═╡ d31477e0-35b0-4968-bfbf-aefa727f3a41
md"""

**Step 4. (optional) Report findings**

Technically, the Bayesian approach has finished once the posterior is finalised. One can ship the full posterior as an answer. However, to answer the particular question of the coin's fairness, we can proceed to *summarise the posterior*.

One way to summarise a posterior distribution is to find the most likely region of the posterior, or **highest probability density interval (HPDI)** such that the enclosed probability within the interval is some number close to 100 %, *e.g.* 90%:

$$p(l \leq \theta \leq u|\mathcal D) = 90 \%.$$

Based on the posterior, we find the corresponding interval is between 0.5 and 0.9 inclusive: *i.e.*

$$p(0.5 \leq \theta \leq 0.9|\mathcal D) \approx 0.90.$$
"""

# ╔═╡ dcc5758e-b1da-48c6-a9cf-5f906fcf76a9
md"""

The fair coin hypothesis, *i.e.* $\theta=0.5$ is within the 90% HPDI. Therefore, we should *not* reject the hypothesis that the coin is fair, which agrees with the frequentist's judgement.

However, compared with Frequentist's confidence interval, **the credible interval is a probability statement about the unknown bias**: the coin's unknown bias is likely to lie within the range 0.5 to 0.9 with a probability of 0.9. 
"""

# ╔═╡ cf054a9c-8617-4d22-a398-f73b2d57139f
md"""

## An extension: multiple parameter model

"""

# ╔═╡ 0ff3bf61-3bd3-449d-94d6-cd6abff3d41b
md"""

As mentioned earlier, the Bayesian approach is very flexible in dealing with customised problems. To demonstrate this, we consider an extension of the previous simple inference problem.

Suppose now we have two coins ``A`` and ``B`` with unknown biases: ``0\leq \theta_A \leq 1, 0\leq\theta_B \leq 1``. The two coins are tossed ``N_A=10`` and ``N_B=100`` times, and the numbers of heads, ``N_{A,h}=9, N_{B,h}=89,`` are recorded. Now the inference question is

> Which coin has a larger bias?


*Remarks. Judging from the empirical frequencies only, one may conclude that coin A has a higher probability of success since ``\frac{9}{10} > \frac{89}{100}``. However, this naive answer clearly does not take all possible uncertainties into account. Coin two's estimator clearly is more reliable and certain than coin one. To answer the question, the frequentist needs to start over to find a suitable test.* 
"""

# ╔═╡ 657a73e1-88c7-4c4d-be22-410aeeb6ef40
Foldable("More application of the two coin problem.", md"The above problem might look bizarre. But it offers a solution to a wide range of practical problems. One variant is an Amazon sellers' review problem. 


!!! note \"Amazon review problem\"
	There are two second-hand booksellers on Amazon. Seller A has sold 10 books in total and 7 out of the 10 are left with positive ratings; whereas seller B has 69 out of 100 positive reviews. Which seller is likely to be better? 

It is not hard to see the connection between the two problems. Both problems can be modelled with the same stochastic model. And a Bayesian model can be specified as the followings.")

# ╔═╡ 397e354d-dd17-4f66-a578-1b6374392953
md"""

**Step 4. Answer questions.**

Bayesian inference can be used to answer questions directly. Rather than framing a null/alternative hypothesis and doing tests like what the frequentists do. We can instead frame the straight-to-the-point question as a posterior probability statement:
*In light of the data, how likely coin ``A`` has a higher bias than coin ``B``?*, *i.e.*

```math
p(\theta_A > \theta_B|\mathcal{D}).
```

The probability can be calculated based on our posterior distribution by summing up the corresponding entries of the joint distribution table:

```math
p(\theta_A > \theta_B|\mathcal{D}) = \sum_{\theta_A > \theta_B:\;\theta_A,\theta_B\in \{0.0, \ldots 1.0\}^2} p(\theta_A, \theta_B|\mathcal{D})
```

"""

# ╔═╡ e100c593-172c-4919-9e5b-b6dd612653f5


md"For our problem, the posterior probability is: "

# ╔═╡ c98177ca-e773-4f00-b27f-79ccb143a5c3
md"""

Figuratively, we are summing up the probability below the line ``\theta_A=\theta_B``, which corresponds to our query's condition. If one happens to want to know the chance that coin ``A``'s bias is twice (or ``k`` times) coin ``B``'s (imagine we are comparing two vaccine's effectiveness rather than two coins), the Bayesian approach can answer that immediately, while the frequentist would need to restart the inference all over again!

```math
p(\theta_A > 2\cdot \theta_B|\mathcal{D}) = \sum_{\theta_A > 2\cdot\theta_B} p(\theta_A, \theta_B|\mathcal{D})
```


"""

# ╔═╡ bc4d9710-7e6c-4bc9-905b-7b3c0c8e9abe
md"""

## What's next?

In the following sections, we are going to discuss topics revolving around the two core building blocks of Bayesian inference: **modelling** and **computation**. 


**Modelling**

We will introduce more Bayesian modelling concepts in the next chapter, *e.g.* generative model and prior choices, and model checking. However, the modelling step is more an art than a science. Ideally, each problem should have its own bespoke model. Therefore, more model-specific details will be introduced later when we introduce the individual models.

**Computation**

In chapter 3, we will introduce Markov Chain Monte Carlo (MCMC), a core algorithm that makes Bayesian inference feasible. We will focus on the intuitions behind some popular MCMC algorithms and also practical applied issues on using MCMC, such as chain diagnostics.


**Probabilistic programming `Turing.jl`**

Implementing each Bayesian model and its inference algorithm from scratch is not practical for applied users. Instead, we can specify a Bayesian model effortless with the help of probabilistic programming languages, such as `Turing.jl`[^1] and `Stan`[^2]. The downstream Bayesian computation tasks can also be done automatically by the packages. We will see how to use `Turing.jl` to do applied Bayesian inference in chapter 4.

**Bayesian models**

After the general concepts being introduced in the first four chapters, the second half of the course will cover a range of commonly used Bayesian models.

* parametric density estimation
* linear regression
* generalised linear regressions
* multi-level models *a.k.a.* hierarchical Bayesian models
* and so on
"""

# ╔═╡ 66159a91-6a82-4a52-b3fb-9749bb66d4e2
md"""

## Notes

[^1]: [Turing.jl: Bayesian inference with probabilistic programming.](https://turing.ml/stable/)

[^2]: [Stan Development Team. 2022. Stan Modeling Language Users Guide and Reference Manual](https://mc-stan.org/users/documentation/)

"""

# ╔═╡ f0a54db9-2cb1-48ad-953f-35d4e0be7b7c
md"
[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section2_modelling.html)"

# ╔═╡ 6a49bd7b-1211-4480-83a3-ca87e26f9b97
md"""
## Appendix

Code used for this chapter (Folded).
"""

# ╔═╡ 14710249-9dc1-4a75-b0b3-25ac565452a5
begin
	head, tail = 1, 0
	𝒟 = [head, head, head, tail, head, tail, head, head, head, tail]
	h_total = sum(𝒟)
	N = length(𝒟)
	function ℓ(θ, N, Nₕ; logLik=false) 
		logℓ = xlogy(Nₕ, θ) + xlogy(N-Nₕ, 1-θ)
		logLik ? logℓ : exp(logℓ)
	end
	θs = 0:0.1:1.0
	likes = ℓ.(θs, N, h_total)
end;

# ╔═╡ 94ae916e-49fb-4426-b261-57b39599a4e7
md"""

**Step 2. Determine the likelihood function: ``p(\mathcal D|\theta)``**

Next, we need to determine the likelihood function: how the observed data, ``\mathcal D``,  *is generated* conditional on ``\theta``. As shown earlier, a coin toss follows a Bernoulli distribution:

$$p(d|\theta) = \begin{cases} \theta, & d = \texttt{h} \\ 1-\theta, & d= \texttt{t},\end{cases}$$

Due to the independence assumption, the likelihood for ``\mathcal D`` is just the product:

$$p(\mathcal D|\theta) = p(d_1|\theta)p(d_2|\theta)\ldots p(d_{10}|\theta)= \prod_{n=1}^{N} p(d_n|\theta) = \theta^{\sum_{n}d_n} (1-\theta)^{N-\sum_{n}d_n},$$

For our case, we have observed ``N_h=\sum_n d_n = 7`` heads out of ``N=10`` total tosses, we can therefore evaluate the likelihood function at the pre-selected 11 values of $\theta$.

$(begin
	like_plt = 
	Plots.plot(0:0.1:1.0, θ -> ℓ(θ, N, h_total), seriestype=:sticks, color=1, markershape=:circle, xlabel=L"θ", ylabel=L"p(𝒟|θ)", label="", title="Likelihood")
end)

"""

# ╔═╡ 009a824f-c26d-43e9-bb6d-fd538a19863b
begin
	l = @layout [a b; c]
	posterior_dis = likes/ sum(likes)
	post_plt = Plots.plot(0:0.1:1.0, posterior_dis, seriestype=:sticks, markershape=:circle, label="", color=2, title="Posterior", xlabel=L"θ", ylabel=L"p(θ|𝒟)", legend=:outerleft)
	Plots.plot!(post_plt, 0:0.1:1.0, posterior_dis, color=2, label ="Posterior", fill=true, alpha=0.5)
	Plots.plot!(post_plt, 0:0.1:1.0, 1/11 * ones(11), seriestype=:sticks, markershape=:circle, color =1, label="")
	Plots.plot!(post_plt, 0:0.1:1.0, 1/11 * ones(11), color=1, label ="Prior", fill=true, alpha=0.5)
	# Plots.plot!(post_plt, 0.5:0.1:0.9, posterior_dis[6:10], seriestype=:sticks, markershape=:circle, label="95% credible interval", legend=:topleft)
	Plots.plot(prior_plt, like_plt, post_plt, layout=l)

	
end

# ╔═╡ 633ad986-dc19-4994-b04e-4ec34432dbf2
begin
	θs_refined = 0:0.01:1
	posterior_dis_refined = ℓ.(θs_refined, N, h_total)
	posterior_dis_refined ./= sum(posterior_dis_refined)
end;

# ╔═╡ be2a2acd-5633-45cc-ab8c-8a064324b287
begin
	cint_normal(n, k; θ=k/n, z=1.96) = max(k/n - z * sqrt(θ*(1-θ)/n),0), min(k/n + z * sqrt(θ*(1-θ)/n),1.0)
	within_int(intv, θ=0.5) = θ<intv[2] && θ>intv[1]
end;

# ╔═╡ bd85c6ef-8623-4e06-8f7e-1e30927b25b7
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

# ╔═╡ b113a7ce-7e00-4df3-b1a4-4cf7af005aaf
begin
	p = hline([0.5], label=L"\mathrm{true}\;θ=0.5", color= 3, linestyle=:dash, linewidth=2,  xlabel="Experiments", ylim =[0,1])
	@gif for i in 1:n_exp
		k_ = outcomes[i]
		# intv = cint_normal(trials, k_; z= 1.645)
		# intv = intvs[i]
		in_out = true_intvs[i]
		col = in_out ? 1 : 2
		θ̂ = k_/trials
		Plots.scatter!([i], [θ̂],  label="", yerror= ([ϵ⁻[i]], [ϵ⁺[i]]), markerstrokecolor=col, color=col)
	end
end

# ╔═╡ 8332304c-2264-46df-baf1-0a1070927152
begin
	first_20_intvs = true_intvs[1:100]
	Plots.scatter(findall(first_20_intvs), outcomes[findall(first_20_intvs)]/trials, ylim= [0,1], yerror =(ϵ⁻[findall(first_20_intvs)], ϵ⁺[findall(first_20_intvs)]), label="true", markerstrokecolor=:auto, legend=:outerbottom,legendtitle = "true "*L"\theta"* " within CI ?")
	Plots.scatter!(findall(.!first_20_intvs), outcomes[findall(.!first_20_intvs)]/trials, ylim= [0,1], yerror =(ϵ⁻[findall(.!first_20_intvs)],ϵ⁺[findall(.!first_20_intvs)]), label="false", markerstrokecolor=:auto)
	hline!([0.5], label=L"\mathrm{true}\;θ=0.5", linewidth =2, linestyle=:dash, xlabel="Experiments")
end

# ╔═╡ 5db12a02-7c7c-47f3-8bea-24e4b5d67cae
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

# ╔═╡ ccc239c6-327f-4905-9040-4a7b4a51e6e1
begin
	post_plt2 = Plots.plot(θs, posterior_dis, seriestype=:sticks, markershape=:circle, label="", title=L"\mathrm{Posterior}\, \, p(\theta|\mathcal{D})", xlabel=L"θ", ylabel=L"p(θ|𝒟)")
	l1, u1, _ = find_hpdi(posterior_dis, 0.87)
	Plots.plot!(post_plt2, θs[l1:u1], posterior_dis[l1:u1], seriestype=:sticks, markershape=:circle, label="90% credible interval", legend=:topleft)
	Plots.plot!(post_plt2, θs[l1:u1], posterior_dis[l1:u1], seriestype=:path, color=2, fill=true, alpha=0.3,  label="")
	Plots.annotate!((0.7, maximum(posterior_dis)/2, ("90 % HDI", 14, :red, :center, "courier")))
end

# ╔═╡ 954ea5d8-ba25-44b9-ac7b-7d3e683cc8e0
begin
	post_plt3 = Plots.plot(θs_refined, posterior_dis_refined, seriestype=:sticks, markershape=:circle, markersize=2, label="", title=L"\mathrm{Posterior}\, \, p(\theta|\mathcal{D}) \mathrm{with\; refined\; space}", xlabel=L"θ", ylabel=L"p(θ|𝒟)")
	l2, u2, _ = find_hpdi(posterior_dis_refined, 0.9)
	Plots.plot!(post_plt3, θs_refined[l2:u2], posterior_dis_refined[l2:u2], seriestype=:sticks, markershape=:circle, markersize=2, label="", legend=:topleft)
	Plots.plot!(post_plt3, θs_refined[l2:u2], posterior_dis_refined[l2:u2], seriestype=:path, fill=true, alpha=0.3, color=2, label="90% credible interval", legend=:topleft)
	Plots.annotate!((0.7, maximum(posterior_dis_refined)/2, ("90 % HDI", 15, :red, :center, "courier")))

end

# ╔═╡ 8f46fde2-c906-4d2f-804d-0ae83fb4c87f
md"""

**A more refined approximation:**
We can have a more refined discretisation of the parameter space. For example, we can choose $$\theta \in [0, 0.01, 0.02, \ldots, 1.0],$$ a step size of ``0.01`` rather than ``0.1``. Check the next chapter for a continuous posterior approach. The corresponding 90% HPDI is 
``p(``$(θs_refined[l2]) ``\leq \theta \leq`` $(θs_refined[u2])``|\mathcal D)= 90\%.`` We still reach the same conclusion.

"""

# ╔═╡ e93a7045-e652-433e-9444-2213b93b57d0
begin
	N₁, N₂ = 10, 100
	Nₕ₁, Nₕ₂ = 9, 89
	dis_size = 51
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)

	function ℓ_two_coins(θ₁, θ₂; N₁=10, N₂=100, Nh₁=7, Nh₂=69, logLik=false)
		logℓ = ℓ(θ₁, N₁, Nh₁; logLik=true) + ℓ(θ₂, N₂, Nh₂; logLik=true)
		logLik ? logℓ : exp(logℓ)
	end

	ℓ_twos = [ℓ_two_coins(xi, yi; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=true) for xi in θ₁s, yi in θ₂s]
	p𝒟 = exp(logsumexp(ℓ_twos))
	ps = exp.(ℓ_twos .- logsumexp(ℓ_twos))
	post_AmoreB = sum([ps[i,j] for j in 1:size(ps)[2] for i in (j+1):size(ps)[1]])
end;

# ╔═╡ 4db4b59e-d59b-4197-befd-0dfa9f703f64
md"""

### A two-parameter Bayesian model

According to the problem statement, we now have a two-parameter model. The two unknowns are the two biases ``\theta_A,\theta_B``. The likelihood is simply the count of tosses and also the counts of heads: ``\mathcal{D} = \{N_A, N_B, N_{A,h}, N_{B,h}\}``.


**Step 1. Prior for the unknown ``p(\theta_A, \theta_B)``.**

Similar to the single coin model, we discrete the two parameters. Now the two-parameter can take any value in a grid of values: ``(\theta_A, \theta_B) \in \{0, 0.01, 0.02, \ldots, 1\}\times \{0, 0.01, 0.02, \ldots, 1\}.`` That means the tuple can take any ``101^2`` possible combinations of the discretised value pairs, such as ``(0.0, 0.0), (0.0, 0.01), (0.0, 0.02)`` and so on.

To show our ignorance, we can further assume a uniform prior over the ``101^2`` choices, *i.e.*

$$p(\theta_A, \theta_B) = \begin{cases} 1/101^2, & \theta_A,\theta_B \in \{0, 0.01, \ldots, 1.0\}^2 \\
0, & \text{otherwise}; \end{cases}$$

The prior distribution is shown below:

$(begin
p1 = Plots.plot(θ₁s, θ₂s,  (x,y) -> 1/(length(θ₁s)^2), st=:surface, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.003, maximum(ps)], colorbar=false, c=:plasma, alpha =0.2, zlabel="density", title="Prior")
end)

**Step 2. Determine the likelihood for ``p(\mathcal{D}|\theta_A, \theta_B)``**

According to the problem statement, the two coins' tosses are all independent. The joint likelihood there is simply a product of two individual single coin's likelihoods.

```math
\begin{align}
p(\mathcal{D}|\theta_A, \theta_B) &= p(\mathcal{D}_A|\theta_A) p(\mathcal{D}_B|\theta_B)\\

&=\theta_A^{N_{A,h}}(1-\theta_A)^{N_A- N_{A,h}}\times\theta_B^{N_{B,h}}(1-\theta_B)^{N_B- N_{B,h}}
\end{align}
```

We can evaluate the likelihood at the pre-selected ``101^2`` values of ``(\theta_A, \theta_B)`` pair.
"""

# ╔═╡ 92d267d5-e808-46a2-aa82-744e66cf5c76
md"""


After finishing specifying the model, what is left is to routinely apply Bayes' rule to find the posterior.

**Step 3. Apply Bayes' rule to find the posterior**

Note the prior is a constant, and the posterior is proportional to the likelihood:

$$p(\theta_A, \theta_B|\mathcal D) \propto p(\theta_A, \theta_B)\cdot p(\mathcal D|\theta_A, \theta_B) = \frac{1}{101^2} \cdot p(\mathcal D|\theta_A, \theta_B).$$

And the normalising constant $p(\mathcal D)$ is

$$p(\mathcal D) = \sum_{\theta_A\in \{0.0, \ldots,1.0\}}\sum_{\theta_B\in \{0.0,\ldots, 1.0\}} p(\theta_A, \theta_B)\cdot p(\mathcal D|\theta_A, \theta_B).$$ 

The updated posterior is plotted below. After observing the data, the posterior now centres around the (0.9, 0.89); however, the ``\theta_A``'s marginal distribution has a much heavier tail than the other, which makes perfect sense. 


$(begin

p2 = Plots.plot(θ₁s, θ₂s,  ps', st=:surface, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.003, maximum(ps)], colorbar=false, c=:plasma, zlabel="density", alpha=0.7, title="Posterior")

end)
"""

# ╔═╡ a2fc25e6-e9e7-4ce7-9532-0449e6423545
md"""
The heatmap of the posterior density is also plotted for reference.

$(begin

Plots.plot(θ₁s, θ₂s,  ps', st=:heatmap, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.003, maximum(ps)], colorbar=false, c=:plasma, zlabel="density", alpha=0.7, title="Posterior's density heapmap")

end)
"""

# ╔═╡ 78ad6108-8058-4c6e-b254-1b508dee2b6f
L"p(\theta_A > \theta_B|\mathcal{D}) \approx %$(round(post_AmoreB, digits=2))"

# ╔═╡ 86b7124e-5042-4f9f-91d7-31b8daad4f98
begin
	post_p =Plots.plot(θ₁s, θ₂s,  ps', st=:contour, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], colorbar=false, c=:thermal, framestyle=:origin)
	plot!((x) -> x, lw=1, lc=:gray, label="")
	equalline = Shape([(0., 0.0), (1,1), (1, 0)])
	plot!(equalline, fillcolor = plot_color(:gray, 0.2), label=L"\theta_A>\theta_B", legend=:bottomright)
	k=2
	kline = Shape([(0., 0.0), (1,1/k), (1, 0)])
	plot!(kline, fillcolor = plot_color(:gray, 0.5), label=L"\theta_A>%$(k)\cdot \theta_B", legend=:bottomright)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.132"
LaTeXStrings = "~1.4.1"
LogExpFunctions = "~1.0.3"
Plots = "~1.41.7"
PlutoTeachingTools = "~0.5.0"
PlutoUI = "~0.7.83"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "1925a59f693bc9a97e5be10e269298d2e65df4d1"

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
# ╟─078e8979-6753-411f-9c34-ba0164ea9cb2
# ╟─b2ad130c-5d40-4a4b-adbe-5880984a9460
# ╟─1e40f26a-6381-4bb7-a70f-b35a51f33dfd
# ╟─3388e67f-a5bb-409a-843d-0958243c3b8f
# ╟─97d325ca-2c41-42b4-82b8-f74659e61dc3
# ╟─300929c0-3279-47ef-b165-0e936b757679
# ╟─41e15a4f-8ddc-47f9-8286-bf583b7d748a
# ╟─c91937d3-1a46-4dc2-9d0d-f6fda482ea36
# ╟─6f57947d-0ecb-4e9c-b8c7-6ca8de2630d8
# ╟─41bd090a-ed31-4ebc-ad57-0785323378d4
# ╟─dc551c95-cae5-4459-bdcd-d98aecb8b5e5
# ╟─07e16486-c07c-450c-8de3-8c088c80816a
# ╟─fcb0fab4-d2f7-411f-85f6-0db536f8ee5a
# ╟─481044fe-07ea-4118-83d8-e9f287406d10
# ╟─198e2787-4901-4478-a092-a84410ad2dd5
# ╟─32c1a567-0b61-4927-81fc-f66773ed3e05
# ╟─ff41d114-0bd4-41ea-9c22-b6af52b3fa21
# ╟─ccd3bb12-02a9-4593-b53c-d7081c4e743f
# ╟─278d302f-6b81-4b00-bdd2-55057417809e
# ╟─b113a7ce-7e00-4df3-b1a4-4cf7af005aaf
# ╟─1c64b953-5a3d-4a80-b523-8b8f213f6849
# ╟─8332304c-2264-46df-baf1-0a1070927152
# ╟─65b17042-1e16-4097-86c9-83c516de803d
# ╟─b42dc5e4-143e-4469-be7f-7b1f73ca9064
# ╟─c8c99b5a-1bd2-4762-beb2-518e0e8c9aa3
# ╟─94ae916e-49fb-4426-b261-57b39599a4e7
# ╟─65ef62da-f095-4bb2-aa0f-120827bed6e0
# ╟─db7f7652-c37d-4f3c-a7c6-0fc57112c3ba
# ╟─785bdafd-4bfc-441b-bd79-f1bbced4efb9
# ╟─009a824f-c26d-43e9-bb6d-fd538a19863b
# ╟─d31477e0-35b0-4968-bfbf-aefa727f3a41
# ╟─ccc239c6-327f-4905-9040-4a7b4a51e6e1
# ╟─dcc5758e-b1da-48c6-a9cf-5f906fcf76a9
# ╟─8f46fde2-c906-4d2f-804d-0ae83fb4c87f
# ╟─633ad986-dc19-4994-b04e-4ec34432dbf2
# ╟─954ea5d8-ba25-44b9-ac7b-7d3e683cc8e0
# ╟─cf054a9c-8617-4d22-a398-f73b2d57139f
# ╟─0ff3bf61-3bd3-449d-94d6-cd6abff3d41b
# ╟─657a73e1-88c7-4c4d-be22-410aeeb6ef40
# ╟─4db4b59e-d59b-4197-befd-0dfa9f703f64
# ╟─92d267d5-e808-46a2-aa82-744e66cf5c76
# ╟─a2fc25e6-e9e7-4ce7-9532-0449e6423545
# ╟─397e354d-dd17-4f66-a578-1b6374392953
# ╟─e100c593-172c-4919-9e5b-b6dd612653f5
# ╟─78ad6108-8058-4c6e-b254-1b508dee2b6f
# ╟─c98177ca-e773-4f00-b27f-79ccb143a5c3
# ╟─86b7124e-5042-4f9f-91d7-31b8daad4f98
# ╟─bc4d9710-7e6c-4bc9-905b-7b3c0c8e9abe
# ╟─66159a91-6a82-4a52-b3fb-9749bb66d4e2
# ╟─f0a54db9-2cb1-48ad-953f-35d4e0be7b7c
# ╟─6a49bd7b-1211-4480-83a3-ca87e26f9b97
# ╟─14710249-9dc1-4a75-b0b3-25ac565452a5
# ╟─bd85c6ef-8623-4e06-8f7e-1e30927b25b7
# ╟─be2a2acd-5633-45cc-ab8c-8a064324b287
# ╟─5db12a02-7c7c-47f3-8bea-24e4b5d67cae
# ╟─e93a7045-e652-433e-9444-2213b93b57d0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
