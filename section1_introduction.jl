### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 078e8979-6753-411f-9c34-ba0164ea9cb2
begin
    using PlutoUI
	using Distributions
	using StatsPlots
    using LogExpFunctions, Random
	using LaTeXStrings
	using Logging; Logging.disable_logging(Logging.Info);
end;

# ╔═╡ b2ad130c-5d40-4a4b-adbe-5880984a9460
TableOfContents()

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
	Plots.scatter(findall(first_20_intvs), outcomes[findall(first_20_intvs)]/trials, ylim= [0,1], yerror =(ϵ⁻[findall(first_20_intvs)], ϵ⁺[findall(first_20_intvs)]), label="true", markerstrokecolor=:auto, legend=:outerbottom,legendtitle = "true θ within CI ?")
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

# ╔═╡ 8076bac8-6501-4792-8e8e-0f57e40cde4d
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

# ╔═╡ ccd3bb12-02a9-4593-b53c-d7081c4e743f
Foldable("Details about the Gaussian based confidence interval.", md"
Based on central limit theory, we know all sample average, regardless of their population distribution, asymptotically converges to a Gaussian: i.e.

$$\hat\theta \sim \mathcal N\left (\theta, \frac{\theta(1-\theta)}{N}\right ).$$

*Note that ``\hat\theta`` (the estimator rather than the parameter) is assumed a random variable here as it is a function of the data ``d_i``, which are assumed to be drawn from a Bernoulli distribution. Therefore, we can find ``\hat\theta``'s mean and variance.* 

And we apply the so called *plug-in principle*: replace all ``\theta`` in the sampling distribution with its estimator ``\hat\theta``.

Note that better methods exist for the model. Ideally, the Gaussian-based confidence interval should only be used when ``N`` is large. We are only illustrating the idea of the frequentist's confidence interval here. 
")

# ╔═╡ db7f7652-c37d-4f3c-a7c6-0fc57112c3ba
Foldable("Why it is called normalising constant?*", md"It *normalises* the unnormalised posterior (the product of prior and likelihood) such that the posterior probability mass add to one:

$$\sum_\theta p(\theta|\mathcal D) =  \sum_\theta \frac{p(\theta) \cdot p(\mathcal D|\theta)}{p(\mathcal D)} = \sum_\theta \frac{p(\theta) \cdot p(\mathcal D|\theta)}{\sum_\theta p(\theta)\cdot p(\mathcal D|\theta)} = 1.$$")

# ╔═╡ 657a73e1-88c7-4c4d-be22-410aeeb6ef40
Foldable("More application of the two coin problem.", md"The above problem might look bizarre. But it offers a solution to a wide range of practical problems. One variant is an Amazon sellers' review problem. 


!!! note \"Amazon review problem\"
	There are two second-hand booksellers on Amazon. Seller A has sold 10 books in total and 7 out of the 10 are left with positive ratings; whereas seller B has 69 out of 100 positive reviews. Which seller is likely to be better? 

It is not hard to see the connection between the two problems. Both problems can be modelled with the same stochastic model. And a Bayesian model can be specified as the followings.")

# ╔═╡ 45973b16-a78e-4709-824d-9b319fae0683
# let
# using GLMakie
# GLMakie.activate!()

## using GeometryBasics

# 	function peaks(; n=49)
# 	    x = LinRange(0, 1, n)
# 	    y = LinRange(0, 1, n)

# 		ℓ_twos = [ℓ_two_coins(xi, yi; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=true) for xi in x, yi in y]
# 		ps = exp.(ℓ_twos .- logsumexp(ℓ_twos))
# 	    return (x, y, ps)
# 	end
	
# 	x, y, z = peaks(; n=20)
# 	δx = (x[2] - x[1]) / 2
# 	δy = (y[2] - y[1]) / 2
# 	z1 = ones(length(x), length(y)) .* (1/(length(x)*length(y)))
# 	cbarPal = :Spectral_11
# 	ztmp = (z .- minimum(z)) ./ (maximum(z .- minimum(z)))
# 	cmap = get(colorschemes[cbarPal], ztmp)
# 	cmap2 = reshape(cmap, size(z))
# 	ztmp2 = abs.(z) ./ maximum(abs.(z)) .+ 0.15


# 	function histogram_or_bars_in_3d()
# 	    fig = Figure(resolution=(2000, 2000), fontsize=30)
# 	    ax1 = Axis3(fig[1, 1]; aspect=(1, 1, 1), elevation=π/6,
# 	        perspectiveness=0.4, xlabel=L"\theta_A", ylabel=L"\theta_B", zlabel="density")
# 	    ax2 = Axis3(fig[2, 1]; aspect=(1, 1, 1), elevation=π/6, perspectiveness=0.4, xlabel=L"\theta_A", ylabel=L"\theta_B", zlabel="density")
# 	    rectMesh = Rect3f(Vec3f(-0.5, -0.5, 0), Vec3f(1, 1, 1))
# 	    meshscatter!(ax1, x, y, 0*z1, marker = rectMesh, color = z1[:],
# 	        markersize = Vec3f.(1.5δx, 1.5δy, z1[:]), 
# 	        shading=false,transparency=true)
# 	    limits!(ax1, 0, 1, 0, 1, -0.02, maximum(z))
# 	    meshscatter!(ax2, x, y, 0*z, marker = rectMesh, color = z[:],
# 	        markersize = Vec3f.(1.5δx, 1.5δy, z[:]), 
# 	        shading=false, transparency=true)
# 		limits!(ax2, 0, 1, 0, 1, -0.02, maximum(z))
# 	    # for (idx, i) in enumerate(x), (idy, j) in enumerate(y)
# 	    #     rectMesh = Rect3f(Vec3f(i - δx, j - δy, 0), Vec3f(2δx, 2δy, z[idx, idy]))
# 	    #     recmesh = GeometryBasics.mesh(rectMesh)
# 	    #     lines!(ax2, recmesh; )
# 	    # end
# 	    fig
# 	end
# 	histogram_or_bars_in_3d()

# end

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
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.66"
LaTeXStrings = "~1.3.0"
LogExpFunctions = "~0.3.18"
PlutoUI = "~0.7.39"
StatsPlots = "~0.14.34"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "ce70ec3d65e8ec0ccfd0a42d75cc8637eff8925b"

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
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

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
git-tree-sha1 = "2dd813e5f2f7eec2d1268c57cf2373d3ee91fcea"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.1"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

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
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "a599cfb8b1909b0f97c5e1b923ab92e1c0406076"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.1"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

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
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c98aea696662d09e215ef7cda5296024a9646c75"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.4"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "3a233eeeb2ca45842fe100e0413936834215abf5"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.4+0"

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
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "cb7099a0109939f16a4d3b572ba8396b1f6c7c31"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.10"

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
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

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
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

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
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

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
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

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
git-tree-sha1 = "891d3b4e8f8415f53108b4918d0183e61e18015b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.0"

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
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
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
git-tree-sha1 = "9a36165cf84cff35851809a40a928e1103702013"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.16+0"

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
git-tree-sha1 = "ca433b9e2f5ca3a0ce6702a032fce95a3b6e1e48"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.14"

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
git-tree-sha1 = "93e82cebd5b25eb33068570e3f63a86be16955be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.1"

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
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

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
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

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
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

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
git-tree-sha1 = "9f8a5dc5944dc7fbbe6eb4180660935653b0a9d9"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2c11d7290036fe7aac9038ff312d3b3a2a5bf89e"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.4.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "48598584bacbebf7d30e20880438ed1d24b7c7d6"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.18"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "43a316e07ae612c461fd874740aeef396c60f5f8"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.34"

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
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

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
# ╟─078e8979-6753-411f-9c34-ba0164ea9cb2
# ╟─b2ad130c-5d40-4a4b-adbe-5880984a9460
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
# ╟─6a49bd7b-1211-4480-83a3-ca87e26f9b97
# ╟─14710249-9dc1-4a75-b0b3-25ac565452a5
# ╟─bd85c6ef-8623-4e06-8f7e-1e30927b25b7
# ╟─be2a2acd-5633-45cc-ab8c-8a064324b287
# ╟─5db12a02-7c7c-47f3-8bea-24e4b5d67cae
# ╟─8076bac8-6501-4792-8e8e-0f57e40cde4d
# ╟─45973b16-a78e-4709-824d-9b319fae0683
# ╟─e93a7045-e652-433e-9444-2213b93b57d0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
