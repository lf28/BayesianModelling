### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ e2f564ae-b741-4532-be01-9255a235793e
using Turing

# ╔═╡ da388a86-19cb-11ed-0c64-95c301c27153
begin
    using PlutoUI
	using StatsPlots
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using PlutoTeachingTools
    using CSV, DataFrames
	using LinearAlgebra
	# using Turing
    using Random
	using LaTeXStrings, Latexify
	using Logging; Logging.disable_logging(Logging.Warn);
end;

# ╔═╡ c1026d6b-4e2e-4045-923e-eb5886b45604
TableOfContents()

# ╔═╡ 13cbb8af-9857-43c1-902c-52ccdbe02384
md"[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section6_logistic_regression.html)
"

# ╔═╡ 4ed16d76-2815-4f8c-8ab0-3819a03a1acc
md"""

# Bayesian linear regressions

Supervised learning in general is to predict a *dependent variable* ``Y`` based on *covariate features* ``X``. In other words, it tries to access how ``X`` affects ``Y``. Depending on the value type of the labelled targets ``Y``, the analysis can be further classified as *regression* and *classification*. When ``Y`` can take a spectrum of continuous values, such as real value, the learning task is known as regression. When ``Y`` take discrete choices, the task is commonly referred to as classification.


In this chapter, we consider the case of ``Y`` being real-valued, ``Y\in \mathbb{R}``, *i.e.* regression. And leave the discussion on other general cases such as ``Y\in \{\texttt{true}, \texttt{false}\}`` or ``Y\in \{0,1,2,\ldots\}`` in the following chapters. We will first briefly review the Frequentist's regression model and then move on to introduce the Bayesian treatment. From a modelling perspective, the Bayesian model is almost the same as the Frequentist's model except for the additional prior distributions. It turns out the additional prior not only offers a way to incorporate the modeller's prior expert knowledge but also has a *regularisation* effect in estimating the model parameters.

Lastly, we are going to see how to do practical Bayesian linear regression analysis with `Turing`. Some extensions are also considered at the end.
"""

# ╔═╡ c8547607-92be-4700-9468-863798c2ddce
md"""
## Frequentist's linear regression


Firstly, we recap the Frequentist's regression model. A typical regression model assumes each observation ``y_n`` is generated based on some deterministic transformation of the independent variable ``\mathbf{x}_n \in \mathbb{R}^D`` plus and some observation noise ``\varepsilon_n``:

```math
y_n =  \mu(\mathbf{x}_n) + \varepsilon_n
```
for ``n=1,\ldots, N``, where the observation noises are usually assumed to be white Gaussian noises:

$$\varepsilon_n \sim \mathcal N(0, \sigma^2).$$


The deterministic function ``\mu`` is usually assumed to be linear:

$$\mu(\mathbf x_n) = \beta_0 + \mathbf{x}_n^\top \boldsymbol{\beta}_1 ,$$ 

where ``\beta_0`` is called intercept and ``\boldsymbol{\beta}_1 \in \mathbb{R}^D`` determine linear relationship between ``\mathbf{x}`` and ``y``. Since ``\mu`` is a linear function, the regression is called **linear regression**. 
"""

# ╔═╡ 8f95b8c8-70a0-471a-be06-3b020db667b3
md"""
**Probabilistic reformulation.**
The Frequentist's model above can also be equivalently written as:

```math
p(y_n|\mathbf{x}_n, \boldsymbol{\beta}, \sigma^2) = \mathcal{N}(y_n; \beta_0+\mathbf{x}_n^\top \boldsymbol{\beta}_1, \sigma^2),
```
where ``\boldsymbol{\beta}^\top = [\beta_0, \boldsymbol{\beta}_1]`` is used to denote the both the intercept and regression parameters. In other words, a **likelihood model** for the observed data ``\mathbf{y} = \{y_n\}`` for ``n=1,\ldots, N``. And each ``y_n`` is Gaussian distributed with mean $\beta_0+\mathbf{x}_n^\top \boldsymbol{\beta}_1$ and variance $\sigma^2$. The model also assumes the independent variables ``\mathbf{x}_n`` are fixed (i.e. non-random).

By using matrix notation, the above model can be compactly written as:

```math
p(\mathbf{y}|\mathbf{X}, \boldsymbol{\beta}, \sigma^2) = \mathcal{N}_N(\mathbf{y}; \beta_0 \mathbf{1}_N + \mathbf{X} \boldsymbol{\beta}_1, \sigma^2\mathbf{I}_N),
```
where ``\mathbf{y} = [y_1, y_2,\ldots,y_N]^\top, \mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n]^\top``, ``\mathbf{1}_N=[1, \ldots, 1]^\top`` is a ``N\times 1`` column vector of ones and ``\mathbf{I}_{N}`` is a ``N\times N`` identity matrix.

The likelihood assumption is illustrated below for a simple linear regression model with a one-dimensional predictor, i.e. ``D=1``. Conditioning on where ``x_n`` is, ``y_n`` is Gaussian distributed with a mean determined by the regression line and an observation variance ``\sigma^2``.

"""

# ╔═╡ 3a46c193-5a25-423f-bcb5-038f3756d7ba
md"""

**OLS (MLE) Estimation*.** Frequentists estimate the model by using maximum likelihood estimation (MLE). The idea is to find point estimators ``\hat{\beta_0}, \hat{\boldsymbol{\beta}}_1, \hat{\sigma^2}`` such that the likelihood function is maximised. 
```math
\hat{\beta_0}, \hat{\boldsymbol{\beta}}_1, \hat{\sigma^2} \leftarrow \arg\max p(\mathbf y|\mathbf X, \beta_0, \boldsymbol{\beta}_1, \sigma^2)
```
It can be shown that the MLE are equivalent to the more widely known ordinary least square (OLS) estimators, in which we are minimising the sum of squared error

```math
\hat{\beta_0}, \hat{\boldsymbol{\beta}}_1 \leftarrow \arg\min \sum_{n=1}^N (y_n - \mathbf{x}_n^\top \boldsymbol{\beta}_1 -\beta_0)^2.
```

By taking the derivative and setting it to zero, the closed-form OLS estimator is:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y},
```

where we have assumed ``\boldsymbol{\beta}^\top = [\beta_0, \boldsymbol{\beta}_1]``, and the design matrix ``\mathbf{X}`` is augmented with ``\mathbf{1}_N`` as the first column (*i.e.* dummy variable for intercept).
"""

# ╔═╡ 1cae04d1-82ca-47c8-b1fa-7fa5a754d4b1
Foldable("Derivation details about the MLE.", md"
Due to the independence assumption,

```math
p(\mathbf{y}|\mathbf{X}, \boldsymbol{\beta}, \sigma^2) = \prod_{n=1}^N p({y}_n|\mathbf{x}_n, \boldsymbol{\beta}, \sigma^2)= \prod_{n=1}^N \mathcal{N}(y_n; \mathbf{x}_n^\top \boldsymbol{\beta}, \sigma^2)
```
Since the logarithm function is a monotonically increasing function, the maximum is invariant under the transformation. We therefore can maximise the log-likelihood function instead. And the log-likelihood function is

$$\begin{align}\mathcal{L}(\beta_0, \boldsymbol{\beta}, \sigma^2) &= \ln \prod_{n} \mathcal{N}(y_n; \mathbf{x}_n^\top \boldsymbol{\beta}, \sigma^2)\\
&= \sum_{n=1}^N \ln  \left\{ (2\pi)^{-\frac{1}{2}}({\sigma^2})^{-\frac{1}{2}} \exp\left ( -\frac{1}{2 \sigma^2} (y_n - \beta_0 - \mathbf{x}_n^\top \boldsymbol{\beta})^2\right )\right \} \\
&= {-\frac{N}{2}}\cdot 2\pi -\frac{N}{2}{\sigma^2} -\frac{1}{2 \sigma^2} \sum_{n=1}^N   (y_n - \beta_0 - \mathbf{x}_n^\top \boldsymbol{\beta})^2\end{align}$$

Maximising the above log-likelihood is the same as minimising its negation, which leads to the least squared error estimation:

```math
\begin{align}
\hat{\beta_0}, \hat{\boldsymbol{\beta}} &\leftarrow \arg\min -\mathcal{L}(\beta_0, \boldsymbol{\beta}, \sigma^2)  \\
&= \arg\min  \sum_{n=1}^N   (y_n - \beta_0 - \mathbf{x}_n^\top \boldsymbol{\beta})^2,
\end{align}
```
where we have assumed ``\sigma^2`` is known.

")

# ╔═╡ effcd3d2-ba90-4ca8-a69c-f1ef1ad697ab
md"We use `GLM.jl` to fit the OLS estimation based on the simulated data. The estimated model and true model is shown below."

# ╔═╡ af404db3-7397-4fd7-a5f4-0c812bd90c4a
begin
	Random.seed!(100)
	β₀, β₁, σ² = 3, 3, 0.5
	N = 50
	X = rand(N)
	μ = β₀ .+ β₁ * X 
	yy = μ + sqrt(σ²) * randn(N)
end;

# ╔═╡ c6938b7f-e6e5-4bea-a273-52ab3916d07c
md"""

**Example.** Consider a simple linear regression model with one predictor, i.e.
```math
y_n = \beta_0 + \beta_1 x_n + \varepsilon_n.
```

To better illustrate the idea, we have simulated a dataset with ``N``=$(N) observations and the regression is defined with the following parameters: ``\beta_0=3, \beta_1=3, \sigma^2=0.5``. The simulated dataset is plotted below.
"""

# ╔═╡ 3e98e9ff-b674-43f9-a3e0-1ca5d8614327
begin
	using GLM
	ols_simulated = lm([ones(N) X], yy)
end

# ╔═╡ f8040fa0-fdff-42da-8b9f-785627f51ee1
let
	p_lr = plot(title="Frequentist's linear regression's probabilistic model", framestyle = :origin,legend=:bottomright)
	β0, σ²0, N = [3, 3], 0.5, 250
	Random.seed!(100)
	X = rand(N)
	μ = β0[1] .+ β0[2] * X 
	yy = μ + sqrt(σ²0) * randn(N)
	plot!(X, yy, st=:scatter, label="")
	plot!([0,1], x->β0[1]+β0[2]*x, c= 1, linewidth=5, label=L"\mu(x) = \beta_0+\beta_1x")
	xis = [0, 0.25, 0.5, 0.75, 0.99, 0.4]
	for i in 1:length(xis)
		x = xis[i]
		μi = dot(β0, [1, x])
		xs_ = μi-3:0.01:μi+3
		ys_ = pdf.(Normal(μi, sqrt(σ²0)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		if i == length(xis)
			plot!([x], [μi], st=:sticks, markerstrokewidth =1, markershape = :diamond, c=:black, label="", markersize=4)
			plot!(ys_ .+x, xs_, c=:red, label=L"\mathcal{N}(\mu(x), \sigma^2)", linewidth=3)
		else
			plot!(ys_ .+x, xs_, c=:gray, label="", linewidth=1)
		end
		
	end
	old_xticks = xticks(p_lr[1])
	new_xticks = ([xis[end]], ["\$x\$"])
	keep_indices = findall(x -> all(x .≠ new_xticks[1]), old_xticks[1])
	merged_xticks = (old_xticks[1][keep_indices] ∪ new_xticks[1], old_xticks[2][keep_indices] ∪ new_xticks[2])
	xticks!(merged_xticks)
	p_lr	
end

# ╔═╡ af2e55f3-08b8-48d4-ae95-e65168a23eeb
let
	pred(x) = coef(ols_simulated)' * [1, x]
	scatter(X, yy, xlabel=L"x", ylabel=L"y", label="")
	plot!(0:0.1:1, pred, lw=2, label="OLS fit", legend=:topleft)
	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
end

# ╔═╡ 43191a7a-f1a2-41df-910d-cf85907e8f7a
md"""

## Bayesian linear regression model
"""

# ╔═╡ 98ef1cca-de03-44c2-bcf4-6e79da139e11
md"""

The Bayesian model reuses the frequentist's likelihood model for ``\mathbf{y}``:

```math
p(y_n|\mathbf{x}_n, \boldsymbol{\beta}, \sigma^2) = \mathcal{N}(y_n; \beta_0+\mathbf{x}_n^\top \boldsymbol{\beta}_1, \sigma^2),
```
where the model parameters are
* ``\beta_0\in \mathbb{R}`` -- intercept
* ``\boldsymbol{\beta}_1 \in \mathbb{R}^D`` -- regression coefficient vector
* ``\sigma^2\in \mathbb{R}^+`` -- Gaussian noise's variance


In addition, the Bayesian model also needs to impose a suitable prior for the unknowns

```math
p(\beta_0, \boldsymbol{\beta}_1, \sigma^2).
```

In most cases, we further assume the joint prior is independent, *i.e.*

```math
p(\beta_0, \boldsymbol{\beta}_1, \sigma^2)= p(\beta_0)p(\boldsymbol{\beta}_1)p( \sigma^2).
```



"""

# ╔═╡ a1421ccb-2d6e-4406-b770-ad7dff007c69
md"""

Based on their value ranges, a commonly used set of prior choices is summarised below.

**Prior for the intercept ``p(\beta_0)``** 

Since ``\beta_0`` takes an unconstrained real value, a common choice is Gaussian 

$$p(\beta_0) = \mathcal N(m_0^{\beta_0}, v_0^{\beta_0});$$ 

where the hyper-parameters ``m_0^{\beta_0}, v_0^{\beta_0}`` can be specified based on the data or independently.

*Data-dependent hyper-parameter:* e.g. ``\mathcal{N}(m_0^{\beta_0}={\bar{\mathbf{y}}}, v_0^{\beta_0}= 2.5^2 \sigma_{\mathbf{y}}^2 )``

* the corresponding prior on ``\beta_0`` centres around the sample average ``\bar{\mathbf{y}}`` with variance 2.5 times the standard deviation of ``\bar{\mathbf{y}}``
* covering a wide range of possible values, the prior is a weakly informative prior


*Data-independent hyper-parameter:* e.g. ``\mathcal{N}(m_0^{\beta_0}=0, v_0^{\beta_0}= 10^2)`` 


* the prior guess centres around ``0`` but with a great amount of uncertainty (very large variance)
* the zero prior mean here encourages the posterior shrinks towards 0 (but very weakly).


"""

# ╔═╡ e1552414-e701-42b5-8eaf-21ae04a829a8
md"""


**Prior for the coefficient ``p(\boldsymbol{\beta}_1)``** 


Since ``\boldsymbol{\beta}_1\in \mathbb{R}^D`` is an unconstrained real vector, a suitable prior choice is a D-dimensional multivariate Gaussian (or other similar distributions such as Student-``t``):

$$p(\boldsymbol{\beta}_1) = \mathcal N_{D}(\mathbf m_0^{\boldsymbol{\beta}_1}, \mathbf V_0^{\boldsymbol{\beta}_1}),$$


where the hyper-parameters are usually set data-independently:

* ``\mathbf{m}_0^{\boldsymbol{\beta}_1}=\mathbf{0}``, the prior encourages the posterior shrinks towards zeros, which has a regularisation effect
* ``\mathbf{V}_0^{\boldsymbol{\beta}_1} = v_0 \mathbf{I}_D``, *i.e.* a diagonal matrix with a common variance ``v_0``; 
    * the common variance ``v_0`` is often set to a large number, *e.g.``v_0=10^2``* to impose a vague prior. 


**Prior for the observation variance ``p(\sigma)``**


``\sigma^2 >0`` is a positive real number, a good choice is truncated real-valued distribution, where the negative part is truncated, such as ``\texttt{Half-Cauchy}`` (Some ``\texttt{Half-Cauchy}`` distributions are plotted below): 

$$p(\sigma^2) = \texttt{HalfCauchy}(s_0^{\sigma^2});$$ 

  * Cauchy distributions have heavy tails, making them suitable choices to express uncertain beliefs on the modelling parameter. And the hyperparameter ``s_0^{\sigma^2}`` controls the tail of a Cauchy. The larger the scale parameter ``s_0^{\sigma^2}`` is, the weaker the prior is. 
  * ``s_0^{\sigma^2} > 2`` usually is good enough.
"""

# ╔═╡ 9387dcb4-3f4e-4ec7-8393-30a483e00c63
let
	plot(-5:0.1:25, (x) -> pdf(truncated(Cauchy(0, 2), lower=0), x), lw=1.5, label=L"\texttt{HalfCauchy}(2.0)")
	plot!(-5:0.1:25, (x) -> pdf(truncated(Cauchy(0, 4), lower=0), x), lw=1.5, label=L"\texttt{HalfCauchy}(4.0)")
	plot!(-5:0.1:25, (x) -> pdf(truncated(Cauchy(0, 6), lower=0), x), lw=1.5, label=L"\texttt{HalfCauchy}(6.0)")
	plot!(-5:0.1:25, (x) -> pdf(truncated(Cauchy(0, 10), lower=0), x), lw=1.5, label=L"\texttt{HalfCauchy}(10.0)")
end

# ╔═╡ d3f4ac7b-1482-4840-b24b-d08066d1d70c
md"""

To put everything together, a fully-specified (data independent) Bayesian model is summarised below.


!!! infor "Bayesian linear regression"
	```math
	\begin{align}
	\text{Priors: }\;\;\;\;\;\;\beta_0 &\sim \mathcal{N}(m_0^{\beta_0}, v_0^{\beta_0})\\
	\boldsymbol{\beta}_1 &\sim \mathcal{N}(\mathbf{m}_0^{\boldsymbol{\beta}_1}, \mathbf{V}_0^{\boldsymbol{\beta}_1 })\\
	\sigma^2 &\sim \texttt{HalfCauchy}(s_0) \\
	\text{Likelihood: }\;\;\text{for } n &= 1,2,\ldots, N:\\
	\mu_n &=\beta_0 + \boldsymbol{\beta}_1^\top  \mathbf{x}_n \\
	y_n &\sim \mathcal{N}(\mu_n, \sigma^2).
	\end{align}
	```
"""

# ╔═╡ bab3a19c-deb0-4b1c-a8f9-f713d66d9199
md"""
### Connection to ridge regression 

"""

# ╔═╡ b433c147-f547-4234-9817-2b29e7d57219
md"""

Setting the prior's mean to zero, *i.e.* ``\mathbf{m}_0^{\beta_1}= \mathbf{0}``, seems an arbitrary choice. However, it can be justified from a regularisation perspective. For regression problems with many predictors, the regression model can be too flexible. In some extreme cases, the regression model can be under-determined, *i.e.* the number of predictors is greater than the number of observations. For problems like this, the ordinary OLS estimator is not stable or not even existed. 


By introducing a zero-mean Gaussian prior for ``\boldsymbol{\beta}``, the log posterior density becomes:

$$\begin{align}
\ln p(\boldsymbol{\beta}|\mathbf y, \mathbf{X}) &= \ln p(\boldsymbol{\beta}) +\ln p(\mathbf y|\boldsymbol{\beta}, \mathbf{X}) + C\\
&= -\frac{1}{2v_0} \|\boldsymbol{\beta}\|^2 - \frac{1}{2\sigma^2}\sum_{n=1}^N( y_n-\mathbf{x}_n^\top \boldsymbol{\beta})^2 + C
\end{align}$$

If one minimises the negative log posterior by:

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} \leftarrow \arg\min_{\boldsymbol{\beta}}\frac{\lambda}{2} \|\boldsymbol{\beta}\|^2 + \frac{1}{2}\sum_{n=1}^N( y_n-\mathbf{x}_n^\top \boldsymbol{\beta})^2,

```
where ``\lambda \triangleq \frac{\sigma^2}{v_0}``, and the estimator is called the **ridge estimator**. The corresponding regression model is Frequentist's **ridge regression**. Note that the first term serves as a penalty that regulates the estimation of ``\boldsymbol{\beta}`` such that large-magnitude estimators are discouraged. 
"""

# ╔═╡ 89414966-5447-4133-80cb-0933c8b0d5d0
Foldable("Derivation details.", md"

The prior is 

$$p(\boldsymbol{\beta}) = \mathcal{N}(\mathbf{0}, v_0 \mathbf{I}) = \frac{1}{\sqrt{(2\pi)^d |v_0 \mathbf{I}|}} \text{exp}\left (-\frac{1}{2v_0 }\boldsymbol{\beta}^\top \boldsymbol{\beta} \right ).$$ 

Take the log and add the log-likelihood function, we have

```math
\begin{align}
\ln p(\boldsymbol{\beta}|\mathbf y, \mathbf{X}) &= \ln p(\boldsymbol{\beta}) +\ln p(\mathbf y|\boldsymbol{\beta}, \mathbf{X}) + C\\
&= -\frac{1}{2v_0}\boldsymbol{\beta}^\top \boldsymbol{\beta} - \frac{1}{2\sigma^2}(\mathbf y-\mathbf{X} \boldsymbol{\beta})^\top (\mathbf y-\mathbf{X} \boldsymbol{\beta}) + C\\
&=-\frac{1}{2v_0} \|\boldsymbol{\beta}\|^2 - \frac{1}{2\sigma^2}\sum_{n=1}^N( y_n-\mathbf{x}_n^\top \boldsymbol{\beta})^2.
\end{align}
```


")

# ╔═╡ 824aa830-c958-4063-9ef7-39eeb743fc06
md"""

### Posterior distribution in analytical form*
"""

# ╔═╡ d825e29d-3e0b-4c25-852f-0c9a544aa916
md"""

After specifying the Bayesian model, what is left is to apply Bayes' rule to find the posterior distribution:

```math
p(\boldsymbol{\beta}, \sigma^2|\mathcal{D}) \propto p(\boldsymbol{\beta}, \sigma^2) p(\mathbf{y}|\boldsymbol{\beta}, \sigma^2,\mathbf{X}),
```
where ``\mathcal{D}`` denotes all the observed data *i.e.* ``\mathcal{D} = \{\mathbf{X}, \mathbf{y}\}``, and ``\boldsymbol{\beta}^\top = [\beta_0, \boldsymbol{\beta}_1]`` denotes the concatenated regression parameter vector. Due to the independence assumption, the joint prior for ``\boldsymbol{\beta}`` can be formed as a ``D+1`` dimensional Gaussian with mean ``\mathbf{m}_0^\top = [m_0^{\beta_0}, \mathbf{m}_0^{\beta_1}]`` and variance ``\mathbf{V}_0 = \text{diag}\{v_0^{\beta_0}, \mathbf{V}_0^{\boldsymbol{\beta}_1}\}.``

Unfortunately, the full joint posterior distribution has no closed-form analytical form, and an approximation method such as MCMC is required to do a full-scale analysis. However, if the observation variance ``\sigma^2`` is assumed known, the *conditional posterior* ``p(\boldsymbol{\beta}|\mathcal{D}, \sigma^2)`` can be computed exactly. And it can be shown that the posterior admits a multivariate Gaussian form:

```math
p(\boldsymbol{\beta}|\mathcal{D}, \sigma^2) =\mathcal{N}_{D+1}(\mathbf{m}_N, \mathbf{V}_N),
```

where 


```math
\mathbf{m}_N = \mathbf{V}_N\left (\mathbf{V}_0^{-1}\mathbf{m}_0 +\frac{1}{\sigma^2}\mathbf{X}^\top\mathbf{y}\right ) ,\;\;\mathbf{V}_N =\left (\mathbf{V}_0^{-1} + \frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}^\top\right )^{-1}.
```


It is one of very few models that the posterior can be evaluated in closed form.
"""

# ╔═╡ f6cedb98-0d29-40f0-b9f3-430dd283fa36
md"""

**Demonstration:**
An animation is created below to demonstrate the posterior update equation. The toy dataset is reused here. Recall the true parameters are ``\beta_0=\beta_1=3``. A zero-mean vague Gaussian prior is placed on ``\boldsymbol{\beta}``:

```math
p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}\right) = \mathcal{N}\left (\begin{bmatrix}0\\ 0\end{bmatrix}, \begin{bmatrix}10^2& 0 \\ 0 & 10^2\end{bmatrix}\right).
```

The posterior distribution is then updated sequentially with the first 10 observations. As can be observed, initially the prior distribution is circular and covers a wide range of possible values. With more data observed, the posterior quickly converges to the posterior centre: ``[3,3]^\top``. Also, note the shrinking posterior variance (or increasing estimation precision) as more data is observed.
"""

# ╔═╡ 71bb2994-7fd2-4e11-b2f1-d88b407f86c1
let
	xs = range(-10, 10, 200)
	ys = range(-10, 10, 200)
	m₀, V₀ = zeros(2), 10^2 * Matrix(1.0I,2,2)
	prior = heatmap(xs, ys, (x,y)-> pdf(MvNormal(m₀, V₀), [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1")
	# lik1 = heatmap(xs, ys, (x,y)-> pdf(Normal(x + y * X[1] , sqrt(σ²)), yy[1]), levels=10,  colorbar=false, ratio=1, color= :jet1, xlim=[-15, 15], xlabel=L"\beta_0", ylabel=L"\beta_1")
	function seq_update(x, y, m0, V0, σ²)
		xx = [1  x]
		mn = m0 + V0 * xx'* (dot(xx, V0, xx') + σ²)^(-1)*(y - dot(xx, m0) )
		Vn = inv(1/σ²* xx'* xx + inv(V0))
		return mn[:], Symmetric(Vn)
	end


	posts = [prior]
	anim = @animate for i in 1:10	
		post = heatmap(xs, ys, (x,y)-> pdf(MvNormal(m₀, V₀), [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with " *L"N=%$(i-1)"*" data")
		m₀, V₀ = seq_update(X[i], yy[i], m₀, V₀, σ²)
		push!(posts, post)
	end

	gif(anim, fps=1)
end

# ╔═╡ d02d5490-cf53-487d-a0c6-651725600f52
md"""

**Interpretation:**
The posterior's parameter provides us with some insights into what Bayesian computation is doing. If we assume the matrix inverse ``(\mathbf{X}^\top\mathbf{X})^{-1}`` exists, 
the posterior's mean can be rewritten as 


```math

\mathbf{m}_N = \left (\mathbf{V}_0^{-1} + \frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}\right )^{-1}\left (\mathbf{V}_0^{-1}\mathbf{m}_0 +\frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}\hat{\boldsymbol{\beta}}\right ) .

```

Defining ``\tilde{\mathbf{V}}^{-1} = \frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}``, the above formula becomes


```math
\mathbf{m}_N = \left (\mathbf{V}_0^{-1} + \tilde{\mathbf{V}}^{-1}\right )^{-1}\left (\mathbf{V}_0^{-1}\mathbf{m}_0 +\tilde{\mathbf{V}}^{-1}\hat{\boldsymbol{\beta}}\right ) .

```
Therefore, the posterior mean is a matrix-weighted average between the prior guess ``\mathbf{m}_0`` and the MLE estimator ``\hat{\boldsymbol{\beta}}``; and the weights are ``\mathbf{V}_0^{-1}``, the precision of the prior guess, and ``\tilde{\mathbf{V}}^{-1}`` the precision for the MLE. If the prior mean is zero, *i.e.* ``\mathbf{m}_0 =\mathbf{0}``, the posterior mean as an average will shrink towards ``\mathbf{0}``.

This can be understood better to consider some de-generate examples. Assume ``D=1``, and both the intercept and observation noise's variance are known, say ``\beta_0=0, \sigma^2=1``. The posterior degenerates to 


```math
\begin{align}
m_N &= \frac{v_0^{-1}}{v_0^{-1} + \tilde{v}^{-1}}m_0 + \frac{\tilde{v}^{-1} }{v_0^{-1} + \tilde{v}^{-1}}\hat{\beta}_1\\
v_N &= \frac{1}{ v_0^{-1} + \tilde{v}^{-1}}
\end{align}
```
where ``\tilde{v}^{-1} = \sum_n x_n^2`` by definition. 

Note that if we assume ``m_0=0``, and the prior precision ``v_0^{-1}`` gets large, say ``v_0^{-1} \rightarrow \infty`` (in other words, we strongly believe the slope is zero), the posterior mean ``m_N`` will get closer to zero: ``m_N\rightarrow m_0=0``.
Also note the posterior variance ``v_N`` is reduced in comparison with the prior variance ``v_0``. It makes sense since the posterior update reduces the estimation uncertainty.
"""

# ╔═╡ 59dd8a13-89c6-4ae9-8546-877bb7992570
md"""
## Implementation in `Turing`



Now we move on to show how to implement the modelling and inference in `Turing`. For simplicity, we consider simple linear regression, *i.e.* regression with one predictor, with simulated data first and move on to see a multiple regression example by using a real-world dataset.



### Simple linear regression

The corresponding Bayesian simple linear regression model is a specific case of the general model in which ``D=1``. The model can be specified as:


```math
\begin{align}
\text{Priors: }\;\;\;\;\;\;\beta_0 &\sim \mathcal{N}(0, v_0^{\beta_0})\\
\beta_1 &\sim \mathcal{N}(0, v_0^{\beta_1})\\
\sigma^2 &\sim \texttt{HalfCauchy}(s_0) \\
\text{Likelihood: }\;\;\text{for } n &= 1,2,\ldots, N:\\
\mu_n &=\beta_0 + \beta_1 x_n \\
y_n &\sim \mathcal{N}(\mu_n, \sigma^2).
\end{align}
```

Note that we have just replaced all the multivariate assumptions for ``\beta_1`` with its uni-variant equivalent.
"""

# ╔═╡ 632575ce-a1ce-4a36-95dc-010229367446
md"""

The model can be "translated" to `Turing` literally. Note that ``\texttt{HalfCauchy}`` distribution is a Cauchy distribution truncated from zero onwards, which can be implemented in `Julia` by:

```julia
truncated(Cauchy(0, s₀), lower=0)  # HalfCauchy distribution with mean 0 and scale s₀ 
```
"""

# ╔═╡ c0f926f1-85e6-4d2c-8e9a-26cd099fd600
md"""

In terms of the prior parameters, we have set weak priors as default. For example, the ``\texttt{HalfCauchy}(s_0=5)`` prior for ``\sigma^2`` easily covers the true value, which is ``0.5``. 

And the prior variances of the regression parameters are set to ``v_0^{\beta_0}= v_0^{\beta_1}=10^2``, leading to a very vague prior (and the true parameters are well covered within the prior's density area).

"""

# ╔═╡ e9bb7a3c-9143-48c5-b33f-e7d6b48cb224
@model function simple_bayesian_regression(Xs, ys; v₀ = 10^2, V₀ = 10^2, s₀ = 5)
	# Priors
	# Gaussian is parameterised with sd rather than variance
	β0 ~ Normal(0, sqrt(v₀)) 
	β ~ Normal(0, sqrt(V₀))
	# Half-Cauchy prior for the observation variance
	σ2 ~ truncated(Cauchy(0, s₀), lower=0)
	# calculate f(x) = β₀ + βx for all observations
	# use .+ to broadcast the intercept to all 
	μs = β0 .+ β * Xs
	
	# Likelihood
	for i in eachindex(ys)
		# Gaussian in `Distributions.jl` is parameterised by std σ rather than variance
		ys[i] ~ Normal(μs[i], sqrt(σ2))
	end
end

# ╔═╡ 1ef001cc-fe70-42e5-8b97-690bb725a734
md"""

Next, we use the above Turing model to infer the simulated dataset. A Turing model is first instantiated with the simulated data and then MCMC sampling algorithms are used to draw posterior samples.
"""

# ╔═╡ 4ae89384-017d-4937-bcc9-3d8c63edaeb5
begin
	Random.seed!(100)
	sim_data_model = simple_bayesian_regression(X, yy)
	chain_sim_data = sample(sim_data_model, NUTS(), MCMCThreads(), 2000, 3; discard_initial=500)
end;

# ╔═╡ 1761f829-6c7d-4186-a66c-b347be7c9a15
md"""
Next, we inspect the chain first to do some diagnostics to see whether the chain has converged. 
"""

# ╔═╡ d57a7a26-a8a5-429d-b1b9-5e0499549544
Turing.summarystats(chain_sim_data)

# ╔═╡ ba598c45-a76e-41aa-bfd6-a64bb6cea875
md"""Based on the fact that `rhat < 1.01` and the `ess` count, the chain has converged well. We can also plot the chain to visually check the chain traces."""

# ╔═╡ 391a1dc8-673a-47e9-aea3-ad79f366460d
md"""
**Result analysis** 
"""

# ╔═╡ d3d8fe25-e674-42de-ac49-b83aac402e2d
md"Recall the true parameters are ``\beta_0=3, \beta_1=3, \sigma^2=0.5``. And the inferred posteriors are summarised below. 
"

# ╔═╡ 748c1ff2-2346-44f4-a799-75fb2002c6fc
Turing.summarystats(chain_sim_data) |> DataFrame

# ╔═╡ 33399b63-dc7d-4690-aecc-c807a3ac69d5
begin
	ss = Turing.summarystats(chain_sim_data)
	b0 = ss[@varname(β0), stat=:mean]
	b1 = ss[@varname(β), stat=:mean] 
end;

# ╔═╡ 08f7be6d-fda9-4013-88f5-92f1b5d26336
md"""
The posterior's means for the three unknowns are around $(round(b0; digits=2)) and $(round(b1; digits=2)), which are almost the same as the OLS estimators, which are expected since very weak-informative priors have been used.

!!! information "Bayesian inference under weak priors"
	As a rule thumb, when non-informative or weak-informative priors are used, the Bayesian inference's posterior mean should be similar to the Frequentist estimators.
"""

# ╔═╡ ab77aef9-920c-423a-9ce0-2b5adead1b7f
begin
	density(chain_sim_data)
end

# ╔═╡ b1d7e28a-b802-4fa1-87ab-7df3369a468a
# md"""
# The above density plots show the posterior distributions. It can be observed the true parameters are all within good credible ranges. 

# The following diagram shows the model ``\mu = \beta_0 + \beta_1 x`` estimated by the Bayesian method:
# * The thick red line shows the posterior mean of the Bayesian model
# * The lighter lines are some posterior samples, which also indicates the uncertainty about the posterior distribution (the true model is within the prediction)
#   * the posterior mean is simply the average of all the posterior samples.
# """

# ╔═╡ 4293298e-52a5-40ca-970d-3f17c2c89adb
# let
# 	parms_turing = describe(chain_sim_data)[1][:, :mean]
# 	β_samples = Array(chain_sim_data[[:β0, :β]])
# 	pred(x) = parms_turing[1:2]' * [1, x]
# 	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="")
# 	plot!(0:0.1:1, pred, lw=2, label=L"\mathbb{E}[\mu|\mathcal{D}]", legend=:topleft)
# 	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
# 	plot!(0:0.1:1, (x) -> β_samples[1, 1] + β_samples[1, 2] *x, lw=0.5, lc=:gray, label=L"\mu^{(r)} \sim p(\mu|\mathcal{D})", legend=:topleft)
# 	for i in 2:100
# 		plot!(0:0.1:1, (x) -> β_samples[i, 1] + β_samples[i, 2] *x, lw=0.15, lc=:gray, label="", legend=:topleft)
# 	end
# 	plt
# end

# ╔═╡ 860c4cf4-59b2-4036-8a30-7fbf44b18648
md"""

#### Predictive checks

To make sure our Bayesian model assumptions make sense, we should always carry out predictive checks. This is relatively straightforward to do the checks with `Turing`. Recall the procedure is to first simulate multiple pseudo samples ``\{\mathbf{y}_{pred}^{(r)}\}`` based on the posterior predictive distribution 

```math
\mathbf{y}_{pred} \sim p(\mathbf{y}|\mathcal{D}, \mathbf{X}),
```

and then the empirical distribution of the simulated data is compared against the observed. If the model assumptions are reasonable, we should expect the observed lies well within the possible region of the simulated.
"""

# ╔═╡ 74a9a678-60b9-4e3f-97a2-56c8fdc7094f
md"""

To simulate the pseudo data, we first create a dummy `Turing` with the targets filled with `missing` types. And then use `predict()` method to simulate the missing data.
"""

# ╔═╡ 435036f6-76fc-458b-b0eb-119de02eabb7
pred_y_matrix = let
	# define the predictive model by passing `missing` targets y such that the values will be samples
	pred_model = simple_bayesian_regression(X, Vector{Union{Missing, Float64}}(undef, length(yy)))
	# simulate the predictive pseudo observations
	preds = predict(pred_model, chain_sim_data) 
	reshape(preds[@varname(ys), stack=true], 6000, 50)
end;

# ╔═╡ 6eff54f2-d2e0-4c18-8eda-3be3124b16a0
md"""
**Kernel density estimation** check. One of the possible visual checks is to use Kernel density estimation (KDE), which is demonstrated below for our model. It can be seen the simulated data agree with the observed.
"""

# ╔═╡ f11a15c9-bb0b-43c1-83e6-dce55f2a772f
let
	pred_check_kde=density(yy, lw=2, label="Observed KDE", xlabel=L"y", ylabel="density")

	for i in 1:20
		label_ = i ==1 ? "Simulated KDE" : ""
		density!(pred_y_matrix[i,:], lw=0.2, lc=:grey, label=label_)
	end
	pred_check_kde
end

# ╔═╡ 03b571e6-9d35-4123-b7bb-5f3b31558c9e
md"""
**Summary statistics** check. Another common visual check is to plot the summary statistics of the simulated data, such as mean and standard deviation (std). Again, the simulated data is similar to the observed.


"""

# ╔═╡ 7cffa790-bf2d-473f-95cc-ed67802d7b4f
let
	# plot one third of the samples for a cleaner visualisation
	first_N = Int(floor(size(pred_y_matrix)[1]/3))
	df_pred_stats = DataFrame(mean= mean(pred_y_matrix, dims=2)[1:first_N], logstd = log.(std(pred_y_matrix, dims=2)[1:first_N]))

	@df df_pred_stats scatter(:mean, :logstd, label="Simulated", ms=3, malpha=0.3, xlabel=L"\texttt{mean}(\mathbf{y})", ylabel=L"\ln(\texttt{std}(\mathbf{y}))")
	scatter!([mean(yy)], [log(std(yy))], label="Observed")
end

# ╔═╡ 21aaa2db-6df0-4573-8951-bdfd9b6be6f9
md"""
### A multiple linear regression example

Consider the *Advertising* dataset which is described in the book [Introduction to Statistical Learning](https://hastie.su.domains/ISLR2/ISLRv2_website.pdf). The dataset records how advertising on TV, radio, and newspaper affects sales of a product. 
"""

# ╔═╡ b1401e7d-9207-4e31-b5ce-df82c8e9b069
begin
	Advertising = DataFrame(CSV.File(download("https://www.statlearning.com/s/Advertising.csv")))
	first(Advertising, 5)
end

# ╔═╡ 241faf68-49b2-404b-b5d5-99061d1dd2a7
md"""
First, we shall plot the data to get a feeling about the dataset. By checking the correlations, it seems both *TV, radio* are effective methods and the correlation between *newspaper* and *sales* seems not strong enough.

$(begin
	@df Advertising cornerplot([:sales :TV :radio :newspaper], compact = true)
end)

"""

# ╔═╡ 796ad911-9c95-454f-95f4-df5370b2496a
md"""
To understand the effects of the three different advertising methods, we can apply **multiple linear regression**. The multiple linear regression is formulated as:

```math
\texttt{sales} = \beta_0 + \beta_1 \times \texttt{TV} + \beta_2 \times \texttt{radio} + \beta_3 \times \texttt{newspaper} + \varepsilon
```

"""

# ╔═╡ ce97fbba-f5d0-42de-8b67-827c3478d8b0
md"A Frequentist's model is fitted by using `GLM` as a reference first."

# ╔═╡ f3b6f3ab-4304-4ef4-933c-728841c17998
ols_advertising = lm(@formula(sales ~ TV+radio+newspaper), Advertising)

# ╔═╡ ab51e2c3-dbd4-43e1-95b5-a875954ac532
md"According to the OLS fit, we can observe that the newspaper's effect is not statistically significant since its lower and upper 95% confidence interval encloses zero.

We now move on to fit a Bayesian model. A `Turing` multiple linear regression model is specified below. Note that the model is almost the same as the simple regression model. The only difference is we allow the number of predictors to vary.
"

# ╔═╡ e1db0ee1-1d91-49c4-be6c-1ea276750bfc
# Bayesian linear regression.
@model function general_linear_regression(X, ys; v₀ = 10^2, V₀ = 10^2, s₀ = 5)
    # Set variance prior.
    σ² ~ truncated(Cauchy(0, s₀), 0, Inf)
    # Set intercept prior.
    intercept ~ Normal(0, sqrt(v₀))
    # Set the priors on our coefficients.
    nfeatures = size(X, 2)
    coefficients ~ MvNormal(nfeatures, sqrt(V₀))
    # Calculate all the mu terms.
    μs = intercept .+ X * coefficients
	for i in eachindex(ys)
		ys[i] ~ Normal(μs[i], sqrt(σ²))
	end
end

# ╔═╡ 80727f81-2440-4da5-8a8b-38ebfdc4ddc9
md"We then fit the Bayesian model with the advertisement dataset:"

# ╔═╡ 929330bf-beb5-4e42-8668-2a10adf13972
chain_adv = let
	xs = Advertising[:,2:4] |> Array # |> cast the dataframe to an array
	advertising_bayes_model = general_linear_regression(xs, Advertising.sales)
	Random.seed!(100)
	chain_adv = sample(advertising_bayes_model, NUTS(), MCMCThreads(), 2000, 4)
	# replacenames(chain_adv,  Dict(["coefficients[1]" => "TV", "coefficients[2]" => "radio", "coefficients[3]" => "newspaper"]))
end;

# ╔═╡ 9668ab51-f86d-4af5-bf1e-3bef7081a53f
Turing.summarystats(chain_adv) |> DataFrame

# ╔═╡ 6a330f81-f581-4ccd-8868-8a5b22afe9b8
md"As can be seen from the summary, the Bayesian results are very similar to the Frequentists again: the posterior means are close to the OLS estimators. By checking the posterior's 95% credible intervals, we can conclude again newspaper (with a 95% credible interval between -0.013 and 0.011) is not an effective method, which is in agreement with the frequentist's result. The trace and density plot of the posterior samples are listed below for reference."

# ╔═╡ b1f6c262-1a2d-4973-bd1a-ba363bcc5c41
plot(chain_adv)

# ╔═╡ 659a3760-0a18-4a95-8168-fc6ca237c4d5
md"""

## Extensions

For both examples we have seen so far, the Bayesian method and the frequentist's method return very similar results. So the reader may wonder why to bother learning the Bayesian inference. The benefit of the Bayesian approach lies in its flexibility: by modifying the model as the modeller sees fit, everyone can do statistical inference like a statistician. With the help of `Turing`, the modeller only needs to do the modelling and leave the computation to the powerful MCMC inference engine.


### Heterogeneous observation σ

To demonstrate the flexibility of the Bayesian approach, we are going to make improvements to the Advertising data's regression model. Here, we consider the effect of *TV* on *sales* only.
"""

# ╔═╡ 9cb32dc0-8c0c-456e-bbcb-7ff6ae63da60
ols_tv = lm(@formula(sales ~ TV), Advertising);

# ╔═╡ 2e7f780e-8850-446e-97f8-51ec26f5e36a
let
	plt=@df Advertising scatter(:TV, :sales, xlabel="TV", ylabel="Sales", label="")
	xis = [25, 150, 280]
	σ²0 = [5, 12, 30]
	pred(x) = coef(ols_tv)' * [1, x]
	for i in 1:length(xis)
		x = xis[i]
		μi = pred(x)
		xs_ = μi-2*sqrt(σ²0[i]):0.01:μi+2*sqrt(σ²0[i])
		ys_ = pdf.(Normal(μi, sqrt(σ²0[i])), xs_)
		ys_ = 20 * ys_ ./ maximum(ys_)
		plot!(ys_ .+x, xs_, c=2, label="", linewidth=2)
	end
	plt
end

# ╔═╡ 65d702d6-abec-43a1-89a8-95c30b3e748a
md"It can be observed that the observation noise's scale ``\sigma^2`` is not constant across the horizontal axis. With a larger investment on TV, the sales are increasing but also the variance of the sales (check the two Gaussians' scales at two ends of the axis). Therefore, the old model which assumes a constant observation variance scale is not a good fit for the data. 
"

# ╔═╡ 611ad745-f92d-47ac-8d58-6f9baaf3077c
let
	# error = Advertising.sales - [ones(length(Advertising.TV)) Advertising.TV] * coef(ols_tv)
	# σ²_ml = sum(error.^2) / (length(Advertising.sales)-2)
	pred(x) = coef(ols_tv)' * [1, x]
	@df Advertising scatter(:TV, :sales, xlabel="TV", ylabel="Sales", label="")
	plot!(0:1:300, pred, lw=2, lc=2, label="OLS", legend=:topleft)
	test_data = DataFrame(TV= 0:300)
	pred_ = predict(ols_tv, test_data, interval = :prediction, level = 0.9)
	# plot!(0:1:300, (x) -> pred(x) + 2 * , lw=2, label="OLS", legend=:topleft)
	# plot!(0:1:300, pred, lw=2, label="OLS", legend=:topleft)
	plot!(0:300, pred_.prediction, linewidth = 0.1,
        ribbon = (pred_.prediction .- pred_.lower, pred_.upper .- pred_.prediction), label=L"90\% "* " OLS prediction interval")
end

# ╔═╡ ddb0d87e-cf0a-4b09-9bb6-4ee267fcdb9d
md"""

The ordinary OLS estimated model with the old model assumption is plotted above. It can be observed that the prediction interval is too wide at the lower end of the input scale but too small on the upper side.

"""

# ╔═╡ 1954544c-48b4-4997-871f-50c9bfa402b7
md"""

**A better "story".**
One possible approach to improve the model is to assume the observation scale ``\sigma`` itself is a function of the independent variable:

```math
\sigma(x) = r_0 + r_1  x,
```
If the slope ``r_1>0``, then the observation scale ``\sigma(x)`` will steadily increase over the input ``x``.

However, note that ``\sigma`` is a constrained parameter, which needs to be strictly positive. To make the transformation valid across the input ``x``, we model ``\sigma``'s transformation ``\rho`` instead. 

To be more specific, we define the following deterministic transformations between ``\rho`` and ``\sigma``:
```math
\begin{align}
&\rho(x) = \gamma_0 + \gamma_1  x,\\
&\sigma = \ln(1+\exp(\rho)).
\end{align}
``` 

The unconstrained observation scale ``\rho \in \mathbb{R}`` is linearly dependent on the input ``x``; and a ``\texttt{softplus}`` transformation ``\ln(1+\exp(\cdot))`` is then applied to ``\rho`` such that the output is always positive.

The second transformation ``\ln(1+ \exp(\cdot))`` is widely known as soft-plus method. The function is plotted below.

$(begin
plot(-10:.1:10, (x)-> log(1+exp(x)), label=L"\ln(1+\exp(x))", legend=:topleft, lw=2, size=(400,300))
end)

``\texttt{Softplus}`` function takes any real-value as input and outputs a positive number, which satisfies our requirement. Note that one can also simply use ``\exp(\cdot)`` instead, which is also a ``\mathbb{R} \rightarrow \mathbb{R}^+`` transformation.
"""

# ╔═╡ 85ae4791-6262-4f07-9796-749982e00fec
md"""

The full Bayesian model is specified below:


```math
\begin{align}
\text{Priors: }\beta_0 &\sim \mathcal{N}(m_0^{\beta_0}, v_0^{\beta_0})\\
\gamma_0 &\sim \mathcal{N}(m_0^{\gamma_0}, v_0^{\gamma_0}) \\
\beta_1 &\sim \mathcal{N}(m_0^{\beta_1}, v_0^{\beta_1})\\
\gamma_1 &\sim \mathcal{N}(m_0^{\gamma_1}, v_0^{\gamma_1}) \\
\text{Likelihood: for } n &= 1,2,\ldots, N:\\
\mu_n &=\beta_0 + \beta_1 x_n \\
\rho_n &= \gamma_0 + \gamma_1 x_n\\
\sigma_n &= \log(1+\exp(\rho_n))\\
y_n &\sim \mathcal{N}(\mu_n, \sigma_n^2),
\end{align}
```

which can be translated to `Turing` as follows. For simplicity, we have assumed all priors are with mean zero and variance ``10^2``. 
"""

# ╔═╡ 08c52bf6-0920-43e7-92a9-275ed298c9ac
@model function hetero_var_model(X, y; v₀=100)
	β₀ ~ Normal(0, sqrt(v₀))
	β₁ ~ Normal(0, sqrt(v₀))
	γ ~ MvNormal(zeros(2), sqrt(v₀))
	μs = β₀ .+ β₁ .* X
	ρs = γ[1] .+ γ[2] .* X
	σs = log.(1 .+  exp.(ρs))
	for i in eachindex(y)
		y[i] ~ Normal(μs[i], σs[i])
	end
	return (;μs, ρs, σs)
end

# ╔═╡ 11021fb7-b072-46ac-8c23-f92825182c8c
begin
	Random.seed!(100)
	model2 = hetero_var_model(Advertising.TV, Advertising.sales; v₀=10)
	chain2 = sample(model2, NUTS(), MCMCThreads(), 2000, 4)
end;

# ╔═╡ cece17c2-bacb-4b4a-897c-4116688812c6
Turing.summarystats(chain2)

# ╔═╡ 29175b82-f208-4fcf-9704-4a1996cc6e3c
begin
	# chain2_ = replacenames(chain2, Dict("β₀" => L"\beta_0", "β₁" => L"\beta_1"))
	plot(chain2)
end

# ╔═╡ 05b90ddb-8479-4fbf-a469-d5b0bf4a91c8
md"""
#### Use `generated_quantities()`

**Analyse internel hidden values.** ``\sigma, \mu`` are of importance in evaluating the model. They represent the observation noise scale and regression-line's expectation respectively. Recall that they depend on the independent variable *TV* and the unknown model parameters ``\gamma_0, \gamma_1, \beta_0, \beta_1`` via deterministic functions
```math
\sigma(\texttt{TV}) = \ln(1+ \exp(\gamma_0 +\gamma_1 \texttt{TV})).
```

and 

```math
\mu(\texttt{TV}) = \beta_0 +\beta_1 \texttt{TV}.
``` 

To gain deeper insights into the inference results, we can analyse the posterior distributions of ``\mu`` and ``\sigma`` directly: i.e.
```math
p(\sigma|\texttt{TV}, \mathcal{D}),\;\; p(\mu|\texttt{TV}, \mathcal{D}). 
```

Based on the Monte Carlo principle, their posterior distributions can be empirically approximated based on their posterior samples, which can be subsequently computed based on the MCMC samples ``\{\gamma_0^{(r)}, \gamma_1^{(r)}\}_{r=1}^R``, and ``\{\beta_0^{(r)},\beta_0^{(r)}\}_{r=1}^R`` respectively.


Take ``\sigma`` for example, its posterior can be approximated based on ``\{\gamma_0^{(r)}, \gamma_1^{(r)}\}_{r=1}^R`` by an empirical distribution:

```math
p(\sigma |\texttt{TV}, \mathcal{D}) \approx \frac{1}{R} \sum_{r=1}^R \delta_{\sigma^{(r)}_{\texttt{TV}}},
```
where 


$$\sigma^{(r)}_{\texttt{TV}} =\ln(1+ \exp( \gamma_0^{(r)} +\gamma_1^{(r)} \texttt{TV})).$$ and ``\delta_{x}`` is a Dirac measured at ``x``. The posterior mean can then be approximated by the Monte Carlo average:

```math
\mathbb{E}[\sigma|\mathcal{D}, \texttt{TV}] \approx \frac{1}{R} \sum_{r=1}^R \sigma^{(r)}_{\texttt{TV}}.
```

To make the idea concrete, we manually implement and plot the posterior approximation below. 


"""

# ╔═╡ 2d7f773e-9fa1-475d-9f74-c13908209aeb
begin
	σ(x; γ) = log(1.0 +  exp(γ[1] + γ[2] * x))
	# obtain the posterior mean of the γ
	# γ_mean = describe(chain2)[1][:,:mean][3:4]
	γ1_mean = mean(chain2)[@varname(γ[1])]
	γ2_mean = mean(chain2)[@varname(γ[2])]
	γ_mean = [γ1_mean, γ2_mean]
	# obtain γ samples from the chain object
	# γ_samples = Array(group(chain2, :γ))
	γ_samples = reshape(chain2[@varname(γ), stack=true], 2000*4,:)
	tv_input = 0:300
	# # calculate each σⁱ
	σ_samples = map(γⁱ -> σ.(tv_input; γ = γⁱ), eachrow(γ_samples))
	# # E[σ|𝒟]: the Monte Carlo average
	σ_mean = mean(σ_samples)
end;

# ╔═╡ 8a0692db-6b85-42f6-8893-4219d58b1032
let
	# Plotting
	plt = plot(tv_input, σ_mean, lw=3, xlabel="TV", ylabel=L"\sigma", label=L"\mathbb{E}[\sigma|\mathcal{D}]", legend=:topleft)
	plot!(tv_input, σ_samples[1], lw=0.2, lc=:gray, xlabel="TV", ylabel=L"\sigma", label=L"\sigma^{(r)}\sim p(\sigma|\mathcal{D})", legend=:topleft)
	for i in 2:25
		plot!(tv_input, σ_samples[i], lw=0.2, lc=:gray, label="")
	end
	plt
end

# ╔═╡ 3c3f6378-ed20-4f40-a780-f9329ace34bc
md"""

It is not practical to implement the approximation for every single internal variable manually. Fortunately, `Turing` provides us with an easy-to-use method:
```
generated_quantities(model, chain)
```  The method takes a `Turing` model and a chain object as input arguments and it returns all internal variables' samples, which are returned at the end of the `Turing` model. For example, to obtain the posterior samples of the hidden variables ``\mu`` and ``\sigma``, simply issue commands:
"""

# ╔═╡ a43189a3-a94f-4e69-85b1-3a586b2cc0eb
begin
	gen_adv = generated_quantities(model2, chain2)
	σs = map((x) -> x.σs, gen_adv)
	μs = map((x) -> x.μs, gen_adv)
end;

# ╔═╡ 4bf4c216-fefb-40fc-9d36-eaa82ff5454b
md"""

And we can plot the posterior samples to visually interpret the results. It is worth noting that the observation's scale ``\sigma`` increases as investment on *TV* increases, which is exactly what we want to model. And also note the uncertainty about the observation scale is not uniform over the scale of the input *TV*. The uncertainty about ``\sigma`` also increases as *TV* gets larger (check the gray lines' spread)."""

# ╔═╡ 5a507f39-a6be-43aa-a909-4c87005ad1d2
begin
	order = sortperm(Advertising.TV)
	plt = plot(Advertising.TV[order], mean(σs)[order], lw=4,  label=L"\mathbb{E}[\sigma|\mathcal{D}]", legend=:topleft, xlabel="TV", ylabel=L"\sigma",ribbon = (2*std(σs)[order], 2*std(σs)[order]))
	plot!(Advertising.TV[order], σs[:][1][order], lw=0.2, lc=:gray, label=L"\sigma^{(r)}\sim p(\sigma|\mathcal{D})")
	for i in 50:75
		plot!(Advertising.TV[order], σs[:][i][order], lw=0.2, lc=:gray, label="")
	end
	plt
end

# ╔═╡ db76ed48-f0c0-4664-b2cf-d47be7faaa3f
let
	plt = plot(Advertising.TV[order], mean(μs)[order], lw=4, lc=2, xlabel="TV", label=L"\mathbb{E}[\mu|\mathcal{D}]", legend=:topleft, ribbon = (2*std(μs)[order], 2*std(μs)[order]))
	plot!(Advertising.TV[order], μs[:][1][order], lw=0.15, lc=:gray, label=L"\mu^{(r)}\sim p(\mu|\mathcal{D})", legend=:topleft)
	for i in 2:25
		plot!(Advertising.TV[order], μs[:][i][order], lw=0.2, lc=:gray, label="")
	end
	@df Advertising scatter!(:TV, :sales, xlabel="TV", ylabel="Sales", label="", alpha=0.3, ms=3)
	plt
end

# ╔═╡ fc324c67-85be-4c50-b315-9d00ad1dc2be
md"""
**Predictions.**
Lastly, we show how to make predictions with `Turing`. To make predictions on new testing input ``x``, we will use `Turing`'s `predict()` method. Similar to prior and posterior predictive checks, we first feed in an array of missing values to the Turing model and use `predict` to draw posterior inference on the unknown targets ``y``.


Take the extended Advertising for example, the code below predicts at testing input at ``\texttt{TV} = 0, 1, \ldots, 300``.

"""

# ╔═╡ 61b1e15f-f3ff-4649-b90b-e85e2d172aaf
begin
	ad_test = collect(0:300)
	missing_mod = hetero_var_model(ad_test, Vector{Union{Missing, Float64}}(undef, length(ad_test)))
	ŷ=predict(missing_mod, chain2)[@varname(y)]
end;

# ╔═╡ 7e992f3e-0337-4b5f-8f4d-c20bdb3b6b66
md"""
After making predictions on the inputs, we can summarise the prediction by calculating the mean and standard deviation. The code below calculates the posterior predictive's mean and standard deviations and then the summary statistics are plotted together with the original dataset. Note how the prediction's variance increases when the input increases.
"""

# ╔═╡ ddc0f5b3-385a-4ceb-82e1-f218299b26d9
begin
	ŷ_mean = mean(ŷ)
	ŷ_std = std(ŷ)
end;

# ╔═╡ 9eef45d5-0cd2-460e-940f-bdc7114106c3
begin
	@df Advertising scatter(:TV, :sales, xlabel="TV", ylabel="Sales", label="", title="Bayesian posterior predictive on the testing data")
	plot!(ad_test, ŷ_mean, lw=3, ribbon = (2 * ŷ_std, 2 * ŷ_std), label=L"E[y|\mathcal{D}]", legend=:topleft)
	plot!(ad_test, ŷ_mean + 2 * ŷ_std, lw=2, lc=3, ls=:dash, label=L"E[y|\mathcal{D}] + 2\cdot \texttt{std}")
	plot!(ad_test, ŷ_mean - 2 * ŷ_std, lw=2, lc=3, ls=:dash, label=L"E[y|\mathcal{D}] - 2\cdot \texttt{std}")
end

# ╔═╡ 1d5835ad-4e04-48f0-90e6-aa1776d9406f
md"""
### Handle outlier -- Bayesian robust linear regression

A handful of outliers might skew the final OLS estimation. For example, some outliers were added to the simulated dataset we consider earlier. The new dataset is plotted together with the estimated OLS estimated model. The model now deviates a lot from the ground true model.
"""

# ╔═╡ ee37602c-229c-4709-8b29-627d32a25823
begin
	Random.seed!(123)
	X_outlier = [0.1, 0.15]
	y_outlier = [13, 13.5]
	X_new = [X; X_outlier]
	yy_new = [yy; y_outlier]

end;

# ╔═╡ 0fc9fd84-9fb1-42a1-bd68-8a9fcdce999d
let
	
	ols_outlier =lm([ones(length(X_new)) X_new], yy_new)
	pred(x) = coef(ols_outlier)' * [1, x]
	scatter(X_new, yy_new, xlabel=L"x", ylabel=L"y", label="",title="OLS estimation with outliers")
	# scatter!(X_outlier, y_outlier, xlabel=L"x", ylabel=L"y", label="")
	plot!(0:0.1:1, pred, lw=2, label="OLS fit", legend=:topleft)
	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topright)
end

# ╔═╡ 9348f3e5-4224-46f5-bc07-dee20b47a8d3
md"""
To deal with outliers, classic frequentist's methods might either suggest identifying and removing the outliers or deriving a new estimator to accommodate the outlier. Bayesian's approach is more consistent: modify the modelling assumption.


One way to achieve **robust Bayesian regression** is to assume the dependent observation is generated with Cauchy noise rather than Gaussian. As demonstrated below, a Cauchy distribution has heavier tails than its Gaussian equivalence, which makes Cauchy more resilient to outliers.
"""

# ╔═╡ 36fb8291-fea9-4f9f-ac52-a0eb61c2c8a8
begin
	plot(-5:0.1:5, Normal(), fill=(0, 0.2), ylabel="density", label=L"\mathcal{N}(0,1)")
	plot!(-5:0.1:5, Cauchy(), fill=(0, 0.2), label=L"\texttt{Cauchy}(0,1)")
end

# ╔═╡ f67d1d47-1e25-4c05-befd-c958dce45168
md"""The robust Bayesian regression model can be specified by replacing the Gaussian likelihood with its Cauchy equivalent. 



!!! infor "Bayesian robust linear regression"
	```math
	\begin{align}
	\text{Priors: }\beta_0 &\sim \mathcal{N}(m_0^{\beta_0}, v_0^{\beta_0})\\
	\beta_1 &\sim \mathcal{N}(m_0^{\beta_1}, v_0^{\beta_1})\\
	\sigma^2 &\sim \texttt{HalfCauchy}(s_0) \\
	\text{Likelihood: for } n &= 1,2,\ldots, N:\\
	\mu_n &=\beta_0 + \beta_1 x_n \\
	y_n &\sim \texttt{Cauchy}(\mu_n, \sigma^2).
	\end{align}
	```

The `Turing` translation of the model is listed below. Note the model is almost the same as the ordinary model except the likelihood part."""

# ╔═╡ e4a67732-ff93-476e-8f5b-8433c1bf015e
@model function simple_robust_blr(Xs, ys; v₀ = 5^2, V₀ = 5^2, s₀ = 5)
	# Priors
	# Gaussian is parameterised with sd rather than variance
	β₀ ~ Normal(0, sqrt(v₀)) 
	β ~ Normal(0, sqrt(V₀))
	σ² ~ truncated(Cauchy(0, s₀), lower=0)
	# calculate f(x) = β₀ + βx for all observations
	# use .+ to broadcast the intercept to all 
	μs = β₀ .+ β * Xs
	
	# Likelihood
	for i in eachindex(ys)
		ys[i] ~ Cauchy(μs[i], sqrt(σ²))
	end
end

# ╔═╡ 92206a26-d361-4be5-b215-3ae388dd7f2f
begin
	Random.seed!(100)
	robust_mod = simple_robust_blr(X_new, yy_new)
	chain_robust = sample(robust_mod, NUTS(), MCMCThreads(), 2000,4)
end;

# ╔═╡ c65e9c09-52c5-4647-9414-ad53841e8ff3
md"""
To compare the results, we plot both ordinary Bayesian model with Gaussian likelihood and the proposed robust Bayesian inference results. It can be seen the ordinary model's posterior (left) has much larger prediction variance due to the influence of the outliers and the prediction mean (the red line) deviates from the true model significantly. On the other hand, the robust model (right) still provides very consistent result.
"""

# ╔═╡ e585393f-e1bd-4199-984f-5a09745171cf
# let
# 	β_mean = describe(chain_robust)[1][:, :mean]
# 	β_samples = Array(chain_robust[[:β₀, :β]])
# 	plt = scatter(X_new, yy_new, xlabel=L"x", ylabel=L"y", label="", title="Robust Bayesian model")
# 	pred(x; β) = x*β[2] + β[1]
# 	plot!(0:0.1:1, (x)->pred(x; β = β_mean), lw=2, label=L"\mathbb{E}[\mu|\mathcal{D}]", legend=:topleft)
# 	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
# 	plot!(0:0.1:1, (x) -> pred(x; β = β_samples[1,:]), lw=0.5, lc=:gray, label=L"\mu^{(r)} \sim p(\mu|\mathcal{D})", legend=:topleft)
# 	for i in 2:50
# 		plot!(0:0.1:1, (x) -> pred(x; β = β_samples[i,:]), lw=0.15, lc=:gray, label="", legend=:topright)
# 	end
# 	plot(plt_outlier_gaussian, plt)

# end

# ╔═╡ 40f528f0-acca-43ff-a500-16aeb97898c8
md"""
The MCMC chain of the robust analysis is also listed below. Recall the true regression parameters are ``\beta_0=\beta_1=3``. The posterior distribution correctly recovers the ground truth.  
"""

# ╔═╡ 9fdaefb8-28b4-4b70-8412-1275dc1ed224
Turing.summarystats(chain_robust)

# ╔═╡ 5193f5aa-91b3-493e-9ecd-8f1865b13f3a
md"[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section6_logistic_regression.html)
"

# ╔═╡ 139b5acb-c938-4fc9-84a5-fdcbd697b9da
md"""
## Appendix

"""

# ╔═╡ 969df742-bc8a-4e89-9e5e-62cb9e7c2215
# begin
# 	chain_outlier_data = sample(simple_bayesian_regression(X_new, yy_new), NUTS(), 2000)
# 	parms_turing = describe(chain_outlier_data)[1][:, :mean]
# 	β_samples = Array(chain_outlier_data[[:β0, :β]])
# 	pred(x) = parms_turing[1:2]' * [1, x]
# 	plt_outlier_gaussian = scatter(X_new, yy_new, xlabel=L"x", ylabel=L"y", label="", title="Ordinary Bayesian model")
# 	plot!(0:0.1:1, pred, lw=2, label=L"\mathbb{E}[\mu|\mathcal{D}]", legend=:topleft)
# 	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
# 	plot!(0:0.1:1, (x) -> β_samples[1, 1] + β_samples[1, 2] *x, lw=0.5, lc=:gray, label=L"\mu^{(r)} \sim p(\mu|\mathcal{D})", legend=:topleft)
# 	for i in 2:50
# 		plot!(0:0.1:1, (x) -> β_samples[i, 1] + β_samples[i, 2] *x, lw=0.15, lc=:gray, label="", legend=:topright)
# 	end
# 	plt_outlier_gaussian
# end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~1.1.0"
DataFrames = "~1.8.2"
GLM = "~1.9.5"
LaTeXStrings = "~1.4.1"
Latexify = "~0.16.12"
Plots = "~1.41.7"
PlutoTeachingTools = "~0.5.0"
PlutoUI = "~0.7.83"
StatsPlots = "~0.15.8"
Turing = "~0.49.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.2"
manifest_format = "2.1"
project_hash = "e875e7e803299716c6749e47e38d21ea16b3d0dc"

[[deps.ADTypes]]
deps = ["PrecompileTools"]
git-tree-sha1 = "629de23e1c16911b439dabd2303c08af9575b226"
registries = "General"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.24.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

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

[[deps.AbstractPPL]]
deps = ["ADTypes", "AbstractMCMC", "Accessors", "BangBang", "DensityInterface", "JSON", "LinearAlgebra", "MacroTools", "OrderedCollections", "Random", "StatsBase"]
git-tree-sha1 = "75c0ff0e0b35513de90def2ff5d3722e5f9cd4ca"
registries = "General"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.15.6"

    [deps.AbstractPPL.extensions]
    AbstractPPLDifferentiationInterfaceExt = ["DifferentiationInterface"]
    AbstractPPLDistributionsExt = ["Distributions", "LinearAlgebra"]
    AbstractPPLForwardDiffExt = ["ForwardDiff", "DiffResults"]
    AbstractPPLMooncakeExt = ["Mooncake"]
    AbstractPPLTestExt = ["Test"]

    [deps.AbstractPPL.weakdeps]
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "LogExpFunctions", "ProgressMeter", "Random", "Setfield", "Statistics", "StatsBase"]
git-tree-sha1 = "f54d8df29de7eacb29211422c2e47cc4358af7e7"
registries = "General"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.8.8"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCADTypesExt = "ADTypes"
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCComponentArraysExt = "ComponentArrays"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqSymplecticRKExt = "OrdinaryDiffEqSymplecticRK"

    [deps.AdvancedHMC.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEqSymplecticRK = "fa646aed-7ef9-47eb-84c4-9443fc8cbfa8"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "DocStringExtensions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "62ddbccf0ce5c26f8ef3cebe4bedef6b1599d616"
registries = "General"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.8.10"

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

    [deps.AdvancedMH.weakdeps]
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.AdvancedVI]]
deps = ["ADTypes", "AbstractPPL", "Accessors", "ChainRulesCore", "DiffResults", "Distributions", "DocStringExtensions", "FillArrays", "Functors", "LinearAlgebra", "LogDensityProblems", "Optimisers", "ProgressMeter", "Random", "StatsBase"]
git-tree-sha1 = "fc42c29943eae508fd28ec620d1faa4d2c1ce044"
registries = "General"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.7.0"

    [deps.AdvancedVI.extensions]
    AdvancedVIDynamicPPLExt = ["DynamicPPL", "Accessors", "Distributions", "LogDensityProblems"]
    AdvancedVIEnzymeExt = ["Enzyme", "ChainRulesCore"]
    AdvancedVIMooncakeExt = ["Mooncake", "ChainRulesCore"]
    AdvancedVIReverseDiffExt = ["ReverseDiff", "ChainRulesCore"]

    [deps.AdvancedVI.weakdeps]
    DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
registries = "General"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.2"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

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

[[deps.Bijectors]]
deps = ["ADTypes", "AbstractPPL", "ArgCheck", "ChangesOfVariables", "Distributions", "DocStringExtensions", "FillArrays", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Roots", "SparseArrays", "Statistics", "Test"]
git-tree-sha1 = "a58fb73182f07a5a561bfadbe40efb9c6c892853"
registries = "General"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.16.3"

    [deps.Bijectors.extensions]
    BijectorsChainRulesCoreExt = "ChainRulesCore"
    BijectorsEnzymeCoreExt = "EnzymeCore"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsMooncakeExt = "Mooncake"
    BijectorsReverseDiffChainRulesExt = ["ChainRules", "ReverseDiff"]
    BijectorsReverseDiffExt = "ReverseDiff"

    [deps.Bijectors.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "DataStrings", "Dates", "Downloads", "Durations", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "Printf", "Tables", "Unicode"]
git-tree-sha1 = "ed05d41d96b1a815d52e9de1ef472971bc509832"
registries = "General"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "1.1.0"

    [deps.CSV.extensions]
    CSVDataDecimalsExt = "DataDecimals"
    CSVFilePathsBaseExt = "FilePathsBase"
    CSVInlineStringsExt = "InlineStrings"

    [deps.CSV.weakdeps]
    DataDecimals = "3e2245cb-6932-498d-a7cf-39d39de8bde3"
    FilePathsBase = "48062228-2e41-5def-b9a4-89aafe57970f"
    InlineStrings = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"

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

[[deps.Chairmarks]]
deps = ["Printf", "Random"]
git-tree-sha1 = "9a49491e67e7a4d6f885c43d00bb101e6e5a434b"
registries = "General"
uuid = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
version = "1.3.1"
weakdeps = ["Statistics"]

    [deps.Chairmarks.extensions]
    StatisticsChairmarksExt = ["Statistics"]

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "83ee8183bd8c4a390ae178385e6c7b3aa4e468b2"
registries = "General"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.11"
weakdeps = ["InverseFunctions", "Test"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"
    ChangesOfVariablesTestExt = "Test"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
registries = "General"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
registries = "General"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

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

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
registries = "General"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStrings]]
git-tree-sha1 = "b8032263d364177a0b192ea15f277df4e6ab07e2"
registries = "General"
uuid = "48204bd6-5611-42ea-b167-fc1d713f9be2"
version = "1.1.0"

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

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
registries = "General"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "0693d8b0a4608ff289d228ab4c598df5894845cd"
registries = "General"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.21"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = ["GPUArraysCore", "Adapt"]
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceHyperHessiansExt = "HyperHessians"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    HyperHessians = "06b494a0-c8e0-40cc-ad32-d99506a00a6c"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DimensionalData]]
deps = ["ConstructionBase", "DataAPI", "Dates", "Extents", "Interfaces", "IntervalSets", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Random", "Statistics", "TableTraits", "Tables"]
git-tree-sha1 = "3e5b58039d35b98f425446c099577d67633a7237"
registries = "General"
uuid = "0703355e-b756-11e9-17c0-8b28908087d0"
version = "0.30.2"

    [deps.DimensionalData.extensions]
    DimensionalDataAbstractFFTsExt = "AbstractFFTs"
    DimensionalDataAdaptExt = "Adapt"
    DimensionalDataAlgebraOfGraphicsExt = "AlgebraOfGraphics"
    DimensionalDataArrayInterfaceExt = "ArrayInterface"
    DimensionalDataCategoricalArraysExt = "CategoricalArrays"
    DimensionalDataChainRulesCoreExt = "ChainRulesCore"
    DimensionalDataDiskArraysExt = "DiskArrays"
    DimensionalDataMakieExt = "Makie"
    DimensionalDataNearestNeighborsExt = "NearestNeighbors"
    DimensionalDataPythonCallExt = "PythonCall"
    DimensionalDataRecipesBaseExt = "RecipesBase"
    DimensionalDataSparseArraysExt = "SparseArrays"
    DimensionalDataStatsBaseExt = "StatsBase"

    [deps.DimensionalData.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiskArrays = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    NearestNeighbors = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

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
weakdeps = ["ChainRulesCore", "DensityInterface", "SparseConnectivityTracer", "Test"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Durations]]
deps = ["Dates"]
git-tree-sha1 = "407b209d76168dcc4355f9dcc4d3b3aaa60dfb3e"
registries = "General"
uuid = "41ac09f5-0a95-4810-aa18-c88bf48ee130"
version = "1.4.1"

    [deps.Durations.extensions]
    DurationsArrowExt = "Arrow"
    DurationsTimeZonesExt = "TimeZones"

    [deps.Durations.weakdeps]
    Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
    TimeZones = "f269a46b-ccf7-5d73-abea-4c690281aa53"

[[deps.DynamicPPL]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "BangBang", "Bijectors", "Chairmarks", "Compat", "ConstructionBase", "Distributions", "DocStringExtensions", "FillArrays", "InteractiveUtils", "LinearAlgebra", "LogDensityProblems", "LogExpFunctions", "MacroTools", "OrderedCollections", "PartitionedDistributions", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "Test"]
git-tree-sha1 = "6e11d83c87b1fff6ff5d940ab29a112b37dff056"
registries = "General"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.42.14"

    [deps.DynamicPPL.extensions]
    DynamicPPLBridgeStanExt = ["BridgeStan"]
    DynamicPPLComponentArraysExt = ["ComponentArrays"]
    DynamicPPLEnzymeCoreExt = ["EnzymeCore"]
    DynamicPPLForwardDiffExt = ["ForwardDiff"]
    DynamicPPLInputProvenanceExt = ["ForwardDiff"]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]
    DynamicPPLMarginalLogDensitiesExt = ["MarginalLogDensities"]
    DynamicPPLMooncakeExt = ["Mooncake"]
    DynamicPPLReverseDiffExt = ["ReverseDiff"]

    [deps.DynamicPPL.weakdeps]
    BridgeStan = "c88b6f0a-829e-4b0b-94b7-f06ab5908f5a"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    MarginalLogDensities = "f0c3360a-fb8d-11e9-1194-5521fd7ee392"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "e611b7fdfbfb5b18d5e98776c30daede41b44542"
registries = "General"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "2.0.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

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

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
registries = "General"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
registries = "General"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

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

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
registries = "General"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

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

[[deps.FindFirstFunctions]]
deps = ["PrecompileTools"]
git-tree-sha1 = "93eb9afa3a53b29d8a49714b83878225402b3281"
registries = "General"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "3.4.1"

    [deps.FindFirstFunctions.extensions]
    FindFirstFunctionsEnzymeCoreExt = "EnzymeCore"
    FindFirstFunctionsMooncakeExt = "Mooncake"

    [deps.FindFirstFunctions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "5031f23e040bf17082e5b52422d77b5e844eefb1"
registries = "General"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.33.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.FlexiChains]]
deps = ["AbstractMCMC", "AbstractPPL", "DelimitedFiles", "DimensionalData", "DocStringExtensions", "LinearAlgebra", "MCMCDiagnosticTools", "OrderedCollections", "PrecompileTools", "Printf", "Random", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "d7bb621eb0b30e713dbe3a6a14612e5d3543e050"
registries = "General"
uuid = "4a37a8b9-6e57-4b92-8664-298d46e639f7"
version = "0.6.43"

    [deps.FlexiChains.extensions]
    FlexiChainsAdvancedHMCExt = ["AdvancedHMC", "DimensionalData"]
    FlexiChainsComponentArraysExt = ["ComponentArrays"]
    FlexiChainsDynamicPPLExt = ["DynamicPPL", "Random", "DimensionalData", "OrderedCollections", "Distributions"]
    FlexiChainsInferenceObjectsExt = ["InferenceObjects", "DimensionalData", "OrderedCollections"]
    FlexiChainsMCMCChainsExt = ["MCMCChains", "OrderedCollections"]
    FlexiChainsMakieExt = ["Makie", "StatsBase", "KernelDensity"]
    FlexiChainsMonteCarloMeasurementsExt = ["MonteCarloMeasurements", "OrderedCollections"]
    FlexiChainsPairPlotsExt = ["PairPlots", "Makie"]
    FlexiChainsPigeonsDynamicPPLExt = ["Pigeons", "DynamicPPL", "AbstractMCMC"]
    FlexiChainsPigeonsExt = ["Pigeons"]
    FlexiChainsPosteriorDBExt = ["PosteriorDB", "OrderedCollections"]
    FlexiChainsPosteriorStatsDynamicPPLExt = ["PosteriorStats", "DynamicPPL"]
    FlexiChainsPosteriorStatsExt = ["PosteriorStats", "DimensionalData", "IntervalSets"]
    FlexiChainsRecipesBaseExt = ["RecipesBase", "StatsBase"]
    FlexiChainsStanSampleExt = ["StanSample"]
    FlexiChainsStatsPlotsExt = ["StatsPlots", "DimensionalData"]

    [deps.FlexiChains.weakdeps]
    AdvancedHMC = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8"
    InferenceObjects = "b5cf5a8d-e756-4ee3-b014-01d49d192c00"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    PairPlots = "43a3c2be-4208-490b-832a-a21dcd55d7da"
    Pigeons = "0eb8d820-af6a-4919-95ae-11206f830c31"
    PosteriorDB = "1c4bc282-d2f5-44f9-b6d1-8c4424a23ad4"
    PosteriorStats = "7f36be82-ad55-44ba-a5c0-b8b5480d7aa5"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    StanSample = "c1514b29-d3a0-5178-b312-660c88baa699"
    StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "3b0f72e2ffef1a139ac450725c9ecd01c0a3c050"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.6"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
registries = "General"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2bcce3ad6f6977d617928d7707fdc86ac83cce03"
registries = "General"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "1.13.0"

    [deps.FunctionWrappersWrappers.extensions]
    FunctionWrappersWrappersEnzymeExt = ["Enzyme", "EnzymeCore"]
    FunctionWrappersWrappersMooncakeExt = "Mooncake"

    [deps.FunctionWrappersWrappers.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "416b521e1158dd93cd99b3adcd872085726b36a1"
registries = "General"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "LogExpFunctions", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsModels"]
git-tree-sha1 = "c963639ae5b9aab54f543bdc7504f42f59880bec"
registries = "General"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.5"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "0b2b3e8b8fb7c1f5ddbf498be2eb2b3b37c9c0ed"
registries = "General"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.1"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
registries = "General"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "06b65886c7577a3784d616e29f1302c2e36e389d"
registries = "General"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.6"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interfaces]]
git-tree-sha1 = "331ff37738aea1a3cf841ddf085442f31b84324f"
registries = "General"
uuid = "85a1e053-f937-4924-92a5-1367d23b7b87"
version = "0.3.2"

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

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

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

[[deps.Libtask]]
deps = ["MistyClosures", "Test"]
git-tree-sha1 = "d045ad5c965fc46c5d4c459d19a99271887de0e9"
registries = "General"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.9.19"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Printf"]
git-tree-sha1 = "a09a85e94e681def2446752cddaf9401a4d82b6d"
registries = "General"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.8.2"

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

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems"]
git-tree-sha1 = "7b83f3ad0a8105f79a067cafbfd124827bb398d0"
registries = "General"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.13.1"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
    LogDensityProblemsADDifferentiationInterfaceExt = ["ADTypes", "DifferentiationInterface"]
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
registries = "General"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

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

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
registries = "General"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

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

[[deps.MistyClosures]]
git-tree-sha1 = "d1a692e293c2a0dc8fda79c04cad60582f3d4de3"
registries = "General"
uuid = "dbe65cb8-6be2-42dd-bbc5-4196aaced4f4"
version = "2.1.0"

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

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "FiniteDiff", "LinearAlgebra"]
git-tree-sha1 = "f96d38936d92d610318dec4d3b5ef37b0373c20f"
registries = "General"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "8.0.1"

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

[[deps.Optim]]
deps = ["ADTypes", "EnumX", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "Statistics"]
git-tree-sha1 = "81f338ad984b25bc82471db4d235e7b8bf21b8d6"
registries = "General"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "2.3.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Compat", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "ce02e5e508795e3df43be2f6a2220e3fd0aae206"
registries = "General"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.11"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"
    OptimisersReactantMLDataDevicesExt = ["Reactant", "MLDataDevices"]

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "OptimizationBase", "Printf", "Reexport", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "b4d47691ca17e5f8e26cc7e453f6ab92d480285f"
registries = "General"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "5.9.1"

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DifferentiationInterface", "DocStringExtensions", "FastClosures", "LinearAlgebra", "PrecompileTools", "SciMLBase", "SciMLLogging", "SparseArrays", "SparseConnectivityTracer", "SparseMatrixColorings", "SymbolicIndexingInterface"]
git-tree-sha1 = "fba46fdcd6e2b5ce63403749c8fe40dc065cec70"
registries = "General"
uuid = "bca83a33-5cc9-4baa-983d-23429ab6bcbb"
version = "5.10.0"

    [deps.OptimizationBase.extensions]
    OptimizationChainRulesCoreExt = "ChainRulesCore"
    OptimizationEnzymeExt = ["ChainRulesCore", "Enzyme"]
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMLDataDevicesExt = "MLDataDevices"
    OptimizationMLUtilsExt = "MLUtils"
    OptimizationMooncakeExt = "Mooncake"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSymbolicAnalysisExt = "SymbolicAnalysis"
    OptimizationZygoteExt = "Zygote"

    [deps.OptimizationBase.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SymbolicAnalysis = "4297ee4d-0239-47d8-ba5d-195ecdf594fe"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationOptimJL]]
deps = ["Optim", "OptimizationBase", "SciMLBase", "SparseArrays"]
git-tree-sha1 = "0aa52ab69b7c4f75c1e28a5311ab4ca834203490"
registries = "General"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.4.22"

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

[[deps.PartitionedDistributions]]
deps = ["Distributions", "FillArrays", "InvertedIndices", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "PDMats", "SpecialFunctions", "StatsBase"]
git-tree-sha1 = "f4fdc1c69051e2610337545416d3777868a98021"
registries = "General"
uuid = "569bd051-8d7b-4221-bcb8-d78512b5866a"
version = "0.1.2"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
registries = "General"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
registries = "General"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "cd00e28071a75c98664663f87c2ae447d25c0503"
registries = "General"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.7.1"

    [deps.PreallocationTools.extensions]
    PreallocationToolsEnzymeCoreExt = "EnzymeCore"
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

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

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
registries = "General"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
registries = "General"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

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

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "e6391348505df3c9c33008c0fec050862b834bf4"
registries = "General"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "4.5.5"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsCUDAExt = "CUDA"
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsFastBroadcastPolyesterExt = ["FastBroadcast", "Polyester"]
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsMooncakeExt = "Mooncake"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

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

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "PrecompileTools", "SHA", "Serialization"]
git-tree-sha1 = "9bb075002b40849582905394bb3031fb77aa102d"
registries = "General"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.27"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FindFirstFunctions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "Markdown", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "Random", "RecipesBase", "RecursiveArrayTools", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "85f85c3c5b949e7d691047e758e3fdb77c21310c"
registries = "General"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "3.58.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseFunctionPropertiesExt = "FunctionProperties"
    SciMLBaseMakieExt = ["Makie", "GeometryBasics"]
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReactantCoreExt = "ReactantCore"
    SciMLBaseReactantExt = "Reactant"
    SciMLBaseReverseDiffExt = ["ReverseDiff", "ForwardDiff"]
    SciMLBaseStaticArraysExt = "StaticArrays"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore", "ZygoteRules", "FillArrays"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    FunctionProperties = "f62d2435-5019-4c03-9749-2d4c77af0cbc"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReactantCore = "a3311ec8-5e00-46d5-b541-4f83e724a433"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "PrecompileTools", "Preferences"]
git-tree-sha1 = "3aafee7edc2ec7e71f50dcc6816cb031fac390a0"
registries = "General"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "2.1.0"

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

    [deps.SciMLLogging.weakdeps]
    Tracy = "e689c965-62c8-4b79-b2c5-8359227902fd"

[[deps.SciMLOperators]]
deps = ["Accessors", "Adapt", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "2712b6eeef282b5cd8c38660e68c001a4b9b24e2"
registries = "General"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.30.3"

    [deps.SciMLOperators.extensions]
    SciMLOperatorsLoopVectorizationExt = "LoopVectorization"
    SciMLOperatorsSparseArraysExt = "SparseArrays"

    [deps.SciMLOperators.weakdeps]
    LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
registries = "General"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "4b8fc0ffe22b4ecba7a02f1fec0b3757db014d0c"
registries = "General"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.6"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
registries = "General"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
registries = "General"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

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

[[deps.SparseConnectivityTracer]]
deps = ["ADTypes", "DocStringExtensions", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "277d8929a56912e7342458d4db22ee45e4c0e75c"
registries = "General"
uuid = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
version = "1.2.4"

    [deps.SparseConnectivityTracer.extensions]
    SparseConnectivityTracerChainRulesCoreExt = "ChainRulesCore"
    SparseConnectivityTracerLogExpFunctionsExt = "LogExpFunctions"
    SparseConnectivityTracerNNlibExt = "NNlib"
    SparseConnectivityTracerNaNMathExt = "NaNMath"
    SparseConnectivityTracerSpecialFunctionsExt = "SpecialFunctions"

    [deps.SparseConnectivityTracer.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Random", "SparseArrays"]
git-tree-sha1 = "0730f14b07c68fd3b3a6fae89a829a5124a7ae7e"
registries = "General"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.4.29"

    [deps.SparseMatrixColorings.extensions]
    SparseMatrixColoringsAMDGPUExt = "AMDGPU"
    SparseMatrixColoringsCUDAExt = ["CUDA", "cuSPARSE"]
    SparseMatrixColoringsCliqueTreesExt = "CliqueTrees"
    SparseMatrixColoringsColorsExt = "Colors"
    SparseMatrixColoringsGPUArraysExt = "GPUArrays"
    SparseMatrixColoringsJuMPExt = ["JuMP", "MathOptInterface"]

    [deps.SparseMatrixColorings.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
    cuSPARSE = "b26da814-b3bc-49ef-b0ee-c816305aa060"

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

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "0db41c4e0d9f3fa195395a6401a8290752c9cd3d"
registries = "General"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.10"

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

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrecompileTools", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "7eb6da9656581ac9fbffaef3ef7950a090a5002a"
registries = "General"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.55"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.Turing]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "AdvancedHMC", "AdvancedMH", "AdvancedVI", "BangBang", "Bijectors", "Compat", "DataStructures", "Distributions", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "FlexiChains", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "Optimization", "OptimizationOptimJL", "OrderedCollections", "Printf", "Random", "Random123", "Reexport", "SciMLBase", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "92b456a228ba157b0d8ee4309f82769e6f672671"
registries = "General"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.49.0"

    [deps.Turing.extensions]
    TuringDistributionsExt = "Distributions"
    TuringDynamicHMCExt = "DynamicHMC"
    TuringMCMCChainsExt = "MCMCChains"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"

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
# ╟─e2f564ae-b741-4532-be01-9255a235793e
# ╟─da388a86-19cb-11ed-0c64-95c301c27153
# ╟─c1026d6b-4e2e-4045-923e-eb5886b45604
# ╟─13cbb8af-9857-43c1-902c-52ccdbe02384
# ╟─4ed16d76-2815-4f8c-8ab0-3819a03a1acc
# ╟─c8547607-92be-4700-9468-863798c2ddce
# ╟─8f95b8c8-70a0-471a-be06-3b020db667b3
# ╟─3a46c193-5a25-423f-bcb5-038f3756d7ba
# ╟─1cae04d1-82ca-47c8-b1fa-7fa5a754d4b1
# ╟─f8040fa0-fdff-42da-8b9f-785627f51ee1
# ╟─c6938b7f-e6e5-4bea-a273-52ab3916d07c
# ╟─effcd3d2-ba90-4ca8-a69c-f1ef1ad697ab
# ╠═3e98e9ff-b674-43f9-a3e0-1ca5d8614327
# ╟─af404db3-7397-4fd7-a5f4-0c812bd90c4a
# ╟─af2e55f3-08b8-48d4-ae95-e65168a23eeb
# ╟─43191a7a-f1a2-41df-910d-cf85907e8f7a
# ╟─98ef1cca-de03-44c2-bcf4-6e79da139e11
# ╟─a1421ccb-2d6e-4406-b770-ad7dff007c69
# ╟─e1552414-e701-42b5-8eaf-21ae04a829a8
# ╟─9387dcb4-3f4e-4ec7-8393-30a483e00c63
# ╟─d3f4ac7b-1482-4840-b24b-d08066d1d70c
# ╟─bab3a19c-deb0-4b1c-a8f9-f713d66d9199
# ╟─b433c147-f547-4234-9817-2b29e7d57219
# ╟─89414966-5447-4133-80cb-0933c8b0d5d0
# ╟─824aa830-c958-4063-9ef7-39eeb743fc06
# ╟─d825e29d-3e0b-4c25-852f-0c9a544aa916
# ╟─f6cedb98-0d29-40f0-b9f3-430dd283fa36
# ╟─71bb2994-7fd2-4e11-b2f1-d88b407f86c1
# ╟─d02d5490-cf53-487d-a0c6-651725600f52
# ╟─59dd8a13-89c6-4ae9-8546-877bb7992570
# ╟─632575ce-a1ce-4a36-95dc-010229367446
# ╟─c0f926f1-85e6-4d2c-8e9a-26cd099fd600
# ╠═e9bb7a3c-9143-48c5-b33f-e7d6b48cb224
# ╟─1ef001cc-fe70-42e5-8b97-690bb725a734
# ╠═4ae89384-017d-4937-bcc9-3d8c63edaeb5
# ╟─1761f829-6c7d-4186-a66c-b347be7c9a15
# ╠═d57a7a26-a8a5-429d-b1b9-5e0499549544
# ╟─ba598c45-a76e-41aa-bfd6-a64bb6cea875
# ╟─391a1dc8-673a-47e9-aea3-ad79f366460d
# ╟─d3d8fe25-e674-42de-ac49-b83aac402e2d
# ╠═748c1ff2-2346-44f4-a799-75fb2002c6fc
# ╠═33399b63-dc7d-4690-aecc-c807a3ac69d5
# ╟─08f7be6d-fda9-4013-88f5-92f1b5d26336
# ╟─ab77aef9-920c-423a-9ce0-2b5adead1b7f
# ╟─b1d7e28a-b802-4fa1-87ab-7df3369a468a
# ╟─4293298e-52a5-40ca-970d-3f17c2c89adb
# ╟─860c4cf4-59b2-4036-8a30-7fbf44b18648
# ╟─74a9a678-60b9-4e3f-97a2-56c8fdc7094f
# ╟─435036f6-76fc-458b-b0eb-119de02eabb7
# ╟─6eff54f2-d2e0-4c18-8eda-3be3124b16a0
# ╟─f11a15c9-bb0b-43c1-83e6-dce55f2a772f
# ╟─03b571e6-9d35-4123-b7bb-5f3b31558c9e
# ╟─7cffa790-bf2d-473f-95cc-ed67802d7b4f
# ╟─21aaa2db-6df0-4573-8951-bdfd9b6be6f9
# ╟─b1401e7d-9207-4e31-b5ce-df82c8e9b069
# ╟─241faf68-49b2-404b-b5d5-99061d1dd2a7
# ╟─796ad911-9c95-454f-95f4-df5370b2496a
# ╟─ce97fbba-f5d0-42de-8b67-827c3478d8b0
# ╠═f3b6f3ab-4304-4ef4-933c-728841c17998
# ╟─ab51e2c3-dbd4-43e1-95b5-a875954ac532
# ╠═e1db0ee1-1d91-49c4-be6c-1ea276750bfc
# ╟─80727f81-2440-4da5-8a8b-38ebfdc4ddc9
# ╠═929330bf-beb5-4e42-8668-2a10adf13972
# ╠═9668ab51-f86d-4af5-bf1e-3bef7081a53f
# ╟─6a330f81-f581-4ccd-8868-8a5b22afe9b8
# ╠═b1f6c262-1a2d-4973-bd1a-ba363bcc5c41
# ╟─659a3760-0a18-4a95-8168-fc6ca237c4d5
# ╟─9cb32dc0-8c0c-456e-bbcb-7ff6ae63da60
# ╟─2e7f780e-8850-446e-97f8-51ec26f5e36a
# ╟─65d702d6-abec-43a1-89a8-95c30b3e748a
# ╟─611ad745-f92d-47ac-8d58-6f9baaf3077c
# ╟─ddb0d87e-cf0a-4b09-9bb6-4ee267fcdb9d
# ╟─1954544c-48b4-4997-871f-50c9bfa402b7
# ╟─85ae4791-6262-4f07-9796-749982e00fec
# ╠═08c52bf6-0920-43e7-92a9-275ed298c9ac
# ╠═11021fb7-b072-46ac-8c23-f92825182c8c
# ╟─cece17c2-bacb-4b4a-897c-4116688812c6
# ╟─29175b82-f208-4fcf-9704-4a1996cc6e3c
# ╟─05b90ddb-8479-4fbf-a469-d5b0bf4a91c8
# ╟─2d7f773e-9fa1-475d-9f74-c13908209aeb
# ╟─8a0692db-6b85-42f6-8893-4219d58b1032
# ╟─3c3f6378-ed20-4f40-a780-f9329ace34bc
# ╠═a43189a3-a94f-4e69-85b1-3a586b2cc0eb
# ╟─4bf4c216-fefb-40fc-9d36-eaa82ff5454b
# ╟─5a507f39-a6be-43aa-a909-4c87005ad1d2
# ╟─db76ed48-f0c0-4664-b2cf-d47be7faaa3f
# ╟─fc324c67-85be-4c50-b315-9d00ad1dc2be
# ╠═61b1e15f-f3ff-4649-b90b-e85e2d172aaf
# ╟─7e992f3e-0337-4b5f-8f4d-c20bdb3b6b66
# ╟─ddc0f5b3-385a-4ceb-82e1-f218299b26d9
# ╟─9eef45d5-0cd2-460e-940f-bdc7114106c3
# ╟─1d5835ad-4e04-48f0-90e6-aa1776d9406f
# ╠═ee37602c-229c-4709-8b29-627d32a25823
# ╟─0fc9fd84-9fb1-42a1-bd68-8a9fcdce999d
# ╟─9348f3e5-4224-46f5-bc07-dee20b47a8d3
# ╟─36fb8291-fea9-4f9f-ac52-a0eb61c2c8a8
# ╟─f67d1d47-1e25-4c05-befd-c958dce45168
# ╠═e4a67732-ff93-476e-8f5b-8433c1bf015e
# ╠═92206a26-d361-4be5-b215-3ae388dd7f2f
# ╟─c65e9c09-52c5-4647-9414-ad53841e8ff3
# ╟─e585393f-e1bd-4199-984f-5a09745171cf
# ╟─40f528f0-acca-43ff-a500-16aeb97898c8
# ╟─9fdaefb8-28b4-4b70-8412-1275dc1ed224
# ╟─5193f5aa-91b3-493e-9ecd-8f1865b13f3a
# ╟─139b5acb-c938-4fc9-84a5-fdcbd697b9da
# ╟─969df742-bc8a-4e89-9e5e-62cb9e7c2215
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
