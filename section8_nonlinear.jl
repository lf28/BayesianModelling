### A Pluto.jl notebook ###
# v0.19.15

using Markdown
using InteractiveUtils

# ╔═╡ f49dc24c-2220-11ed-0ff2-2df7e9f3a0cf
begin
    using PlutoUI
	using StatsPlots
    using CSV, DataFrames
	using LinearAlgebra
	using RDatasets
	using StatsBase
	using StatsFuns:logistic
	using Turing
    using GLM
	using Random
	using LaTeXStrings
	using Logging; 
	# Logging.disable_logging(Logging.War);
end;

# ╔═╡ ff6fcfa7-dcd1-4529-ac83-46f9f1e17bc7
using Splines2

# ╔═╡ 709e2b13-67d2-4e5b-b148-aba16431b0ae
TableOfContents()

# ╔═╡ 81a47ad3-18de-4437-a5b2-c803e7938842
md"""

In this chapter, we are going to see how non-linear analysis can be achieved via a technique called fixed basis expansion. And it turns out the Bayesian approach works well with non-linear basis expansion. Compared with the Frequentist estimation, Bayesian inference is more sparse and generalises better in an unseen dataset.
"""

# ╔═╡ 7f290bd5-06be-4627-b34a-4748521c48e8
md"""
## Non-linear models


The simplest approach to obtaining a non-linear model is to apply **non-linear** functions to the original input predictors first and then estimate a model with the expanded features. And such a technique is called *basis expansion*. A lot of more advanced techniques, such as neural networks, support vector machines, and splines, can be viewed as special cases of this technique.

More formally, given input vector ``\mathbf{x}_n \in R^D`` and its target ``y_n``, instead of fitting a linear regression model with ``\mathbf{x}_n``, *i.e.*

```math
y_n = \beta_0 +\mathbf{x}_n^\top \boldsymbol{\beta}_1 + \varepsilon_n
```


we fit the model

```math

\begin{align}
y_n &= \beta_0 +\beta_1 \phi_1(\mathbf{x}_n) + \beta_2\phi_2(\mathbf{x}_n) + \ldots + \beta_K \phi_K(\mathbf{x}_n) + \varepsilon_n
\end{align}
```

where ``\varepsilon_n`` are still random white Gaussian noises; and the functions ``\{\phi_k\}_{k=1}^K`` are called **basis function**, and each ``\phi_k`` is
* a ``R^D\rightarrow R`` function that transforms a ``D`` dimensional input observation ``\mathbf{x}_n`` to a scalar;
* and the function has to be non-linear.
"""

# ╔═╡ 89a89522-52d4-4591-a22f-5f0737176cf8
md"""
*Remarks. The idea can also be applied to generalised linear models (GLMs), such as logistic regression to achieve generalised non-linear models. The model assumption becomes*
```math 
g(\mathbf{E}[y_n|\mathbf{x}_n]) = \beta_0 +\beta_1 \phi_1(\mathbf{x}_n) + \beta_2\phi_2(\mathbf{x}_n) + \ldots + \beta_K \phi_K(\mathbf{x}_n),
```
*where ``g`` is the appropriate link function. For example, ``g^{-1}`` is a logistic function for the logistic regression.*
"""

# ╔═╡ 9c53adc3-8ecc-4c45-9c12-addfb30edf8d
md"""

By using matrix notation, the regression model with basis expansion can be compactly written as:

```math
p(\mathbf{y}|\mathbf{X}, \boldsymbol{\beta}, \sigma^2) = \mathcal{N}_N(\mathbf{y}; \mathbf{\Phi} \boldsymbol{\beta}, \sigma^2\mathbf{I}_N),
```
where ``\mathbf{y} = [y_1, y_2,\ldots,y_N]^\top``, ``\mathbf{I}_{N}`` is a ``N\times N`` identity matrix, and ``\mathbf{X}`` is original ``N\times D`` design matrix where each row corresponds to one observation, and ``D`` is number of input feature:

```math
\mathbf{X} = \begin{bmatrix}\rule[.5ex]{3.5 ex}{0.5pt} & \mathbf{x}_1^\top & \rule[.5ex]{3.5ex}{0.5pt} \\
\rule[.5ex]{3.5ex}{0.5pt} & \mathbf{x}_2^\top & \rule[.5ex]{3.5ex}{0.5pt} \\
&\vdots& \\
\rule[.5ex]{3.5ex}{0.5pt} & \mathbf{x}_N^\top & \rule[.5ex]{3.5ex}{0.5pt}
\end{bmatrix},
```

whereas ``\mathbf\Phi`` is the expanded ``N\times K`` matrix:
```math

 \mathbf{\Phi} = \begin{bmatrix}
   \phi_1(\mathbf{x}_1) & \phi_2(\mathbf{x}_1) & \ldots & \phi_K(\mathbf{x}_1) \\
  \phi_1(\mathbf{x}_2) & \phi_2(\mathbf{x}_2) & \ldots & \phi_K(\mathbf{x}_2) \\
\vdots & \vdots & \ddots & \vdots \\
  \phi_1(\mathbf{x}_N) & \phi_2(\mathbf{x}_N) & \ldots & \phi_K(\mathbf{x}_N) \\
  \end{bmatrix} = \begin{bmatrix}\rule[.5ex]{3.5 ex}{0.5pt} & \boldsymbol{\phi}_1^\top & \rule[.5ex]{3.5ex}{0.5pt} \\
\rule[.5ex]{3.5ex}{0.5pt} & \boldsymbol{\phi}_2^\top & \rule[.5ex]{3.5ex}{0.5pt} \\
&\vdots& \\
\rule[.5ex]{3.5ex}{0.5pt} & \boldsymbol{\phi}_N^\top & \rule[.5ex]{3.5ex}{0.5pt}
\end{bmatrix},
```
where each row is the ``n``-th observation's ``K`` dimensional expanded feature ``\boldsymbol{\phi}_n \in R^K``.

It can be immediately observed that the new regression model is still linear with respect to ``\mathbf\Phi``, *i.e.*
```math
\mathbf{y} = \mathbf{\Phi}\boldsymbol{\beta} + \boldsymbol{\varepsilon}.
```
Therefore, the ordinary linear model's inference algorithms can be applied directly. One only needs to replace ``\mathbf{X}`` with the new design matrix ``\mathbf{\Phi}``.

"""

# ╔═╡ 4d0c7d09-4eac-40ab-9824-5b8739d4b146
md"""
**Example (Polynomial regression)** For simplicity, consider a simple regression problem with one predictor here: *i.e.*
```math
y_n = \beta_0 +\beta_1 x_n
```


A K-th degree polynomial regression can be achieved by assuming
```math
\phi_k(x) = x^k,\;\; \text{for }k = 0,\ldots, K.
``` 
Note here we have introduced a dummy basis function ``\phi_0(x) = 1`` here to serve as intercept. Substitute in the basis function, we have a non-linear regression 

```math
y_n = \beta_0 + \beta_1 x_n + \beta_2 x_n^2 +\ldots, \beta_K x_n^K.
```
"""

# ╔═╡ eb9e22f8-2867-4320-9c81-5d63262089cc
md"As a concrete example, we fit a ``K=4``-th degree Polynomial regression to the Wage data."

# ╔═╡ 8b3660d5-e3a6-4f25-9c5b-553ca36a6b28
md" And the fitted polynomial model is plotted below."

# ╔═╡ b7981b85-d045-442b-89d0-d20cca6381f3
md"""

### Popular basis functions
"""

# ╔═╡ 988fb829-5ddb-43af-8836-f4fc95a945af
md"""

Technically speaking, any non-linear function can be used as a basis function. However, there are a few function choices that are more popular and successful. For example, the following are commonly used in the machine learning community:

* radial basis function (RBF):
```math
\phi_k(x) = \exp\left \{-\frac{1}{2}\left (\frac{x-\mu_k}{s}\right )^2\right \}
```

* sigmoid function (or logistic function)

```math
\phi_k(x) = \frac{1}{1 + \exp(-\frac{x-\mu_k}{s})}
```

* tanh function

```math
\phi_k(x) = \tanh\left (\frac{x-\mu_k}{s}\right )
```


* `ReLU` function

```math
\phi_k(x) = \max\left(0, \frac{x-\mu_k}{s}\right)
```
"""

# ╔═╡ 9e54b0ce-baaa-4116-a08e-2cc93e12026c
md"""
Some of the basis functions are plotted below for your reference. It is worth noting that all four functions are *local functions* which are parameterised with
* a location parameter ``\mu_k`` and 
* a scale parameter ``s``.

Compared with the polynomial basis functions which are *global* functions, the local basis functions are more flexible. To overcome this limitation, one possible solution is to fit multiple polynomial functions in multiple truncated locations (predefined by knots). And the corresponding basis functions are commonly known as **splines functions**.

"""

# ╔═╡ df342965-f2eb-456f-b4d4-1fc76615e52a
begin
	plt_poly_basis = plot(title="Polynomial basis functions")
	for k in 1:10
		plot!(-1:0.05:1, (x) -> x^k, label="")
	end


	plt_rbf_basis = plot(title="RBF basis functions")
	s = 0.25
	for μ in -1:0.2:1
		plot!(-1:0.01:1, (x) -> exp(-(x-μ)^2/(2*s^2)), label="")
	end



	plt_sigmoid_basis = plot(title="Sigmoid basis functions")
	s_sig = 0.1
	for μ in -1:0.2:1
		plot!(-1:0.01:1, (x) -> logistic((x-μ)/s_sig), label="")
	end


	plt_relu_basis = plot(title="ReLu basis functions")
	for μ in -1:0.2:1
		plot!(-1:0.01:1, (x) -> max(0,x-μ), label="")
	end
	plot(plt_poly_basis, plt_rbf_basis, plt_sigmoid_basis, plt_relu_basis)
end

# ╔═╡ d6c1074b-530b-453a-a1e4-cb41b09f4fdf
md"""

#### Specification of basis function parameters 

For **fixed basis** expansion models, the user needs to decide what the expansion locations ``\{\mu_k\}_{k=1}^K ``and scale ``s`` (which is usually shared among the ``K`` functions) are. One option is to choose the ``k/(K+1)`` percentile of the input data as the expanded locations. For example, for ``K=3``, the 25%, 50% and 75% percentile of the input data will be used as the expansion locations. 

A **non-parametric option** is to choose all the data points as expansion locations, *i.e.* ``\mu_n= \mathbf{x}_n`` for ``n=1,\ldots, N``. The new design matrix ``\mathbf{\Phi}`` then is a ``N\times (N+1)`` matrix. The expanded regression has ``N+1`` parameters, which potentially grow unbounded with the observation size. As will be demonstrated soon, this over-parameterised model proves to be too flexible to handle for the ordinary frequentist method (*i.e.* maximum likelihood estimation). 

Lastly, the hyperparameters can also be learnt based on the data. The model becomes an **adaptive basis function** model. And the model is more commonly known as **Neural networks**.
"""

# ╔═╡ 47d9675b-4427-45af-a02a-950961692d5d
md"""

### Frequentist MLE estimation

To demonstrate the idea, we consider a one-dimensional toy dataset first. The data is generated with a non-linear true function:

```math
f(x) = -5\tanh(0.5x) \cdot (1- (\tanh(0.5x))^2).
```
and 18 noisy observations are simulated based on some randomly selected input locations. The code to simulate the data is listed below. 

"""

# ╔═╡ 2aebebfd-7009-4def-8362-1617d18b5c64
begin
	function true_f(x)
		-5*tanh(0.5*x) * (1- tanh(0.5*x)^2)
	end
	Random.seed!(100)
	x_input = [range(-4, -0.4, 10); range(0.8, 4, 8)]
	y_output = true_f.(x_input) .+ sqrt(0.05) * randn(length(x_input))
end;

# ╔═╡ 0edef66b-e323-475b-bd83-972afe153c23
md"The simulated data is plotted below."

# ╔═╡ e9dfde47-306d-4f04-b468-2178c00cce74
begin
	scatter(x_input, y_output, label="Observations")	
	plot!(true_f, lw=2, xlim=[-10, 10], framestyle=:default,  lc=:gray, ls=:dash,label="true function", xlabel=L"x", ylabel=L"y")
end

# ╔═╡ 61398b43-d307-4e03-9097-1aae3414a8e7
md"""

We choose `RBF` as the basis function and choose basis expansion locations non-parametrically. That means we apply `RBF` basis functions on all input locations. Specifically, ``\mu_n=x_n, s^2=1.0`` are used to generate the expanded design matrix ``\mathbf \Phi``.

This expansion can be accomplished easily in `Julia`. The code below expand the given input vector ``\mathbf{X}``, which is a ``N`` element vector to a ``N\times (N+1)`` design matrix ``\mathbf\Phi``.
"""

# ╔═╡ e21db2b0-2c2f-4bfd-9e5a-05b06db4d0fe
begin
	# Radial basis function
	# called Gaussian kernel in the paper
	function rbf_basis(x, μ, s)
		exp.(-1 ./(2 .* s^2) .* (x .- μ).^2)
	end

	# vectorised version of basis expansion by RBF
	function basis_expansion(xtest, xtrain, s=1, intercept = false)
		# number of rows of xtrain, which is the size of basis
		n_basis = size(xtrain)[1] 
		n_obs = size(xtest)[1]
		X = rbf_basis(xtest, xtrain', s)
		intercept ? [ones(n_obs) X] : X
	end
	intercept = true
	sϕ = 1.0
	Φ = basis_expansion(x_input, x_input, sϕ, intercept);
end;

# ╔═╡ 2a0c544f-0442-41d7-84b7-b716b2a909ec
md"""
Next, we apply the frequentist maximum likelihood estimation (MLE) method (a.k.a. ordinary least square for linear regression) to fit the model with the new design matrix ``\mathbf\Phi`` and the output ``\mathbf y``.
"""

# ╔═╡ bf7857fd-1b69-4f54-86b7-4f6082081693
freq_ols_model = lm(Φ, y_output);

# ╔═╡ 2097b04a-c6a2-433a-8393-a5c4794db6c1
md"""

The model is then used to predict some testing data ``\mathbf{X}_{\text{test}}``. Note that to use the fitted model, one needs to apply the same basis expansion function to the testing data, and then make the prediction. The estimated model is plotted with the true signal function and also the observations.
"""

# ╔═╡ 53c7c995-c9c3-409c-9a77-1bea31bd9fa9
begin
	# apply the same expansion on the testing dataset
	x_test = -10:0.1:10
	Φₜₑₛₜ = basis_expansion(x_test, x_input, sϕ, intercept)
	tₜₑₛₜ = true_f.(x_test)
	# predict on the test dataset
	βₘₗ = coef(freq_ols_model)
	pred_y_ols = Φₜₑₛₜ*βₘₗ # or equivalently, one can use predict(freq_ols_model, Φₜₑₛₜ) from the GLM.jl package
end;

# ╔═╡ cd3c7fa8-dc4c-4101-b9d5-47e92c8df6f3
let
	plot(x_test, tₜₑₛₜ, linecolor=:black, ylim= [-3, 3], lw=2, linestyle=:dash, lc=:gray,framestyle=:default, label="true signal")
	scatter!(x_input, y_output, markershape=:xcross, markersize=3, markerstrokewidth=3, markercolor=:gray, markeralpha=2, label="Obvs")
	plot!(x_test, pred_y_ols, linestyle=:solid, lw=2, xlabel=L"x", ylabel=L"y", legend=:topright, label="Frequentist Est.")
end

# ╔═╡ c2ac1425-cff9-4cd9-8b6e-6465a9299274
md"""

**Frequentist MLE estimation overfits the data.** It can be observed that the frequentist's estimation fits the data too perfectly! The prediction goes through every single observation. However, the prediction is very bumpy and extravagant at many locations. This behaviour is called **overfitting**. An overfitting model has a poor generalisation performance on an unseen dataset.
"""

# ╔═╡ 8975b5af-3e5b-45ba-8954-6336addaa246
md"""
## Bayesian inference

"""

# ╔═╡ 10dac645-8078-45d7-ad66-64160f99a9a4
md"""

Since the model is still linear after fixed basis expansion, the Bayesian model specified before can be immediately reused here.  Take regression as an example, a Bayesian model with Gaussian priors on the regression parameter can be specified as follows:

!!! infor "Bayesian linear regression"
	```math
	\begin{align}
	\text{Priors: }\;\;\;\;\;\;\beta_0 &\sim \mathcal{N}(0, v_0^{\beta_0})\\
	\boldsymbol{\beta}_1 &\sim \mathcal{N}(0, \lambda^2\mathbf{I})\\
	\sigma^2 &\sim \texttt{HalfCauchy}(s_0) \\
	\text{Likelihood: }\;\;\text{for } n &= 1,2,\ldots, N:\\
	\mu_n &=\beta_0 + \boldsymbol{\beta}_1^\top  \boldsymbol{\phi}_n \\
	y_n &\sim \mathcal{N}(\mu_n, \sigma^2).
	\end{align}
	```

Compared with the Bayesian model used earlier, the only difference is that we have used the generated feature ``\boldsymbol{\phi}_n`` instead of ``\mathbf{x}_n`` in the likelihood part.


"""

# ╔═╡ 100e5718-52df-4451-a42a-58c54ca5dfe5
md"""

### Hierarchical Bayesian modelling

One drawback of the above model is that the user needs to specify the parameters of the priors, which are called hyper-parameters. For example, a zero mean Gaussian prior is imposed for the regression parameter ``\boldsymbol{\beta}_1``:
```math
\boldsymbol{\beta}_1 \sim \mathcal{N}(0, \lambda^2\mathbf{I}),
```


where ``\lambda^2 >0``, the prior variance, is a hyper-parameter that needs to be specified. In previous chapters, we have set all hyperparameters manually such that the priors are weakly-informative. 


However, the Bayesian approach offers us an alternative more principled approach. The idea is to introduce an additional layer of prior, called **hyper-priors**, for the hyperparameters. We usually set non-informative priors or weakly informative priors for the hyper-parameters. 

For our regression problem, the prior scale parameter ``\lambda`` controls the complexity of the regression. It, therefore, makes better sense to let the data determine its value. Since ``\lambda`` is a positive real value, a suitable prior is ``\texttt{HalfCauchy}(1)``.


!!! infor "Hierarchical Bayesian linear regression"
	```math
	\begin{align}
	\text{Hyper-prior: }\;\;\;\;\;\;\;\lambda  &\sim \texttt{HalfCauchy}(1.0)  \\
	
	\text{Priors: }\;\;\;\;\;\;\beta_0 &\sim \mathcal{N}(0, v_0^{\beta_0})\\
	\boldsymbol{\beta}_1 &\sim \mathcal{N}(\mathbf{0}, \lambda^2\mathbf{I})\\
	\sigma^2 &\sim \texttt{HalfCauchy}(s_0) \\
	\text{Likelihood: }\;\;\text{for } n &= 1,2,\ldots, N:\\
	\mu_n &=\beta_0 + \boldsymbol{\beta}_1^\top  \boldsymbol{\phi}_n \\
	y_n &\sim \mathcal{N}(\mu_n, \sigma^2).
	\end{align}
	```


The hierarchical Bayesian model is translated in `Turing` as follows.
"""

# ╔═╡ f969414b-6439-489d-af7a-fbb80cc9fff3
# Bayesian linear regression.
@model function hier_linear_regression(X, ys; v₀ = 10^2,  s₀ = 5)
    # Set hyperprior for the λ
	λ ~ truncated(Cauchy(0.0,1.0), lower=0.0)
	 # Set variance prior.
    σ² ~ truncated(Cauchy(0, s₀), 0, Inf)
    intercept ~ Normal(0, sqrt(v₀))
    # Set the priors on our coefficients.
    nfeatures = size(X, 2)
    coefficients ~ MvNormal(nfeatures, λ)
    # Calculate all the mu terms.
    μs = intercept .+ X * coefficients

	for i in eachindex(ys)
		ys[i] ~ Normal(μs[i], sqrt(σ²))
	end
	# ys .~ Normal.(μs, sqrt(σ²))
	# ys ~ arraydist(Normal.(μs, sqrt(σ²)))
	# ys ~ MvNormal(μs, sqrt(σ²))
	return (; μs, ys)
end

# ╔═╡ 7c84879e-7bfe-41da-81f8-eab4f1360511
md"""

Next, we apply the `Turing` model to fit our simulated dataset.
"""

# ╔═╡ ef0da715-15b3-4dee-8a8a-dccd2bcc36f6
chain_hier = let
	Random.seed!(100)
	# note that we exclude the first column i.e. the intercept since it has already been included in the Turing model
	sample(hier_linear_regression(Φ[:, 2:end], y_output; v₀ = 5, s₀= 2),
		# NUTS{Turing.TrackerAD}(),
		NUTS(),
		1000)
end;

# ╔═╡ 33803b7b-a60d-4c1b-a5ac-92b5e8aae8cd
md"""

Lastly, we compare the Bayesian prediction with the frequentist. To use `Turing` to make predictions on new testing input, we can use `generated_quantities()`. For example, the following code will return samples from the posterior predictive ``\mathbf{y}_{\texttt{test}}\sim p(\mathbf{y}_{\texttt{test}}|\mathcal{D}, \mathbf{X}_{\texttt{test}})`` and also the prediction means ``\mathbf{\mu}_{\texttt{test}}\sim p(\mathbf{\mu}_{\texttt{test}}|\mathcal{D}, \mathbf{X}_{\texttt{test}})`` based on the posterior samples.


```julia
pred_model = hier_linear_regression(Φₜₑₛₜ[:, 2:end], 
	Vector{Union{Missing, Float64}}(undef, size(Φₜₑₛₜ)[1]);
	v₀ = 5, s₀= 2)
generated_quantities(pred_model, chain_hier)
```
"""

# ╔═╡ 1e7628d4-e698-4469-b433-a0b9c3ffaa4a
md"""

Lastly, we plot the Bayesian prediction with the frequentist together to draw some comparison. It can be observed that the Bayesian approach automatically avoids **overfitting**. Noticeably, the Bayesian prediction is more reasonable at the two ends, where the data is scarce.
"""

# ╔═╡ 0dd33f5f-7d6c-4ebc-aff4-4a545f41e9bc
begin
	# extract the posterior samples as a R × D matrix; the third to last are the samples for β
	βs_samples = Array(chain_hier[:,:,1])[:, 3:end]
	μ_preds = Φₜₑₛₜ * βs_samples'
	plot(x_test, tₜₑₛₜ, linecolor=:black, ylim= [-3, 3], linestyle=:dash, lw=2, lc=:gray, framestyle=:default, label="true signal")
	scatter!(x_input, y_output, markershape=:xcross, markersize=3, markerstrokewidth=3, markercolor=:gray, markeralpha=2, label="Obvs")
	plot!(x_test, pred_y_ols, linestyle=:solid, lw=2, label="Frequentist")
	plot!(x_test, mean(μ_preds, dims=2),  linestyle=:solid, lw=2, label="Bayesian's mean")
end

# ╔═╡ 0165ce73-cad2-4e21-ab1c-1539e853bdc2
md"""

What is shown below is the 95% prediction credible intervals for the regression function ``\mu``. It can be observed that the uncertainty grows at the two ends and shrinks when there are enough data.
"""

# ╔═╡ 81040385-7216-4b79-af50-14098754a27f
begin
	stats_μ_pred = describe(Chains(μ_preds'))[2]
	lower = stats_μ_pred[:, Symbol("2.5%")]
	upper = stats_μ_pred[:, Symbol("97.5%")]
	middle = stats_μ_pred[:, Symbol("50.0%")]
	plot(x_test, tₜₑₛₜ, linecolor=:black, ylim= [-3, 3], linestyle=:dash, lw=2, lc=:gray, framestyle=:default, label="true signal")
	scatter!(x_input, y_output, markershape=:xcross, markersize=3, markerstrokewidth=3, markercolor=:gray, markeralpha=2, label="Obvs")
	βs_hier = Array(chain_hier[:,:,1])[:, 3:end]
	bayes_preds2 = Φₜₑₛₜ * βs_hier'
	plot!(x_test, middle,  linestyle=:solid, lw=2, lc=4, ribbon = (middle-lower, upper-middle), label="Bayesian's mean ± 95% CI")
end

# ╔═╡ 60f86e5e-98c1-475a-badd-6eeab9ebaaf7
md"""
### Why does Bayesian work?

Bayesian makes better-generalised predictions almost out-of-box. Compared with the frequentist approach, the Bayesian approach has two unique features that make it stand out. The first is the inclusion of prior. As discussed before, zero-mean prior distributions play a key role in *regularising* extreme values of the regression parameters. And the second factor is Bayesian's *ensemble estimation* approach versus Frequentist's plug-in principle. 
"""

# ╔═╡ 38fade9c-fb40-40cd-8c43-d70d896c1f6f
βml_mean = sum(abs.(βₘₗ)); βml_bayes = mean(sum(abs.(βs_samples), dims=1));

# ╔═╡ 7ffdb827-59aa-4742-95cc-d0ff9b9f1fed
md"""

**Regularisation effect**

As discussed in one of the previous chapters, the zero mean Gaussian prior regulates the estimation such that large extreme values of ``\boldsymbol{\beta}`` are discouraged. As a result, the posterior mean of the regression parameters is also shrunken towards ``\mathbf{0}``, the prior mean. 

To draw a more direct comparison, we compare the averaged ``L_1`` norms,
```math
\Vert \boldsymbol{\beta}\Vert_1 = \sum_{d=1}^D |\beta_d|
```
of the two estimations. For the Bayesian method, we report the average of samples' ``L_1`` norms.

|  | Frequentist | Bayesian|
|---|---|---|
|``{\Vert\beta \Vert_1}``|$(round(βml_mean, digits=2))|$(round(βml_bayes, digits=2))|

As expected, the Bayesian estimator is much smaller in magnitude than the unregulated OLS estimator. The posterior samples of the regression parameters are also plotted below in a `boxplot`.
"""

# ╔═╡ 7e8d71db-6cdd-4b09-b93a-9d58e74c08e3
boxplot("β" .* string.([18:-1:1]'...), Array(chain_hier[:, end:-1:4, :]), leg=false, permute=(:x, :y), outliers=false, ylim=[-10,10], title="Box plot on Bayesian β MCMC samples")

# ╔═╡ 100d76d5-61ed-4e78-8103-a450b3834647
md"""

It can be observed that the majority of the parameters are close to zero except for a handful of parameters, such as ``\beta_8, \beta_9, \beta_{11}, \beta_{12}``, which corresponds to the 8-th, 9-th, 11-th and 12-th observations' basis functions (check the plot below). In other words, the corresponding basis (or input features) are considered more important in predicting the target. And the *importance* is automatically determined by the Bayesian algorithm. 


The posterior means of the 18 parameters are plotted below together with the observed data and their corresponding basis functions. The important bases, *i.e.* ``\beta_d`` estimated with larger magnitudes, are highlighted.
"""

# ╔═╡ fa34843a-dafc-49f4-b3c3-b5426843176b
let
	mean_β = mean(βs_samples, dims=1)[2:end]
	plt = plot(x_input, mean_β, st=:sticks, xticks=true, ylabel=L"\beta", xlim=[-7,7], xlabel=L"x", marker=:circle, label=L"\mathbb{E}[\beta|\mathcal{D}]")
	for i in 1: 50
		lbl = i == 1 ? L"\beta^{(r)} \sim p(\beta|\mathcal{D})" : ""
		plot!(x_input, βs_samples[i, 2:end], lw=0.2, alpha=0.5, st=:line, ls=:dash, label=lbl, lc=:gray)
	end
	colors= cgrad(:tab20, length(x_input),scale = :linear, categorical = true)
	important_basis = [8,9, 11,12]
	for (i, μ) in enumerate(x_input)
		scatter!([μ], [y_output[i]], markershape=:xcross, markersize=3, markerstrokewidth=3, markercolor=colors[i], markeralpha=2, label="")
		plot!(-7:0.1:7, (x) -> exp(-(x-μ)^2/(2*sϕ^2)), lw=0.8, lc=colors[i], label="")
		
		if i ∈ important_basis
			plot!([μ], [mean_β[i]], st=:sticks,  marker=:circle, c=:red, label="")
		end
	end
	plt
end

# ╔═╡ 4de2a978-d13b-4968-b45c-645c2da210ff
md"""

**Ensemble learning** 

As discussed before, when it comes to prediction at a new testing input location ``\mathbf{x}_\ast``, Bayesian predicts by integration rather than plug-in estimation:

```math
\begin{align}
p(\mu_\ast |\mathbf{x}_\ast, \mathcal{D}) &= \int p(\mu_\ast|\boldsymbol{\beta}, \mathbf{x}_\ast)p(\boldsymbol{\beta}|\mathcal{D}) \mathrm{d}\boldsymbol{\beta}\\
&= \int {\boldsymbol{\phi}_\ast}^\top \boldsymbol{\beta}\cdot p(\boldsymbol{\beta}|\mathcal{D}) \mathrm{d}\boldsymbol{\beta}\\
&\approx \frac{1}{R} \sum_{r=1}^R \boldsymbol{\phi}_\ast^\top \boldsymbol{\beta}^{(r)},
\end{align}
```
where ``\{\boldsymbol{\beta}^{(r)}\}_r`` are the MCMC posterior samples. In other words, there are ``R`` models, each indexed by the MCMC samples, that make contributions to the prediction. Bayesian simple takes the average of them. The ensemble prediction idea is illustrated below. The posterior mean of the regression function ``\mathbb E[\mu^{(r)}(x_\ast)|\mathcal{D}]`` (the thick solid line) is plotted with a few individual models' predictions (gray lighter lines).

"""

# ╔═╡ 2c2779a7-9983-4dec-9f99-c110c2d15fef
let
	plt =plot(x_test, tₜₑₛₜ, linecolor=:black, ylim= [-3, 3], linestyle=:dash, lw=2, lc=:gray, framestyle=:default, label="true signal")
	scatter!(x_input, y_output, markershape=:xcross, markersize=3, markerstrokewidth=3, markercolor=:gray, markeralpha=2, label="Obvs")
	# plot!(xs, pred_y_ols, linestyle=:solid, lw=2, label="Frequentist")
	βs_hier = Array(chain_hier[:,:,1])[:, 3:end]
	bayes_preds2 = Φₜₑₛₜ * βs_hier'
	plot!(x_test, mean(bayes_preds2, dims=2),  linestyle=:solid, lw=2, lc=4, label="Bayesian's mean")

	for i in 1:35
		plot!(x_test, bayes_preds2[:, i],  linestyle=:dash, lw=0.3, lc=4, label="")
	end
	plt
end

# ╔═╡ 2770759d-1da9-4194-a3fc-0e6de9241bc5
md"""
## Bayesian non-linear classification

The basis expansion idea can also be applied to solve non-linear classification problems. As a concrete example, consider the following simulated dataset. The two classes clearly cannot be classified by a linear model since class 2 data is scattered in two centres.
"""

# ╔═╡ 850d643d-f34f-44dd-9a19-e214280d9f21
md"""

### Basis expansion with RBF
"""

# ╔═╡ 5f053027-33ce-4915-b4ba-3cafb99001a6
md"""
As a reference, we fit a logistic regression with the expanded design matrix. The estimated model is plotted below. The frequentist method returns an irregular decision boundary that follows the shape of class 2's data. And the posterior prediction is very clean-cut (or overconfident), *i.e.* either 0% or 100% classification predictions are made for all input locations. However, one may argue input locations further away from the data centre should be given less confident predictions.
"""

# ╔═╡ 9837843d-c1a2-4782-8872-3547de23dc8f
md"""

### Bayesian inference with `Turing`
"""

# ╔═╡ dc4715c7-83f5-48bb-9990-1aacdd3050d5
md"""

Next, we apply the Bayesian model to solve the problem. A hierarchical Bayesian logistic regression model is specified below in `Turing`. The prior structure is almost the same as the regression model and the only difference is the Bernoulli likelihood. The model is then inferred with a `NUTS()` sampler. As a quick demonstration, only 1000 samples were drawn. 
"""

# ╔═╡ d46e2d45-92ea-47df-99d6-9e4b11fedfba
begin
	@model function hier_logistic_reg(X, y; v₀=10^2)
		# Set hyperprior for the λ
		λ ~ truncated(Cauchy(0.0,1.0), lower=0.0)
		# priors
		β₀ ~ Normal(0, sqrt(v₀))
		nfeatures = size(X)[2]
		β ~ MvNormal(zeros(nfeatures), λ)
		# Likelihood
		μs = β₀ .+ X * β
		# logistic transformations
		σs = logistic.(μs)
		for i in eachindex(y)
			y[i] ~ Bernoulli(σs[i])
		end

		# y ~ arraydist(Bernoulli.(σs))
		return (; σs)
	end

end;

# ╔═╡ 3e7742e7-9faf-4a2c-bd05-1c5eb386492d
md"""
### Comparison

"""

# ╔═╡ fbfa39b0-d3f9-455f-bcf5-30c877a44e21
md"""

Next, we compare the two methods' prediction performance. Note that to make a prediction at a new testing location, we need to first apply the basis expansion functions and then use the Monte Carlo estimation for the Bayesian approach.

One can make a prediction at ``\mathbf{x}_\ast`` by using `predict()`. As an example, the following code makes predictions on randomly generated testing data ``\mathbf{X}_\text{test}``. 

```julia
Nₜₑₛₜ = 10
Xₜₑₛₜ = rand(Nₜₑₛₜ,2)
ϕₜₑₛₜ = apply_rbs_expansion(Xₜₑₛₜ, D[xknots,:], σ²_rbf)
pred_model = hier_logistic_reg(ϕₜₑₛₜ, Vector{Union{Missing, Bool}}(undef, Nₜₑₛₜ))
predict(pred_model,  chain_bayes_class)
```
"""

# ╔═╡ eaaaaf70-78f5-4da7-a3c4-b10757702991
md"The Bayesian's expected predictions are plotted below together with the Frequentists prediction. It can be observed that Bayesian prediction is more reasonable. 

* a circular decision boundary is formed (check the .5 contour line) rather than an irregular one
* the prediction is no longer black and white; 
  * at input locations further away from the observed data, the predictions are around 0.7 rather than 1.0 "

# ╔═╡ e1f91394-8da7-490a-acb9-d39b95ebdf33
md"""
## Appendix

"""

# ╔═╡ 3c06eb27-6147-4092-9ac5-312d7332cebd
md"The wage dataset"

# ╔═╡ a0b03301-805e-4d05-98ce-a320efff9667
begin
	wage_df = dataset("ISLR", "Wage");
end;

# ╔═╡ 4cc79cbf-68cd-4376-8fd8-22f44a0fe3f8
begin
	K_degree = 4
	N_wage = length(wage_df.Age)
	x_wage = wage_df.Age
	# create a new design matrix Φ
	Φ_poly = Float64.(hcat([x_wage.^k for k in 0:K_degree]...))
	lm(Φ_poly, wage_df.Wage) # fit with GLM.jl
end;

# ╔═╡ e0d46820-da53-47e0-83eb-6f6503b3b3fb
let
	dt = fit(ZScoreTransform, Φ_poly[:, 2:end], dims=1)
	Φ_poly_t= [ones(N_wage) StatsBase.transform(dt, Φ_poly[:, 2:end])]
	poly_reg = lm(Φ_poly_t, wage_df.Wage)
	@df wage_df scatter(:Age, :Wage, ms=3, malpha=0.1, mc=1,markershape=:circle, label="", xlabel="Age", ylabel="Wage", yguidefontsize=8, xguidefontsize=8, title="$(K_degree)-th degree polynomial regression")
	order=sortperm(wage_df.Age)
	plot!(wage_df.Age[order], predict(poly_reg)[order], lw=3, lc=2, label="")
end

# ╔═╡ c9c15c9e-7400-4047-b8c1-2bd0bf7f4dfb
begin
	wage = Float64.(wage_df.Age)
	B = bs(wage, df=7, intercept=false)
end;

# ╔═╡ 2ab8821e-393f-4687-ac38-5f55b6263944
lbm=lm([ones(size(B)[1]) B], wage_df.Wage);

# ╔═╡ 5303c9ea-95c9-49fb-a9ff-0b085c2afae0
logrbm=glm([ones(size(B)[1]) B], wage_df.Wage .> 250, Binomial(), LogitLink());

# ╔═╡ b1d7ad9f-5019-47ea-84cc-7fc53153033b
plt_wage_reg = let
	@df wage_df scatter(:Age, :Wage, ms=3, malpha=0.1, mc=1,markershape=:circle, label="", xlabel="Age", ylabel="Wage", yguidefontsize=8, xguidefontsize=8)
	order=sortperm(wage_df.Age)
	plot!(wage_df.Age[order], predict(lbm)[order], lw=3, lc=2, label="")
end;

# ╔═╡ 41c95a38-d477-4047-bb29-c9d001fc3593
plt_wage_class=let
	order=sortperm(wage_df.Age)
	plot(wage_df.Age[order], predict(logrbm)[order], lw=3, lc=2, label="", ylim=[-0.01, 0.22], xlabel="Age", ylabel=L"p(\texttt{Wage} >250|\texttt{Age})", yguidefontsize=8, xguidefontsize=8)
	id_w250 = wage_df.Wage .> 250
	scatter!(wage_df.Age[id_w250] + 0.2*randn(sum(id_w250)), 0.2 .* ones(sum(id_w250)), markershape=:vline, markersize=3,  mc=:gray, label="")
	scatter!(wage_df.Age[.!id_w250] + 0.2*randn(sum(.!id_w250)), zeros(sum(.!id_w250)), markershape=:vline, markersize=3, mc=:gray, label="")
end;

# ╔═╡ 2c50b998-12e7-4bba-a29c-5b892aa1612a
md"""

# Bayesian sparse models

So far we have focused on linear models, *i.e.* the relationship between the predictors and the target (or its transformation) is linear. The linear assumption makes sense for some simple cases. However, real-world data usually exhibit a more complicated correlation structure, and the relationships usually are **non-linear**. 

As an example, we consider the `Wage` dataset discussed in the book [Introduction to Statistical Learning](https://hastie.su.domains/ISLR2/). The data records a number of factors that are considered to relate to wages for a group of men from the Atlantic region of the United States. The `Age` factor and `Wage` are plotted below on the left. It seems that wage increases with age initially but declines again after approximately age 60. We have also plotted `Age` against the binary variable `Wage > 250` on the right. The chance of earning over 250k is also non-linear with respect to the age factor. 

$(begin

plot(plt_wage_reg, plt_wage_class, size=(650,330))

end)

"""

# ╔═╡ 09ed0b3a-1a92-441d-80da-d4ac2b7f80b3
md"Non-linear classification dataset"

# ╔═╡ 50007061-33bc-408f-8d8a-21b05e668cc3
begin
	Random.seed!(123)
	n_= 30
	D1 = [0 0] .+ randn(n_,2)
	# D1 = [D1; [-5 -5] .+ randn(n_,2)]
	D2 = [5 5] .+ randn(n_,2)
	D2 = [D2; [-5 -5] .+ randn(n_,2)]
	D = [D1; D2]
	targets_ = [repeat(["class 1"], n_); repeat(["class 2"], n_*2)]
	targets = [zeros(n_); ones(n_*2)]
	df_class = DataFrame(x₁ = D[:, 1], x₂ = D[:, 2], y=targets_)
end;

# ╔═╡ 31647188-cb5b-4e92-8ae3-91247c15d976
begin
	Random.seed!(100)
	nknots = 25
	xknots = randperm(size(D)[1])[1:nknots]
	rbf(xs, μ, σ²) = exp.(- 0.5 .* sum((xs .- μ).^2, dims=2) ./ σ²)[:]
	function apply_rbs_expansion(D, μs, σ²)
		hcat(map((x)-> rbf(D, x', σ²) , eachrow(μs))...)
	end
	σ²_rbf = 1.0
	Φ_class = Array{Float64}(undef, size(D)[1], length(xknots))
	for j in 1:length(xknots)
		Φ_class[:, j] = rbf(D, D[xknots[j], :]', σ²_rbf)
	end
end

# ╔═╡ 009ff641-1cf6-475b-8e63-2594bb40878f
md"""

To solve the problem, we apply (multi-dimensional) RBF basis expansions with randomly selected $(nknots) data points as centres and a fixed scale ``s^2=1.0``:

```math
\text{rbf}(\mathbf{x}, \boldsymbol{\mu}, s^2) = \exp\left( -\frac{1}{2s^2} (\mathbf{x}- \boldsymbol{\mu})^\top (\mathbf{x}- \boldsymbol{\mu})\right)
```

The randomly picked expansion locations and the contour plots of the corresponding RBF functions are plotted below. Each circle represents a new feature in the expanded design matrix ``\mathbf{\Phi}_{\text{class}}``.
"""

# ╔═╡ 267c8824-ad28-4e50-b331-7b6174778562
let
	plt=@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", alpha=0.2, ylabel=L"x_2", size=(400,350) )

	scatter!(D[xknots, 1], D[xknots, 2], markershape=:xcross,alpha=1, color=:gray,markerstrokewidth=2,  ms=5, label=L"{\mu}")

	for i in 1:length(xknots)
		contour!(-7:0.1:7, -7:0.1:7, (x,y)->rbf([x y], D[xknots[i],:]', σ²_rbf)[1], levels=1, alpha=0.5)
	end

	plt
	# b = glm([ones(size(D)[1]) D], targets, Bernoulli(), LogitLink()) |> coef
	# contourf!(-10:0.1:10, -10:0.1:10, (x, y) -> logistic(b[1] + x*b[2] + y*b[3]), fill=false, alpha=0.1)
end

# ╔═╡ e0067b9a-dc9e-4dcb-95b9-71f040dd3d5c
rbf_freq_fit=glm([ones(length(targets)) Φ_class], targets, Binomial(), LogitLink());

# ╔═╡ 30547131-c50b-4ab0-a3ce-c55385f070cd
pred_class_feq, pred_class_freq_surf=let
	xs = -10:0.2:10
	ys = -10:0.2:10

	function pred(x, y)
		predict(rbf_freq_fit, [1 apply_rbs_expansion([x y], D[xknots,:], σ²_rbf)])[1]
	end

	plt = contour(xs, ys, pred, levels=10, fill=true,  c=:jet1, alpha=0.5, title="Frequentist prediction: "*L"p(y=1|x)")

	plt2 = surface(xs, ys, pred, levels=10, fill=true,  c=:jet1, alpha=0.5, title="Frequentist prediction: "*L"p(y=1|x)")
	@df df_class scatter!(plt, :x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", alpha=0.2, ylabel=L"x_2", size=(400,350), framestyle=:box )
	plt, plt2
end;

# ╔═╡ 73718d33-7a81-41fd-bb80-66f805e08c50
plot(pred_class_feq, pred_class_freq_surf, size=(990,400))

# ╔═╡ 1d3b5326-be29-4c8d-b32a-853a7b46fd2b
chain_bayes_class=let
	Random.seed!(123)
	sample(hier_logistic_reg(Φ_class, targets), NUTS(), 1000; discard_init=500)
end;

# ╔═╡ fd46999c-ad42-4ad3-a94c-271263a5d7ad
pred_class_bayes, pred_class_bayes_surf =let

	pred_class_mod = hier_logistic_reg(Φ_class, Vector{Union{Missing, Bool}}(undef, length(targets)))

	β_samples = Array(chain_bayes_class[:,2:end,:])
	xs = -10:0.2:10
	ys = -10:0.2:10
	
	function pred(x, y, βs)
		ϕₓ = [1 apply_rbs_expansion([x y], D[xknots,:], σ²_rbf)]
		mean(logistic.(βs * ϕₓ'))
	end

	plt = contour(xs, ys, (x,y)-> pred(x, y, β_samples), levels= 10,fill=true, alpha=0.5, c=:jet, lw=1, colorbar=true, xlim=[-10,10], framestyle=:box, title="Bayesian prediction: "*L"p(y=1|x)")

	plt2 = surface(xs, ys, (x,y)-> pred(x, y, β_samples), levels= 10,fill=true, alpha=0.5, c=:jet, lw=1, colorbar=true, xlim=[-10,10], framestyle=:box, title="Bayesian prediction: "*L"p(y=1|x)")
	@df df_class scatter!(plt, :x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", alpha=0.2, ylabel=L"x_2", size=(400,350) )
	plt, plt2
end;

# ╔═╡ fa87351a-80dc-4486-85e6-e852ec244497
plot(pred_class_feq, pred_class_bayes, size=(990,400))

# ╔═╡ 84af3bc4-e82f-4049-96a6-2d5ee018bd97
plot(pred_class_freq_surf, pred_class_bayes_surf, size=(990,400))

# ╔═╡ 2951083d-79c3-4850-9d00-8a6fe030edd0
p_nl_cls = let
	@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2")
	
	# b = glm([ones(size(D)[1]) D], targets, Bernoulli(), LogitLink()) |> coef
	# contourf!(-10:0.1:10, -10:0.1:10, (x, y) -> logistic(b[1] + x*b[2] + y*b[3]), fill=false, alpha=0.1)
end;

# ╔═╡ db1c1f87-c460-4e32-a965-13bff50fcd40
md"""

$(begin
plot(p_nl_cls, size=(400,350), title="A non-linear classification data")
end)
"""

# ╔═╡ 99506fba-05a5-4701-9e38-c8aee4c845ca
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

# ╔═╡ 3f66d862-46f7-43b4-a93f-e4b5d861b5b5
Foldable("Find the posterior samples manually.", md"""

Since the problem is relatively simple, we can manually calculate the posterior samples for the prediction mean ``\mu`` based on the following deterministic relationship:

```math
\boldsymbol{\mu}^{(r)} = \mathbf{\Phi}_{\texttt{test}}\boldsymbol{\beta}^{(r)}
``` 

which can be implemented as 
```julia
# extract the posterior samples as a R × (N+1) matrix; the third to last are the samples for β
βs_samples = Array(chain_hier[:,:,1])[:, 3:end]
# need to apply a transpose on the β samples such that the matrix multiplication makes sense
μ_preds = Φₜₑₛₜ * βs_samples'
```
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Splines2 = "5a560754-308a-11ea-3701-ef72685e98d6"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.4"
GLM = "~1.8.0"
LaTeXStrings = "~1.3.0"
PlutoUI = "~0.7.39"
RDatasets = "~0.7.7"
Splines2 = "~0.2.1"
StatsBase = "~0.33.21"
StatsFuns = "~1.0.1"
StatsPlots = "~0.15.1"
Turing = "~0.21.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "501ffe42f70684df02e7d0563b10de854bdaffd3"

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

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Setfield", "SparseArrays"]
git-tree-sha1 = "6320752437e9fbf49639a410017d862ad64415a5"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.5.2"

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
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "345effa84030f273ee86fcdd706d8484ce9a1a3c"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.5"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "d7a7dabeaef34e5106cdf6c2ac956e9e3f97f666"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.8"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "9ff1247be1e2aa2e740e84e8c18652bd9d55df22"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.3.8"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "e743af305716a527cdb3a67b31a33a7c3832c41f"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.5"

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

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "40debc9f72d0511e12d817c7ca06a721b6423ba3"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.17"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "a1e2cf6ced6505cbad2490532388683f1e88c3ed"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.0"

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

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "875f3845e1256ee1d9e0c8ca3993e709b32c0ed1"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.10.3"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

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

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5f5a975d996026a8dd877c35fe26a7b8179c02ba"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.6"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "650415ad4c2a007b17f577cb081d9376cc908b6f"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.44.2"

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

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "332a332c97c7071600984b3c31d9067e1a4e6e25"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "78bee250c6826e1cf805a88b7f1e86025275d208"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.0"

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
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

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
git-tree-sha1 = "6180800cebb409d7eeef8b2a9a562107b9705be5"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.67"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "74dd5dac82812af7041ae322584d5c2181dead5c"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.42"

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

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "MacroTools", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "b1d485d5e92a16545d14775d08eb22ca7a840515"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.20.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterfaceCore", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4cda4527e990c0cc201286e0a0bfbbce00abcfc2"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.0.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

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

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "94f5101b96d2d968ace56f7f2db19d0a5f592e28"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.15.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "129b104185df66e408edd6625d480b7f9e9823a0"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.18"

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
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

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

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "2da4f223fbc4328b389bcce5f3e93dbe71678590"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.0"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "039118892476c2bf045a43b88fcb75ed566000ff"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2d908286d120c584abbe7621756c341707096ba4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.2+0"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

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

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "076bb0da51a8c8d1229936a1af7bdfacd65037e1"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.2"

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

[[deps.LRUCache]]
git-tree-sha1 = "d64a0aff6691612ab9fb0117b0995270871c5dfc"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.3.0"

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

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "dfa6c5f2d5a8918dd97c7f1a9ea0de68c2365426"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.7.5"

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

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8cb9b8fb081afd7728f5de25b9025bff97cb5c7a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.3.1"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "59ac3cc5c08023f58b9cd6a5c447c4407cede6bc"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.4"

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

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

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

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "6d019f5a0465522bbfdd68ecfad7f86b535d6935"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.0"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "415108fd88d6f55cedf7ee940c7d4b01fad85421"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.9"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

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
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.RData]]
deps = ["CategoricalArrays", "CodecZlib", "DataFrames", "Dates", "FileIO", "Requires", "TimeZones", "Unicode"]
git-tree-sha1 = "19e47a495dfb7240eb44dc6971d660f7e4244a72"
uuid = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
version = "0.8.3"

[[deps.RDatasets]]
deps = ["CSV", "CodecZlib", "DataFrames", "FileIO", "Printf", "RData", "Reexport"]
git-tree-sha1 = "2720e6f6afb3e562ccb70a6b62f8f308ff810333"
uuid = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
version = "0.7.7"

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

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "3004608dc42101a944e44c1c68b599fa7c669080"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.0"

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

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "50f945fb7d7fdece03bbc76ff1ab96170f64a892"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "f1bc477b771a75178da44adb252fdc70b4b22e24"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.50.1"

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

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

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

[[deps.Splines2]]
deps = ["LinearAlgebra", "OffsetArrays", "Statistics"]
git-tree-sha1 = "0c929daf7cb741b611aa9f89f81d53e2fd9c291a"
uuid = "5a560754-308a-11ea-3701-ef72685e98d6"
version = "0.2.1"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "85bc4b051546db130aeb1e8a696f1da6d4497200"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.5"

[[deps.StaticArraysCore]]
git-tree-sha1 = "5b413a57dd3cea38497d745ce088ac8592fbb5be"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.1.0"

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

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "f8ba54b202c77622a713e25e7616d618308b34d3"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.31"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "2b35ba790f1f823872dcf378a6d3c3b520092eac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

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

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Scratch", "Unicode"]
git-tree-sha1 = "d634a3641062c040fc8a7e2a3ea17661cc159688"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.9.0"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0874c1b5de1b5529b776cfeca3ec0acfada97b1b"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.20"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "4ad90ab2bbfdddcae329cba59dab4a8cdfac3832"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.7"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "DiffResults", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "c43c5b5e2c6dcebee8705c25dbf22f4411e992ab"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.10"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

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
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─f49dc24c-2220-11ed-0ff2-2df7e9f3a0cf
# ╟─709e2b13-67d2-4e5b-b148-aba16431b0ae
# ╟─2c50b998-12e7-4bba-a29c-5b892aa1612a
# ╟─81a47ad3-18de-4437-a5b2-c803e7938842
# ╟─7f290bd5-06be-4627-b34a-4748521c48e8
# ╟─89a89522-52d4-4591-a22f-5f0737176cf8
# ╟─9c53adc3-8ecc-4c45-9c12-addfb30edf8d
# ╟─4d0c7d09-4eac-40ab-9824-5b8739d4b146
# ╟─eb9e22f8-2867-4320-9c81-5d63262089cc
# ╠═4cc79cbf-68cd-4376-8fd8-22f44a0fe3f8
# ╟─8b3660d5-e3a6-4f25-9c5b-553ca36a6b28
# ╟─e0d46820-da53-47e0-83eb-6f6503b3b3fb
# ╟─b7981b85-d045-442b-89d0-d20cca6381f3
# ╟─988fb829-5ddb-43af-8836-f4fc95a945af
# ╟─9e54b0ce-baaa-4116-a08e-2cc93e12026c
# ╟─df342965-f2eb-456f-b4d4-1fc76615e52a
# ╟─d6c1074b-530b-453a-a1e4-cb41b09f4fdf
# ╟─47d9675b-4427-45af-a02a-950961692d5d
# ╠═2aebebfd-7009-4def-8362-1617d18b5c64
# ╟─0edef66b-e323-475b-bd83-972afe153c23
# ╟─e9dfde47-306d-4f04-b468-2178c00cce74
# ╟─61398b43-d307-4e03-9097-1aae3414a8e7
# ╠═e21db2b0-2c2f-4bfd-9e5a-05b06db4d0fe
# ╟─2a0c544f-0442-41d7-84b7-b716b2a909ec
# ╠═bf7857fd-1b69-4f54-86b7-4f6082081693
# ╟─2097b04a-c6a2-433a-8393-a5c4794db6c1
# ╠═53c7c995-c9c3-409c-9a77-1bea31bd9fa9
# ╟─cd3c7fa8-dc4c-4101-b9d5-47e92c8df6f3
# ╟─c2ac1425-cff9-4cd9-8b6e-6465a9299274
# ╟─8975b5af-3e5b-45ba-8954-6336addaa246
# ╟─10dac645-8078-45d7-ad66-64160f99a9a4
# ╟─100e5718-52df-4451-a42a-58c54ca5dfe5
# ╠═f969414b-6439-489d-af7a-fbb80cc9fff3
# ╟─7c84879e-7bfe-41da-81f8-eab4f1360511
# ╠═ef0da715-15b3-4dee-8a8a-dccd2bcc36f6
# ╟─33803b7b-a60d-4c1b-a5ac-92b5e8aae8cd
# ╟─3f66d862-46f7-43b4-a93f-e4b5d861b5b5
# ╟─1e7628d4-e698-4469-b433-a0b9c3ffaa4a
# ╟─0dd33f5f-7d6c-4ebc-aff4-4a545f41e9bc
# ╟─0165ce73-cad2-4e21-ab1c-1539e853bdc2
# ╟─81040385-7216-4b79-af50-14098754a27f
# ╟─60f86e5e-98c1-475a-badd-6eeab9ebaaf7
# ╟─7ffdb827-59aa-4742-95cc-d0ff9b9f1fed
# ╟─38fade9c-fb40-40cd-8c43-d70d896c1f6f
# ╟─7e8d71db-6cdd-4b09-b93a-9d58e74c08e3
# ╟─100d76d5-61ed-4e78-8103-a450b3834647
# ╟─fa34843a-dafc-49f4-b3c3-b5426843176b
# ╟─4de2a978-d13b-4968-b45c-645c2da210ff
# ╟─2c2779a7-9983-4dec-9f99-c110c2d15fef
# ╟─2770759d-1da9-4194-a3fc-0e6de9241bc5
# ╟─db1c1f87-c460-4e32-a965-13bff50fcd40
# ╟─850d643d-f34f-44dd-9a19-e214280d9f21
# ╟─009ff641-1cf6-475b-8e63-2594bb40878f
# ╠═31647188-cb5b-4e92-8ae3-91247c15d976
# ╟─267c8824-ad28-4e50-b331-7b6174778562
# ╟─5f053027-33ce-4915-b4ba-3cafb99001a6
# ╠═e0067b9a-dc9e-4dcb-95b9-71f040dd3d5c
# ╟─30547131-c50b-4ab0-a3ce-c55385f070cd
# ╟─73718d33-7a81-41fd-bb80-66f805e08c50
# ╟─9837843d-c1a2-4782-8872-3547de23dc8f
# ╟─dc4715c7-83f5-48bb-9990-1aacdd3050d5
# ╠═d46e2d45-92ea-47df-99d6-9e4b11fedfba
# ╠═1d3b5326-be29-4c8d-b32a-853a7b46fd2b
# ╟─3e7742e7-9faf-4a2c-bd05-1c5eb386492d
# ╟─fbfa39b0-d3f9-455f-bcf5-30c877a44e21
# ╟─eaaaaf70-78f5-4da7-a3c4-b10757702991
# ╟─fd46999c-ad42-4ad3-a94c-271263a5d7ad
# ╟─fa87351a-80dc-4486-85e6-e852ec244497
# ╟─84af3bc4-e82f-4049-96a6-2d5ee018bd97
# ╟─e1f91394-8da7-490a-acb9-d39b95ebdf33
# ╟─3c06eb27-6147-4092-9ac5-312d7332cebd
# ╠═a0b03301-805e-4d05-98ce-a320efff9667
# ╠═ff6fcfa7-dcd1-4529-ac83-46f9f1e17bc7
# ╠═c9c15c9e-7400-4047-b8c1-2bd0bf7f4dfb
# ╠═2ab8821e-393f-4687-ac38-5f55b6263944
# ╠═5303c9ea-95c9-49fb-a9ff-0b085c2afae0
# ╠═b1d7ad9f-5019-47ea-84cc-7fc53153033b
# ╠═41c95a38-d477-4047-bb29-c9d001fc3593
# ╟─09ed0b3a-1a92-441d-80da-d4ac2b7f80b3
# ╠═50007061-33bc-408f-8d8a-21b05e668cc3
# ╠═2951083d-79c3-4850-9d00-8a6fe030edd0
# ╠═99506fba-05a5-4701-9e38-c8aee4c845ca
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
