### A Pluto.jl notebook ###
# v0.19.22

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

# ╔═╡ da388a86-19cb-11ed-0c64-95c301c27153
begin
    using PlutoUI
	using StatsPlots
    using CSV, DataFrames
	using LinearAlgebra
	using Turing
    using Random
	using LaTeXStrings, Latexify
	using PlutoTeachingTools
	using MLDatasets
	using Logging; Logging.disable_logging(Logging.Warn);
end;

# ╔═╡ c1026d6b-4e2e-4045-923e-eb5886b45604
TableOfContents()

# ╔═╡ 96708550-225d-4312-bf74-737ab8fe0b4d
present_button()

# ╔═╡ 684f63ec-1e2f-4384-8ddd-f18d2469ebc3
md"""# Bayesian linear regression 




$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang (lf28@st-andrews.ac.uk)

*School of Computer Science*

*University of St Andrews, UK*

*March 2023*
"""

# ╔═╡ b94583a3-7a04-4b30-ba7b-4ff5a72baf5f
md"""

## Notations


Super-index with brackets ``.^{(i)}``: ``i \in \{1,2,\ldots, n\}`` index for observations/data
* ``n`` total number of observations
* *e.g.* ``y^{(i)}`` the i-th observation's label
* ``\mathbf{x}^{(i)}`` the i-th observation's predictor vector

Feature/predictor index ``j \in \{1,2,,\ldots, m\} ``
* ``m`` total number of features
* *e.g.* ``\mathbf{x}^{(i)}_2``: the second entry/predictor/feature of ``i``-th observation vector


Vectors: **Bold-face** smaller case:
* ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
* ``\mathbf{x}^\top``: row vector

Matrices: **Bold-face** capital case: 
* ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  


Scalars: normal letters
* ``x,y,\beta,\gamma``

"""

# ╔═╡ 4ed16d76-2815-4f8c-8ab0-3819a03a1acc
md"""

## Supervised learning

Supervised learning in general 

* predict *targets* ``Y`` with *covariates* ``X``
* it tries to access how ``X`` affects ``Y``




Depending on the type of the labelled targets ``Y``
* regression*: ``Y`` is continuous real values
* and *classification*: ``Y`` is categorical 

In this chapter, we consider regression in Bayesian approach
"""

# ╔═╡ c21f64bb-a934-4ec0-b1eb-e3fd6695d116
md"""

## Regression example


Example: **House price** prediction

The data is ``\mathbf{x}^{(i)}, y^{(i)}`` for ``i=1,2,\ldots, n``

* ``\mathbf{x}^{(i)}`` has 14 features: such as average number of rooms, crime rate, *etc*

* ``y^{(i)} \in R``:  house price of the ``i``- observation


"""

# ╔═╡ 0d3d98c4-fed4-4f4b-be17-bd733e808256
begin
	X_housing = MLDatasets.BostonHousing.features()
	df_house = DataFrame(X_housing', MLDatasets.BostonHousing.feature_names())
	df_house[!, :target] = MLDatasets.BostonHousing.targets()[:]
end;

# ╔═╡ 94e5a7ed-6332-4f92-8b77-7e0ce7b88a84
md"""

Consider the relationship between `room` and `price`
"""

# ╔═╡ 215c4d7f-ef58-4682-893b-d41b7de75afa
@df df_house scatter(:rm, :target, xlabel="room", ylabel="price", label="", title="House price prediction: regression")

# ╔═╡ c6c3e3aa-fee6-418f-b304-8d5b353bd2d7
md"""

## Linear regression 


!!! note "Linear regression assumption"
	Linear regression: prediction function ``\mu(\cdot)`` is assumed linear 

	```math
	\mu(x_{\text{room}}) = \beta_0 + \beta_1 x_{\text{room}} 
	```


``\mu(x)`` is called **prediction** function or **regression function**
* ``\beta_0, \beta_1``: model parameters
* sometimes we write ``\mu(x; \beta_0, \beta_1)`` or ``\mu_{\beta_0, \beta_1}(x)`` to emphasise ``\mu`` is parameterised with ``\beta_0, \beta_1``
"""

# ╔═╡ ab53480e-a9f3-4a86-91cd-aa3168128696
md"""

## Multiple linear regression


When the covariate ``\mathbf{x} \in R^m``

* we have ``m=14`` predictors in our house data, *e.g.* crime, room, *etc*


!!! note "Multiple linear regression "
	Prediction function ``\mu(\mathbf{x})`` becomes a hyperplane

	```math
	\mu(\mathbf{x}) = \beta_0 + \beta_1 x_{1} + \beta_2 x_2 + \ldots + \beta_m x_m  = \boldsymbol{\beta}^\top \mathbf{x}
	```

	* for convenience, we add 1 dummy predictor one to ``\mathbf{x}``

	```math
		\mathbf{x} =\begin{bmatrix}1 & x_1 & x_2 & \ldots& x_m \end{bmatrix}^\top
	```

    * we sometimes write ``\mu(\mathbf{x}; \boldsymbol{\beta}) = \boldsymbol{\beta}^\top \mathbf{x}`` or ``\mu_{\boldsymbol{\beta}}(\mathbf{x})``

"""

# ╔═╡ a00eb60a-4e90-4002-a758-799fbceab48c
md"""

## Hyperplane ``\mu(\mathbf{x}) = \boldsymbol{\beta}^\top \mathbf{x}``


Geometrically, ``\mu(\mathbf{x})`` forms a hyperplane 
"""

# ╔═╡ 41f6c4fa-89b9-492d-9276-b1651ba92236
md"""
## Frequentist's linear regression model


In other words, 

$$y^{(i)} = \boldsymbol{\beta}^\top \mathbf{x}^{(i)} + \epsilon^{(i)}, \;\; \epsilon^{(i)} \sim  \mathcal{N}(0, \sigma^2)$$

which implies **a probability distribution** for ``y^{(i)}`` 

$p(y^{(i)}|\mathbf{x}^{(i)}, \boldsymbol{\beta}, \sigma^2) = \mathcal{N}(y^{(i)};  \boldsymbol{\beta}^\top \mathbf{x}^{(i)} , \sigma^2)= \frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}\left(-\frac{(y^{(i)}-{\boldsymbol{\beta}}^\top\mathbf{x}^{(i)})^2}{2\sigma^2}\right)$

* ``y^{(i)}`` is a univariate Gaussian with mean $\boldsymbol{\beta}^\top \mathbf{x}^{(i)}$ and variance $\sigma^2$ 

"""

# ╔═╡ f6cb99dd-f25c-4770-bba2-8a2496016316
md"``x_i`` $(@bind xᵢ0 Slider(-0.5:0.1:1, default=0.15));	``\sigma^2`` $(@bind σ²0 Slider(0.005:0.01:0.15, default=0.05))"

# ╔═╡ 546e2142-41d2-4c4e-b997-adf8262c3345
md"input $x^{(i)}=$ $(xᵢ0); and ``\sigma^2=`` $(σ²0)"

# ╔═╡ b0008150-7aae-4310-bfa8-950ba7bc9092
md"""

## Likelihood ``p(\mathbf{y}|\boldsymbol{\beta}, \sigma, \mathbf{X})``



For the linear regression model

* the unknown are: ``\boldsymbol{\beta}, \sigma^2``

* the data observations are: ``\mathcal{D} =\{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}``

* note that ``\{\mathbf{x}^{(i)}\}`` are assumed fixed



Therefore, the likelihood function is

```math
p(\mathcal{D}|\boldsymbol{\beta}, \sigma^2, \{\mathbf{x}^{i}\}) = \prod_{i=1}^n p(y^{(i)}|\boldsymbol{\beta}, \sigma^2, \mathbf{x}^{(i)})
```

* conditional independence assumption (also i.i.d assumption)
"""

# ╔═╡ 3a46c193-5a25-423f-bcb5-038f3756d7ba
md"""
## MLE and OLS

Frequentists estimate the model by using maximum likelihood estimation (MLE):


```math
\hat{\beta_0}, \hat{\boldsymbol{\beta}}_1, \hat{\sigma^2} \leftarrow \arg\max p(\mathbf y|\mathbf X, \beta_0, \boldsymbol{\beta}_1, \sigma^2)
```
It can be shown that the MLE are equivalent to the ordinary least square (OLS) estimators

```math
\hat{\beta_0}, \hat{\boldsymbol{\beta}}_1 \leftarrow \arg\min \sum_{n=1}^N (y_n - \mathbf{x}_n^\top \boldsymbol{\beta}_1 -\beta_0)^2.
```

The closed-form OLS estimator is:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y},
```

* the design matrix ``\mathbf{X}`` is augmented with ``\mathbf{1}_N`` as the first column (*i.e.* dummy variable for intercept).
"""

# ╔═╡ c6938b7f-e6e5-4bea-a273-52ab3916d07c
md"""

## Example


Consider a simple linear regression model with one predictor, i.e.
```math
y_n = \beta_0 + \beta_1 x_n + \varepsilon_n.
```

* we have simulated a dataset with the following parameters: ``\beta_0=3, \beta_1=3, \sigma^2=0.5``
"""

# ╔═╡ effcd3d2-ba90-4ca8-a69c-f1ef1ad697ab
md"
* use `GLM.jl` to fit the OLS estimation "

# ╔═╡ af404db3-7397-4fd7-a5f4-0c812bd90c4a
begin
	Random.seed!(100)
	β₀, β₁, σ² = 3, 3, 0.5
	N = 50
	X = rand(N)
	μ = β₀ .+ β₁ * X 
	yy = μ + sqrt(σ²) * randn(N)
end;

# ╔═╡ 3e98e9ff-b674-43f9-a3e0-1ca5d8614327
begin
	using GLM
	ols_simulated = lm([ones(N) X], yy)
end

# ╔═╡ 06f941c1-53c8-4279-8214-0d3ef5c81c4b
linear_reg_normal_eq(X, y) = lm(X, y) |> coef

# ╔═╡ 75567e73-f48a-477c-bc9f-91ce1630468c
begin
	@df df_house scatter(:rm, :target, xlabel="room", ylabel="price", label="", title="House price prediction: regression")
	x_room = df_house[:, :rm]
	# x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room]

	c, b = linear_reg_normal_eq(X_train_room, df_house.target)

	plot!(3.5:0.5:9, (x) -> b* x+ c, lw=3, label=L"\mu({x}) = \beta_0+ \beta_1 x", legend=:outerbottom)
end

# ╔═╡ 510cc569-08eb-4deb-b695-2f3044d758e5
let
	gr()


	n_obs = 100
	# the input x is fixed; non-random
	xs = range(-0.5, 1; length = n_obs)
	true_w = [1.0, 1.0]
	true_σ² = 0.05
	ys = zeros(n_obs)
	for (i, xⁱ) in enumerate(xs)
		hⁱ = true_w' * [1, xⁱ]
		ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
	end
	
	b_1 = 3.0
	p_lr = plot(title="Linear regression's probabilistic model",legend=:bottomright)
	β0 = true_w
	n0 = n_obs
	xx = [ones(n0) collect(xs)]
	yy = xx * β0 + sqrt(σ²0) * randn(n0)
	plot!(xs, yy, st=:scatter, ylim=[0, 3],framestyle=:origin, label="observations", legend=:topleft)
	plot!(-0.5:0.1:1.0, x->β0[1]+β0[2]*x, c= 1, linewidth=5, label="",  ylim=[0, 3],framestyle=:origin)
	# xis = [-0.35, -0.2, 0, 0.25, 0.5, 0.75, 0.99, xᵢ0]
	xis = [range(-0.5, 1.0, 8)...]
	push!(xis, xᵢ0)
	for i in 1:length(xis)
		x = xis[i]
		μi = dot(β0, [1, x])
		σ = sqrt(σ²0)
		xs_ = μi- 4*σ :0.01:μi+ 4 *σ
		ys_ = pdf.(Normal(μi, sqrt(σ²0)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		if i == length(xis)
			scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, c=:red, label=L"\mu(x)", markersize=6)
			plot!(ys_ .+x, xs_, c=:red, label="", linewidth=3)
		else
			plot!(ys_ .+x, xs_, c=:gray, label="", linewidth=1)
			# scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, label="μ @ x="*string(x))
		end
		
	end
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
* ``\beta_0\in R`` -- intercept
* ``\boldsymbol{\beta}_1 \in R^D`` -- regression coefficient vector
* ``\sigma^2\in R^+`` -- Gaussian noise's variance


**In addition**, the Bayesian model also imposes priors on the unknowns

```math
p(\beta_0, \boldsymbol{\beta}_1, \sigma^2).
```

* a simple independent prior 

```math
p(\beta_0, \boldsymbol{\beta}_1, \sigma^2)= p(\beta_0)p(\boldsymbol{\beta}_1)p( \sigma^2).
```



"""

# ╔═╡ bc597dc8-dbfe-4a75-81b5-ea37655f95eb
md"""

## Graphical model


The frequentist model is on the **left** 

* where the parameters are not assumed random

```math
p(\mathbf{y}|\beta_0, \boldsymbol{\beta}, \sigma^2,  \mathbf{X}) = \prod_i p(y^{(i)} |\mathbf{x}^{(i)}, \beta_0, \boldsymbol{\beta}, \sigma^2)
```

The Bayesian model is on the right

* all parameters are assumed random (with priors)
* and based on the factoring property

```math
p(\beta_0, \boldsymbol{\beta}, \sigma^2, \mathbf{y}, \mathbf{X}) = p(\beta_0) p(\boldsymbol{\beta})p(\sigma^2)\prod_i p(y^{(i)} |\mathbf{x}^{(i)}, \beta_0, \boldsymbol{\beta}, \sigma^2)
```
"""

# ╔═╡ f240d7b7-4ea8-4836-81ae-ba1cd169b87d
TwoColumn(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/bayes/regression_freq.png", :height=>310, :align=>"left"), Resource("https://leo.host.cs.st-andrews.ac.uk/figs/bayes/regression_bayes.png", :height=>310, :align=>"right"))

# ╔═╡ a1421ccb-2d6e-4406-b770-ad7dff007c69
md"""

## Prior choices


**Prior for the intercept ``p(\beta_0)``** 

``\beta_0 \in R``, a common choice is Gaussian 

$$p(\beta_0) = \mathcal N(m_0^{\beta_0}, v_0^{\beta_0});$$ 

* where the hyper-parameters ``m_0^{\beta_0}, v_0^{\beta_0}`` can be specified based on the data or independently.


**Data-dependent hyper-parameter:** e.g. 

$\mathcal{N}(m_0^{\beta_0}={\bar{\mathbf{y}}}, v_0^{\beta_0}= 2.5^2 \sigma_{\mathbf{y}}^2 )$

* covering a wide range of possible values, the prior is a weakly informative prior


**Data-independent hyper-parameter:** e.g. 

$\mathcal{N}(m_0^{\beta_0}=0, v_0^{\beta_0}= 10^2)$


* the prior guess centres around ``0`` but with a great amount of uncertainty (large variance)
* zero mean here encourages the posterior shrinks towards 0 


"""

# ╔═╡ e1552414-e701-42b5-8eaf-21ae04a829a8
md"""

## Prior choice (cont.)

**Prior for**  $$p(\boldsymbol{\beta}_1)$$: ``\boldsymbol{\beta}_1\in R^D`` is also unconstrained

* a common prior choice is a D-dimensional multivariate Gaussian 


$$p(\boldsymbol{\beta}_1) = \mathcal N_{D}(\mathbf m_0^{\boldsymbol{\beta}_1}, \mathbf V_0^{\boldsymbol{\beta}_1}),$$

* or other similar location-scale distributions (such as Student-``t``, Cauchy)



The hyper-parameters are usually set data-independently:

* ``\mathbf{m}_0^{\boldsymbol{\beta}_1}=\mathbf{0}``, the prior encourages the posterior shrinks towards zeros, which has a regularisation effect
* ``\mathbf{V}_0^{\boldsymbol{\beta}_1} = v_0 \mathbf{I}_D``, *i.e.* a diagonal matrix with a common variance ``v_0``; 
    * the common variance ``v_0`` is often set to a large number, *e.g.``v_0=10^2``* to impose a vague non-informative prior 

## Prior choice (cont.)


**Prior for the observation variance ``p(\sigma)``**


``\sigma^2 >0`` is a positive real number


* a good choice is truncated real-valued distribution such as $\texttt{Half-Cauchy}$ 

$$p(\sigma^2) = \texttt{HalfCauchy}(s_0^{\sigma^2});$$ 

  * Cauchy distributions have heavy tails, suitable choices to express uncertainty 

* ``s_0^{\sigma^2}`` controls the tail of a Cauchy; the larger ``s_0^{\sigma^2}``, the weaker the prior
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

## The full model

To put everything together, the model and the graphical model are

"""

# ╔═╡ 3a21c148-9538-4a0b-92df-67857c8099d7
TwoColumn(
	
md"""

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
""", 
	
	Resource("https://leo.host.cs.st-andrews.ac.uk/figs/bayes/linreg_bayes_priors.png", :height=>240, :align=>"right"))

# ╔═╡ bab3a19c-deb0-4b1c-a8f9-f713d66d9199
md"""
## Connection to ridge regression 

"""

# ╔═╡ b433c147-f547-4234-9817-2b29e7d57219
md"""

Note that 

$$p(\boldsymbol{\beta}_1) = \mathcal N_{D}(\mathbf m_0^{\boldsymbol{\beta}_1}, \mathbf V_0^{\boldsymbol{\beta}_1}),$$

* we usually set ``\mathbf{m}_0^{\beta_1}= \mathbf{0}``, 

$$p(\boldsymbol{\beta}_1) = \mathcal N_{D}(\mathbf{0}, \mathbf V_0^{\boldsymbol{\beta}_1}),$$

* it has a regularisation perspective (also shrinking to zero)


The log posterior density becomes:

$$\begin{align}
\ln p(\boldsymbol{\beta}|\mathbf y, \mathbf{X}) &= \ln p(\boldsymbol{\beta}) +\ln p(\mathbf y|\boldsymbol{\beta}, \mathbf{X}) + C\\
&= -\frac{1}{2v_0} \|\boldsymbol{\beta}\|^2 - \frac{1}{2\sigma^2}\sum_{n=1}^N( y_n-\mathbf{x}_n^\top \boldsymbol{\beta})^2 + C
\end{align}$$

maximising the log posterior is the same as the ridge regression:

```math
\hat{\boldsymbol{\beta}}_{\text{ridge}} \leftarrow \arg\min_{\boldsymbol{\beta}}\frac{\lambda}{2} \|\boldsymbol{\beta}\|^2 + \frac{1}{2}\sum_{n=1}^N( y_n-\mathbf{x}_n^\top \boldsymbol{\beta})^2,

```
* where ``\lambda \triangleq \frac{\sigma^2}{v_0}``, and the estimator is called the **ridge estimator**
"""

# ╔═╡ 824aa830-c958-4063-9ef7-39eeb743fc06
md"""

## Posterior distribution in analytical form*
"""

# ╔═╡ d825e29d-3e0b-4c25-852f-0c9a544aa916
md"""

If we assume ``\sigma^2`` is known, the posterior has a closed-form analytical form (conjugacy again)


```math
p(\boldsymbol{\beta}|\mathcal{D}, \sigma^2) =\mathcal{N}_{D+1}(\mathbf{m}_N, \mathbf{V}_N),
```

where 


```math
\mathbf{m}_N = \mathbf{V}_N\left (\mathbf{V}_0^{-1}\mathbf{m}_0 +\frac{1}{\sigma^2}\mathbf{X}^\top\mathbf{y}\right ) ,\;\;\mathbf{V}_N =\left (\mathbf{V}_0^{-1} + \frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}\right )^{-1}.
```

"""

# ╔═╡ f6cedb98-0d29-40f0-b9f3-430dd283fa36
md"""

## Demonstration

Recall the true parameters are ``\beta_0=\beta_1=3``

A zero-mean vague Gaussian prior is placed on ``\boldsymbol{\beta}``:

```math
p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}\right) = \mathcal{N}\left (\begin{bmatrix}0\\ 0\end{bmatrix}, \begin{bmatrix}5^2& 0 \\ 0 & 5^2\end{bmatrix}\right).
```

* the posterior distribution is sequentially updated

* the posterior converges to ``[3,3]^\top``

* the posterior variance shrinks (or increasing estimation precision)
"""

# ╔═╡ 323a6e91-4cf7-4554-ae6a-2e9bb6621114
function seq_update(x, y, m0, V0, σ²)
	xx = [1  x]
	mn = m0 + V0 * xx'* (dot(xx, V0, xx') + σ²)^(-1)*(y - dot(xx, m0) )
	Vn = inv(1/σ²* xx'* xx + inv(V0))
	return mn[:], Symmetric(Vn)
end

# ╔═╡ 8c9b4743-140d-4806-9552-e117f9956f08
md"""

## Posterior update (Step 0)


```math
p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}\right) = \mathcal{N}\left (\begin{bmatrix}0\\ 0\end{bmatrix}, \begin{bmatrix}5^2& 0 \\ 0 & 5^2\end{bmatrix}\right)
```

* the gray lines are 50 prior samples  

```math
\begin{bmatrix}\beta_0^{(i)}\\ \beta_1^{(i)}\end{bmatrix} \sim p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}\right)= \mathcal{N}\left (\begin{bmatrix}0\\ 0\end{bmatrix}, \begin{bmatrix}5^2& 0 \\ 0 & 5^2\end{bmatrix}\right)
```

"""

# ╔═╡ ac461dcd-9829-4d1d-912a-7a5b8c077ad6
begin
	pMvns = MvNormal[]
	m₀, V₀ = zeros(2), 5^2 * Matrix(1.0I,2,2)
	push!(pMvns, MvNormal(m₀, V₀))

	for i in 1:10
		m₀, V₀ = seq_update(X[i], yy[i], m₀, V₀, σ²)
		push!(pMvns, MvNormal(m₀, V₀))
	end
	
end

# ╔═╡ 8945b750-915d-485b-8dd6-5c77594a17c6
let
	Random.seed!(123)
	iter = 1
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 100)
	ys = range(-10, 10, 100)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ bc4ef6f5-c854-4d6f-9fff-9fcca968bea7
md"""

## Posterior update (Step 1)

After observing the one observation ``\{x^{(1)}, y^{(1)}\}``

```math
\begin{bmatrix}\beta_0^{(i)}\\ \beta_1^{(i)}\end{bmatrix} \sim p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}  \middle \vert \{x^{(1)}, y^{(1)}\} \right)
```

"""

# ╔═╡ 893a9e0d-1dbf-488a-9dd0-32d1ceaaff87
let
	Random.seed!(123)
	iter = 2
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 100)
	ys = range(-10, 10, 100)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ 7cb6b90f-17ff-445e-aa4e-158893f3cf3b
md"""
## Posterior update (Step 2)

After observing the two observation ``\{x^{(1:2)}, y^{(1:2)}\}``

```math
\begin{bmatrix}\beta_0^{(i)}\\ \beta_1^{(i)}\end{bmatrix} \sim p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}  \middle \vert \{x^{(1:2)}, y^{(1:2)}\} \right)
```

"""

# ╔═╡ 4eea6db3-40c9-4dbe-87ef-7e1025de46de
let
	Random.seed!(123)
	iter = 3
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 100)
	ys = range(-10, 10, 100)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ ff93d036-18b5-4afc-94b9-e4ea15c37711
md"""

After observing the two observation ``\{x^{(1:3)}, y^{(1:3)}\}``

```math
\begin{bmatrix}\beta_0^{(i)}\\ \beta_1^{(i)}\end{bmatrix} \sim p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}  \middle \vert \{x^{(1:3)}, y^{(1:3)}\} \right)
```

"""

# ╔═╡ 8b835a09-8e13-4927-93fd-dbcc16226956
let
	Random.seed!(123)
	iter = 4
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 100)
	ys = range(-10, 10, 100)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ 691fe1c6-66a2-45e4-a3ac-8d586493a61f
let
	Random.seed!(123)
	iter = 5
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 100)
	ys = range(-10, 10, 100)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ 01e91145-6738-4b49-831a-3934f37209fb
let
	Random.seed!(123)
	iter = 6
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 100)
	ys = range(-10, 10, 100)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ dbaf2f13-c4a7-47ae-a4a3-fd183632cc23
let
	Random.seed!(123)
	iter = 7
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 100)
	ys = range(-10, 10, 100)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ d1ee75e8-0797-4372-91e5-7d1021ece2f9
let
	Random.seed!(123)
	iter = 11
	mm = 50
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after N=$(iter-1) data")
	plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	spls = rand(pMvns[iter], mm)
	for i in 1:mm
		b, k =  spls[:, i]
		plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
	end
	xs = range(-10, 10, 150)
	ys = range(-10, 10, 150)
	plt_ = heatmap(xs, ys, (x,y)-> pdf(pMvns[iter], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(iter-1) data")	
	plot(plt, plt_, size=(800,400))
end

# ╔═╡ 77817fdb-1b22-49ce-998a-a8de157bf8c4
md"""

## Animation
"""

# ╔═╡ fac530cc-8ad8-4319-a72e-b7c381d656ac
let
	# plts = [plt0, plt1, plt2, plt3, plt4, plt5, plt6, plt10]
	anim = @animate for (iter, mvn) in enumerate(pMvns)
		mm = 25
		plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="", xlim=[-0.5,1.5], ylim=extrema(yy) .+ (-0.5, 0.5), title="Samples after observing "*string(iter-1)*" data" , size=(500, 400))
		plot!(-0.5:0.1:1.5, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
		spls = rand(mvn, mm)
		for i in 1:mm
			b, k =  spls[:, i]
			plot!(-0.5:0.1:1.5, (x) -> k*x+b,  lw=1, ls=:dash, lc=:gray, label="")
		end
	end

	gif(anim, fps=2)
end

# ╔═╡ 71bb2994-7fd2-4e11-b2f1-d88b407f86c1
let
	xs = range(-10, 10, 200)
	ys = range(-10, 10, 200)
	# m₀, V₀ = zeros(2), 10^2 * Matrix(1.0I,2,2)
	posts = []
	anim = @animate for i in 1:10	
		post = heatmap(xs, ys, (x,y)-> pdf(pMvns[i], [x,y]), levels=20, colorbar=false , fill=true, ratio=1, color= :jet1, xlim=[-10, 10], xlabel=L"\beta_0", ylabel=L"\beta_1", title="Update with N=$(i-1) data")
		# m₀, V₀ = seq_update(X[i], yy[i], m₀, V₀, σ²)
		push!(posts, post)
	end

	gif(anim, fps=1)
end

# ╔═╡ d02d5490-cf53-487d-a0c6-651725600f52
md"""

## Interpretation


If we assume the matrix inverse ``(\mathbf{X}^\top\mathbf{X})^{-1}`` exists, 
the posterior's mean can be rewritten as 


```math

\mathbf{m}_N = \left (\mathbf{V}_0^{-1} + \frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}\right )^{-1}\left (\mathbf{V}_0^{-1}\mathbf{m}_0 +\frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}\hat{\boldsymbol{\beta}}\right ) .

```

Defining ``\tilde{\mathbf{V}}^{-1} = \frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}``, the above formula becomes


```math
\mathbf{m}_N = \left (\mathbf{V}_0^{-1} + \tilde{\mathbf{V}}^{-1}\right )^{-1}\left (\mathbf{V}_0^{-1}\mathbf{m}_0 +\tilde{\mathbf{V}}^{-1}\hat{\boldsymbol{\beta}}\right ) .

```


* the posterior mean is a **matrix-weighted average** between 
  * the prior guess ``\mathbf{m}_0`` and 
  * the MLE estimator ``\hat{\boldsymbol{\beta}}``; 

* and the weights are ``\mathbf{V}_0^{-1}``, the precision of the prior guess, and ``\tilde{\mathbf{V}}^{-1}`` the precision for the MLE. 

## Interpretation (cont.)

Consider ``D=1`` (and ``\beta_0=0, \sigma^2=1`` are known)


The posterior degenerates to 


```math
p(\beta_1|\mathcal{D}) = \mathcal{N}(m_N, v_N)
```

where

```math
\begin{align}
m_N &= \frac{v_0^{-1}}{v_0^{-1} + \tilde{v}^{-1}}m_0 + \frac{\tilde{v}^{-1} }{v_0^{-1} + \tilde{v}^{-1}}\hat{\beta}_1\\
v_N^{-1} &=  v_0^{-1} + \tilde{v}^{-1}
\end{align}
```

* ``m_N``: the posterior mean is a weighted average
* ``v_N^{-1}``: the posterior precision is the sum of the precisons


**Shrinkage** given ``m_0``, and if the prior precision ``v_0^{-1}`` is large
* ``m_N \rightarrow m_0``

"""

# ╔═╡ 7dcf736f-958c-43bf-8c15-ec5b27a4650e
md"""


## Demon on the shrinkage effects

If we impose a very strong prior on ``\boldsymbol{\beta}``:

```math
p\left(\begin{bmatrix}\beta_0\\ \beta_1\end{bmatrix}\right) = \mathcal{N}\left (\begin{bmatrix}0\\ 0\end{bmatrix}, \begin{bmatrix}0.1^2& 0 \\ 0 & 0.1^2\end{bmatrix}\right).
```

* the variance is now ``0.1^2``


!!! question "Question"
	What the posterior should look like?

"""

# ╔═╡ 49790b58-9f66-4dd2-bfbc-415c916ae2ab
md"""


The posterior


```math
p(\boldsymbol{\beta}|\mathcal{D}, \sigma^2) =\mathcal{N}(\mathbf{m}_N, \mathbf{V}_N),
```

"""

# ╔═╡ 965faf88-1d33-4c1d-971c-6763cd737145
σ²_prior = 10^2

# ╔═╡ 5dee6633-3100-418c-af3a-d9843e093eab
begin
	σ²_ = σ²_prior
	m₀_, V₀_ = zeros(2), σ²_ * Matrix(1.0I,2,2)
	X_ = [ones(size(X)[1]) X] 
	VN_ = (inv(V₀_) + (1/ σ²) * X_' * X_)^(-1)
	mN_ = VN_ * (inv(V₀_) *  m₀_ + 1/σ² * X_' * yy)
end;

# ╔═╡ 378f8401-310f-4506-bd3b-f9e5e4dae124
mN_

# ╔═╡ 2fd3cddf-12be-40be-b793-142f8f22de39
begin
	plot(Normal(mN_[2], VN_[2, 2]), label=L"p(\beta_1|\mathcal{D})")
	plot!(Normal(mN_[1], VN_[1, 1]), label=L"p(\beta_0|\mathcal{D})")
end

# ╔═╡ b3b1dc37-4ce9-4b3d-b59d-74412cd63c1e
begin


	plot(-5:0.05:5, -5:0.05:5, (x,y)-> pdf(MvNormal(mN_, VN_), [x,y]), seriestype=:contour, colorbar=false , fill=false, ratio=1,  xlim=[-5, 5], levels=10,  xlabel=L"\beta_0", ylabel=L"\beta_1")

	
	# plot!(-5:0.1:5, -5:0.1:5, (x,y)-> pdf(MvNormal(m₀_, V₀_), [x,y]) * 100, levels=3, seriestype=:contour, colorbar=false , fill=false, ratio=1)
end

# ╔═╡ 59dd8a13-89c6-4ae9-8546-877bb7992570
md"""
# With `Turing.jl`



## Simple linear regression 

The model can be specified as:


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

"""

# ╔═╡ 632575ce-a1ce-4a36-95dc-010229367446
md"""

* ``\texttt{HalfCauchy}`` distribution in `Julia`:

```julia
truncated(Cauchy(0, s₀), lower=0)  # HalfCauchy distribution with mean 0 and scale s₀ 
```
"""

# ╔═╡ c0f926f1-85e6-4d2c-8e9a-26cd099fd600
md"""

* ``\texttt{HalfCauchy}(s_0=5)`` has a reasonable coverage

* ``v_0^{\beta_0}= v_0^{\beta_1}=10^2``, leading to a very vague prior (and the true parameters are well covered within the prior's density area).

"""

# ╔═╡ e9bb7a3c-9143-48c5-b33f-e7d6b48cb224
@model function simple_bayesian_regression(Xs, ys; v₀ = 10^2, V₀ = 10^2, s₀ = 5)
	# Priors
	# Gaussian is parameterised with sd rather than variance
	β₀ ~ Normal(0, sqrt(v₀)) 
	β ~ Normal(0, sqrt(V₀))
	# Half-Cauchy prior for the observation variance
	σ² ~ truncated(Cauchy(0, s₀), lower=0)
	# calculate f(x) = β₀ + βx for all observations
	# use .+ to broadcast the intercept to all 
	μs = β₀ .+ β * Xs
	
	# Likelihood
	for i in eachindex(ys)
		# Gaussian in `Distributions.jl` is parameterised by std σ rather than variance
		ys[i] ~ Normal(μs[i], sqrt(σ²))
	end
end

# ╔═╡ 1ef001cc-fe70-42e5-8b97-690bb725a734
md"""

## MCMC inference
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
summarystats(chain_sim_data)

# ╔═╡ ba598c45-a76e-41aa-bfd6-a64bb6cea875
md"""Based on the fact that `rhat < 1.01` and the `ess` count, the chain has converged well. We can also plot the chain to visually check the chain traces."""

# ╔═╡ 391a1dc8-673a-47e9-aea3-ad79f366460d
md"""
## Analysis
"""

# ╔═╡ d3d8fe25-e674-42de-ac49-b83aac402e2d
md"Recall the true parameters are ``\beta_0=3, \beta_1=3, \sigma^2=0.5``. And the inferred posteriors are summarised below. 
"

# ╔═╡ 748c1ff2-2346-44f4-a799-75fb2002c6fc
describe(chain_sim_data)[1]

# ╔═╡ ae720bcd-1607-4155-9a69-bfb6944d5dde
b0, b1 = describe(chain_sim_data)[1][:, :mean][1:2];

# ╔═╡ 08f7be6d-fda9-4013-88f5-92f1b5d26336
md"""

* the posterior's means for the three unknowns are around $(round(b0; digits=2)) and $(round(b1; digits=2)), which are almost the same as the OLS estimators

* which are expected since very weak-informative priors have been used.

"""

# ╔═╡ 40daa902-cb85-4cda-9b9f-7a32ee9cbf1c
describe(chain_sim_data)[2]

# ╔═╡ ab77aef9-920c-423a-9ce0-2b5adead1b7f
density(chain_sim_data)

# ╔═╡ b1d7e28a-b802-4fa1-87ab-7df3369a468a
md"""
## Further analysis

The following diagram shows the model 

$\mu = \beta_0 + \beta_1 x$ 

inferred by the Bayesian method
* The thick red line shows the posterior mean of the Bayesian model
* The lighter lines are some posterior samples, which also indicates the uncertainty about the posterior distribution (the true model is within the prediction)
  * the posterior mean is simply the average of all the posterior samples.
"""

# ╔═╡ 4293298e-52a5-40ca-970d-3f17c2c89adb
let
	parms_turing = describe(chain_sim_data)[1][:, :mean]
	β_samples = Array(chain_sim_data[[:β₀, :β]])
	pred(x) = parms_turing[1:2]' * [1, x]
	plt = scatter(X, yy, xlabel=L"x", ylabel=L"y", label="")
	plot!(0:0.1:1, pred, lw=2, label=L"\mathbb{E}[\mu|\mathcal{D}]", legend=:topleft)
	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	plot!(0:0.1:1, (x) -> β_samples[1, 1] + β_samples[1, 2] *x, lw=0.5, lc=:gray, label=L"\mu^{(r)} \sim p(\mu|\mathcal{D})", legend=:topleft)
	for i in 2:100
		plot!(0:0.1:1, (x) -> β_samples[i, 1] + β_samples[i, 2] *x, lw=0.15, lc=:gray, label="", legend=:topleft)
	end
	plt
end

# ╔═╡ 860c4cf4-59b2-4036-8a30-7fbf44b18648
md"""

## Predictive checks

Recall the procedure is to simulate pseudo samples ``\{\mathbf{y}_{pred}^{(r)}\}`` based on predictive distributions 

```math
\mathbf{y}_{pred} \sim p(\mathbf{y}|\mathcal{D}, \mathbf{X}),
```

* and then the empirical distribution of the simulated data is compared against the observed
"""

# ╔═╡ 74a9a678-60b9-4e3f-97a2-56c8fdc7094f
md"""

* to simulate the pseudo data, we first create a dummy `Turing` with the targets filled with `missing` types. 

* and then use `predict()` method to simulate the missing data.
"""

# ╔═╡ 435036f6-76fc-458b-b0eb-119de02eabb7
pred_y_matrix = let
	# define the predictive model by passing `missing` targets y such that the values will be samples
	pred_model = simple_bayesian_regression(X, Vector{Union{Missing, Float64}}(undef, length(yy)))
	# simulate the predictive pseudo observations
	predict(pred_model, chain_sim_data) |> Array
end;

# ╔═╡ 6eff54f2-d2e0-4c18-8eda-3be3124b16a0
md"""
**Kernel density estimation** check
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
**Summary statistics** check 

* another common visual check is to plot the summary statistics 
  * such as mean and standard deviation (std)


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
## A multiple linear regression example

Consider the *Advertising* dataset which is described in the book [Introduction to Statistical Learning](https://hastie.su.domains/ISLR2/ISLRv2_website.pdf). 


The dataset records how advertising on TV, radio, and in newspapers affects the **sales** of a product
"""

# ╔═╡ b1401e7d-9207-4e31-b5ce-df82c8e9b069
begin
	Advertising = DataFrame(CSV.File(download("https://www.statlearning.com/s/Advertising.csv")))
	first(Advertising, 5)
end

# ╔═╡ 241faf68-49b2-404b-b5d5-99061d1dd2a7
md"""

## Exploratory data analysis 
First, we shall plot the data 

$(begin
	@df Advertising cornerplot([:sales :TV :radio :newspaper], compact = true)
end)

"""

# ╔═╡ 796ad911-9c95-454f-95f4-df5370b2496a
md"""

## Multiple linear model

The multiple linear regression is formulated as:

```math
\texttt{sales} = \beta_0 + \beta_1 \times \texttt{TV} + \beta_2 \times \texttt{radio} + \beta_3 \times \texttt{newspaper} + \varepsilon
```

"""

# ╔═╡ ce97fbba-f5d0-42de-8b67-827c3478d8b0
md"A Frequentist's model is fitted by using `GLM` as a reference first."

# ╔═╡ f3b6f3ab-4304-4ef4-933c-728841c17998
ols_advertising = lm(@formula(sales ~ TV+radio+newspaper), Advertising)

# ╔═╡ ab51e2c3-dbd4-43e1-95b5-a875954ac532
md"

## Bayesian model with `Turing.jl`
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
	replacenames(chain_adv,  Dict(["coefficients[1]" => "TV", "coefficients[2]" => "radio", "coefficients[3]" => "newspaper"]))
end;

# ╔═╡ 9668ab51-f86d-4af5-bf1e-3bef7081a53f
summarystats(chain_adv)

# ╔═╡ ead10892-a94c-40a4-b8ed-efa96d4a32b8
describe(chain_adv)[2]

# ╔═╡ 6a330f81-f581-4ccd-8868-8a5b22afe9b8
md"

* By checking the posterior's 95% credible intervals, we can conclude again newspaper (with a 95% credible interval between -0.013 and 0.011) is not an effective method, 

* which is in agreement with the frequentist's result. The trace and density plot of the posterior samples are listed below for reference."

# ╔═╡ b1f6c262-1a2d-4973-bd1a-ba363bcc5c41
plot(chain_adv)

# ╔═╡ 659a3760-0a18-4a95-8168-fc6ca237c4d5
md"""

# Extensions



## Heterogeneous observation σ

The bayesian approach offers great flexibility

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
md"It can be observed that the observation noise's scale ``\sigma^2`` is not constant across the horizontal axis. 

* with a larger investment on TV, the sales are increasing 
* but also the variance of the sales (check the two Gaussians' scales at two ends of the axis)
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

# ╔═╡ 1954544c-48b4-4997-871f-50c9bfa402b7
md"""

## A better "story"


Assume the observation scale ``\sigma`` itself is a function of the independent variable:

```math
\sigma(x) = r_0 + r_1  x,
```
* if the slope ``r_1>0``, then the observation scale ``\sigma(x)`` will steadily increase over the input ``x``

However, note that ``\sigma`` has to be positive 


## Aside: soft-plus function


``\ln(1+ \exp(\cdot))`` is called 
$(begin
plot(-10:.1:10, (x)-> log(1+exp(x)), label=L"\ln(1+\exp(x))", legend=:topleft, lw=2, size=(400,300))
end)

* ``\texttt{Softplus}``: ``R \rightarrow R^+`` transformation.
"""

# ╔═╡ 85f1d525-3822-47ed-81c0-723d312f8f3f
md"""


## A better "story"

We model ``\sigma``'s transformation ``\rho`` instead, where 

```math
\begin{align}
&\rho(x) = \gamma_0 + \gamma_1  x,\\
&\sigma = \ln(1+\exp(\rho)).
\end{align}
``` 



* the unconstrained observation scale ``\rho \in R`` is linearly dependent on the input ``x``; 

* and a ``\texttt{softplus}`` transformation ``\ln(1+\exp(\cdot))`` is then applied to ``\rho`` such that the output is always positive


"""

# ╔═╡ 85ae4791-6262-4f07-9796-749982e00fec
md"""

## The new Bayesian model


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
	model2 = hetero_var_model(Advertising.TV, Advertising.sales)
	chain2 = sample(model2, NUTS(), MCMCThreads(), 2000, 3)
end;

# ╔═╡ cece17c2-bacb-4b4a-897c-4116688812c6
describe(chain2)

# ╔═╡ 29175b82-f208-4fcf-9704-4a1996cc6e3c
plot(chain2)

# ╔═╡ 05b90ddb-8479-4fbf-a469-d5b0bf4a91c8
md"""
## `generated_quantities()`

**Analyse internel hidden values.** ``\sigma, \mu`` are of importance in evaluating the model:

```math
\sigma(\texttt{TV}) = \ln(1+ \exp(\gamma_0 +\gamma_1 \texttt{TV})).
```

and 

```math
\mu(\texttt{TV}) = \beta_0 +\beta_1 \texttt{TV}.
``` 

we can analyse the posterior distributions of ``\mu`` and ``\sigma`` directly: i.e.
```math
p(\sigma|\texttt{TV}, \mathcal{D}),\;\; p(\mu|\texttt{TV}, \mathcal{D}). 
```

Their posterior distributions can be approximated based on posterior samples

* ``\{\gamma_0^{(r)}, \gamma_1^{(r)}\}_{r=1}^R``, 
* and ``\{\beta_0^{(r)},\beta_0^{(r)}\}_{r=1}^R`` respectively.


"""

# ╔═╡ 2d7f773e-9fa1-475d-9f74-c13908209aeb
begin
	σ(x; γ) = log(1.0 +  exp(γ[1] + γ[2] * x))
	# obtain the posterior mean of the γ
	γ_mean = describe(chain2)[1][:,:mean][3:4]
	# obtain γ samples from the chain object
	γ_samples = Array(group(chain2, :γ))
	tv_input = 0:300
	# calculate each σⁱ
	σ_samples = map(γⁱ -> σ.(tv_input; γ = γⁱ), eachrow(γ_samples))
	# E[σ|𝒟]: the Monte Carlo average
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
## Use `generated_quantities()`

`Turing` provides us with an easy-to-use method:

```
generated_quantities(model, chain)
```  
"""

# ╔═╡ a43189a3-a94f-4e69-85b1-3a586b2cc0eb
begin
	gen_adv = generated_quantities(model2, chain2)
	σs = map((x) -> x.σs, gen_adv)
	μs = map((x) -> x.μs, gen_adv)
end;

# ╔═╡ 4bf4c216-fefb-40fc-9d36-eaa82ff5454b
md"""

And we can plot the posterior samples to visually interpret the results


* the observation's scale ``\sigma`` now increases as investment on *TV* increases

"""

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
## Predictions

We can use `Turing`'s `predict()` method. 

* which is very similar to prior and posterior predictive checks

* we first feed in an array of `missing` values to the Turing model 

* and use `predict` to draw posterior inference on the unknown targets ``y``.


Take the extended Advertising for example, 

* the code below predicts at testing input at 
$\texttt{TV} = 0, 1, \ldots, 300$

"""

# ╔═╡ 61b1e15f-f3ff-4649-b90b-e85e2d172aaf
begin
	ad_test = collect(0:300)
	missing_mod = hetero_var_model(ad_test, Vector{Union{Missing, Float64}}(undef, length(ad_test)))
	ŷ=predict(missing_mod, chain2)
end;

# ╔═╡ 7e992f3e-0337-4b5f-8f4d-c20bdb3b6b66
md"""
**Visualisation**
"""

# ╔═╡ ddc0f5b3-385a-4ceb-82e1-f218299b26d9
begin
	tmp=Array(ŷ)
	ŷ_mean = mean(tmp, dims=1)[:]
	ŷ_std = std(tmp, dims=1)[:]
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
## Handle outlier -- Bayesian robust linear regression


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
**Robust Bayesian regression** is 


* assume the dependent observation is generated with Cauchy noise rather than Gaussian
"""

# ╔═╡ 36fb8291-fea9-4f9f-ac52-a0eb61c2c8a8
begin
	plot(-5:0.1:5, Normal(), fill=(0, 0.2), ylabel="density", label=L"\mathcal{N}(0,1)")
	plot!(-5:0.1:5, Cauchy(), fill=(0, 0.2), label=L"\texttt{Cauchy}(0,1)")
end

# ╔═╡ f67d1d47-1e25-4c05-befd-c958dce45168
md"""
## Bayesian robust regression

The robust Bayesian regression model can be specified by replacing the Gaussian likelihood with its Cauchy equivalent. 



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
## Comparison
"""

# ╔═╡ 40f528f0-acca-43ff-a500-16aeb97898c8
md"""
The MCMC chain of the robust analysis is also listed below. Recall the true regression parameters are ``\beta_0=\beta_1=3``. The posterior distribution correctly recovers the ground truth.  
"""

# ╔═╡ 9fdaefb8-28b4-4b70-8412-1275dc1ed224
describe(chain_robust)[1]

# ╔═╡ 559660b8-1e69-41fe-9139-d031eb26e31c
describe(chain_robust)[2]

# ╔═╡ 139b5acb-c938-4fc9-84a5-fdcbd697b9da
md"""
## Appendix

"""

# ╔═╡ c5ff903e-d220-4e0a-901b-acdece61e465
begin
	Random.seed!(111)
	num_features = 2
	num_data = 25
	true_w = rand(num_features+1) * 10
	# simulate the design matrix or input features
	X_train = [ones(num_data) rand(num_data, num_features)]
	# generate the noisy observations
	y_train = X_train * true_w + randn(num_data)
end;

# ╔═╡ c3b2732f-b9f1-4747-bce3-703a2f03f7d2
let
	plotly()
	# plot(X_train[:,2], y_train, st=:scatter, label="Observations")
	scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="observations", title="Linear regression assumption", xlabel="x₁", ylabel="x₂", zlabel="y")
	surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], true_w),  colorbar=false, xlabel="x₁", ylabel="x₂", zlabel="y", alpha=0.5, label="h(x)")
end

# ╔═╡ d8dae6b8-00fb-4519-ae45-36d36a9c90bb
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

# ╔═╡ 969df742-bc8a-4e89-9e5e-62cb9e7c2215
begin
	chain_outlier_data = sample(simple_bayesian_regression(X_new, yy_new), NUTS(), 2000)
	parms_turing = describe(chain_outlier_data)[1][:, :mean]
	β_samples = Array(chain_outlier_data[[:β₀, :β]])
	pred(x) = parms_turing[1:2]' * [1, x]
	plt_outlier_gaussian = scatter(X_new, yy_new, xlabel=L"x", ylabel=L"y", label="", title="Ordinary Bayesian model")
	plot!(0:0.1:1, pred, lw=2, label=L"\mathbb{E}[\mu|\mathcal{D}]", legend=:topleft)
	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	plot!(0:0.1:1, (x) -> β_samples[1, 1] + β_samples[1, 2] *x, lw=0.5, lc=:gray, label=L"\mu^{(r)} \sim p(\mu|\mathcal{D})", legend=:topleft)
	for i in 2:50
		plot!(0:0.1:1, (x) -> β_samples[i, 1] + β_samples[i, 2] *x, lw=0.15, lc=:gray, label="", legend=:topright)
	end
	plt_outlier_gaussian
end;

# ╔═╡ e585393f-e1bd-4199-984f-5a09745171cf
let
	β_mean = describe(chain_robust)[1][:, :mean]
	β_samples = Array(chain_robust[[:β₀, :β]])
	plt = scatter(X_new, yy_new, xlabel=L"x", ylabel=L"y", label="", title="Robust Bayesian model")
	pred(x; β) = x*β[2] + β[1]
	plot!(0:0.1:1, (x)->pred(x; β = β_mean), lw=2, label=L"\mathbb{E}[\mu|\mathcal{D}]", legend=:topleft)
	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
	plot!(0:0.1:1, (x) -> pred(x; β = β_samples[1,:]), lw=0.5, lc=:gray, label=L"\mu^{(r)} \sim p(\mu|\mathcal{D})", legend=:topleft)
	for i in 2:50
		plot!(0:0.1:1, (x) -> pred(x; β = β_samples[i,:]), lw=0.15, lc=:gray, label="", legend=:topright)
	end
	plot(plt_outlier_gaussian, plt)

end

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
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.4"
GLM = "~1.8.0"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.16"
MLDatasets = "~0.7.9"
PlutoTeachingTools = "~0.2.8"
PlutoUI = "~0.7.39"
StatsPlots = "~0.15.1"
Turing = "~0.21.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "54822bd69db92b2dbb3484e4d78ead5bfa3c4125"

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

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "StaticArrays", "Test"]
git-tree-sha1 = "4a98a9491dd44348664c371998a75074a6938145"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.27"

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

[[deps.BufferedStreams]]
git-tree-sha1 = "bb065b14d7f941b8617bc323063dbe79f55d16ea"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.1.0"

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
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics"]
git-tree-sha1 = "f9d6dd293ed05233d37a5644f880f5def9fdfae3"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.42.0"

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

[[deps.Chemfiles]]
deps = ["Chemfiles_jll", "DocStringExtensions"]
git-tree-sha1 = "9126d0271c337ca5ed02ba92f2dec087c4260d4a"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.31"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d4e54b053fc584e7a0f37e9d3a5c4500927b343a"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.3+0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "0683f086e2ef8e2fdacd3f246b35c59e7088b283"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.0"

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
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

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

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

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

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "bc0a264d3e7b3eeb0b6fc9f6481f970697f29805"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.10"

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

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "ec811a2688b3504ce5b315fe7bc86464480d5964"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.41"

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
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "Distributions", "LinearAlgebra", "MacroTools", "Random", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "c6f574d855670c2906af3f4053e6db10224e5dda"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.19.3"

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

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

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

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

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
git-tree-sha1 = "425e126d13023600ebdecd4cf037f96e396187dd"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.31"

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
git-tree-sha1 = "d88b17a38322e153c519f5a9ed8d91e9baa03d8f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.1"

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

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

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

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "3dab31542b3da9f25a6a1d11159d4af8fdce7d67"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.14"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

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

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "ce28c68c900eed3cdbfa418be66ed053e54d4f56"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.7"

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

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "23e651bbb8d00e9971015d0dd306b780edbdb6b9"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.3"

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

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "42c17b18ced77ff0be65957a591d34f4ed57c631"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.31"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "SnoopPrecompile", "StructTypes", "UUIDs"]
git-tree-sha1 = "84b10656a41ef564c39d2d477d7236966d2b5683"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.12.0"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6a125e6a4cb391e0b9adbd1afa9e771c2179f8ef"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.23"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

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

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

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

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "6eff5740c8ab02c90065719579c7aa0eb40c9f69"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.4"

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

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Tables"]
git-tree-sha1 = "498b37aa3ebb4407adea36df1b244fa4e397de5e"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.9"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "16fa7c2e14aa5b3854bc77ab5f1dbe2cdc488903"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.6.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase", "Transducers"]
git-tree-sha1 = "824e9dfc7509cab1ec73ba77b55a916bb2905e26"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.11"

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

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

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

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

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

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pickle]]
deps = ["DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e6a34eb1dc0c498f0774bbfbbbeff2de101f4235"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.2"

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
git-tree-sha1 = "79830c17fe30f234931767238c584b3a75b3329d"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.6"

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
git-tree-sha1 = "b970826468465da71f839cdacc403e99842c18ea"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.8"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

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
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "ZygoteRules"]
git-tree-sha1 = "4ce7584604489e537b2ab84ed92b4107d03377f0"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.31.2"

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

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "feafdc70b2e6684314e188d95fe66d116de834a7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.2"

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
deps = ["ArrayInterfaceCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "RecipesBase", "RecursiveArrayTools", "StaticArraysCore", "Statistics", "Tables"]
git-tree-sha1 = "3077587613bd4ba73e2acd4df2d1300ef19d8513"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.47.0"

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

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

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
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

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

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "33c0da881af3248dafefb939a21694b97cfece76"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.6"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

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

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0874c1b5de1b5529b776cfeca3ec0acfada97b1b"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.20"

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

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "DiffResults", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "d5e128d1a8db72ebdd2b76644d19128cf90dda29"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.21.9"

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

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

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
# ╟─da388a86-19cb-11ed-0c64-95c301c27153
# ╟─c1026d6b-4e2e-4045-923e-eb5886b45604
# ╟─96708550-225d-4312-bf74-737ab8fe0b4d
# ╟─684f63ec-1e2f-4384-8ddd-f18d2469ebc3
# ╟─b94583a3-7a04-4b30-ba7b-4ff5a72baf5f
# ╟─4ed16d76-2815-4f8c-8ab0-3819a03a1acc
# ╟─c21f64bb-a934-4ec0-b1eb-e3fd6695d116
# ╟─0d3d98c4-fed4-4f4b-be17-bd733e808256
# ╟─94e5a7ed-6332-4f92-8b77-7e0ce7b88a84
# ╟─215c4d7f-ef58-4682-893b-d41b7de75afa
# ╟─c6c3e3aa-fee6-418f-b304-8d5b353bd2d7
# ╟─75567e73-f48a-477c-bc9f-91ce1630468c
# ╟─06f941c1-53c8-4279-8214-0d3ef5c81c4b
# ╟─ab53480e-a9f3-4a86-91cd-aa3168128696
# ╟─a00eb60a-4e90-4002-a758-799fbceab48c
# ╟─c3b2732f-b9f1-4747-bce3-703a2f03f7d2
# ╟─41f6c4fa-89b9-492d-9276-b1651ba92236
# ╟─546e2142-41d2-4c4e-b997-adf8262c3345
# ╟─f6cb99dd-f25c-4770-bba2-8a2496016316
# ╟─510cc569-08eb-4deb-b695-2f3044d758e5
# ╟─b0008150-7aae-4310-bfa8-950ba7bc9092
# ╟─3a46c193-5a25-423f-bcb5-038f3756d7ba
# ╟─c6938b7f-e6e5-4bea-a273-52ab3916d07c
# ╟─effcd3d2-ba90-4ca8-a69c-f1ef1ad697ab
# ╠═3e98e9ff-b674-43f9-a3e0-1ca5d8614327
# ╟─af404db3-7397-4fd7-a5f4-0c812bd90c4a
# ╟─af2e55f3-08b8-48d4-ae95-e65168a23eeb
# ╟─43191a7a-f1a2-41df-910d-cf85907e8f7a
# ╟─98ef1cca-de03-44c2-bcf4-6e79da139e11
# ╟─bc597dc8-dbfe-4a75-81b5-ea37655f95eb
# ╟─f240d7b7-4ea8-4836-81ae-ba1cd169b87d
# ╟─a1421ccb-2d6e-4406-b770-ad7dff007c69
# ╟─e1552414-e701-42b5-8eaf-21ae04a829a8
# ╟─9387dcb4-3f4e-4ec7-8393-30a483e00c63
# ╟─d3f4ac7b-1482-4840-b24b-d08066d1d70c
# ╟─3a21c148-9538-4a0b-92df-67857c8099d7
# ╟─bab3a19c-deb0-4b1c-a8f9-f713d66d9199
# ╟─b433c147-f547-4234-9817-2b29e7d57219
# ╟─824aa830-c958-4063-9ef7-39eeb743fc06
# ╟─d825e29d-3e0b-4c25-852f-0c9a544aa916
# ╟─f6cedb98-0d29-40f0-b9f3-430dd283fa36
# ╟─323a6e91-4cf7-4554-ae6a-2e9bb6621114
# ╟─8c9b4743-140d-4806-9552-e117f9956f08
# ╟─8945b750-915d-485b-8dd6-5c77594a17c6
# ╟─ac461dcd-9829-4d1d-912a-7a5b8c077ad6
# ╟─bc4ef6f5-c854-4d6f-9fff-9fcca968bea7
# ╟─893a9e0d-1dbf-488a-9dd0-32d1ceaaff87
# ╟─7cb6b90f-17ff-445e-aa4e-158893f3cf3b
# ╟─4eea6db3-40c9-4dbe-87ef-7e1025de46de
# ╟─ff93d036-18b5-4afc-94b9-e4ea15c37711
# ╟─8b835a09-8e13-4927-93fd-dbcc16226956
# ╟─691fe1c6-66a2-45e4-a3ac-8d586493a61f
# ╟─01e91145-6738-4b49-831a-3934f37209fb
# ╟─dbaf2f13-c4a7-47ae-a4a3-fd183632cc23
# ╟─d1ee75e8-0797-4372-91e5-7d1021ece2f9
# ╟─77817fdb-1b22-49ce-998a-a8de157bf8c4
# ╟─fac530cc-8ad8-4319-a72e-b7c381d656ac
# ╟─71bb2994-7fd2-4e11-b2f1-d88b407f86c1
# ╟─d02d5490-cf53-487d-a0c6-651725600f52
# ╟─7dcf736f-958c-43bf-8c15-ec5b27a4650e
# ╟─49790b58-9f66-4dd2-bfbc-415c916ae2ab
# ╠═965faf88-1d33-4c1d-971c-6763cd737145
# ╠═378f8401-310f-4506-bd3b-f9e5e4dae124
# ╟─5dee6633-3100-418c-af3a-d9843e093eab
# ╟─2fd3cddf-12be-40be-b793-142f8f22de39
# ╟─b3b1dc37-4ce9-4b3d-b59d-74412cd63c1e
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
# ╟─ae720bcd-1607-4155-9a69-bfb6944d5dde
# ╟─08f7be6d-fda9-4013-88f5-92f1b5d26336
# ╠═40daa902-cb85-4cda-9b9f-7a32ee9cbf1c
# ╠═ab77aef9-920c-423a-9ce0-2b5adead1b7f
# ╟─b1d7e28a-b802-4fa1-87ab-7df3369a468a
# ╟─4293298e-52a5-40ca-970d-3f17c2c89adb
# ╟─860c4cf4-59b2-4036-8a30-7fbf44b18648
# ╟─74a9a678-60b9-4e3f-97a2-56c8fdc7094f
# ╠═435036f6-76fc-458b-b0eb-119de02eabb7
# ╟─6eff54f2-d2e0-4c18-8eda-3be3124b16a0
# ╠═f11a15c9-bb0b-43c1-83e6-dce55f2a772f
# ╟─03b571e6-9d35-4123-b7bb-5f3b31558c9e
# ╠═7cffa790-bf2d-473f-95cc-ed67802d7b4f
# ╟─21aaa2db-6df0-4573-8951-bdfd9b6be6f9
# ╠═b1401e7d-9207-4e31-b5ce-df82c8e9b069
# ╟─241faf68-49b2-404b-b5d5-99061d1dd2a7
# ╟─796ad911-9c95-454f-95f4-df5370b2496a
# ╟─ce97fbba-f5d0-42de-8b67-827c3478d8b0
# ╠═f3b6f3ab-4304-4ef4-933c-728841c17998
# ╟─ab51e2c3-dbd4-43e1-95b5-a875954ac532
# ╠═e1db0ee1-1d91-49c4-be6c-1ea276750bfc
# ╟─80727f81-2440-4da5-8a8b-38ebfdc4ddc9
# ╠═929330bf-beb5-4e42-8668-2a10adf13972
# ╠═9668ab51-f86d-4af5-bf1e-3bef7081a53f
# ╠═ead10892-a94c-40a4-b8ed-efa96d4a32b8
# ╟─6a330f81-f581-4ccd-8868-8a5b22afe9b8
# ╠═b1f6c262-1a2d-4973-bd1a-ba363bcc5c41
# ╟─659a3760-0a18-4a95-8168-fc6ca237c4d5
# ╟─9cb32dc0-8c0c-456e-bbcb-7ff6ae63da60
# ╟─2e7f780e-8850-446e-97f8-51ec26f5e36a
# ╟─65d702d6-abec-43a1-89a8-95c30b3e748a
# ╟─611ad745-f92d-47ac-8d58-6f9baaf3077c
# ╟─1954544c-48b4-4997-871f-50c9bfa402b7
# ╟─85f1d525-3822-47ed-81c0-723d312f8f3f
# ╟─85ae4791-6262-4f07-9796-749982e00fec
# ╠═08c52bf6-0920-43e7-92a9-275ed298c9ac
# ╠═11021fb7-b072-46ac-8c23-f92825182c8c
# ╠═cece17c2-bacb-4b4a-897c-4116688812c6
# ╟─29175b82-f208-4fcf-9704-4a1996cc6e3c
# ╟─05b90ddb-8479-4fbf-a469-d5b0bf4a91c8
# ╠═2d7f773e-9fa1-475d-9f74-c13908209aeb
# ╟─8a0692db-6b85-42f6-8893-4219d58b1032
# ╟─3c3f6378-ed20-4f40-a780-f9329ace34bc
# ╠═a43189a3-a94f-4e69-85b1-3a586b2cc0eb
# ╟─4bf4c216-fefb-40fc-9d36-eaa82ff5454b
# ╟─5a507f39-a6be-43aa-a909-4c87005ad1d2
# ╟─db76ed48-f0c0-4664-b2cf-d47be7faaa3f
# ╟─fc324c67-85be-4c50-b315-9d00ad1dc2be
# ╠═61b1e15f-f3ff-4649-b90b-e85e2d172aaf
# ╟─7e992f3e-0337-4b5f-8f4d-c20bdb3b6b66
# ╠═ddc0f5b3-385a-4ceb-82e1-f218299b26d9
# ╠═9eef45d5-0cd2-460e-940f-bdc7114106c3
# ╟─1d5835ad-4e04-48f0-90e6-aa1776d9406f
# ╟─ee37602c-229c-4709-8b29-627d32a25823
# ╟─0fc9fd84-9fb1-42a1-bd68-8a9fcdce999d
# ╟─9348f3e5-4224-46f5-bc07-dee20b47a8d3
# ╟─36fb8291-fea9-4f9f-ac52-a0eb61c2c8a8
# ╟─f67d1d47-1e25-4c05-befd-c958dce45168
# ╠═e4a67732-ff93-476e-8f5b-8433c1bf015e
# ╠═92206a26-d361-4be5-b215-3ae388dd7f2f
# ╟─c65e9c09-52c5-4647-9414-ad53841e8ff3
# ╟─e585393f-e1bd-4199-984f-5a09745171cf
# ╟─40f528f0-acca-43ff-a500-16aeb97898c8
# ╠═9fdaefb8-28b4-4b70-8412-1275dc1ed224
# ╠═559660b8-1e69-41fe-9139-d031eb26e31c
# ╟─139b5acb-c938-4fc9-84a5-fdcbd697b9da
# ╠═c5ff903e-d220-4e0a-901b-acdece61e465
# ╟─d8dae6b8-00fb-4519-ae45-36d36a9c90bb
# ╠═969df742-bc8a-4e89-9e5e-62cb9e7c2215
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
