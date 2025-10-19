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

# ╔═╡ da388a86-19cb-11ed-0c64-95c301c27153
begin
    using PlutoUI
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using StatsPlots
    using CSV, DataFrames
	using LinearAlgebra
	using Turing
    using Random
	using LaTeXStrings, Latexify
	using PlutoTeachingTools
	using MLDatasets
	using GLM
	using Logging; Logging.disable_logging(Logging.Warn);
end;

# ╔═╡ c1026d6b-4e2e-4045-923e-eb5886b45604
TableOfContents()

# ╔═╡ 96708550-225d-4312-bf74-737ab8fe0b4d
ChooseDisplayMode()

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
# md"""

# ## Supervised learning

# Supervised learning in general 

# * predict *targets* ``Y`` with *covariates* ``X``
# * it tries to access how ``X`` affects ``Y``




# Depending on the type of the labelled targets ``Y``
# * regression*: ``Y`` is continuous real values
# * and *classification*: ``Y`` is categorical 

# In this chapter, we consider regression in Bayesian approach
# """

# ╔═╡ 65f8b976-2151-447a-bdcf-8f0430d5757a
md"""

## What is regression ?

"""

# ╔═╡ 0d3d98c4-fed4-4f4b-be17-bd733e808256
begin
	X_housing = MLDatasets.BostonHousing.features()
	df_house = DataFrame(X_housing', MLDatasets.BostonHousing.feature_names())
	df_house[!, :target] = MLDatasets.BostonHousing.targets()[:]
end;

# ╔═╡ c6c3e3aa-fee6-418f-b304-8d5b353bd2d7
md"""

## Linear regression 


!!! note "Linear regression assumption"
	Linear regression: prediction function ``\mu(\cdot)`` is assumed **linear**

	```math
	\large
	\mu(x_{\text{room}}) = \beta_0 + \beta_1 x_{\text{room}} 
	```
	``\mu(x)`` is called **prediction** function or **regression function**
	* ``\beta_0, \beta_1``: model parameters
"""

# ╔═╡ 06f941c1-53c8-4279-8214-0d3ef5c81c4b
linear_reg_normal_eq(X, y) = GLM.lm(X, y) |> coef

# ╔═╡ 7ca631e2-aafa-4f3f-909b-ba73775ec8c4
TwoColumn(md"""
!!! information "Regression"
	_Supervised learning_ with _continuous_ targets ``y^{(i)} \in \mathbb{R}``
    * input feature ``\mathbf{x}^{(i)}``
    * target ``y^{(i)}``

**Example**: *house price* prediction:  data ``\{\mathbf{x}^{(i)}, y^{(i)}\}`` for ``i=1,2,\ldots, n``

* ``y^{(i)} \in \mathbb{R}``:  house _price_ is continuous
* ``\mathbf{x}^{(i)}``: the average number of rooms""", 
	
	let
	@df df_house scatter(:rm, :target, xlabel="room", ylabel="price", label="", title="House price prediction", size=(350,300))
	x_room = df_house[:, :rm]
	# x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room]
	c, b = linear_reg_normal_eq(X_train_room, df_house.target)

	plot!(3.5:0.5:9, (x) -> b* x+ c, lw=3, label="")
end)

# ╔═╡ 75567e73-f48a-477c-bc9f-91ce1630468c
begin
	@df df_house scatter(:rm, :target, xlabel="room", ylabel="price", label="", title="House price prediction: regression")
	x_room = df_house[:, :rm]
	# x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room]

	c, b = linear_reg_normal_eq(X_train_room, df_house.target)

	plot!(3.5:0.5:9, (x) -> b* x+ c, lw=3, label=L"\mu({x}) = \beta_0+ \beta_1 x", legend=:outerbottom)
end

# ╔═╡ ab53480e-a9f3-4a86-91cd-aa3168128696
md"""

## Multiple linear regression


When the covariate ``\mathbf{x} \in \mathbb{R}^m``

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


#### ``\mu(\mathbf{x})`` now forms a _hyperplane_
"""

# ╔═╡ 41f6c4fa-89b9-492d-9276-b1651ba92236
md"""
## Frequentist's linear regression model


$$\Large y^{(i)} = \boldsymbol{\beta}^\top \mathbf{x}^{(i)} + \epsilon^{(i)}, \;\; \epsilon^{(i)} \sim  \mathcal{N}(0, \sigma^2)$$

* which implies **a probability distribution** for ``y^{(i)}``
  * **Gaussian**: mean $\boldsymbol{\beta}^\top \mathbf{x}^{(i)}$ and variance $\sigma^2$ 


$p(y^{(i)}|\mathbf{x}^{(i)}, \boldsymbol{\beta}, \sigma^2) = \mathcal{N}(y^{(i)};  \boldsymbol{\beta}^\top \mathbf{x}^{(i)} , \sigma^2)= \frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}\left(-\frac{(y^{(i)}-{\boldsymbol{\beta}}^\top\mathbf{x}^{(i)})^2}{2\sigma^2}\right)$

"""

# ╔═╡ f6cb99dd-f25c-4770-bba2-8a2496016316
md"``x_i`` $(@bind xᵢ0 Slider(-0.5:0.1:1, default=0.15));	``\sigma^2`` $(@bind σ²0 Slider(0.005:0.01:0.15, default=0.05))"

# ╔═╡ 546e2142-41d2-4c4e-b997-adf8262c3345
md"input $x^{(i)}=$ $(xᵢ0); and ``\sigma^2=`` $(σ²0)"

# ╔═╡ 510cc569-08eb-4deb-b695-2f3044d758e5
let
	gr()

	Random.seed!(123)
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

# ╔═╡ b0008150-7aae-4310-bfa8-950ba7bc9092
md"""

## Likelihood ``p(\mathbf{y}|\boldsymbol{\beta}, \sigma, \mathbf{X})``








"""

# ╔═╡ fe7503ef-0ad6-4192-b3af-ddcc94e2c36b
TwoColumn(
md"""


The **likelihood**:

```math
\large
p(\mathcal{D}|\boldsymbol{\beta}, \sigma^2, \{\mathbf{x}^{(i)}\}) = \prod_{i=1}^n p(y^{(i)}|\boldsymbol{\beta}, \sigma^2, \mathbf{x}^{(i)})
```

* conditional independence assumed (*i.i.d*)

* the unknown: ``\beta_0, \boldsymbol{\beta}, \sigma^2``

* the observed: ``\mathcal{D} =\{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}``

  * note that ``\{\mathbf{x}^{(i)}\}`` are assumed fixed
""",

Resource("https://leo.host.cs.st-andrews.ac.uk/figs/bayes/regression_freq.png", :height=>310)	

	
)

# ╔═╡ 3a46c193-5a25-423f-bcb5-038f3756d7ba
md"""
## MLE and OLS

Frequentists estimate the model by using maximum likelihood estimation (MLE):


```math
\large
\hat{\beta_0}, \hat{\boldsymbol{\beta}}_1, \hat{\sigma}^2 \leftarrow \arg\max p(\mathbf y|\mathbf X, \beta_0, \boldsymbol{\beta}_1, \sigma^2)
```
It can be shown that the MLE are equivalent to the ordinary least square (OLS) estimator

The closed-form OLS estimator is:

```math
\large
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
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
# md"
# * use `GLM.jl` to fit the OLS estimation "

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
	ols_simulated = lm([ones(N) X], yy)
end;

# ╔═╡ af2e55f3-08b8-48d4-ae95-e65168a23eeb
let
	pred(x) = coef(ols_simulated)' * [1, x]
	scatter(X, yy, xlabel=L"x", ylabel=L"y", label="")
	plot!(0:0.1:1, pred, lw=2, label="OLS fit", legend=:topleft)
	plot!(0:0.1:1, (x) -> β₀ + β₁*x, lw=2, label="True model", legend=:topleft)
end

# ╔═╡ 43191a7a-f1a2-41df-910d-cf85907e8f7a
md"""

## Full Bayesian linear regression model
"""

# ╔═╡ 98ef1cca-de03-44c2-bcf4-6e79da139e11
md"""

The **Bayesian model** reuses the frequentist's likelihood model for ``\mathbf{y}``:

```math
p(y_n|\mathbf{x}_n, \boldsymbol{\beta}, \sigma^2) = \mathcal{N}(y_n; \beta_0+\mathbf{x}_n^\top \boldsymbol{\beta}_1, \sigma^2),
```
where the model parameters are
* ``\beta_0\in \mathbb{R}`` -- intercept
* ``\boldsymbol{\beta}_1 \in \mathbb{R}^D`` -- regression coefficient vector
* ``\sigma^2\in \mathbb{R}^+`` -- Gaussian noise's variance


**In addition**, the Bayesian model also imposes **priors** on the unknowns

```math
p(\beta_0, \boldsymbol{\beta}_1, \sigma^2)
```

* *e.g.* a simple independent prior:
```math
p(\beta_0, \boldsymbol{\beta}_1, \sigma^2)= p(\beta_0)p(\boldsymbol{\beta}_1)p( \sigma^2)
```



"""

# ╔═╡ bc597dc8-dbfe-4a75-81b5-ea37655f95eb
md"""

## Graphical model



"""

# ╔═╡ d910a745-ba64-4ca0-ab92-acdda7262444
TwoColumn(md"""
### Frequentist's model

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/bayes/regression_freq.png", :height=>310, :align=>"left"))
\
\
\
\
\
\
\
\
\
\
\
\


* fixed unknowns
""", 

md"""

### Bayesian model

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/bayes/regression_bayes.png", :height=>310, :align=>"right"))

* prior for the unknown

```math
p(\beta_0, \boldsymbol{\beta}, \sigma^2) = p(\beta_0) p(\boldsymbol{\beta})p(\sigma^2)
```

""")

# ╔═╡ a1421ccb-2d6e-4406-b770-ad7dff007c69
md"""

## Prior choices: ``p(\beta_0)``


**Prior for the intercept ``p(\beta_0)``** 

``\beta_0 \in \mathbb{R}``, a common choice is Gaussian 

$$p(\beta_0) = \mathcal N(m_0^{\beta_0}, v_0^{\beta_0});$$ 

* where the hyper-parameters ``m_0^{\beta_0}, v_0^{\beta_0}`` can be specified based on the data or independently.

## Prior choices: ``p(\beta_0)``


**Prior for the intercept ``p(\beta_0)``** 

``\beta_0 \in \mathbb{R}``, a common choice is Gaussian 

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

**Prior for**  $$p(\boldsymbol{\beta}_1)$$: ``\boldsymbol{\beta}_1\in \mathbb{R}^D`` is also unconstrained

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

# ╔═╡ 2ad5a031-5be1-46c0-8c27-b037217e5b21
md"""

# Conjugate Bayesian linear regression analysis *




"""

# ╔═╡ d825e29d-3e0b-4c25-852f-0c9a544aa916
md"""

If we assume ``\sigma^2`` is known, and we assume the prior on ``\boldsymbol{\beta}`` is Gaussian distributed, *i.e.*

```math
p(\boldsymbol{\beta})\sim \mathcal{N}(\mathbf{m}_0, \mathbf{V}_0)
```



Then it can be shown that the posterior admits a closed-form analytical form (conjugacy again)


```math
p(\boldsymbol{\beta}|\mathcal{D}, \sigma^2) =\mathcal{N}(\mathbf{m}_N, \mathbf{V}_N),
```

where 


```math
\mathbf{m}_N = \mathbf{V}_N\left (\mathbf{V}_0^{-1}\mathbf{m}_0 +\frac{1}{\sigma^2}\mathbf{X}^\top\mathbf{y}\right ) ,\;\;\mathbf{V}_N =\left (\mathbf{V}_0^{-1} + \frac{1}{\sigma^2}\mathbf{X}^\top \mathbf{X}\right )^{-1}.
```

"""

# ╔═╡ 2e46aa25-60c3-487c-bb11-625fd9cffac9
md"""

*Note that here we assume ``\boldsymbol{\beta}`` includes both the bias and the slope parameter.*
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

	# scatter!([X[1]], [yy[1]], ms=8, m=:circle, mc=:red, label="")
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

		scatter!([X[1:2]], [yy[1:2]], ms=8, m=:circle, mc=:red, label="")

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

	scatter!([X[1:5]], [yy[1:5]], ms=8, m=:circle, mc=:red, label="")

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

# ╔═╡ 8f51d185-fabf-4199-9373-62e30f6eb0f5
md"``v_0:`` $(@bind σ²_prior Slider(10.0.^(2:-.2:-5); show_value=true))"

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

# ╔═╡ b3b1dc37-4ce9-4b3d-b59d-74412cd63c1e
begin


	plot(-5:0.05:5, -5:0.05:5, (x,y)-> pdf(MvNormal(mN_, VN_), [x,y]), seriestype=:contour, colorbar=false , fill=false, ratio=1,  xlim=[-5, 5], levels=10,  xlabel=L"\beta_0", ylabel=L"\beta_1")

	
	# plot!(-5:0.1:5, -5:0.1:5, (x,y)-> pdf(MvNormal(m₀_, V₀_), [x,y]) * 100, levels=3, seriestype=:contour, colorbar=false , fill=false, ratio=1)
end

# ╔═╡ 2fd3cddf-12be-40be-b793-142f8f22de39
# begin
# 	plot(Normal(mN_[2], VN_[2, 2]), label=L"p(\beta_1|\mathcal{D})", xlim=[-0.5, 4])
# 	plot!(Normal(mN_[1], VN_[1, 1]), label=L"p(\beta_0|\mathcal{D})")
# end

# ╔═╡ bab3a19c-deb0-4b1c-a8f9-f713d66d9199
md"""
## Connection to ridge regression*

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
begin
	chain_sim_data_ = replacenames(chain_sim_data, Dict("β₀"=> L"\beta_0", "σ²" => L"\sigma^2"))
	density(chain_sim_data_)
end

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
let
	chain_adv_ = replacenames(chain_adv, Dict("σ²" => L"\sigma^2"))
	plot(chain_adv_)
end

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
@model function hetero_var_model(X, y; v₀=10)
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
let
	chain2_ = replacenames(chain2, Dict("β₀" => L"\beta_0", "β₁" => L"\beta_1"))
	plot(chain2_)
end

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
	gr()
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
end;

# ╔═╡ e585393f-e1bd-4199-984f-5a09745171cf
let
	gr()
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
	plt
	plot(plt_outlier_gaussian, plt)
	# plt_outlier_gaussian
	# plt_outlier_gaussian
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
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.10.11"
DataFrames = "~1.7.0"
GLM = "~1.9.0"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.1"
MLDatasets = "~0.7.14"
Plots = "~1.40.8"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.54"
StatsPlots = "~0.15.6"
Turing = "~0.34.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.0"
manifest_format = "2.0"
project_hash = "6c9f7b66148cbe743136ce0dbf17d0aa5c8500c9"

[[deps.ADTypes]]
git-tree-sha1 = "eea5d80188827b35333801ef97a40c2ed653b081"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.9.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "FillArrays", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "c732dd9f356d26cc48d3b484f3fd9886c0ba8ba3"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.5.0"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "Accessors", "DensityInterface", "Random"]
git-tree-sha1 = "6380a9a03a4207bac53ac310dd3a283bb4df54ef"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.8.4"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "ProgressMeter", "Random", "Requires", "Setfield", "SimpleUnPack", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "1da0961a400c28d1e5f057e922ff75ec5d6a5747"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.6.2"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqExt = "OrdinaryDiffEq"

    [deps.AdvancedHMC.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "66ac4c7b320d2434f04d48116db02e73e6dabc8b"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.8.3"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Random", "Random123", "Requires", "SSMProblems", "StatsFuns"]
git-tree-sha1 = "5dcd3de7e7346f48739256e71a86d0f96690b8c8"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.6.0"
weakdeps = ["Libtask"]

    [deps.AdvancedPS.extensions]
    AdvancedPSLibtaskExt = "Libtask"

[[deps.AdvancedVI]]
deps = ["ADTypes", "Bijectors", "DiffResults", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "c217a9b531b4b752eb120a9f820527126ba68fb9"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.2.8"

    [deps.AdvancedVI.extensions]
    AdvancedVIEnzymeExt = ["Enzyme"]
    AdvancedVIFluxExt = ["Flux"]
    AdvancedVIReverseDiffExt = ["ReverseDiff"]
    AdvancedVIZygoteExt = ["Zygote"]

    [deps.AdvancedVI.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "3640d077b6dafd64ceb8fd5c1ec76f7ca53bcf76"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.16.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "995c2b6b17840cd87b722ce9c6cdd72f47bab545"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.3.5"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "2c7cc21e8678eff479978a0a2ef5ce2f51b63dff"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

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

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijections]]
git-tree-sha1 = "d8b0439d2be438a5f2cd68ec158fe08a7b2595b7"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.9"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRules", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "DocStringExtensions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "92edc3544607c4fda1b30357910597e2a70dc5ea"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.13.18"

    [deps.Bijectors.extensions]
    BijectorsDistributionsADExt = "DistributionsAD"
    BijectorsEnzymeExt = "Enzyme"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsReverseDiffExt = "ReverseDiff"
    BijectorsTapirExt = "Tapir"
    BijectorsTrackerExt = "Tracker"
    BijectorsZygoteExt = "Zygote"

    [deps.Bijectors.weakdeps]
    DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tapir = "07d77754-e150-4737-8c94-cd238a1fb45b"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "6c834533dc1fabd820c1db03c839bf97e45a3fab"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.14"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "be227d253d132a6d57f9ccf5f67c0fb6488afd87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.71.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "799b25ca3a8a24936ae7b5c52ad194685fc3e6ef"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.9"
weakdeps = ["InverseFunctions", "Test"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"
    ChangesOfVariablesTestExt = "Test"

[[deps.Chemfiles]]
deps = ["AtomsBase", "Chemfiles_jll", "DocStringExtensions", "PeriodicTable", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "82fe5e341c793cb51149d993307da9543824b206"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.41"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

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

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

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

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0df00546373af8eee1598fb4b2ba480b1ebe895c"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.10"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "Scratch", "p7zip_jll"]
git-tree-sha1 = "8ae085b71c462c2cb1cfedcb10c3c877ec6cf03f"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.13"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

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

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "35b66b6744b2d92c778afd3a88d2571875664a2a"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.2"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

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
weakdeps = ["ChainRulesCore", "DensityInterface", "Test"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "02c2e6e6a137069227439fe884d729cca5b70e56"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.57"

    [deps.DistributionsAD.extensions]
    DistributionsADForwardDiffExt = "ForwardDiff"
    DistributionsADLazyArraysExt = "LazyArrays"
    DistributionsADReverseDiffExt = "ReverseDiff"
    DistributionsADTrackerExt = "Tracker"

    [deps.DistributionsAD.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "490392af2c7d63183bfa2c8aaa6ab981c5ba7561"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.14"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DynamicPPL]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "BangBang", "Bijectors", "Compat", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MacroTools", "OrderedCollections", "Random", "Requires", "Test"]
git-tree-sha1 = "6e5aa7546a0281a1b4bb51a26d0145bb92813a34"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.28.5"

    [deps.DynamicPPL.extensions]
    DynamicPPLChainRulesCoreExt = ["ChainRulesCore"]
    DynamicPPLEnzymeCoreExt = ["EnzymeCore"]
    DynamicPPLForwardDiffExt = ["ForwardDiff"]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]
    DynamicPPLReverseDiffExt = ["ReverseDiff"]
    DynamicPPLZygoteRulesExt = ["ZygoteRules"]

    [deps.DynamicPPL.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    ZygoteRules = "700de1a5-db45-46bc-99cf-38207098b444"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Reexport", "Test"]
git-tree-sha1 = "bbf1ace0781d9744cb697fb856bd2c3f6568dadb"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.6.0"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "e611b7fdfbfb5b18d5e98776c30daede41b44542"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "2.0.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

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

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Expronicon]]
deps = ["MLStyle", "Pkg", "TOML"]
git-tree-sha1 = "fc3951d4d398b5515f91d7fe5d45fc31dccb3c9b"
uuid = "6b7a57c9-7cc1-4fdf-b7f5-e857abae3636"
version = "0.8.5"

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

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "62ca0547a14c57e98154423419d8a342dca75ca9"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

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

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "b10bdafd1647f57ace3885143936749d61638c3b"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.26.0"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "64d8e93700c7a3f28f717d265382d52fac9fa1c1"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.12"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

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

[[deps.GZip]]
deps = ["Libdl", "Zlib_jll"]
git-tree-sha1 = "0085ccd5ec327c077ec5b91a5f937b759810ba62"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.6.2"

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

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

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

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dd3b49277ec2bb2c6b94eb1604d4d0616016f7a6"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+0"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "6c3e57bc26728b99f470b267a437f0d380eac4fc"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.16"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "aeab5c68eb2cf326619bf71235d8f4561c62fe22"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.5"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

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

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "04e52f596d0871fa3890170fa79cb15e481e4cd8"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.28"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "9253429e28cceae6e823bec9ffde12460d79bb38"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LBFGSB]]
deps = ["L_BFGS_B_jll"]
git-tree-sha1 = "e2e6f53ee20605d0ea2be473480b7480bd5091b5"
uuid = "5be7bae1-8223-5378-bac3-9e7378a2f6e6"
version = "0.4.1"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "4ad43cb0a4bb5e5b1506e1d1f48646d7e0c80363"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.2"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LRUCache]]
git-tree-sha1 = "b3cc6698599b10e652832c2f23db3cab99d51b59"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.1"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.L_BFGS_B_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "77feda930ed3f04b2b0fbb5bea89e69d3677c6b0"
uuid = "81d17ec3-03a1-5e46-b53e-bddc35a13473"
version = "3.0.1+0"

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

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

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

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "ed1f362b3fd13f00b65e61d98669c652c17663ab"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.8.7"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ee79c3208e55786de58f8dcccca098ced79f743f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.3"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "4e0128c1590d23a50dcdb106c7e2dbca99df85c0"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.2"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems"]
git-tree-sha1 = "3092250f021aca6d3e24036019cbb24c0c5d89df"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.11.0"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
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
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"
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
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "260dc274c1bc2cb839e758588c63d9c8b5e639d1"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.5"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "d28056379864318172ff4b7958710cfddd709339"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.6"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "8ba8b1840d3ab5b38e7c71c23c3193bb5cbc02b5"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.10"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "361c2692ee730944764945859f1a6b31072e275d"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.18"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ceaff6618408d0e412619321ae43b33b40c1a733"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "b45738c2e3d0d402dffa32b2c1654759a2ac35a4"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.4"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "7715e65c47ba3941c502bffb7f266a41a7f54423"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "70e830dab5d0775183c99fc75e4c24c614ed7142"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.ManifoldDiff]]
deps = ["LinearAlgebra", "ManifoldsBase", "Markdown", "Random", "Requires"]
git-tree-sha1 = "62fccd67f9fb83526bff40a1432f63464eb6d282"
uuid = "af67fdf4-a580-4b9f-bbec-742ef357defd"
version = "0.3.12"

    [deps.ManifoldDiff.extensions]
    ManifoldDiffFiniteDiffExt = "FiniteDiff"
    ManifoldDiffFiniteDifferencesExt = "FiniteDifferences"
    ManifoldDiffForwardDiffExt = "ForwardDiff"
    ManifoldDiffReverseDiffExt = "ReverseDiff"
    ManifoldDiffZygoteExt = "Zygote"

    [deps.ManifoldDiff.weakdeps]
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Manifolds]]
deps = ["Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldDiff", "ManifoldsBase", "Markdown", "MatrixEquations", "Quaternions", "Random", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "58eb09899273a3ed17aae1f56435f440669b810c"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.9.20"

    [deps.Manifolds.extensions]
    ManifoldsBoundaryValueDiffEqExt = "BoundaryValueDiffEq"
    ManifoldsNLsolveExt = "NLsolve"
    ManifoldsOrdinaryDiffEqDiffEqCallbacksExt = ["DiffEqCallbacks", "OrdinaryDiffEq"]
    ManifoldsOrdinaryDiffEqExt = "OrdinaryDiffEq"
    ManifoldsRecipesBaseExt = ["Colors", "RecipesBase"]
    ManifoldsTestExt = "Test"

    [deps.Manifolds.weakdeps]
    BoundaryValueDiffEq = "764a87c0-6b3e-53db-9096-fe964310641d"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    DiffEqCallbacks = "459566f4-90b8-5000-8ac3-15dfb0a30def"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown", "Printf", "Random", "Requires"]
git-tree-sha1 = "4259c5f29dbe9d7441ec0f5ce31c2a6895285495"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.15.17"
weakdeps = ["Plots", "Quaternions", "RecursiveArrayTools", "Statistics"]

    [deps.ManifoldsBase.extensions]
    ManifoldsBasePlotsExt = "Plots"
    ManifoldsBaseQuaternionsExt = "Quaternions"
    ManifoldsBaseRecursiveArrayToolsExt = "RecursiveArrayTools"
    ManifoldsBaseStatisticsExt = "Statistics"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "f765b4eda3ea9be8e644b9127809ca5151f3d9ea"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.4.2"

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

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f12a29c4400ba812841c6ace3f4efbb6dbb3ba01"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "8d39779e29f80aa6c071e7ac17101c6e31f075d7"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.7"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "90077f1e79de8c9c7c8a90644494411111f4e07b"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.5.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "da09a1e112fd75f9af2a5229323f01b56ec96a4c"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.24"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "58e317b3b956b8aaddfd33ff4c3e33199cd8efce"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.3"

[[deps.NamedDims]]
deps = ["LinearAlgebra", "Pkg", "Statistics"]
git-tree-sha1 = "90178dc801073728b8b2d0d8677d10909feb94d8"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "1.2.2"

    [deps.NamedDims.extensions]
    AbstractFFTsExt = "AbstractFFTs"
    ChainRulesCoreExt = "ChainRulesCore"
    CovarianceEstimationExt = "CovarianceEstimation"
    TrackerExt = "Tracker"

    [deps.NamedDims.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    CovarianceEstimation = "587fd27a-f159-11e8-2dae-1979310e6154"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

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

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e25c1778a98e34219a00455d6e4384e017ea9762"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.6+0"

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

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6572fe0c5b74431aaeb0b18a4aa5ef03c84678be"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.3"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LBFGSB", "LinearAlgebra", "Logging", "LoggingExtras", "OptimizationBase", "Printf", "ProgressLogging", "Reexport", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "0b2a631276dc92ab147535689fa43f1e22a657b8"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "3.28.0"

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "PDMats", "Reexport", "Requires", "SciMLBase", "SparseArrays", "SymbolicAnalysis", "SymbolicIndexingInterface", "Symbolics"]
git-tree-sha1 = "3e5e5e8cbe572200dcd94a6084a63ca68fe76279"
uuid = "bca83a33-5cc9-4baa-983d-23429ab6bcbb"
version = "1.5.0"

    [deps.OptimizationBase.extensions]
    OptimizationEnzymeExt = "Enzyme"
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMTKExt = "ModelingToolkit"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSparseDiffExt = ["SparseDiffTools", "ReverseDiff"]
    OptimizationTrackerExt = "Tracker"
    OptimizationZygoteExt = "Zygote"

    [deps.OptimizationBase.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseDiffTools = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationOptimJL]]
deps = ["Optim", "Optimization", "Reexport", "SparseArrays"]
git-tree-sha1 = "43870d726f883a47d158beebb1fc3c9fab1da9d6"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.3.2"

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

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PeriodicTable]]
deps = ["Base64", "Unitful"]
git-tree-sha1 = "238aa6298007565529f911b734e18addd56985e1"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.2.1"

[[deps.Pickle]]
deps = ["BFloat16s", "DataStructures", "InternedStrings", "Mmap", "Serialization", "SparseArrays", "StridedViews", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e99da19b86b7e1547b423fc1721b260cfbe83acb"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.5"

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

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "1a9cfb2dc2c2f1bd63f1906d72af39a79b49b736"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.11"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

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

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "b034171b93aebc81b3e1890a036d13a9c4a9e3e0"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.27.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

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

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "3a7c7e5c3f015415637f5debdf8a674aa2c979c4"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.1"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SSMProblems]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "f640e4e8343c9d5f470e2f6ca6ce79f708ab6376"
uuid = "26aad666-b158-4e64-9d35-0e672562fa48"
version = "0.1.1"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "Expronicon", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "50ed64cd5ad79b0bef71fdb6a11d10c3448bfef0"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.56.1"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "e39c5f217f9aca640c8e27ab21acf557a3967db5"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.10"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "25514a6f200219cd1073e4ff23a6324e4a7efe64"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.5.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

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
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

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

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

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

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

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
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "5b765c4e401693ab08981989f74a36a010aa1d8e"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.2.2"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"
weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

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

[[deps.SymbolicAnalysis]]
deps = ["DSP", "DataStructures", "Dictionaries", "Distributions", "DomainSets", "IfElse", "LinearAlgebra", "LogExpFunctions", "Manifolds", "PDMats", "RecursiveArrayTools", "StatsBase", "SymbolicUtils", "Symbolics"]
git-tree-sha1 = "64f26bb4a666bb97baa16f063164ade83ca29ec9"
uuid = "4297ee4d-0239-47d8-ba5d-195ecdf594fe"
version = "0.3.0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "4bc96df5d71515b1cb86dd626915f06f4c0d46f5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.33"

[[deps.SymbolicLimits]]
deps = ["SymbolicUtils"]
git-tree-sha1 = "fabf4650afe966a2ba646cabd924c3fd43577fc3"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "0.2.2"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "ArrayInterface", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TermInterface", "TimerOutputs", "Unityper"]
git-tree-sha1 = "04e9157537ba51dad58336976f8d04b9ab7122f0"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "3.7.2"

    [deps.SymbolicUtils.extensions]
    SymbolicUtilsLabelledArraysExt = "LabelledArrays"
    SymbolicUtilsReverseDiffExt = "ReverseDiff"

    [deps.SymbolicUtils.weakdeps]
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.Symbolics]]
deps = ["ADTypes", "ArrayInterface", "Bijections", "CommonWorldInvalidations", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "IfElse", "LaTeXStrings", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "PrecompileTools", "Primes", "RecipesBase", "Reexport", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "aa3218c29b81384531631b2e5354fdf034a13ec2"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "6.14.1"

    [deps.Symbolics.extensions]
    SymbolicsForwardDiffExt = "ForwardDiff"
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsLuxExt = "Lux"
    SymbolicsNemoExt = "Nemo"
    SymbolicsPreallocationToolsExt = ["PreallocationTools", "ForwardDiff"]
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
    Nemo = "2edaba10-b0f1-5616-af89-8c11ac63239a"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

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

[[deps.TermInterface]]
git-tree-sha1 = "d673e0aca9e46a2f63720201f55cc7b3e7169b16"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "2.0.0"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3a6f063d690135f5c1ba351412c82bae4d1402bf"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.25"

[[deps.Tracker]]
deps = ["Adapt", "ChainRulesCore", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "da45269e1da051c2a13624194fcdc74d6483fad5"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.35"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.Turing]]
deps = ["ADTypes", "AbstractMCMC", "Accessors", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "Compat", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MCMCChains", "NamedArrays", "Optimization", "OptimizationOptimJL", "OrderedCollections", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f69f33f7862a66674c279fe9b86b457a96767e35"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.34.1"

    [deps.Turing.extensions]
    TuringDynamicHMCExt = "DynamicHMC"
    TuringOptimExt = "Optim"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "2d17fabcd17e67d7625ce9c531fb9f40b7c42ce4"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.2.1"

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
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

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

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46bf7be2917b59b761247be3f317ddf75e50e997"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.2+0"

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
# ╟─da388a86-19cb-11ed-0c64-95c301c27153
# ╟─c1026d6b-4e2e-4045-923e-eb5886b45604
# ╟─96708550-225d-4312-bf74-737ab8fe0b4d
# ╟─684f63ec-1e2f-4384-8ddd-f18d2469ebc3
# ╟─b94583a3-7a04-4b30-ba7b-4ff5a72baf5f
# ╟─4ed16d76-2815-4f8c-8ab0-3819a03a1acc
# ╟─65f8b976-2151-447a-bdcf-8f0430d5757a
# ╟─7ca631e2-aafa-4f3f-909b-ba73775ec8c4
# ╟─0d3d98c4-fed4-4f4b-be17-bd733e808256
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
# ╟─fe7503ef-0ad6-4192-b3af-ddcc94e2c36b
# ╟─3a46c193-5a25-423f-bcb5-038f3756d7ba
# ╟─c6938b7f-e6e5-4bea-a273-52ab3916d07c
# ╟─af2e55f3-08b8-48d4-ae95-e65168a23eeb
# ╟─effcd3d2-ba90-4ca8-a69c-f1ef1ad697ab
# ╟─3e98e9ff-b674-43f9-a3e0-1ca5d8614327
# ╟─af404db3-7397-4fd7-a5f4-0c812bd90c4a
# ╟─43191a7a-f1a2-41df-910d-cf85907e8f7a
# ╟─98ef1cca-de03-44c2-bcf4-6e79da139e11
# ╟─bc597dc8-dbfe-4a75-81b5-ea37655f95eb
# ╟─d910a745-ba64-4ca0-ab92-acdda7262444
# ╟─a1421ccb-2d6e-4406-b770-ad7dff007c69
# ╟─e1552414-e701-42b5-8eaf-21ae04a829a8
# ╟─9387dcb4-3f4e-4ec7-8393-30a483e00c63
# ╟─d3f4ac7b-1482-4840-b24b-d08066d1d70c
# ╟─3a21c148-9538-4a0b-92df-67857c8099d7
# ╟─2ad5a031-5be1-46c0-8c27-b037217e5b21
# ╟─d825e29d-3e0b-4c25-852f-0c9a544aa916
# ╟─2e46aa25-60c3-487c-bb11-625fd9cffac9
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
# ╟─8f51d185-fabf-4199-9373-62e30f6eb0f5
# ╟─378f8401-310f-4506-bd3b-f9e5e4dae124
# ╟─b3b1dc37-4ce9-4b3d-b59d-74412cd63c1e
# ╟─5dee6633-3100-418c-af3a-d9843e093eab
# ╟─2fd3cddf-12be-40be-b793-142f8f22de39
# ╟─bab3a19c-deb0-4b1c-a8f9-f713d66d9199
# ╟─b433c147-f547-4234-9817-2b29e7d57219
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
# ╟─ab77aef9-920c-423a-9ce0-2b5adead1b7f
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
# ╟─b1f6c262-1a2d-4973-bd1a-ba363bcc5c41
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
# ╟─969df742-bc8a-4e89-9e5e-62cb9e7c2215
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
