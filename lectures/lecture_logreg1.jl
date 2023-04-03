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

# ╔═╡ 6a1f3828-667a-11ed-2ea0-154032459b81
begin
	using PlutoTeachingTools
	using PlutoUI
	using Plots
	using LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using Distributions
end

# ╔═╡ 7c49ffb7-dc5f-4cd5-ada5-44719e24b8a6
begin
	# using DataFrames, CSV
	using MLDatasets
	# using Images
end

# ╔═╡ 18a77849-f40d-4eb6-aa5d-07f7d4ec007d
using Images

# ╔═╡ b741192c-d6b8-435f-b553-0aa5ebed984d
using MCMCChains

# ╔═╡ 0b01ac40-7054-4671-a370-483cce6b2231
TableOfContents()

# ╔═╡ 07d11cca-30a6-414c-bdd6-176de80e8b33
md"""

# Bayesian logistic regression 1


$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang (lf28@st-andrews.ac.uk)

*School of Computer Science*

*University of St Andrews, UK*

*April 2023*

"""

# ╔═╡ 7547a998-0652-4c8a-ac63-c0daf03b72cf
md"""
## Binary classification



* input features: ``\mathbf{x} \in R^m``

* output label: ``y^{(i)} \in \{0,1\}``


Below is a two-dimensional example ``m=2``
"""

# ╔═╡ 8879917c-072c-41b2-947f-7e65f12faa58
md"""

## Logistic function 


> Idea: apply some "activation" function to squeeze the hyperplane to 0 and 1
> * which can then be interpreted as **probability** of ``y^{(i)}`` being class 1

"""

# ╔═╡ 38bf8ca4-2507-45cf-8036-1f9b6ba0a66a
html"<center><img src='https://carpentries-incubator.github.io/ml4bio-workshop/assets/Linreg-vs-logit_.png
' width = '400' /></center>"

# ╔═╡ a6ad102a-9ff7-4ed9-8ab6-fd7790d8d37a
md"""


## Logistic function

Such an activation function is **logistic function** (also known as **sigmoid**)

```math
\texttt{logistic}(x) \triangleq \sigma(x) = \frac{1}{1+e^{-x}}.
``` 

* we use ``\sigma`` as a shorthand notation for the logistic activation

$(begin
gr()
plot(-10:0.2:10, logistic, c=7, xlabel=L"x",  label=L"\texttt{logistic}(x)", legend=:topleft, lw=3, size=(450,300), ylabel=L"P(y=1)")
end)



* it squeezes a line to 0 and 1
  * ``x\rightarrow \infty``, ``\sigma(x) \rightarrow 1``
   * ``x\rightarrow -\infty``, ``\sigma(x) \rightarrow 0``
"""

# ╔═╡ 0f225525-1a86-4fab-9a8f-3d6a3a5135f6
md"""

## Logistic regression function



```math
\sigma(h(x)) = \sigma(w_1 x+ w_0) = \frac{1}{1+e^{-w_1x -w_0}}
```
"""

# ╔═╡ 3b33095d-b4be-4924-95ee-2b70ace93352
@bind w₁_ Slider([(0.0:0.0001:1.0)...; (1.0:0.005:20)...], default=1.0)

# ╔═╡ 69f07beb-4110-4f7e-87cf-efd2fd5f88a8
md"Slope: ``w_1`` = $(round(w₁_; digits=2))"

# ╔═╡ 5ed3dc0a-8a73-4024-9a8a-c333936329c3
md"Intercept: ``w_0``"

# ╔═╡ 0685a343-9a21-4b1a-80bc-d41f6615f22c
@bind w₀_ Slider(-10:0.5:10, default=0, show_value=true)

# ╔═╡ adebb65b-200e-4f07-93f0-934c55a66dcc
begin
	k, b= w₁_, w₀_
	gr()

	plot(-20:0.1:20, (x) -> k* x+ b, ylim=[-0.2, 1.2], label=L"h(x) =%$(round(w₁_; digits=2)) x + %$(w₀_)", lw=2, l=:gray, ls=:dash, ylabel=L"\sigma(x) \triangleq P(y=1|x)")
	
	plot!(-20:0.1:20, (x) -> logistic( x ), xlabel=L"x", label=L"\sigma(x)", legend=:topleft, lw=2, size=(450,300), c=1)
	
	plot!(-20:0.1:20, (x) -> logistic(k * x + b), xlabel=L"x", label=L"\sigma(%$(round(w₁_; digits=2)) x + %$(w₀_))", legend=:topleft, lw=2, c=7, size=(450,300))
end

# ╔═╡ a0474cb7-be98-449b-a5a7-ef8bac52fda6
md"""

## Logistic regression function


Now feed a hyperplane 

```math
\mu(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}
```


```math
\texttt{logistic}(\mu(\mathbf{x})) = σ(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{x}}}
```
"""

# ╔═╡ 06bfedf9-f4e2-4885-b466-842362e4d739
wv_ = [1, 1] * 1

# ╔═╡ 6d9ff057-1ebd-43bd-ac2e-cbfb1da6dd98
md"""

## Logistic regression's prediction function


The model 
```math
\sigma(\mathbf{x}) = \frac{1}{1+ e^{- \mathbf{w}^\top\mathbf{x}}}
``` 

can be represented as a **computational graph**

* the output is between 0 and 1
  * which can be interpreted as a probability!
* a building block of a neural network

"""

# ╔═╡ e759ec49-c1ae-491d-b5bd-1043e934aa87
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/logistic_reg_neuron.png
' width = '400' /></center>"

# ╔═╡ ee737d2c-d7d9-464b-8cd5-91b0b97f07d0
html"<center><img src='https://carpentries-incubator.github.io/ml4bio-workshop/assets/logit_nodes.png' width = '400' /></center>";

# ╔═╡ ad9019e9-5ffd-4d18-8f67-0d7a41d08a56
md"""

## Probabilistic model for logistic regression



Since ``y^{(i)} \in \{0,1\}``, a natural choice of likelihood model is Bernoulli (coin tossing)

* replace the bias with the sigmoid output!


```math
p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
\\
1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
```

* for notational convenience, we use short hand ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``


"""

# ╔═╡ 9c844e0d-6396-46e6-8de2-4c7bb18254c2
function ∇logistic_loss(w, X, y)
	σ = logistic.(X * w)
	X' * (σ - y)
end

# ╔═╡ 4c0fc54b-9184-4f0b-8224-bf0774d844f1
logistic_loss(w, data, targets) = - sum(logpdf.(BernoulliLogit.(data * w), targets))

# ╔═╡ 1d87e445-984a-4544-88e3-780c681fb737
md"""

## MLE -- gradient descent


"""

# ╔═╡ 74b9b15d-a8ab-42b9-a5c3-74e78994e4e6
md"""

## Case study: real-world dataset -- MNIST


Let's classify `0` (positive) from non-zeros `{1,2,…,9}` (negative)

* it becomes a **binary classification**

Flatten the input ``28 \times 28`` to a long ``28^2+1`` vector 
* +1: dummy one for the intercept
* and feed it to a logistic regression/single neuron
"""

# ╔═╡ bbbf3716-ec62-4f41-97a1-1fa2563481e3
md"""
## Visualise the learnt parameter

The learnt features ``\hat{\mathbf{w}}`` is a ``28^2+1`` long vector
* we can view it as an image
* the classifier *does not like* the blue pixels for an image to be `0`
  * *e.g.* anything in the centre 
  * or the centre of an image should be blank
"""

# ╔═╡ 3a4df100-cf7b-46fa-8b72-bc0aa52d1c6a
mnist_train_X, mnist_train_ys = MLDatasets.MNIST.traindata();

# ╔═╡ 10d0b2bd-222c-46f7-abc3-b70b8425add7
let
	mnist_idx_by_digits = [findall(mnist_train_ys .== d) for d in 0:9];
	num_each_digit = 1
	reshape([Gray.(mnist_train_X[:,:,i]) for d in 1:length(mnist_idx_by_digits) for i in mnist_idx_by_digits[d][1:num_each_digit]], 10, num_each_digit)'
end

# ╔═╡ 12297994-3faa-4d6c-9cfa-b1cb69ac5550
mnist_test_X, mnist_test_ys = MLDatasets.MNIST.testdata();

# ╔═╡ 1dff88ba-9573-479f-abe8-5079acdc2654
function flatten_images(X) # flatten the matrix input to a vector
	vcat([X[:,:,i][:]' for i in 1:size(X)[3]]...)
end;

# ╔═╡ 2d32c35b-5ee1-44a6-ad0f-c96750f5c946
mnist_zeros_xs_train, mnist_zeros_ys_train, mnist_zeros_xs_test, mnist_zeros_ys_test = let #prepare the dataset
	Random.seed!(123)
	train_size = 500
	zeros_ids = mnist_train_ys .== 0
	zeros_indices = shuffle!(findall(zeros_ids))[1:train_size]
	non_zero_indices = shuffle!(findall(.!zeros_ids))[1: 5*train_size]
	x_zeros = mnist_train_X[:, :, zeros_indices]
	X_zeros = flatten_images(x_zeros)
	x_nzeros = mnist_train_X[:, :, non_zero_indices]
	X_nzeros = flatten_images(x_nzeros)
	Xs_zeros = [X_zeros; X_nzeros]
	ys_zeros = [ones(Bool, length(zeros_indices)); zeros(Bool, length(non_zero_indices))]
	Xs_zeros_ = [ones(size(Xs_zeros)[1]) Xs_zeros]
	X_zeros_test = [ones(length(mnist_test_ys)) flatten_images(mnist_test_X)]
	ys_zeros_test = mnist_test_ys .== 0
	Xs_zeros_, ys_zeros, X_zeros_test, ys_zeros_test
end;

# ╔═╡ 80df1e62-5b92-437d-b409-b974186cdf19
accuracy(y, ŷ) = mean(y .== ŷ);

# ╔═╡ 1f23ecaa-c5f6-4a25-8c12-115a96868a93
predict(w, X) = logistic.(X * w) .> 0.5;

# ╔═╡ 3cb006a6-5c80-4300-907f-9eaae67c979a
md"""

## MLE is problematic

### Another dataset

* 10 data points in total
"""

# ╔═╡ f7b465c7-951b-4abd-9252-5260c1e2c8b2
md"""

## MLE overfitting



When the data is linearly separable

* MLE leads to very large scale ``\mathbf{w}``

* ``\mathbf{w} \rightarrow \infty``, ``\text{loss} \rightarrow 0``


The sharper the better! overfitting 
"""

# ╔═╡ 55ffe10e-899d-456f-ba62-12d51a072843
md"""
## Frequentist plug-in prediction

The prediction function for a new test data ``\mathbf{x}_{test}`` is 
* *i.e.* MLE plug-in prediction

```math
p(y_{test} =1 |\hat{\mathbf{w}}_{\text{ML}}, \mathbf{x}_{test}) =\sigma(\hat{\mathbf{w}}_{\text{ML}}^\top\mathbf{x}_{test})
```
* black and white classification
"""

# ╔═╡ 2ad1858b-fcd3-4cc8-be70-97746480eacd
md"""


## Ridge regularisation -- overfitting



!!! note "Solution: regularised or MAP"
	Solution: add a penalty term or MAP estimator
	```math
		\hat{\mathbf{w}}_{\text{MAP}}\leftarrow \arg\min_{\mathbf{w}} \underbrace{L(\mathbf{w}) + \frac{\lambda }{2} \mathbf{w}^\top \mathbf{w}}_{\text{regularised loss}}
	```


The new gradient becomes

```math
\nabla_{\mathbf{w}}L(\mathbf{w})  + \lambda \cdot \mathbf{w}
```
* apply gradient descent with the new gradient
"""

# ╔═╡ cf104a92-22e8-4995-96d1-869b1ea1c4a2
function ∇logistic_regularised(w, X, y; λ = 0.02)
	∇logistic_loss(w, X, y) + λ * w
end;

# ╔═╡ dbf31225-55a1-4d3d-8758-bcab0e653f55
md"""
## Comparison 


* MLE (``\textcolor{blue}{\text{blue}}`` lines) converge to larger scale
* MAP (the ridge logistic regression)(``\textcolor{orange}{\text{orange}}`` lines) more regularised; thanks to regularisation (or add the zero mean Gaussian prior)

"""

# ╔═╡ daeaf752-7f43-4afb-abdc-73a3ad7f2301
md"""

## Comparison

"""

# ╔═╡ b2888f9b-c40a-4ee9-bdbd-898eb89df5be
md"""


The regularised or MAP prediction, still a plug-in prediction *b.t.w.*, 

```math
\begin{align}
p(y_{test} =1 |\hat{\mathbf{w}}_{\text{ML}}, \mathbf{x}_{test}) &=\sigma(\hat{\mathbf{w}}_{\text{ML}}^\top\mathbf{x}_{test}) \\

&vs\\

p(y_{test} =1 |\hat{\mathbf{w}}_{\text{MAP}}, \mathbf{x}_{test}) &=\sigma(\hat{\mathbf{w}}_{\text{MAP}}^\top\mathbf{x}_{test})

\end{align}
```
"""

# ╔═╡ e30b60a4-0625-4f68-83bd-481b623616c9
md"""

## MAP is not perfect either


The MAP prediction is still just **plug in**
* the straight-line extrapolated prediction is **not natural**
  * poor generalisation performance
* data points further away from the training data should be ``\Rightarrow`` less certain

"""

# ╔═╡ 4840b4b2-f8ad-4a43-95c1-750d2cb39524
md"""

## Bayesian inference

Full Bayesian preview

* provides a very nuanced, reasonable and intelligent prediction 

* attention to the finest details

* uncertainties well accounted for -- good generalisation performance!
"""

# ╔═╡ e71c8dd8-2f5e-4038-88cd-ea2863adc3b0
md"""

## Comparison

"""

# ╔═╡ 7717a633-822e-4347-bc39-17caa6cbcbd0
md"""

## Bayesian logistic regression in BN

"""

# ╔═╡ 03471888-cdf2-4e32-a06c-381e1b66e6d7
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/logistic_first.png' width = '220' /></center>"

# ╔═╡ 924b8e0e-dc4e-43b6-a948-554e5bc956d3
md"""

The CPF for ``y^{(i)}`` given its parents: ``\mathbf{x}^{(i)}``, and ``\mathbf{w}`` is
Bernoulli with a logistic transformed bias

```math
p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}) = \begin{cases} \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})& y^{(i)} = 1\\ 1- \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})& y^{(i)} = 0\end{cases} 
```

The prior ``p(\mathbf{w})`` 

```math
p(\mathbf{w}) = \prod_{j=1}^m \mathcal{N}({0}, 1/\lambda)
```
* ``m`` variate (``m`` features) independent zero mean Gaussians with variance ``1/\lambda``



"""

# ╔═╡ fc448fa2-f764-4d0a-b50c-6283ecebee81
aside(tip(md"""``\lambda = \frac{1}{\sigma^2}`` is the precision of the Gaussian prior.""" ))

# ╔═╡ 1dd1b222-6e5e-47f1-9c5c-cfa2f847075f
md"""


## Bayesian logistic regression in BN with prediction

Now what if we want to predict ``\mathbf{x}_{test} \in \mathbf{D}_{test}``?
* add an unknown node
* the hyperparameter ``\lambda`` (Gaussian prior's precision) is also added for completeness
"""

# ╔═╡ 340944a7-7a14-4a68-984e-d6924fc13cf1
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/logistic_pred.png' width = '220' /></center>"

# ╔═╡ a67aa14f-f4c0-4214-813e-fcc56aee852c
md"""

## Bayesian prediction


The objective is (other fixed nodes are omitted)

```math
p(y_{test} |\mathbf{x}_{test}, \mathbf{y}, \mathbf{X})
```

Apply exact inference routinely:


```math
\begin{align}
p(y_{test} |\mathbf{x}_{test}, \mathbf{y}, \mathbf{X})
&= \int  p(y_{test}|\mathbf{x}_{test}, \mathbf{w})  \underbrace{p(\mathbf{w}|\mathbf{y}, \mathbf{X})}_{\text{posterior!}} d\mathbf{w}
\end{align}
```


A weighted average of multiple predictions

* ensemble method: democratic and balanced
* we do not know the true ``\mathbf{w}``: we may as well just let every ``\mathbf{w} \in R^m`` to predict then take average
  * the average is weighted by ``p(\mathbf{w}|\mathbf{y}, \mathbf{X})``
"""

# ╔═╡ b710b9c1-f842-4c97-bda3-b4e6f324a9c9
md"""

## Markov Chain Monte Carlo estimation


We can resort to MCMC method, ``M`` samples are drawn from the posterior

```math
\mathbf{w}^{(m)} \sim p(\mathbf{w}|\mathbf{X}, \mathbf{y})
```

* check appendix for details: where a Metropolis sampler is implemented

Then approximate the integration with Monte Carlo method

```math
\begin{align}
\hat{p}_{MC}(y_{test}=1 |\mathbf{x}_{test}, \mathbf{y}, \mathbf{X}) 
&= \frac{1}{M}\sum_{m=1}^M p(y_{test}=1|\mathbf{x}_{test}, \mathbf{w}^{(m)})  \\
&= \frac{1}{M}\sum_{m=1}^M \sigma((\mathbf{w}^{(m)})^\top\mathbf{x}_{test}) 
\end{align}

```
"""

# ╔═╡ af74c191-0176-4197-945e-174c0456bd66
begin
	# monte carlo prediction
	# ws: MCMC samples, assumed a mc × 3 vector
	# demonstration for 2-d case
	prediction(ws, x1, x2) = mean(logistic.(ws * [1.0, x1, x2]))
end

# ╔═╡ f09f3b01-623d-4cd0-9d6f-939787aed445
md"""

## Demonstration


100 ``\mathbf{w}^{(m)}`` induced predictions (dash gray lines) are plotted below

Full Bayesian use all of them to make a prediction!
* the final prediction is an average of all ``M`` ensemble's prediction


* a very natural curved prediction surface emerges by this average
"""

# ╔═╡ 66e8bc6e-61dd-4b5d-8130-b96f03d92bf8
md"""

# Appendix


Reading for Bayesian machine learning

* [Bayesian Regression and Classification by *Christopher M. Bishop* and *Michael E. Tipping* 2003](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-nato-bayes.pdf)
"""

# ╔═╡ b07efeea-75c9-4847-ba45-752ff9b34d53
md"""

### Data generation
"""

# ╔═╡ af30c343-64bd-45e6-be64-59f1f0413cc5
D₂, targets_D₂, targets_D₂_=let
	Random.seed!(123)
	D_class_1 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .+2
	D_class_2 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .-2
	data₂ = [D_class_1; D_class_2]
	D₂ = [ones(size(data₂)[1]) data₂]
	targets_D₂ = [ones(size(D_class_1)[1]); zeros(size(D_class_2)[1])]
	targets_D₂_ = [ones(size(D_class_1)[1]); -1 *ones(size(D_class_2)[1])]
	D₂, targets_D₂,targets_D₂_
end

# ╔═╡ a8c421e9-369e-438b-9388-9b4c5fd03484
let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", c=2)
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
end

# ╔═╡ 1e6c5c71-1b42-4b76-a6a3-3e183365716d
begin
	D1 = [
	    7 4;
	    5 6;
	    8 6;
	    9.5 5;
	    9 7
	]

	# D1 = randn(5, 2) .+ 2
	
	D2 = [
	    2 3;
	    3 2;
	    3 6;
	    5.5 4.5;
	    5 3;
	]

	# D2 = randn(5,2) .- 2

	D = [D1; D2]
	D = [ones(10) D]
	targets = [ones(5); zeros(5)]
	AB = [1.5 8; 10 1.9]
end;

# ╔═╡ aaadebf2-294c-4e97-9a02-56e758485fcb
let
	plot(D1[:,1], D1[:,2],xlim =[0, 11], ylim=[-1,10], xlabel=L"$x_1$", ylabel=L"x_2", label="class 1", seriestype=:scatter, ratio=1.0, c=2)
	plot!(D2[:,1], D2[:,2], xlabel=L"$x_1$", ylabel=L"x_2", label="class 2", seriestype=:scatter, ratio=1.0, c=1)
end

# ╔═╡ c5ec2d52-cf7f-403f-bd21-a68ab3b97a9e
wws_map = let
	λ = 0.02
	wwₗ₂ = randn(3) * 0.25
	γ = 0.02
	iters = 20000
	# optₗ₂ = Descent(ηₗ₂)
	# logLikGradl2(x) = Zygote.gradient((w) -> logPosteriorLogisticR(w, m0, V0, D, targets)[1], x)
	wwsₗ₂ = Matrix(undef, 3, iters)
	for i in 1:iters
		gt = ∇logistic_regularised(wwₗ₂, D, targets)
		wwₗ₂ = wwₗ₂ - γ * gt
		wwsₗ₂[:, i] = wwₗ₂
	end
	wwsₗ₂
end;

# ╔═╡ 216a2619-8eb3-4edc-988f-7af88eb121b3
wws_mle = let
	ww = randn(3) * 0.25
	γ = 0.02
	iters = 20000
	wws = Matrix(undef, 3, iters)
	for i in 1:iters
		gt = ∇logistic_loss(ww, D, targets)
		ww = ww - γ * gt
		wws[:, i] = ww
	end
	wws
end;

# ╔═╡ 338ae8dc-b1b4-4953-9aee-f97e15421f97
let
	step = 1000
	plot(wws_mle[1,1:step:end], c=1, ls=:dash,  marker = (:d, 0.5, 0.8, Plots.stroke(0.5, :gray)), label="w0: MLE")
	plot!(wws_mle[2,1:step:end], c=1, ls=:dash,  marker = (:circle, 0.5, 0.8, Plots.stroke(0.5, :gray)), label="w1: MLE")
	plot!(wws_mle[3,1:step:end], c=1, ls=:dash, marker = (:xcross, 0.5, 0.8, Plots.stroke(0.5, :gray)), label ="w2: MLE")

	plot!(wws_map[1,1:step:end], c=2, ls=:dot,  marker = (:d, 0.5, 0.8, Plots.stroke(0.5, :gray)) , label="w0: MAP")

	plot!(wws_map[2,1:step:end], c=2, ls=:dash,  marker = (:circle, 0.5, 0.8, Plots.stroke(0.5, :gray)), label="w1: MAP")
	plot!(wws_map[3,1:step:end], c=2, marker = (:xcross, 0.5, 0.8, Plots.stroke(0.5, :gray)), label ="w2: MAP", legend=:outerright, xlabel="Iterations (× 1000)", ylabel="w")
end

# ╔═╡ 28c04f67-699d-4665-90ef-ed402eed0b2c
begin
	using GLM
	glm_fit = glm(D, targets, Bernoulli(), LogitLink())
end

# ╔═╡ d05c93b1-d1fb-473c-b269-fce2bb48f8e0
let
	p2 = plot(D[targets .== 1, 2], D[targets .== 1, 3], ones(5), st=:scatter, label="class 1", c=2)
	plot!(D[targets .== 0, 2], D[targets .== 0, 3], zeros(5), st=:scatter, label="class 2",  legend=:topleft, c=1)
	w₀, w₁, w₂ = ww = coef(glm_fit)[:]
	plot!(2:0.1:10, 2:0.1:10, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5,  xlabel="x₁", ylabel="x₂", title="Loss: " *L"%$(round(logistic_loss(ww, D, targets) ; digits=20))")


	p1 = plot(D[targets .== 1, 2], D[targets .== 1, 3], ones(5), st=:scatter, label="class 1", c=2)
	plot!(D[targets .== 0, 2], D[targets .== 0, 3], zeros(5), st=:scatter, label="class 2",  legend=:topleft, c=1)
	w₀, w₁, w₂ = ww0 = ww/20
	plot!(2:0.1:10, 2:0.1:10, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5,  xlabel="x₁", ylabel="x₂", title="Loss: " *L"%$(round(logistic_loss(ww0, D, targets) ; digits=20))")
	plot(p1, p2)
end

# ╔═╡ 87e7270a-431e-44ee-8ace-5c9b5a5036f8
let
	gr()
	freq_coef = coef(glm_fit)'
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+1
	plt_heat = Plots.heatmap(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="MLE Prediction: heatmap", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2")
	plt_heat
end

# ╔═╡ 774a9247-1db7-48d6-b1f9-63bcce3924cd
p_map_surface_1 = let
	plotly()
	wmle = coef(glm_fit)[:]
	w = wws_map[:,end]
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+0.5
	p1=Plots.surface(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> logistic(wmle' * [1, x, y]), legend=:best, title="MLE Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2", zlabel=L"p(y=1|\ldots)", colorbar=:false)
	p2=Plots.surface(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> logistic(w' * [1, x, y]), legend=:best, title="MAP Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2", zlabel=L"p(y=1|\ldots)", colorbar=:false)
	plot(p1, p2)
end;

# ╔═╡ cfc53ba8-dc4e-4929-9866-ea879f33d720
p_map_surface_1

# ╔═╡ b2609e7c-5654-40b5-aa75-80fa174d4bcf
p_map_surface_1

# ╔═╡ 2cec4c51-1686-4166-ac67-c317075d79a4
begin
	n3_ = 30
	extraD = randn(n3_, 2)/2 .+ [2 -6]
	D₃ = [copy(D₂); [ones(n3_) extraD]]
	targets_D₃ = [targets_D₂; zeros(n3_)]
	targets_D₃_ = [targets_D₂; -ones(n3_)]
end

# ╔═╡ 9e00e2b5-4c8f-41f6-b433-934edfc742ef
losses, wws=let
	# a very bad starting point: completely wrong prediction function
	ww = [0, -5, -5]
	γ = 0.02
	iters = 2000
	losses = zeros(iters+1)
	wws = Matrix(undef, 3, iters+1)
	losses[1] = logistic_loss(ww, D₃, targets_D₃)
	wws[:, 1] = ww 
	for i in 1:iters
		gw = ∇logistic_loss(ww, D₃, targets_D₃)
		# Flux.Optimise.update!(opt, ww, -gt)
		ww = ww - γ * gw
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, D₃, targets_D₃)
	end
	losses, wws
end;

# ╔═╡ 7101c997-d26a-491c-82c3-ab5a27721578
plot(losses[1:100], label="loss", xlabel="Iteration")

# ╔═╡ 4476bdd5-e16d-4130-a157-25c39d46cf71
anim_logis=let

	gr()
	# xs_, ys_ = meshgrid(range(-5, 5, length=20), range(-5, 5, length=20))

	anim = @animate for t in [1:25...]
		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, label="class 1", c=2)
		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contour, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel="x₁", ylabel="x₂", title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
	end

	anim
	# ∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * 1
	# quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
	# w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
	# w_d₂
	# plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=:gray, label="Decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier fails")
end;

# ╔═╡ 037ad762-08f1-4b7e-9582-5b1b094afeb4
anim_logis2=let

	gr()

	anim = @animate for t in [1:25...]
		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], ones(sum(targets_D₃ .== 1)), st=:scatter, label="class 1", c=2)
		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], zeros(sum(targets_D₃ .== 0)), st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel="x₁", ylabel="x₂", title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
	end

	anim

end;

# ╔═╡ 6ebd4400-23dc-4b94-b637-d82f174a2c6d
gif(anim_logis2, fps=5)

# ╔═╡ 60237e6f-a695-461c-b27a-c0e461d29227
md"""
### Extra functions

"""

# ╔═╡ bde57876-f675-4b1d-8cb6-33ed2790d31b
# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow3d!(x, y, z,  u, v, w; as=0.1, lc=:black, la=1, lw=0.4, scale=:identity)
    (as < 0) && (nv0 = -maximum(norm.(eachrow([u v w]))))
    for (x,y,z, u,v,w) in zip(x,y,z, u,v,w)
        nv = sqrt(u^2 + v^2 + w^2)
        v1, v2 = -[u,v,w]/nv, nullspace(adjoint([u,v,w]))[:,1]
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        (as < 0) && (nv = nv0) 
        v4, v5 = -as*nv*v4, -as*nv*v5
        plot!([x,x+u], [y,y+v], [z,z+w], lc=lc, la=la, lw=lw, scale=scale, label=false)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], [z+w,z+w-v5[3]], lc=lc, la=la, lw=lw, label=false)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], [z+w,z+w-v4[3]], lc=lc, la=la, lw=lw, label=false)
    end
end

# ╔═╡ 56b87cec-2869-4588-b933-bfa40eedb745
let
	plotly()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel="x₁", ylabel="x₂", title="h(x) = wᵀ x")
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel="x₁", ylabel="x₂", title="σ(wᵀx)")

	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot(p1, p2)
end

# ╔═╡ 8b9248c4-7811-4d9a-bf29-068c3eb4beef
linear_reg(X, y; λ = 1) = (X' * X + λ *I) \ X' * y

# ╔═╡ a69507c4-2a0a-42a1-95fd-62501f693691
ws_ones = linear_reg(mnist_zeros_xs_train, 2 * mnist_zeros_ys_train .-1 ; λ=0.05);

# ╔═╡ f79ed613-84a8-4944-a458-97cb095e85ab
losses_, ww_ = let
	X = mnist_zeros_xs_train
	y = mnist_zeros_ys_train
	ww = ws_ones
	γ = 0.05
	iters = 2000
	losses = zeros(iters+1)
	losses[1] = logistic_loss(ww, X, y)
	for i in 1:iters
		gw = ∇logistic_loss(ww, X, y)/length(y)
		ww = ww - γ * gw
		losses[i+1] = logistic_loss(ww, X, y)
	end
	losses, ww
end;

# ╔═╡ 9579c682-835c-4155-84cc-a9227f793432
train_acc = accuracy(mnist_zeros_ys_train, predict(ww_, mnist_zeros_xs_train))

# ╔═╡ 9f8d8b7c-cb34-4ca4-ad0f-01aa2826d29f
md"""

### Training accuracy is $(round(train_acc; digits=3))
"""

# ╔═╡ b9d60608-e9ee-4e7f-a2cc-90d9d7adc56c
test_acc = accuracy(mnist_zeros_ys_test, predict(ww_, mnist_zeros_xs_test))

# ╔═╡ 2c37c0da-31f3-4730-a5d7-6bdef1393b96
md"""

### Testing accuracy is $(test_acc)

"""

# ╔═╡ b2a3def4-ebee-4b7a-a986-ab91cfed1d69
plot(reshape(ww_[2:end], 28, 28)', st=:heatmap, r=1, c=:jet, size=(380,400))

# ╔═╡ ad8ff0d7-f8f9-4c37-a769-81d6e14b7cdf
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ 8da31795-a4ca-4177-9d78-0f09170d6710
begin
	function ∇σ(w, x) 
		wx = w' * x
		logistic(wx) * (1-logistic(wx)) * x
	end
end

# ╔═╡ 55ebd87c-7789-4a4d-9d49-99e0e5d9a5c7
let
	plotly()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.7, xlim=[-5, 5], ylim=[-5, 5],  xlabel="x₁", ylabel="x₂", title="contour plot of "* "σ(wᵀx)", ratio=1)
	α = 2
	xs_, ys_ = meshgrid(range(-5, 5, length=20), range(-5, 5, length=20))
	∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
end

# ╔═╡ fd35b73a-fdbc-4ae9-aa45-0eeeb91698c5
md"""

### MCMC - Bayesian inference on single neuron

A basic Metropolis Hasting sampler which samples the following posterior

```math
\mathbf{w}^{(m)} \sim p(\mathbf{w}|\mathbf{X}, \mathbf{y})
```
"""

# ╔═╡ f054c1ba-651c-450b-b2e8-2d0f0efb18a7
begin
	function MCMCLogisticR(X, y, dim; mc= 1000, burnin=0, thinning=10, m₀= zeros(dim), λ = 1/1e2, qV= nothing)
		if isnothing(qV)
			qV = Symmetric(inv(6/(π^2) * X' * X))
		end
		postLRFun(w) = -(logistic_loss(w, X, y) + 0.5 * λ * (w' * w))
		mcLR = MHRWMvN((w) -> postLRFun(w), dim; logP = true, Σ = qV, x0=m₀, mc=mc, burnin=burnin, thinning= thinning)
		return mcLR
		# return wt, Ht
	end


	# Metropolis Hastings with simple Gaussian proposal
	function MHRWMvN(pf, dim; logP = true, Σ = 10. *Matrix(I, dim, dim), x0=zeros(dim), mc=5000, burnin =0, thinning = 1)
		samples = zeros(dim, mc)
		C = cholesky(Σ)
		L = C.L
		pfx0 = pf(x0) 
		j = 1
		for i in 1:((mc+burnin)*thinning)
			xstar = x0 + L * randn(dim)
			pfxstar = pf(xstar)
			if logP
				Acp = pfxstar - pfx0 
				Acp = exp(Acp)
			else
				Acp = pfxstar / pfx0 
			end
			if rand() < Acp
				x0 = xstar
				pfx0 = pfxstar
			end
			if i > burnin && mod(i, thinning) ==0
				samples[:,j] = x0
				j += 1
			end
		end
		return samples
	end
	
end

# ╔═╡ 46f1fbb9-59f4-4521-b4f2-9f4584abe1bf
mcmcLR = MCMCLogisticR(D, targets, 3; mc=2000);

# ╔═╡ fc1a2ca5-ab39-44e6-8199-86f1656a0c03
p_bayes_pred=begin
	gr()
	ppf(x, y) = prediction(mcmcLR[:,1:1000]', x, y)
	contour(-0:0.1:10, 0:0.1:10, ppf, xlabel=L"x_1", ylabel=L"x_2", fill=true,  connectgaps=true, line_smoothing=0.85, title="Bayesian prediction", c=:jet, alpha=0.9)
end

# ╔═╡ 0a7e8f24-b4aa-46a9-8e8c-2bca3b9c36fe
p_bayes_pred

# ╔═╡ effd0f06-ddfb-478b-b04e-e0fb8b72cead
mcmcLR

# ╔═╡ 2910d53f-dacb-4868-8b80-11f9365b1597
plot(Chains(mcmcLR', [:w₀ , :w₁, :w₂]))

# ╔═╡ a36b3e58-f705-4bc4-b9a5-2b6e8dfd0f89
md"""

### Extra plots
"""

# ╔═╡ a452c99a-08b5-4334-8a3d-3ab1403e9e32
p_freq_1= let
	gr()
	freq_coef = coef(glm_fit)'
	# prediction(ws, x1, x2) = logistic(ws * [1.0, x1, x2])
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+1
	p_freq_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="MLE Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2")
	Plots.plot!(p_freq_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=2, label="class 1")
	Plots.plot!(p_freq_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=1, label="class 2")
	Plots.plot!(p_freq_1, AB[1:2, 1], AB[1:2, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, annotate = [(AB[1,1], AB[1,2], text("A", :top,:red, 9)), (AB[2,1], AB[2,2], text("B", :top, :red, 9))], markerstrokewidth = 1, markercolor=:red)
	markers = [2, 5, 6, 9]
	for m in markers
		p0 = round(prediction(freq_coef, D[m, 2], D[m, 3]), digits=2)
		annotate!(D[m, 2], D[m,3], text(L"\hat{\sigma}="*"$(p0)", :bottom, 10))
	end

	for i in 1:2
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1], AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, :red, 10))
	end
	p_freq_1
end;

# ╔═╡ 4b0d5bd7-2e3a-4ebf-8e1d-6cb67de267b5
p_freq_1

# ╔═╡ eee886f5-3712-4a58-91d2-a01e2ca70d53
p_freq_1

# ╔═╡ 2440623f-04ee-4d13-8282-4b56768cb2c6
p_freq_2= let
	gr()
	freq_coef = w = wws_map[:,end]'
	# prediction(ws, x1, x2) = logistic(ws * [1.0, x1, x2])
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+1
	p_freq_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="MAP/Regularised Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2")
	Plots.plot!(p_freq_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=2, label="class 1")
	Plots.plot!(p_freq_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=1, label="class 2")
	Plots.plot!(p_freq_1, AB[1:2, 1], AB[1:2, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, annotate = [(AB[1,1], AB[1,2], text("A", :top,:red, 9)), (AB[2,1], AB[2,2], text("B", :top, :red, 9))], markerstrokewidth = 1, markercolor=:red)
	markers = [2, 5, 6, 9]
	for m in markers
		p0 = round(prediction(freq_coef, D[m, 2], D[m, 3]), digits=2)
		annotate!(D[m, 2], D[m,3], text(L"\hat{\sigma}="*"$(p0)", :bottom, 10))
	end

	for i in 1:2
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1], AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, :red, 10))
	end
	p_freq_1
end;

# ╔═╡ 5c4c8b0d-313f-4e63-ab2d-1d399ed69ddd
p_freq_2

# ╔═╡ 5067de18-678f-4b32-87a1-e1fa84c69942
chain_array = mcmcLR';

# ╔═╡ 046c29ed-afad-4f7b-bf2c-ea7fa3a3ceef
let
	plotly()
	Plots.surface(-40:0.5:40, -40:0.5:40, (x, y) -> prediction(chain_array, x, y), legend=:best, title="Bayesian Prediction", c=:jet, ratio=1.0, xlabel="x1", ylabel="x2", zlabel="p(y=1|x)")
end

# ╔═╡ 26a378d5-735f-49ee-8dfb-33074a2dfa49
p_bayes_2 = let
	gr()
	freq_coef = w = wws_map[:,end]'
	as_ = collect(range(-10, 15, 6))
	f(x, b) = -(w[1]+b) /w[3] - w[2] / w[3] * x
	bs = f.(as_, +3)
	AB = [as_ bs]
	minx, maxx = minimum(D[:,2])-20, maximum(D[:,2])+20
	miny, maxy = minimum(D[:,3])-20, maximum(D[:,3])+20
	p_bayes_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(chain_array, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="Bayesian Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2")
	Plots.plot!(p_bayes_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=1, label="class 1")
	Plots.plot!(p_bayes_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=2, label="class 2")
	for i in 1:size(AB)[1]
		Plots.plot!(p_bayes_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		
		p0 = round(prediction(chain_array, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]-1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :right, :blue, 10))
	end

	as2_ = collect(range(0, 23, 7))
	bs2 = f.(as2_, -5)
	AB = [as2_ bs2]
	for i in 1:size(AB)[1]
		Plots.plot!(p_bayes_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		p0 = round(prediction(chain_array, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]+1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :left, :red, 10))
	end
	p_bayes_1
end;

# ╔═╡ 03d93a3e-3af5-468d-90ef-ee1e7f587dd7
p_bayes_2

# ╔═╡ eade1785-e759-4e23-ab97-50d94360687d
let

	p_bayes_pred = Plots.plot(D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 5, markercolor=2, label="class 1", legend=:topright, xlim=[-1, 11], ylim=[-1,11])
	Plots.plot!(p_bayes_pred, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 5,  markercolor=1, label="class 2")
	mean_pred = mean(chain_array, dims=1)[:]

	b, k =  -chain_array[1, 1:2] ./	chain_array[1, 3]
	plot!(-2:12, (x) -> k*x+b,  lw=0.1, lc=:gray, label=L"\sigma^{(m)}\sim p(\sigma|\mathcal{D})")
	for i in 2:2:200
		b, k =  -chain_array[i, 1:2] ./	chain_array[i, 3]
		plot!(-2:12, (x) -> k*x+b,  lw=0.2, ls=:dash, lc=:gray, label="", title="Bayesian ensemble predictions")
	end
	# b, k =  - mean_pred[1:2] ./	mean_pred[3]
	# plot!(-2:12, (x) -> k*x+b,  lw=3, lc=3, label=L"\texttt{mean}(\sigma^{(r)})")
	p_bayes_pred

end

# ╔═╡ 53b09b80-f10c-493c-8127-19a06bf248d2
p_bayes_1 = let
	p_bayes_1 = Plots.contour(2-6:0.1:9+7, 2-6:0.1:8+7, (x, y) -> prediction(chain_array, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="Bayesian Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2")
	Plots.plot!(p_bayes_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=1, label="class 1")
	Plots.plot!(p_bayes_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=2, label="class 2")
	# Plots.plot!(p_bayes_1, AB[1:2, 1], AB[1:2, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, annotate = [(AB[1,1], AB[1,2], text("A", :top,:red, 9)), (AB[2,1], AB[2,2], text("B", :top, :red, 9))], markerstrokewidth = 1, markercolor=:red)
	# markers = [2, 5, 6, 9]
	# for m in markers
	# 	p0 = round(prediction(chain_array, D[m, 1], D[m, 2]), digits=2)
	# 	annotate!(D[m, 1], D[m,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, 10))
	# end

	# for i in 1:2
	# 	p0 = round(prediction(chain_array, AB[i, 1], AB[i, 2]), digits=2)
	# 	annotate!(AB[i, 1], AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :bottom, :red, 10))
	# end
	p_bayes_1
end;

# ╔═╡ 23cd44a0-beb5-4d82-89e2-a7d77aaf8b63
p_freq_surface_1 = let
	gr()
	freq_coef = coef(glm_fit)'
	minx, maxx = minimum(D[:,2])-0.5, maximum(D[:,2])+0.5
	miny, maxy = minimum(D[:,3])-0.5, maximum(D[:,3])+0.5
	Plots.surface(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), legend=:best, title="MLE Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2", zlabel=L"p(y=1|\ldots)")
end;

# ╔═╡ c9302e0c-786a-498a-9a66-a6b160ebfe57
p_freq_surface_1

# ╔═╡ f7c1e631-a25f-409b-826e-3c272da9b56d
p_bayes_surface_1 = let
	Plots.surface(0:0.1:10, 0:0.1:10, (x, y) -> prediction(chain_array, x, y), legend=:best, title="Bayesian Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2", zlabel=L"p(y=1|\ldots)")
end;

# ╔═╡ 690ae8aa-f816-4b8c-8c2c-d09f5025fab6
p_freq_3= let
	gr()
	freq_coef = w = wws_map[:,end]'
	as_ = collect(range(-10, 15, 5))
	f(x, b) = -(w[1]+b) /w[3] - w[2] / w[3] * x

	bs = f.(as_, +3)
	AB = [as_ bs]
	minx, maxx = minimum(D[:,2])-20, maximum(D[:,2])+20
	miny, maxy = minimum(D[:,3])-20, maximum(D[:,3])+20
	p_freq_1 = Plots.contour(minx:0.1:maxx, miny:0.1:maxy, (x, y) -> prediction(freq_coef, x, y), fill=false, connectgaps=true, levels=8, line_smoothing = 0.85, legend=:left, title="MAP/Regularised Prediction", c=:jet, ratio=1.0, xlabel=L"x_1", ylabel=L"$x_2")
	Plots.plot!(p_freq_1, D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 3, markercolor=2, label="class 1")
	Plots.plot!(p_freq_1, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 3,  markercolor=1, label="class 2")


	for i in 1:size(AB)[1]
		Plots.plot!(p_freq_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]-1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :right, :blue, 10))
	end
	as2_ = collect(range(0, 23, 5))
	bs2 = f.(as2_, -5)
	AB = [as2_ bs2]
	for i in 1:size(AB)[1]
		Plots.plot!(p_freq_1, AB[:, 1], AB[:, 2], label="", seriestype=:scatter, markersize = 2, markershape =:star4, markerstrokewidth = 4, markercolor=:red)
		
		p0 = round(prediction(freq_coef, AB[i, 1], AB[i, 2]), digits=2)
		annotate!(AB[i, 1]+1.2, AB[i,2], text(L"\hat{\sigma}="*"$(p0)", :left, :red, 10))
	end
	p_freq_1
end;

# ╔═╡ e8e2a39c-bc18-42dd-86bf-684426ad6b35
p_freq_3

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.78"
GLM = "~1.8.1"
Images = "~0.25.2"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.18"
MCMCChains = "~5.5.0"
MLDatasets = "~0.5.16"
Plots = "~1.36.1"
PlutoTeachingTools = "~0.2.5"
PlutoUI = "~0.7.48"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "11a03d045be5c9c34e69204d0f58443bbd500dac"

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
git-tree-sha1 = "52b3b436f8f73133d7bc3a6c71ee7ed6ab2ab754"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.3"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "eb7a1342ff77f4f9b6552605f27fd432745a53a3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.22"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

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
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[deps.BitFlags]]
git-tree-sha1 = "629c6e4a7be8f427d268cebef2a5e3de6c50d462"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.6"

[[deps.BufferedStreams]]
git-tree-sha1 = "bb065b14d7f941b8617bc323063dbe79f55d16ea"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.1.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "c5fd7cd27ac4aed0acf4b73948f0110ff2a854b2"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.7"

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

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "cc4bd91eba9cdbbb4df4746124c22c0832a460d6"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.1.1"

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
git-tree-sha1 = "aaabba4ce1b7f8a9b34c015053d3b1edf60fa49c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.4.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

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

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "bc0a264d3e7b3eeb0b6fc9f6481f970697f29805"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.10"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "0f44494fe4271cc966ac4fea524111bef63ba86c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.3"

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
git-tree-sha1 = "7fe1eff48e18a91946ff753baf834ff4d5c03744"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.78"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

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
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "802bfc139833d2ba893dd9e62ba1767c88d708ae"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.5"

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
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "884477b9886a52a84378275737e2823a5c98e349"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "00a9d4abadc05b9476e937a5557fcce476b9e547"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.69.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Glob]]
git-tree-sha1 = "4df9f7e06108728ebf00a0a11edee4b29a482bb2"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.0"

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

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "19effd6b5af759c8aaeb9c77f89422d3f975ab65"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.12"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "e1acc37ed078d99a714ed8376446f92a5535ca65"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.5.5"

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

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b1798a4a6b9aafb530f8f0c4a7b2eb5501e2f2a3"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.16"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "8b251ec0582187eff1ee5c0220501ef30a59d2f7"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "124626988534986113cfd876e3093e4a03890f58"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "e7c68ab3df4a75511ba33fc5d8d9098007b579a8"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.2"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "Statistics"]
git-tree-sha1 = "0c703732335a75e683aec7fdfc6d5d1ebd7c596f"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.3"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "b563cf9ae75a635592fc73d3eb78b86220e55bd8"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.6"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "03d1301b7ec885b266c0f816f338368c6c0b81bd"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

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
git-tree-sha1 = "0cf92ec945125946352f3d46c96976ab972bde6f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.3.2"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

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
git-tree-sha1 = "842dd89a6cb75e02e85fdd75c760cdc43f5d6863"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.6"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "3f91cd3f56ea48d4d2a75c2a65455c5fc74fa347"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

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
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "18dd357553912b6adc23b5f721e4be19930140c6"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.28"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0f960b1404abb0b244c1ece579a0ec78d056a5d1"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.15"

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
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "dedbebe234e06e1ddad435f5c6f4b85cd8ce55f7"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.2"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "971be550166fe3f604d28715302b58a3f7293160"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.3"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "f5f347b828fd95ece7398f412c81569789361697"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.5.0"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "dab8e9e9cd714fd3adc2344a97504e7e64e78546"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLDatasets]]
deps = ["BinDeps", "CSV", "ColorTypes", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "JLD2", "JSON3", "MAT", "MLUtils", "Pickle", "Requires", "SparseArrays", "Tables"]
git-tree-sha1 = "862c3a31a5a6dfc68e78e2e1634dac1d3b0f654e"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.5.16"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "4040c0da2bd05130687cc258c1318acd32bace90"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.7.1"

[[deps.MLStyle]]
git-tree-sha1 = "060ef7956fef2dc06b0e63b294f7dbfbcbdc7ea2"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.16"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase", "Transducers"]
git-tree-sha1 = "824e9dfc7509cab1ec73ba77b55a916bb2905e26"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.11"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "2af69ff3c024d13bde52b34a2a7d6887d4e7b438"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "4d5917a26ca33c66c8e5ca3247bd163624d35493"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "efe9c8ecab7a6311d4b91568bd6c88897822fabe"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.0"

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

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "440165bf08bc500b8fe4a7be2dc83271a00c0716"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.12"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "df6830e37943c7aaa10023471ca47fb3065cc3c4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6e9dba33f9f2c44e08a020b0caf6903be540004"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.19+0"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "b64719e8b4504983c7fca6cc9db3ebc8acc2a4d6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.1"

[[deps.Pickle]]
deps = ["DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e6a34eb1dc0c498f0774bbfbbbeff2de101f4235"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "47e70b391ff314cc36e7c2400f7d2c5455dc9496"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.36.1"

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
git-tree-sha1 = "ea3e4ac2e49e3438815f8946fa7673b658e35bdb"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

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
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "d8ed354439950b34ab04ff8f3dfd49e11bc6c94b"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.1"

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

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "fcebf40de9a04c58da5073ec09c1c1e95944c79b"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.6.1"

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
deps = ["SnoopPrecompile"]
git-tree-sha1 = "d12e612bba40d189cead6ff857ddb67bd2e6a387"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "dad726963ecea2d8a81e26286f625aee09a91b7c"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.4.0"

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

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "793b6ef92f9e96167ddbbd2d9685009e200eb84f"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.3"

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
git-tree-sha1 = "efd23b378ea5f2db53a55ae53d3133de4e080aa9"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.16"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

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
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "4e051b85454b4e4f66e6a6b7bdc452ad9da3dcf6"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.10"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

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
git-tree-sha1 = "a5e15f27abd2692ccb61a99e0854dfb7d48017db"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.33"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e0d5bc26226ab1b7648278169858adcfbd861780"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.4"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "50ccd5ddb00d19392577902f0079267a72c5ab04"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.5"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

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
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

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
git-tree-sha1 = "f53e34e784ae771eb9ccde4d72e578aa453d0554"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.6"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "f8cd5b95aae14d3d88da725414bdde342457366f"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "77fea79baa5b22aeda896a8d9c6445a74500a2c2"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.74"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

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
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

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
git-tree-sha1 = "ef4f23ffde3ee95114b461dc667ea4e6906874b2"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.0"

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

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

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

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

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
# ╟─6a1f3828-667a-11ed-2ea0-154032459b81
# ╟─7c49ffb7-dc5f-4cd5-ada5-44719e24b8a6
# ╟─18a77849-f40d-4eb6-aa5d-07f7d4ec007d
# ╟─0b01ac40-7054-4671-a370-483cce6b2231
# ╟─07d11cca-30a6-414c-bdd6-176de80e8b33
# ╟─7547a998-0652-4c8a-ac63-c0daf03b72cf
# ╟─a8c421e9-369e-438b-9388-9b4c5fd03484
# ╟─8879917c-072c-41b2-947f-7e65f12faa58
# ╟─38bf8ca4-2507-45cf-8036-1f9b6ba0a66a
# ╟─a6ad102a-9ff7-4ed9-8ab6-fd7790d8d37a
# ╟─0f225525-1a86-4fab-9a8f-3d6a3a5135f6
# ╟─69f07beb-4110-4f7e-87cf-efd2fd5f88a8
# ╟─3b33095d-b4be-4924-95ee-2b70ace93352
# ╟─5ed3dc0a-8a73-4024-9a8a-c333936329c3
# ╟─0685a343-9a21-4b1a-80bc-d41f6615f22c
# ╟─adebb65b-200e-4f07-93f0-934c55a66dcc
# ╟─a0474cb7-be98-449b-a5a7-ef8bac52fda6
# ╠═06bfedf9-f4e2-4885-b466-842362e4d739
# ╟─56b87cec-2869-4588-b933-bfa40eedb745
# ╟─55ebd87c-7789-4a4d-9d49-99e0e5d9a5c7
# ╟─6d9ff057-1ebd-43bd-ac2e-cbfb1da6dd98
# ╟─e759ec49-c1ae-491d-b5bd-1043e934aa87
# ╟─ee737d2c-d7d9-464b-8cd5-91b0b97f07d0
# ╟─ad9019e9-5ffd-4d18-8f67-0d7a41d08a56
# ╟─9c844e0d-6396-46e6-8de2-4c7bb18254c2
# ╟─4c0fc54b-9184-4f0b-8224-bf0774d844f1
# ╟─1d87e445-984a-4544-88e3-780c681fb737
# ╟─7101c997-d26a-491c-82c3-ab5a27721578
# ╠═6ebd4400-23dc-4b94-b637-d82f174a2c6d
# ╟─9e00e2b5-4c8f-41f6-b433-934edfc742ef
# ╟─4476bdd5-e16d-4130-a157-25c39d46cf71
# ╟─037ad762-08f1-4b7e-9582-5b1b094afeb4
# ╟─74b9b15d-a8ab-42b9-a5c3-74e78994e4e6
# ╟─10d0b2bd-222c-46f7-abc3-b70b8425add7
# ╟─9f8d8b7c-cb34-4ca4-ad0f-01aa2826d29f
# ╟─9579c682-835c-4155-84cc-a9227f793432
# ╟─2c37c0da-31f3-4730-a5d7-6bdef1393b96
# ╟─b9d60608-e9ee-4e7f-a2cc-90d9d7adc56c
# ╟─bbbf3716-ec62-4f41-97a1-1fa2563481e3
# ╟─b2a3def4-ebee-4b7a-a986-ab91cfed1d69
# ╟─3a4df100-cf7b-46fa-8b72-bc0aa52d1c6a
# ╟─12297994-3faa-4d6c-9cfa-b1cb69ac5550
# ╟─1dff88ba-9573-479f-abe8-5079acdc2654
# ╟─2d32c35b-5ee1-44a6-ad0f-c96750f5c946
# ╟─a69507c4-2a0a-42a1-95fd-62501f693691
# ╟─f79ed613-84a8-4944-a458-97cb095e85ab
# ╟─80df1e62-5b92-437d-b409-b974186cdf19
# ╟─1f23ecaa-c5f6-4a25-8c12-115a96868a93
# ╟─3cb006a6-5c80-4300-907f-9eaae67c979a
# ╟─aaadebf2-294c-4e97-9a02-56e758485fcb
# ╟─f7b465c7-951b-4abd-9252-5260c1e2c8b2
# ╟─d05c93b1-d1fb-473c-b269-fce2bb48f8e0
# ╟─55ffe10e-899d-456f-ba62-12d51a072843
# ╟─c9302e0c-786a-498a-9a66-a6b160ebfe57
# ╟─87e7270a-431e-44ee-8ace-5c9b5a5036f8
# ╟─4b0d5bd7-2e3a-4ebf-8e1d-6cb67de267b5
# ╟─2ad1858b-fcd3-4cc8-be70-97746480eacd
# ╟─cf104a92-22e8-4995-96d1-869b1ea1c4a2
# ╟─c5ec2d52-cf7f-403f-bd21-a68ab3b97a9e
# ╟─216a2619-8eb3-4edc-988f-7af88eb121b3
# ╟─774a9247-1db7-48d6-b1f9-63bcce3924cd
# ╟─dbf31225-55a1-4d3d-8758-bcab0e653f55
# ╟─338ae8dc-b1b4-4953-9aee-f97e15421f97
# ╟─daeaf752-7f43-4afb-abdc-73a3ad7f2301
# ╟─cfc53ba8-dc4e-4929-9866-ea879f33d720
# ╟─b2888f9b-c40a-4ee9-bdbd-898eb89df5be
# ╟─eee886f5-3712-4a58-91d2-a01e2ca70d53
# ╟─5c4c8b0d-313f-4e63-ab2d-1d399ed69ddd
# ╟─e30b60a4-0625-4f68-83bd-481b623616c9
# ╟─e8e2a39c-bc18-42dd-86bf-684426ad6b35
# ╟─4840b4b2-f8ad-4a43-95c1-750d2cb39524
# ╟─0a7e8f24-b4aa-46a9-8e8c-2bca3b9c36fe
# ╟─03d93a3e-3af5-468d-90ef-ee1e7f587dd7
# ╟─e71c8dd8-2f5e-4038-88cd-ea2863adc3b0
# ╟─b2609e7c-5654-40b5-aa75-80fa174d4bcf
# ╟─046c29ed-afad-4f7b-bf2c-ea7fa3a3ceef
# ╟─26a378d5-735f-49ee-8dfb-33074a2dfa49
# ╟─7717a633-822e-4347-bc39-17caa6cbcbd0
# ╟─03471888-cdf2-4e32-a06c-381e1b66e6d7
# ╟─924b8e0e-dc4e-43b6-a948-554e5bc956d3
# ╟─fc448fa2-f764-4d0a-b50c-6283ecebee81
# ╟─1dd1b222-6e5e-47f1-9c5c-cfa2f847075f
# ╟─340944a7-7a14-4a68-984e-d6924fc13cf1
# ╟─a67aa14f-f4c0-4214-813e-fcc56aee852c
# ╟─b710b9c1-f842-4c97-bda3-b4e6f324a9c9
# ╟─af74c191-0176-4197-945e-174c0456bd66
# ╟─f09f3b01-623d-4cd0-9d6f-939787aed445
# ╟─eade1785-e759-4e23-ab97-50d94360687d
# ╟─fc1a2ca5-ab39-44e6-8199-86f1656a0c03
# ╟─66e8bc6e-61dd-4b5d-8130-b96f03d92bf8
# ╟─b07efeea-75c9-4847-ba45-752ff9b34d53
# ╠═af30c343-64bd-45e6-be64-59f1f0413cc5
# ╠═1e6c5c71-1b42-4b76-a6a3-3e183365716d
# ╠═2cec4c51-1686-4166-ac67-c317075d79a4
# ╟─60237e6f-a695-461c-b27a-c0e461d29227
# ╠═bde57876-f675-4b1d-8cb6-33ed2790d31b
# ╠═8b9248c4-7811-4d9a-bf29-068c3eb4beef
# ╠═ad8ff0d7-f8f9-4c37-a769-81d6e14b7cdf
# ╠═8da31795-a4ca-4177-9d78-0f09170d6710
# ╟─fd35b73a-fdbc-4ae9-aa45-0eeeb91698c5
# ╠═f054c1ba-651c-450b-b2e8-2d0f0efb18a7
# ╠═46f1fbb9-59f4-4521-b4f2-9f4584abe1bf
# ╟─effd0f06-ddfb-478b-b04e-e0fb8b72cead
# ╠═b741192c-d6b8-435f-b553-0aa5ebed984d
# ╟─2910d53f-dacb-4868-8b80-11f9365b1597
# ╟─a36b3e58-f705-4bc4-b9a5-2b6e8dfd0f89
# ╠═a452c99a-08b5-4334-8a3d-3ab1403e9e32
# ╠═2440623f-04ee-4d13-8282-4b56768cb2c6
# ╠═5067de18-678f-4b32-87a1-e1fa84c69942
# ╠═53b09b80-f10c-493c-8127-19a06bf248d2
# ╠═23cd44a0-beb5-4d82-89e2-a7d77aaf8b63
# ╠═f7c1e631-a25f-409b-826e-3c272da9b56d
# ╠═690ae8aa-f816-4b8c-8c2c-d09f5025fab6
# ╟─28c04f67-699d-4665-90ef-ed402eed0b2c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
