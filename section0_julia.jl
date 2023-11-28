### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ f3770e42-edff-4c9f-9a13-47edaf8a8cae
begin
	using PlutoUI
	using LaTeXStrings,Latexify
	using PlutoTeachingTools
end

# ╔═╡ bcf4086f-1469-49d4-a3f5-730a30771da4
using Distributions

# ╔═╡ d8764bd8-7d7d-4910-80ee-2187ad31f336
using LinearAlgebra

# ╔═╡ 58a419b4-4017-474f-a3e9-cc58d2aaa29a
using Plots

# ╔═╡ 0ee0e688-9bc7-489b-a4ae-a4f09477fd33
let
	using StatsPlots
	dist = Gamma(2, 0.5)
	pdf_plt = plot(dist, label= L"\texttt{Gamma}(2, 0.5)", legend=:right)
	plot!(dist, func =cdf, label= L"\texttt{CDF}")
	cdf_plt = scatter(dist, leg=false)
	bar!(dist, func=cdf, alpha=0.3)
	plot(pdf_plt, cdf_plt, layouts=(2,1))
end

# ╔═╡ fe7f2df3-8c27-4e80-a633-386e8643e6e1
TableOfContents()

# ╔═╡ 35d6a20a-6ab1-4d46-be79-c84d664a6119
md"[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section1_introduction.html)
"

# ╔═╡ baf83ba6-2079-11ed-1505-cb44aea4377e
md"""# A quick introduction to Julia $(Resource("https://julialang.org/assets/infra/logo.svg", :height=>40, :align=>:right))
"""

# ╔═╡ c85214aa-83b1-41f0-b856-f85435db6e1d
md"""

[Julia](https://docs.julialang.org/en/v1/) is a general-purpose, open-source, dynamic, and high-performance language. The language is easy to use like other script languages such as R and Python but works as fast as C/Fortran. The language also offers high-level, easy-to-use and expressive syntax that is particularly convenient for scientific or numerical computing. One often finds code written in Julia look like the maths equations they have derived and written on paper. 


In this course, we are going to use Julia. Although Bayesian modelling concepts transcends the underlying implementation language, the reader should gain better understanding of the subjects if she/he can understand the basic Julia syntax and do some basic programming. This note covers some basic features of Julia that are used later in the course. To fully understand the note, some prior programming experience is however assumed.
"""

# ╔═╡ e6c8aca8-1907-49b0-ba11-62fda98418fc
md"""

## Install `Julia`


Install Julia (1.8 over greater)
* Download and install Julia by following the instructions at [https://julialang.org/downloads/](https://julialang.org/downloads/).
* Choose the latest stable version (the current one is v1.8.5) 


"""

# ╔═╡ a5654402-f51d-468e-99fb-09a5c69a6421
md"""## Interact with `Julia`
"""

# ╔═╡ b145ea5b-9fc0-4a57-b41b-ad8c825eaccc
md"""There are many different ways to interact with Julia. For example: the interactive command line interface REPL (read-eval-print loop) .
"""

# ╔═╡ de6fc3a6-78dc-489a-8ac3-3642e0760d6a
md"""  * The `Julia` REPL is a built-in means to interact with `Julia`. The command line interface may look something like the following where `1+2` is evaluated:
"""

# ╔═╡ a3a93c83-94c8-44e7-a62f-46cae602ac77
md"""```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.0 (2022-08-17)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 1 + 2
3
```"""

# ╔═╡ 1635054e-3e02-4f3b-8726-bef468c11a82
md""" 
**Pluto** Notebook (such as this notebook)
  * The [Pluto](https://github.com/fonsp/Pluto.jl) package provides a *reactive* notebook interface that runs in a browser. 
  * The notebook can be used to write both markup texts (including latex equations) and Julia code.
  * All notes are written with Pluto notebook in this course. And you can download and run them locally on your desktop.
"""

# ╔═╡ 2b52dcf7-f8aa-49ea-a8e5-df4d819fc970
md"""
Other interaction options include 

* Integrated development environments (IDE)s, which are particularly useful for large-scale developments. [VS Code](https://code.visualstudio.com/docs/languages/julia) is recommended.
* and [Jupyter-notebook](https://github.com/JuliaLang/IJulia.jl).

"""

# ╔═╡ 3adfcebb-e620-4c41-b9a7-3584fbfd61c1
md"""### Package manager `Pkg`

Unlike Python which relies on external package managers, Julia has its native package manager `Pkg`. It is very straightforward to use an add-on package in Julia. Follow the steps below to install an add-on package. 

Step 1: Open the Julia Command-Line 
  * either by double-clicking the Julia executable or 
  * running Julia from the command line
And you should see a command line interface like this

```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.0 (2022-08-17)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 
```

  
Step 2: Use `Pkg` to add packages. For example, to install `Turing.jl` and `Pluto.jl` (or any registered package)
```juliarepl
julia> using Pkg
julia> Pkg.add("Turing")
```

or equivalently

```juliarepl
julia> ]
(@v1.8) pkg> add Turing
```

Step 3: import the package and start using it 
```juliarepl
using Turing
```
  * The `using` command loads the specified package and makes all its *exported* values available for direct use.
"""

# ╔═╡ 0f9ce7d3-ccf3-40f6-9573-ec40dc643749
md"""

### More on Pluto notebook
It is recommended that you use Pluto notebook to interact with Julia. To install and use Pluto, please follow the steps below.

* Step 1: Install `Pluto.jl`; 
```juliarepl
julia> Using Pkg
julia> Pkg.add("Pluto")
```


* Step 2: Use Pluto 
```juliarepl
julia> using Pluto
julia> Pluto.run()
```

* Step 3: To get started, you can either run this notebook (recommended) or start your own new notebook

Some worth-noting features of Pluto are:
* To use add-on packages, one only needs to add `using package_name` directly without installing them first with `pkg`. Pluto does all the package maintenance behind the scene.  For example, to use package `Distributions.jl` (no matter it has been installed or not), one simply add `using Distributions` in a cell:
"""

# ╔═╡ 6be2784a-cf0c-4d75-9858-b063880c4e98
md"""
* On a given line, anything after a # is a *comment*
* Code blocks with more than one line of code should be included within a `begin` and `end` block.
* Pluto notebook is **reactive**, which means when one "cell" is modified and executed, the new values cascade to all other dependent cells which in turn are updated. The working mechanism is like a spreadsheet. Therefore, all variables in one cell are accessible in the rest of the notebook (a.k.a global variable). 
* To force variables to have a local cell scope (so they cannot be accessed elsewhere), one can use `let ... end` block instead of `begin ... end`.

"""

# ╔═╡ 3520ad46-7ac5-47bf-9389-62a46f6a26f8
md"""

## `Julia` language
"""

# ╔═╡ 65ae3b8f-a33a-49ba-b135-ab20674ec67a
md"""

### Numerical variable types

`Julia` offers a wide range of numerical variable types that are commonly used in maths and statistics. These include:
* Boolean: `true, false`
* Integers: `1,2`
* Rational numbers: `1//2`
* Floating number: `2.0`
* and even Complex numbers: `2 + 0im`

`Julia`'s parsers can automatically infer the appropriate type for the value. The following examples demonstrate how the number 2 can be represented as different forms: *i.e.* as an integer, a rational number, a floating number, and a complex number.
"""

# ╔═╡ c3fe7693-8425-4f6b-8529-c9b242073334
begin
	2 # as an integer
	2//1 # as a rational number
	2.0 # as a floating number; or 2e0
	2 + 0im # as a complex number where the imaginary part is zero
end;

# ╔═╡ 045474d2-18f1-45b1-8c5d-9a82980e96bd
md"""
All above values are equal if compared with `==`, which checks their values numerically.
"""

# ╔═╡ 287d41f1-9ade-4a73-b8ea-1e11aaf62fbe
2 == 2//1 == 2.0 == 2 + 0im

# ╔═╡ d51ba0c0-9ca1-47aa-8364-080abc8175c3
md"""To check variable's physical internal representations in computer memory, one should use "`===`" instead. In other words, "`===`" checks whether two variables refer to the same object in the computer memory."""

# ╔═╡ ae200380-7c83-4fa6-9661-9cf1999732e3
2 === 2//1, 2 === 2.0, 2 === 2 + 0im, 2//1 === 2.0, 2//1=== 2+0im

# ╔═╡ b3919bcb-c2e7-442b-a06c-93d172878f9c
md"""

Variables assignments are done with the"`=`" operator
* Variable names can be Unicode, 
* Greek letters are widely used; for example, α: type `\alpha` + `tab`
"""

# ╔═╡ ed8457c7-9547-4953-ae72-121572fe8495
begin
	α, β = 1.0, 2
end

# ╔═╡ 49a993c4-27c9-4c89-b4c4-04c294f044c0
md"""

### Standard mathematical operations

All standard mathematical operations are implemented, such as `+, -, *, /`. And Parentheses are used for grouping operations as usual.

Some other useful operations include

* `x^y`, exponential ``x^y``
* `sqrt(x)`, square root of a number, i.e. ``x^{0.5}``
* `log(x)`, logarithm operation
* `sin(x), cos(x)`, trigonometric functions
* `abs`: absolute value
* `sign`: is ``|x|/x`` except at ``x=0``
* `floor, ceil`: flooring or ceiling of a floating number
* `max(a,b), min(a,b)`: returns the greater or smaller betwee `a` and `b`
* `maximum(xs), minimum(xs)`: the largest or smallest of the input array

Check [the user manual](https://docs.julialang.org/en/v1/manual/mathematical-operations/) for a more detailed list of built-in operations.
"""

# ╔═╡ a091f1d5-29c0-4a98-b1b8-4177cd6cd2f7
md"""

!!! information "Getting help within Pluto"
	If you want to know more about a specific function, simply type `? function_name` in a cell. 

	Try typing `?sign` in a cell and see the live document section.
"""

# ╔═╡ a6c186ae-9d7a-452e-bfb2-b303c0cfea5a
md"""

### Vectors

A vector in `Julia` is defined as an indexed array of similarly typed values. A  vector (or array), can be created with square brackets:
"""

# ╔═╡ cbd40c3a-cf65-4d1c-833a-29a1e39dc893
[1,2,3,4,5]

# ╔═╡ a08b431d-86b4-4f03-b0aa-eadba6284e95
md"""

There are other convenient ways to create a vector. 

* `zeros(n), ones(n)`: create an array of `n` zeros or ones;

To create a sequence of numbers with some fixed gap:
* `a:b` or `a:h:b`: which create a sequence of numbers starting with `a` and ending with `b` inclusive and the gap is `h` ;
* `range(a,b,length=n)`: offers an alternative way to produce `n` values between `a` and `b`;

Both above methods return a generator rather than the real values. To cast them to an array, one can use `collect()` or a splat `...` operator.
"""

# ╔═╡ 82b73754-f164-4bc0-82d9-fd0311a59919
begin
	1:10, collect(1:10), [(1:10)...]
end

# ╔═╡ a12d7a2e-5a4e-46db-b802-1af1331456dd
range(1, 10, length=12), collect(range(1, 10, length=12))

# ╔═╡ 911bf98c-130c-4e20-a353-bc290c865a27
md"""

One can also declare an array with the specified element type and length when creating it. For example, to create an array of 10 integers of uninitialised values:
"""

# ╔═╡ 5c15f435-8ea6-446e-a27d-3270b463f681
Array{Int}(undef, 10)

# ╔═╡ b8d452b1-b75b-4012-872a-3baced9c1cb6
md"""

To accommodate missing values, `Julia` has a special data type called `Missing`. An element of either missing type or concrete observation can be created with `Union`. For example, `Union{Missing, Float64}` can either be a missing value or a floating number. An array of mixed missing and observed values can therefore be created by:
"""

# ╔═╡ 054d075f-4e1c-42b2-955e-70ac1e36144b
Array{Union{Missing, Float64}}(undef, 10)

# ╔═╡ 2d934969-c540-4e88-b137-d22e836e7daa
md"""
### Matrices

Matrices are simply multi-dimensional arrays. It is recommended to import the add-on `LinearAlgebra` package when dealing with matrices.

"""

# ╔═╡ d78b7c17-1f6d-424c-aeed-e35a9d3db49e
md"To create matrix

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4\end{bmatrix},
```
one has the following options:
"

# ╔═╡ d349aa84-4b2a-42c9-98a4-5849ece5e6ae
[1 2; 3 4], hcat([1,3], [2,4]), vcat([1 2], [3 4]), reshape([1,2,3,4], 2, 2)'

# ╔═╡ 6950f0da-d61a-465b-aaaf-8a802f50452a
md"""
Where `vcat` and `hcat` concatenate the inputs horizontally (or vertically), also note
* `[1,3]` is a column vector
* while `[1 2]` is a row vector
"""

# ╔═╡ c68f7d30-3061-40db-8af3-369149cd5ff9
md"""

Here are some commonly used special matrices:

* `Matrix{Float64}(undef, m, n)`, returns an `m × n` uninitialised real valued matrix 
* `ones(m, n)`, returns an `m × n` matrix with ones
* `zeros(m, n)`, returns an `m × n` zero matrix 
* `Matrix(1.0I, n, n)`, returns an `n × n` identity matrix 
* `Diagonal(V::AbstractVector)`, returns a diagonal matrix with `V` as its diagonal 
"""

# ╔═╡ 84710a37-34c3-4c20-bd5c-b293d23bc7cf
begin
	Matrix{Float64}(undef, 2, 3), ones(2,3), zeros(2,3)
end

# ╔═╡ 45037f4e-5744-4de7-885d-f0a678b72ec1
begin
	Matrix(1.0I, 3, 3), Diagonal(ones(3))
end

# ╔═╡ b3d46870-ef00-4fec-a5f2-2f3b6b11687c
md"""

Standard matrix algebra operations such as addition/subtraction, multiplication and matrix inversion are all supported.

* `A'`: matrix transpose: ``A^{\top}``
* `A + B`: matrix addition or subtraction (`-`)
* `A * B`: matrix multiplication
* `A^(-1)` or `inv(A)`: inverse a matrix ``A^{-1}``
"""

# ╔═╡ 52a6e45f-b458-463a-95b8-0f8a184ce171
begin
	A = rand(5,5)
	B = rand(5)
	A + A == 2 * A
end

# ╔═╡ b078d4c9-7638-43a4-bd77-57a40b1511ef
A * B, B' * A # matrix multiplication is not commutative

# ╔═╡ 2b305ef2-551e-4327-ad56-2bd3a12f51fc
A^(-1) == inv(A) # inverting a matrix

# ╔═╡ 84713022-8e34-4e57-aa2f-cf9d6cf61cad
md"""

Note that most mathematical operations have an optional argument `dims` to specify along which dimension of the input matrix to apply the operation. For example, to sum a matrix `C` along the row (dim =1):
"""

# ╔═╡ 36824286-375a-48db-af65-b6dcaf6497b8
C = ones(4,2);

# ╔═╡ 46be754e-ec2f-4e0d-aef3-865492d2dd0c
md"C="

# ╔═╡ 1af4b04a-a7a1-4808-9cf0-038c95037fa5
latexify(C)

# ╔═╡ 7d897f13-7358-446f-b87a-60ad945769ea
md"And the row sum is:"

# ╔═╡ 7cd155b2-f76e-4152-9d79-ac9ea8c32b16
sum(C, dims=1)

# ╔═╡ d0b5cc5a-f0cf-44e8-9e2a-e28b94d4429f
md"Note that the returned object is a ``1\times n`` matrix rather than a vector. To cast the matrix to a vector, append `[:]` at the end."

# ╔═╡ 04065f11-4748-4cf0-abf4-01484454da0b
sum(C, dims=1)[:]

# ╔═╡ aed0ed19-0183-4eb8-83c9-3d2419d299b8
md"To apply the operation to the columns, simply specify `dims=2`."

# ╔═╡ 02938204-99d2-4209-abcf-cbcfd62f9c04
md"""

### Vectorised operation

A function can be applied to each element of a vector through a mechanism called **broadcasting**. The latter is implemented in a succinct `.` notation. For example, to calculate `sin` at an array of values in `xs`: `sin.(xs)`. 
"""

# ╔═╡ 8548c4cb-9e9b-4ca7-bd4e-97e16058606d
let
	xs = -2π : 0.1 : 2π
	sin.(xs)
end

# ╔═╡ c2ee8d41-eec8-43ad-b4c0-2f859ad03b93
md"""
Broadcasting also works for operators such as `*.`, `+.`, `.^`, `.==` etc. For example,
`xs .^2` and `xs .* xs` both square array `xs` element-wisely. And `.==` applies element-wise comparisons.
"""

# ╔═╡ c195aa69-89f6-4417-b602-67922a76b87a
let
	xs = 1:5
	# element wise multiplication and element-wise squares
 	xs .* xs, xs .^2, xs .* xs .== xs .^2
end

# ╔═╡ 4a51bb66-f4d4-4a7d-8e07-91d3ff5f4f28
md"Broadcasting is a very handy operator. Without it, one would have to write a for loop to achieve the same result."


# ╔═╡ 17b5759c-deab-4bc0-b7b6-fc2c0e64eebb
md"""

### Looping

Iterating over a collection can be done with the traditional `for` loop.
"""

# ╔═╡ c52294ca-49fb-43f7-87bb-860c8088506d
let
	results = zeros(10)
	for x in 1:10
		results[x] = x^2
	end
	results
end

# ╔═╡ fcd72da9-e175-4fd1-a047-7dd3711dcbf4
md" However, there are list comprehensions to mimic the definition of a set. The above `for` loop can be replaced with:"

# ╔═╡ 6df9180f-59bf-4462-ae9d-a2623379071a
[x^2 for x in 1:10]

# ╔═╡ a1aa6ab4-f183-4adc-9ad8-70d80deaa15f
md"""

### User defined functions

User-defined functions can be easily created with 

```julia
function function_name()
	...
	...
end
```

or in one-line 

```julia
function_name(x) = ...
```

or anonymously


```julia
(x) -> ...

```

For example, to create a third-order polynomial with coefficients `a,b,c,d`

$$f(x; a,b,c, d) = a x^3 + bx^2+cx + d$$
"""

# ╔═╡ dfa4201c-19da-4ca9-b537-8396c2899d42
function f1(x; a=1, b=2, c=3, d=4)
	return a * x^3 + b * x^2 + c * x + d
end

# ╔═╡ dbc643ae-d68c-416f-b1b9-dc8a0025dfe0
md"""
The default values of the coefficients are a = 1, b = 2, c = 3, d = 4. Or equivalently:
"""

# ╔═╡ e1de498d-d66f-42c8-bb22-e80d623c4554
f2(x; a=1,b=2,c=3,d=4) = a * x^3 + b * x^2 + c * x + d

# ╔═╡ 6c4ede4f-a566-4dfc-bfdf-f4feadac84aa
md"""
To use the created function, we simply feed in the required input value.
"""

# ╔═╡ 14fc0ff8-2284-412f-99af-6264d24d6198
f1(0), f2(0) 

# ╔═╡ 4958c15f-5725-4645-a71d-e8a26a94a894
f1.(1:10) .== f2.(1:10) 

# ╔═╡ a5c62a3c-2a59-4388-adb5-b0d93be2360d
md"""

!!! danger "Exercise"
	Write a function to calculate the area of a circle with radius `r`: ``\text{area}(r)= \pi r^2``.

!!! hint "Answer"
	```julia
		area(r) = π * r^2
	```

	Julia provides very mathematical syntax. You program in Julia the same way as you would have written the equations on paper.
"""

# ╔═╡ 5f31fe56-ff6c-4cae-b835-15e100771a82
md"""

Sometimes, it is handy to write a one-off **anonymous function**, such as

"""

# ╔═╡ a1b6c516-0d3a-4add-990c-9ea755d6c7b6
((x) -> x^2)(5)

# ╔═╡ 9fbbb50d-ebb8-4e13-b28c-fbde63926cae
md"where `((x) -> x^2)` is an anonymous function, and 5 is the input argument."

# ╔═╡ 3f9138da-c21a-458f-8eaa-a10079a2ee35
md"""

An anonymous function has no name, therefore it cannot be reused later. However, the function can be passed around like a variable. In Julia, functions are first-class objects (treated the same way as a variable). Therefore, an anonymous function can be assigned a name, if needed.
"""

# ╔═╡ d0d5068b-f58d-4835-81eb-7cdc3273961e
my_square = (x) -> x^2

# ╔═╡ bcaf0dba-c26c-41f2-b5ad-34fb81b4e34d
my_square(5) # can be used the same as a normal function

# ╔═╡ a748b9e1-dd0c-4899-bf04-42a831e95434
md"""
### Map


`map(f, c)` is usually used to apply some function `f` to the elements of an iterable object, such as a list or array `c`. 

For example, to transform 1,2,...,5 to their cubic root:
"""

# ╔═╡ 22aa7b65-94f7-4cd7-a3f6-49d70c88e8e1
map((x) -> x^(1/3), 1:5)  

# ╔═╡ 3cfdb5c7-5ca3-4f86-ac9f-b3456b03303b
md"""
For example, to transform an array of strings of `"Yes"` and `"No"` to `1`/`0`.
"""

# ╔═╡ 60c7ec9a-a6e4-43a5-921a-b6ba3ba511d0
map((x) -> x == "Yes" ? 1 : 0, ["Yes", "Yes", "No", "Yes"])

# ╔═╡ 8e0c8524-3142-498b-bac1-60f591ede438
md"""
Find the sum along the column of a matrix:
"""

# ╔═╡ 42d47b65-5b68-4f81-af5f-17a1df35c1cb
M = latexify(ones(6, 3))

# ╔═╡ 0ff926f9-a294-4846-b7ab-d02203a73a13
md"Apply `sum` to each column of the matrix `M`."

# ╔═╡ 482b83c6-fe77-41ea-bc4a-4a9b76bf30ac
map(sum, eachcol(ones(6,3)))

# ╔═╡ 77ec658a-0fe5-4fc4-81ac-70336e46096b
md"Similarly, apply `sum` to each row:"

# ╔═╡ 528d9adb-1888-4f31-91eb-450879053bf6
map(sum, eachrow(ones(6,3)))

# ╔═╡ 49c595cf-4a6f-43fe-9c0e-365effd70f96
md"""

### Plotting
 
`Julia` provides very easy-to-use plotting methods to visualise functions. To plot a function, one should first import `Plots` package.
"""

# ╔═╡ f38cf720-5346-4590-b466-c42c86fe1555
md"For example, to visualise the `sin` function between -3π to 3π."

# ╔═╡ b6637bc1-a63c-44cc-bf57-ae2c7252e176
plot(sin, label="sin(x)", legend=:topright, linewidth=2)

# ╔═╡ 9d25c7c8-252c-4998-9d7c-f93be58ec671
md"""

To plot a series of `sin` functions with different frequencies: ``\sin(k x)``
  * note we use `plot!()` to modify the existing plot directly rather than creating a new plot

"""

# ╔═╡ 8416fec3-5460-441a-b415-40d04075ccf3
let
	plt = plot(-π: 0.01: π, sin, xlabel=L"x", ylabel=L"\sin(x)", label=L"\sin(x)", lw=1.5, legend=:outerbottom)

	for k in [2, 3]
		plot!(-π: 0.01: π, (x) -> sin(k*x), label=L"\sin(%$(k)x)", lw=1.5)
	end
	plt
end

# ╔═╡ 0eb81388-75c6-40c6-ad2e-04baf59a7d71
md"""

!!! danger "Exercise"
	Plot the user-defined polynomial function with parameters ``a=0, b=1, c=2, d=5``

!!! hint "Answer"
	```julia
		plot(0:0.1:10, (x) -> f(x; a=0, b=1, c=2, d=5))
	```
"""

# ╔═╡ 56961f01-40b2-4a18-90a2-a0ebeeebd16c
md"""

**Multi-dimensional plots.** For functions with multiple variables, it is convenient to visualise the function either with a contour plot or the surface 3-D plot.
"""

# ╔═╡ e65a6edb-12ab-44c4-a1ff-94a457583c49
let
	# a quadratic function
	qf(x, y) = x^2 - y^2
	xs = -5:.1:5
	ys = -5:.1:5
	# theme(:ggplot2)
	cont_plot = plot(xs, ys, (x,y) -> qf(x, y), st=:contour, ratio =1, legend=false)
	surf_plot = plot(xs, ys, (x,y) -> qf(x, y), st=:surface, ratio =1, legend=false)
	# plot side by side
	plot(cont_plot, surf_plot)
end

# ╔═╡ c8ab39c8-a3e0-4ef5-8a47-365b5c484042
md"""

### Animations

We can also put a series of plots (known as frames) together to form an animation. This is particularly useful to visually demonstrate an algorithm's procedures. `Julia` provides two macros: `@animate` and `@gif`, to simplify the animation creation process.

Here we use the derivative as an example to show how to create an animation. The derivative of a continuous function ``f(x)`` at ``x_0`` is defined as the limit of a ratio:

```math
f'(x_0) = \lim_{\Delta{x} \rightarrow 0} \frac{f(x_0 + \Delta{x}) - f(x_0)}{\Delta x}= \lim_{\Delta{x} \rightarrow 0} \frac{\Delta{f}}{\Delta x}
```

* when ``\Delta{x}`` approaches zero, the ratio ``\frac{f(x_0 + \Delta{x}) - f(x_0)}{\Delta x}`` gets closer to the exact derivative, 
* and the derivative can be interpreted as the slope of the limiting approximate linear function of ``f`` when ``\Delta{x}`` approaches zero.

The idea is demonstrated below in an animation.

"""

# ╔═╡ 9f420af3-f080-411d-bc14-e2e45572ac77
let
	x₀ = 0.0
	xs = -π : 0.1: π
	f, ∇f = sin, cos
	anim = @animate for Δx in π:-0.1:0.0
		plot(xs, sin, label=L"f(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:topleft)
		df = f(x₀ + Δx)-f(x₀)
		k = Δx == 0 ? ∇f(x₀) : df/Δx
		b = f(x₀) - k * x₀ 
		# the approximating linear function with Δx 
		plot!(xs, (x) -> k*x+b, label="", lw=2)
		# the location where the derivative is defined
		scatter!([x₀], [f(x₀)], ms=3, label=L"x_0, f(x_0)")
		scatter!([x₀+Δx], [f(x₀+Δx)], ms=3, label=L"x_0+Δx, f(x_0+Δx)")
		plot!([x₀, x₀+Δx], [f(x₀), f(x₀)], lc=:gray, label="")
		plot!([x₀+Δx, x₀+Δx], [f(x₀), f(x₀+Δx)], lc=:gray, label="")
		font_size = Δx < 0.8 ? 7 : 10
		annotate!(x₀+Δx, 0.5 *(f(x₀) + f(x₀+Δx)), text(L"Δf=%$(round(df, digits=1))", font_size, :top, rotation = 90))
		annotate!(0.5*(x₀+x₀+Δx), 0, text(L"Δx=%$(round(Δx, digits=1))", font_size,:top))
		annotate!(0, 1, text(L"\frac{Δf}{Δx}=%$(round(k, digits=2))", 10,:top))
	end

	gif(anim, fps=5)
end

# ╔═╡ da4fb082-1f42-41ac-a0eb-55a087b4b4a5
md"""
### Distributions


Julia's [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/) implements a wide range of popular distributions, such as
* `Normal(μ, σ)`, note that it is parameterised with standard deviation rather than variance
* `Gamma(α, θ)`, [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) parameterised with shape and scale 
* `Beta(α, β)`, [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)
* `Uniform(a, b)`, [Uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) between `a` and `b` where `b>a`;

Also popular discrete random variables:

* `Poisson(λ)`: where `λ` is the rate or mean of the Poisson distribution
* `Binomial(n, p)`: where `n` is the total number of experiments and `p` is individual experiment's bias
* `Bernoulli(p)`: where `p` is the bias

To see how to use a distribution, simply type `?distribution_name` in a cell. Or equivalently, use `@doc distribution_name`. For example, Gaussian distribution's document is listed below:
"""

# ╔═╡ d0646d89-64a4-4eb3-8739-2cac268abcf3
@doc Normal

# ╔═╡ a5212ff7-0416-440d-8ad6-9b6e0ac35a8d
md"""
`Julia` provides standard interfaces to view and manipulate a distribution. To random sample from a distribution, one can use `rand(dist, n)`, where `n` is the number of draws required.

"""

# ╔═╡ de16ea53-0391-41b5-950f-0c60654d0cd7
let
	dist = Normal(0, 1)
	gaussian_draws = rand(dist, 1000)
	plot(gaussian_draws, st=:hist, nbins=30, normed=true, label=L"x\sim \mathcal{N}(0,1)")
	plot!(dist, lw=2, fill=(0, 0.5), c=1, label=L"\mathcal{N}(0,1)")
end

# ╔═╡ fcdfed67-380d-4b11-8881-2575df2008b0
md"""

[`StatsPlots.jl`](https://github.com/JuliaPlots/StatsPlots.jl) provides handy methods to visualise a distribution. 
* `plot(dist,func)`: plot the probability density function (or probability mass function) of a distribution
* `scatter(dist, func)`: scatter plot of the density function
* `bars(dist, func)`: bar plot of distribution

where `func` can be 
* `pdf`: probability density function (default choice)
* `logpdf`: log probability density function, 
* `cdf`: cumulative density function

Check the code below for an example. Note you can replace `Gamma` with any distribution of your choice.
"""

# ╔═╡ 65dbb2b9-961b-43e0-9ff9-62790d60aa8e
md"---"

# ╔═╡ 1f699130-fb0f-434b-b0c2-5229e4745e1c
md"
[**↩ Home**](https://lf28.github.io/BayesianModelling/) 


[**↪ Next Chapter**](./section1_introduction.html)
"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.103"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.54"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "97c2221ddf05bca42e67ff6d2c0814cc5a5eb83b"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "793501dcd3fa7ce8d375a2c878dca2296232686e"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

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

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "05f9816a77231b07e634ab8715ba50e5249d6f76"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.5"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

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
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "5225c965635d8c21168e32a12954675e7bea1151"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.10"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a6c00f894f24460379cb7136633cef54ac9f6f4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.103"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "28e4e9c4b7b162398ec8004bdabe9a90c78c122d"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.8.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

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
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

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
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

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
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "0592b1810613d1c95eeebcd22dc11fba186c2a57"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.26"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

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

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

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
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

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
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4e5be6bb265d33669f98eb55d2a57addd1eeb72c"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.30"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

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
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "a38e7d70267283888bc83911626961f0b8d5966f"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.9"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

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
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "5ef59aea6f18c25168842bded46b16662141ab87"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.7.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

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
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "242982d62ff0d1671e9029b52743062739255c7e"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.18.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

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
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "da69178aacc095066bad1f69d2f59a60a1dd8ad1"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.0+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

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
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

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

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─f3770e42-edff-4c9f-9a13-47edaf8a8cae
# ╟─fe7f2df3-8c27-4e80-a633-386e8643e6e1
# ╟─35d6a20a-6ab1-4d46-be79-c84d664a6119
# ╟─baf83ba6-2079-11ed-1505-cb44aea4377e
# ╟─c85214aa-83b1-41f0-b856-f85435db6e1d
# ╟─e6c8aca8-1907-49b0-ba11-62fda98418fc
# ╟─a5654402-f51d-468e-99fb-09a5c69a6421
# ╟─b145ea5b-9fc0-4a57-b41b-ad8c825eaccc
# ╟─de6fc3a6-78dc-489a-8ac3-3642e0760d6a
# ╟─a3a93c83-94c8-44e7-a62f-46cae602ac77
# ╟─1635054e-3e02-4f3b-8726-bef468c11a82
# ╟─2b52dcf7-f8aa-49ea-a8e5-df4d819fc970
# ╠═3adfcebb-e620-4c41-b9a7-3584fbfd61c1
# ╟─0f9ce7d3-ccf3-40f6-9573-ec40dc643749
# ╠═bcf4086f-1469-49d4-a3f5-730a30771da4
# ╟─6be2784a-cf0c-4d75-9858-b063880c4e98
# ╟─3520ad46-7ac5-47bf-9389-62a46f6a26f8
# ╟─65ae3b8f-a33a-49ba-b135-ab20674ec67a
# ╠═c3fe7693-8425-4f6b-8529-c9b242073334
# ╟─045474d2-18f1-45b1-8c5d-9a82980e96bd
# ╠═287d41f1-9ade-4a73-b8ea-1e11aaf62fbe
# ╟─d51ba0c0-9ca1-47aa-8364-080abc8175c3
# ╠═ae200380-7c83-4fa6-9661-9cf1999732e3
# ╟─b3919bcb-c2e7-442b-a06c-93d172878f9c
# ╠═ed8457c7-9547-4953-ae72-121572fe8495
# ╟─49a993c4-27c9-4c89-b4c4-04c294f044c0
# ╟─a091f1d5-29c0-4a98-b1b8-4177cd6cd2f7
# ╟─a6c186ae-9d7a-452e-bfb2-b303c0cfea5a
# ╠═cbd40c3a-cf65-4d1c-833a-29a1e39dc893
# ╟─a08b431d-86b4-4f03-b0aa-eadba6284e95
# ╠═82b73754-f164-4bc0-82d9-fd0311a59919
# ╠═a12d7a2e-5a4e-46db-b802-1af1331456dd
# ╟─911bf98c-130c-4e20-a353-bc290c865a27
# ╠═5c15f435-8ea6-446e-a27d-3270b463f681
# ╟─b8d452b1-b75b-4012-872a-3baced9c1cb6
# ╠═054d075f-4e1c-42b2-955e-70ac1e36144b
# ╟─2d934969-c540-4e88-b137-d22e836e7daa
# ╠═d8764bd8-7d7d-4910-80ee-2187ad31f336
# ╟─d78b7c17-1f6d-424c-aeed-e35a9d3db49e
# ╠═d349aa84-4b2a-42c9-98a4-5849ece5e6ae
# ╟─6950f0da-d61a-465b-aaaf-8a802f50452a
# ╟─c68f7d30-3061-40db-8af3-369149cd5ff9
# ╠═84710a37-34c3-4c20-bd5c-b293d23bc7cf
# ╠═45037f4e-5744-4de7-885d-f0a678b72ec1
# ╟─b3d46870-ef00-4fec-a5f2-2f3b6b11687c
# ╠═52a6e45f-b458-463a-95b8-0f8a184ce171
# ╠═b078d4c9-7638-43a4-bd77-57a40b1511ef
# ╠═2b305ef2-551e-4327-ad56-2bd3a12f51fc
# ╟─84713022-8e34-4e57-aa2f-cf9d6cf61cad
# ╟─36824286-375a-48db-af65-b6dcaf6497b8
# ╟─46be754e-ec2f-4e0d-aef3-865492d2dd0c
# ╟─1af4b04a-a7a1-4808-9cf0-038c95037fa5
# ╟─7d897f13-7358-446f-b87a-60ad945769ea
# ╠═7cd155b2-f76e-4152-9d79-ac9ea8c32b16
# ╟─d0b5cc5a-f0cf-44e8-9e2a-e28b94d4429f
# ╠═04065f11-4748-4cf0-abf4-01484454da0b
# ╟─aed0ed19-0183-4eb8-83c9-3d2419d299b8
# ╟─02938204-99d2-4209-abcf-cbcfd62f9c04
# ╠═8548c4cb-9e9b-4ca7-bd4e-97e16058606d
# ╟─c2ee8d41-eec8-43ad-b4c0-2f859ad03b93
# ╠═c195aa69-89f6-4417-b602-67922a76b87a
# ╟─4a51bb66-f4d4-4a7d-8e07-91d3ff5f4f28
# ╟─17b5759c-deab-4bc0-b7b6-fc2c0e64eebb
# ╠═c52294ca-49fb-43f7-87bb-860c8088506d
# ╟─fcd72da9-e175-4fd1-a047-7dd3711dcbf4
# ╠═6df9180f-59bf-4462-ae9d-a2623379071a
# ╟─a1aa6ab4-f183-4adc-9ad8-70d80deaa15f
# ╠═dfa4201c-19da-4ca9-b537-8396c2899d42
# ╟─dbc643ae-d68c-416f-b1b9-dc8a0025dfe0
# ╠═e1de498d-d66f-42c8-bb22-e80d623c4554
# ╟─6c4ede4f-a566-4dfc-bfdf-f4feadac84aa
# ╠═14fc0ff8-2284-412f-99af-6264d24d6198
# ╠═4958c15f-5725-4645-a71d-e8a26a94a894
# ╟─a5c62a3c-2a59-4388-adb5-b0d93be2360d
# ╟─5f31fe56-ff6c-4cae-b835-15e100771a82
# ╠═a1b6c516-0d3a-4add-990c-9ea755d6c7b6
# ╟─9fbbb50d-ebb8-4e13-b28c-fbde63926cae
# ╟─3f9138da-c21a-458f-8eaa-a10079a2ee35
# ╠═d0d5068b-f58d-4835-81eb-7cdc3273961e
# ╠═bcaf0dba-c26c-41f2-b5ad-34fb81b4e34d
# ╟─a748b9e1-dd0c-4899-bf04-42a831e95434
# ╠═22aa7b65-94f7-4cd7-a3f6-49d70c88e8e1
# ╟─3cfdb5c7-5ca3-4f86-ac9f-b3456b03303b
# ╠═60c7ec9a-a6e4-43a5-921a-b6ba3ba511d0
# ╟─8e0c8524-3142-498b-bac1-60f591ede438
# ╟─42d47b65-5b68-4f81-af5f-17a1df35c1cb
# ╟─0ff926f9-a294-4846-b7ab-d02203a73a13
# ╠═482b83c6-fe77-41ea-bc4a-4a9b76bf30ac
# ╟─77ec658a-0fe5-4fc4-81ac-70336e46096b
# ╠═528d9adb-1888-4f31-91eb-450879053bf6
# ╟─49c595cf-4a6f-43fe-9c0e-365effd70f96
# ╠═58a419b4-4017-474f-a3e9-cc58d2aaa29a
# ╟─f38cf720-5346-4590-b466-c42c86fe1555
# ╠═b6637bc1-a63c-44cc-bf57-ae2c7252e176
# ╟─9d25c7c8-252c-4998-9d7c-f93be58ec671
# ╠═8416fec3-5460-441a-b415-40d04075ccf3
# ╟─0eb81388-75c6-40c6-ad2e-04baf59a7d71
# ╟─56961f01-40b2-4a18-90a2-a0ebeeebd16c
# ╠═e65a6edb-12ab-44c4-a1ff-94a457583c49
# ╟─c8ab39c8-a3e0-4ef5-8a47-365b5c484042
# ╠═9f420af3-f080-411d-bc14-e2e45572ac77
# ╟─da4fb082-1f42-41ac-a0eb-55a087b4b4a5
# ╠═d0646d89-64a4-4eb3-8739-2cac268abcf3
# ╟─a5212ff7-0416-440d-8ad6-9b6e0ac35a8d
# ╠═de16ea53-0391-41b5-950f-0c60654d0cd7
# ╟─fcdfed67-380d-4b11-8881-2575df2008b0
# ╠═0ee0e688-9bc7-489b-a4ae-a4f09477fd33
# ╟─65dbb2b9-961b-43e0-9ff9-62790d60aa8e
# ╟─1f699130-fb0f-434b-b0c2-5229e4745e1c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
