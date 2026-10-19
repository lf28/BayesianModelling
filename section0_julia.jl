### A Pluto.jl notebook ###
# v1.0.4

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
* Choose the latest stable version 


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
Distributions = "~0.25.132"
LaTeXStrings = "~1.4.1"
Latexify = "~0.16.12"
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
project_hash = "d2242eaa29c29d9846ad4b4cf97f3467ac1ca508"

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
