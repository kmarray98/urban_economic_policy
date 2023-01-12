### A Pluto.jl notebook ###
# v0.19.9

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

# ╔═╡ ff95d6a8-e56b-48c9-9d64-61a134e6effc
using PlutoUI, Plots, Random, Distributions, LinearAlgebra, LaTeXStrings, Optim

# ╔═╡ 561b4e30-2916-47c5-ae90-1c2396343366
md" # Urban economics policy and challenges: econometrics refresher"

# ╔═╡ d4dd4e40-8130-11ed-1d7f-1f21a70ab743
TableOfContents()

# ╔═╡ 66765dfc-7cc3-43af-a282-08dcc08e15a9
md" Here, we cover concepts in econometrics that are useful for the course. We go through each quickly, as you should have seen these concepts before. We cover

+ data preparation,
+ estimation strategies (parametric vs. semi-parametric vs. non-parametric, structural vs. reduced-form)
+ estimation methods,
+ causal inference, and
+ ordinal data.

If you do not know about estimators, testing, or the general idea of statistical inference, we detail these briefly in the appendix.

For more, I recommend

+ *Microeconometrics: methods and applications* (2005) by Colin Cameron and Pravin Trivedi (the econometrics bible)
+ *Mostly harmless econometrics: an empiricist's companion* (2009) by Joshua Angrist and Jorn-Steffen Pishke
+ *Causal inference: the mixtape* (2021) by Scott Cunningham (available in web form [here](https://mixtape.scunning.com/)).

"

# ╔═╡ 98e0a048-1a7e-4242-b2a5-982d273d6e87
md" As an example throughout, we generate the data from a log-linear hedonic pricing model

```math
p_{i} = e^{\alpha + \beta x_{i} + \gamma x^{2}_{i} + \epsilon_{i}}
```
depending on some variable $x$ plus the parameters $(\alpha, \beta, \gamma)$. "

# ╔═╡ a1595bed-ecf7-4042-bff6-b63fd4fd351f
begin
	alpha_slide = @bind α Slider(-2:0.1:2.0, 0.5, true)
	beta_slide = @bind β Slider(-1.0:0.1:1.0, 1.0, true)
	gamma_slide = @bind γ Slider(-1.0:0.1:1.0, 1.0, true)
	md"alpha: $(alpha_slide) \
	beta: $(beta_slide)\
	gamma: $(gamma_slide)"
end

# ╔═╡ dcb102b5-1cef-48e4-a3b6-53f70a33ecf1
begin

	rng = MersenneTwister(2023) # setting seed

	#β = 0.5
	#γ = 0.2

	d_price = Normal(1, 1) # setting distributions
    d_int = Normal(0,0.5)
	
    X = rand(rng, d_price, 10000) # prices
	#α = 0.5 # individual-specific intercepts drawn Gaussian
	error = rand(rng, d_int, 10000)

	Y = exp.(α .* ones(10000) .+ β .* X+ γ .* X.^2 + error) # now simulating Y

	scatter(X, log.(Y), label="Data")
	xlabel!("x")
	ylabel!("Price")

	
	
end

# ╔═╡ 7e4e549c-ce8d-47c6-ab82-bf0d570dcacb
md"## Preparing your data"

# ╔═╡ c824a69f-637a-4b61-acee-e91410ac4f63
md" **Before you start doing any estimation**, there are two things you need to think about.

+ What **specific** question do you want to answer.
+ Is the data that you have **error-free** and **appropriate to answer your specific question**.

To guide you in doing this, the first thing you should do in any analysis is **plot your data**. You should plot a **histogram of each of the main variables you care about** and a **scatterplot of the main variables you think are related**. 
"

# ╔═╡ 46ec542c-de39-42cd-929f-dccccc2c462d
md"The first will show you if the data is in units that you did not realise (e.g logs) or will reveal data entry errors that would mess up your estimates. In extreme cases, [it can also reveal fraud](https://datacolada.org/98). For example, if someone accidentally switched the units on some of the house price quality data I generated above, the basic histogram would look like this."

# ╔═╡ e5a98d77-a512-4c58-8ca4-4dc384971df2
begin
   X_error = copy(X)
	X_error[50:300] = X_error[50:300] .* 1000
	histogram(X_error, label="")
	
end

# ╔═╡ 73ee62a1-de4e-40b8-8d62-56bb07e292e8
md" From how the variable is defined, I know the top observation should be around 4 so the histogram should look more like this."

# ╔═╡ eae896fe-c1c8-4ae2-a896-0641f472d6be
histogram(X, label="")

# ╔═╡ a54e2053-fb8f-4c20-acfb-8043aaa3c416
md" The difference between what I see and what I expect tells me that something is wrong, so I should look more carefully at the data to find any coding errors etc, or check if I have misunderstood it.

The second shows you if the relationship you want to estimate is actually there. If our price data and quality data is related, we should see a relatively clear trend in the data like the first scatterplot above. If instead we see something like this"

# ╔═╡ 69ac43a3-54dc-412b-9595-64abeb80c567
begin

	Y_error = rand(rng, d_price, 10000)
	scatter(X, Y_error, label="")
end

# ╔═╡ a9f50d52-fda7-4bb3-9104-9901eb75b257
md" then maybe our variables are not actually be related and we need to think about another research question."

# ╔═╡ 3b853bc9-e9a5-4b59-ba83-976a3ddd4fd1
md" When we make histograms, we might see a few extreme data points (what people often refer to as **outliers**). This is especially true if we are dealing with a variable that is very skewed like house prices e.g"

# ╔═╡ 19f8ca7e-97b1-44e2-87d4-a269d2597675
histogram(log.(Y), label="Log house prices")

# ╔═╡ e34fe2b8-fe93-470c-bb40-23777f6d761a
md"Extreme observations might drive the results of our parameter estimates. So, what we do depends on our specific question. Do we care about the effect of quality on house prices in general, including mansions? Or do we care about the effect of quality on 'normal' house prices, excluding mansions? If we care about the second, we might want to **drop the top observations** so our estimates reflect the trend in most of the data. This is called **censoring** our data. If we censor the house price data above to exclude prices greater than 1,000,000 and then plot in logs, we get something less skewed."

# ╔═╡ 64309bbb-3c2e-4e1a-b1de-5b7626cd9b31
begin
	Y_censored = Y[Y .< 1000000]
    histogram(log.(Y_censored), label="Log house prices")
end

# ╔═╡ fab276f4-4067-4c2b-916f-98d5278ffd9d
md" Whatever you do, **justify your choices!** **Do not arbitrarily delete data points**! If you think the person who created the data has entered a point incorrectly, you should be able to provide a convincing reason why you think this. If you want to censor your data, you should be explicit about which individuals your results then apply to."

# ╔═╡ 94d0747f-be6f-4055-8def-1914a02a95c5
md" ## Estimation strategies

Once we have our cleaned data and know what question we want to answer, we need to pick a way that we will estimate parameters. Which we pick depends on how much prior information we have about our true **data generating process**, and how confident we are that this prior information is correct. We face a fundamental trade-off. The more prior information we include, the more efficient our estimator of the parameters will be. But if our prior information is incorrect, our estimates will also be incorrect. This splits into two related choices: what type of statistical model to use, and how much economic theory to use in our model."

# ╔═╡ e8612d42-aa11-43fa-ae67-24fca5c21a11
md" ### Types of statistical model"

# ╔═╡ 8364256f-0a18-4dc1-997c-142a8fc93f34
md" To estimate a parameter $\beta$, we typically fit some kind of model $f(x, \beta, \epsilon)$ where $\epsilon$ is our error term that takes some distribution. Our estimation strategy depends on how much prior knowledge we are willing to assume we have.
+ To do **parametric** estimation, we assume that we know the true $f()$ and the true distribution of $\epsilon$. Then we fit the model using maximum likelihood.
+ To do **semi-parametric** estimation, we assume that we know some properties of $f()$ at the true parameter values. Then we fit the model using these properties.
+ To do **non-parametric** estimation, we assume that we know nothing about $f()$. Instead, we estimate functions of $f()$ (e.g the conditional mean of y given x) as their sample values, appropriately weighted.
"

# ╔═╡ 803df173-04c2-491a-8450-361ea126be8e
md" ### Types of economic model
We might also have some model from economic theory that gives us a specific form of $f()$. There are two ways of using the theory to estimate parameters.
+ **Structural estimation** takes the form of $f()$ given by our economic theory and fit it to data.
+ **Reduced form estimation** uses the form of $f()$ given by the theory to motivate the choice of variables in a simpler $f()$, e.g a linear regression, that we then fit to data.

Structural estimation gives us a more accurate parameter estimates if the structure given by economic theory accurately describes the world. But it typically involves making quite strong assumptions on people's expectations, preferences etc that we need to solve the economic model. We might be sceptical of these assumptions. Reduced form estimation might give less correct parameter estimates if the true structure is very non-linear, but it allows us to describe the data without making these assumptions.
"

# ╔═╡ 2f0d6238-7e3b-4669-9d7e-72a2994456dc
md" ## Common estimators"

# ╔═╡ bf4a7cbb-3778-4557-94ec-9fdf382a330d
md" ### Least-squares "

# ╔═╡ 7d1a97f7-d1b0-43b4-8f93-68d89e9b0ce7
md" We typically use a **regression** model -- estimating the conditional mean of our dependent variable given some independent variables as a function of the independent variables. We estimate the conditional mean based on some **loss function**, which tells us how much we care about different errors the model could make. 

Call our regression function $f(x, \beta)$ where $\beta$ is the parameter that we are trying to estimate.

The most common loss function we use is **least-squares** loss 

```math
\hat{\beta} = \text{arg\_min}_{\beta}\sum_{i=1}^{N}(y_{i}-f(x_{i}, \beta))^{2}.
```

This loss function says that we care about large errors more than small ones, and do not care about the sign of the errors.
"

# ╔═╡ 8f8a6c1d-4fc0-4d76-a9d7-d00782941a49
begin

	ϵ = collect(-5:0.1:5)
	plot(ϵ, ϵ.^2, label="Least-squares loss")
	xlabel!(L"y_i-f(x_i,\beta)")
	ylabel!("Loss")

end

# ╔═╡ 87b02c57-7f4b-413f-a0d3-018f004ae6ad
md"Other loss functions are possible, but we will not get into them here."

# ╔═╡ ac263e35-99d6-4491-8bdb-04fd4c94a460
md"The most common specification of $f(x,\beta)$ is to fit a straight line -- known as **linear regression**

```math
f(x, \beta) = \alpha + \beta x.
```

A linear regression model plus least-squares loss plus the assumption that all errors have the same distribution gives us the **ordinary least-squares** regression we see most commonly in applied econometrics

```math
y_{i} = \alpha + \beta x_{i} + \epsilon_{i}.
```

We fit

```math
\hat{\alpha},\hat{\beta} = \text{arg\_min}_{\alpha, \beta}\sum_{i=1}^{N}(y_{i} - \alpha + \beta x_{i})^{2}.
```"

# ╔═╡ 37d475f6-9bd1-4a7b-bd5d-5b9a189fdefd
    md" For example, if we log-transform our simulated house-price variables, we     would fit the log-linear regression

```math
\ln(p_i) = \alpha + \beta x_{i} + \epsilon_{i}.
```

Below, we fit two of these models, both including and not including the quadratic term.	

"	

# ╔═╡ 35b51e6a-f4e2-4ae1-ba2b-16f2734f556a
begin

X_mat = [ones(10000) X log.(Y)]
X_mat_2 = [ones(10000) X X.^2 log.(Y)]	

sort!(X_mat, dims=1, by = x -> x[1])	
sort!(X_mat_2, dims=1, by = x -> x[1])		
params_1 = (X_mat[:,1:2]' * X_mat[:,1:2])^-1 * X_mat[:,1:2]' * X_mat[:,3]
params_2 = (X_mat_2[:,1:3]' * X_mat_2[:,1:3])^-1 * X_mat_2[:,1:3]' * X_mat_2[:,4]

#Y_2 = exp.(X_mat_2 * params_2)	

	# Error somewhere above!
scatter(X, log.(Y), label="Data", alpha=0.6)

# now need to order by X

	
plot!(X_mat[:,2], X_mat[:,1:2] * params_1, label="OLS - without quadratic term", lwd=5, color=:red)	
plot!(X_mat_2[:,2], X_mat_2[:,1:3] * params_2, label="OLS - with quadratic term", lwd=5, color=:orange)		
end

# ╔═╡ e4ccd357-535f-4d2a-a832-ab02202f66bc
Print("Parameters in first model are: alpha = " * string(params_1[1]) * ", and beta =  " *  string(params_1[2]))

# ╔═╡ 70404453-6444-4ba4-8b8e-f7cb43bd6b70
Print("Parameters in second model are: alpha = " * string(params_2[1]) * ", beta =  " *  string(params_2[2]) * ", and gamma = " * string(params_2[3]))

# ╔═╡ b16801b2-0e2b-44ba-922e-3653a0baef70
md" ### Maximum-likelihood "

# ╔═╡ 6f261bf9-dee9-4a0a-91b7-e6b1adf484fd
md" Instead of specifying a fixed loss function, we can specify a probability density function for our data. For example, assume that our data comes from a normal (Gaussian) distribtuion. Then the probability density $g(y)$ is
```math

g(y) = \frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^{2}}.

```
With a model for $\mu$, we can then estimate $\beta$ by maximising the product of the densities of all the observed data i.e

```math
\hat{\beta} = \text{arg\_max}_{\beta} L(y|x, \beta, \sigma) := \prod_{i=1}^{N}g(y_{i}|x_{i},\beta, \sigma).
```
Each density represents how likely each observed data point is given the data, so if each parameter and the model are correct, the data points should be very likely. The product is called the **likelihood**, so this method is called **maximum-likelihood**. If the likelihood is correct, maximum likelihood is the most efficient method.

Maximum likelihood with a normal distribution is equivalent to least-squares estimation.
"

# ╔═╡ d609e18a-6508-4867-86ab-b45691d92f41
md" ### Methods of moments"

# ╔═╡ 57a4e042-8aac-4f5c-a668-8a8060403b51
md" Instead of using a loss function or specifying a likelihood, we might specify some facts based on summary statistics of the data that must be true in the population at the true parameter values. These facts are called **population moments**. An example is that the conditional expectation of the errors should be $0$ i.e

```math
E(y-\beta x|x) = 0.
```

We can then fit a model by constructing the **sample analogue** of the population moment conditions i.e in our example the mean error

```math
\sum_{i=1}^{N}(y_{i}-\beta x_{i}|x_{i})
```

and then picking our parameters to **minimise the distance of the sample analogues from the population moment condition**. Typically we have more moment conditions than parameters. So if we stack our moment conditions in a vector $G$, weight them using a weight matrix $W$, and use squared distance, we can write

```math
\hat{\beta} = \text{arg\_min}G(\beta, x)'WG(\beta, x).
```

Our estimator is called the **generalised method of moments** estimator."

# ╔═╡ 03c57489-3b0e-436d-8e8b-24393c640a2a
md" ### Non-parametric regression
We might instead try to estimate a potentially non-linear conditional mean

```math
E(y|x) = m(x)
```

where $m()$ is some unknown, potentially very complicated, function.

A standard way of doing this is a **locally weighted regression**. Here, we estimate the conditional mean function at some point in the data as a **weighted average of the across the whole dataset** where the **weights decrease the further we go from our current point**. 

```math
\bar{y} = \frac{\sum_{i=1}^{m}w_{i}(x_{i}, h)y_{i}}{\sum_{i=1}^{m}w_{i}(x_{i},h)}.
```

The weighting function is a function of $x$ and typically the **kernel** of some probability density. So we only use data that is 'close' to the point we care about, and use the data more the 'closer' it is. The parameter `h' is the **bandwidth**, and determines the points we look at. The simplest example is **k-nearest neighbours**

```math
w_{i}(k) = \frac{1}{k+1} \text{ for j in} \{i-\frac{k}{2}\,..., i+\frac{k}{2}\}, 
0 \text{ else}.
```

i.e the average of the k nearest neighbours to the point.
"

# ╔═╡ 393f8fa0-a9f5-4dc6-b54f-f460b709ddb1
begin
    h_slide = @bind h Slider(2:2:200, 6, true)
	md"h: $(h_slide)
	"
end

# ╔═╡ 9518b760-2112-437f-bdc7-eee26cb54d8c

# computing k nearest neighbour fit

begin

	h_slice = Int64(h/2)

	log_y = X_mat[:,3]


    y_plot = [ mean(@view log_y[(i-(h_slice)):(i+(h_slice))]) for i in (h_slice)+1:(length(log_y)-h_slice - 1)]

	# does not seem to be going up enough at the end...

    scatter(X, log.(Y), label="Data", alpha=0.6)

# now need to order by X

	
     plot!(X_mat[(h_slice)+1:(length(log_y)-h_slice - 1),2], y_plot, label="K nearest neighbour fit", lwd=5, color=:red)	



end

# ╔═╡ 75a0e831-e61a-4637-ba78-c6b32fe2db96
md" ## Casual inference"

# ╔═╡ c7d7aeb1-ab48-4985-a7ae-efd6385f5db2
md"We often want to measure the effect that some intervention has on an outcome that we care about. Thus, we want to estimate the **causal effect** of the intervention on our outcome. To do this, we should first quickly define what we mean by a causal effect. Econometricians typically use a **counterfactual account of causality** - roughly that $x$ causes $y$ if and only if if $x$ had not occurred, $y$ would not have occurred (other definitions are possible, e.g see [here](https://plato.stanford.edu/entries/causal-models/)).

Consider a binary treatment $D \in \{0,1\}$, and consider some outcome $Y$. Define the **potential outcomes** $Y_{i}^{1}, Y_{i}^{0}$ under different treatments. Following our definition, we think of the causal effect of $D$ on $Y$ for individual $i$ as the **difference in potential outcomes**

```math
\delta_{i} = Y_{i}^{1} - Y_{i}^{0}
```

under the different treatments. The outcome we observe is

```math
Y_{i} = Y_{i}^{0} + D_{i}(Y_{i}^{1} - Y_{i}^{0}).
```

Imagine we try to estimate the causal effect of $D$ on $Y$ by simply comparing average differences across groups. Using our expression for the observed outcome above, we have


```math
\begin{align*}
E(Y_{i}|D_{i} = 1) - E(Y_{i}|D_{i}=0) = &E(Y_{i}^{1}|D_{i}=1) - E(Y_{i}^{0}|D_{i}=1) +\\ 
&E(Y_{i}^{0}|D_{i} = 1) - E(Y_{i}^{0}|D_{i}=0)
\end{align*}
```

i.e the average treatment effect on the treated plus **selection bias**. To estimate the first, we need a way of accounting for the second.
"

# ╔═╡ cc022b3e-4bd5-4ac9-91fc-c833625fc6d4
md" Now imagine we have a reconstruction project where individuals get a binary treatment if their house quality is less than 1 and increases log house prices by $\delta$."

# ╔═╡ 0fafd79e-fd1f-4639-b9d2-f74ded42b54b
begin
    d_slide = @bind d Slider(0:0.1:5, 1.7, true)
	bx_slide = @bind bx Slider(0:0.1:5, 1.7, true)
	md"delta: $(d_slide)\
	beta\_x: $(bx_slide)
	"
end

# ╔═╡ 20f67f15-ec5b-4f31-ba26-553ff06d56be
begin

	
    X_caus = rand(rng, d_price, 10000) # prices
	D = zeros(10000)
	D[X_caus .>= 1] .= 1
	#α = 0.5 # individual-specific intercepts drawn Gaussian
	error_caus = rand(rng, d_int, 10000)

	Y_caus = exp.(bx .* X_caus + d .* D + error_caus) # now simulating Y

	scatter(X_caus, log.(Y_caus), label="Data")
	xlabel!("x")
	ylabel!("Log Price")
	vline!([1], lwd=3, label="Treatment threshold")
end

# ╔═╡ 2cdde6b5-05c9-4511-bbbd-054fd0bb2d9f
md" Now if we compute the difference in means, we get"

# ╔═╡ 330a3993-e057-45b2-9836-711032a86db2
begin
   X_caus_mat = [X_caus D log.(Y_caus)]
   sort!(X_caus_mat, dims=1, by = x -> x[1])	
	Print(mean(X_caus_mat[X_caus_mat[:,1] .>= 1,3]) - mean(X_caus_mat[X_caus_mat[:,1] .< 1,3]))
   #Print("Parameter estimate by OLS: " * string(params_caus[2]))
end

# ╔═╡ 4e5f42e6-d1d0-4233-8f41-a339db0c0be4
md" which is very wrong."

# ╔═╡ 235dac45-c8a0-4f32-aa11-f9f6d5b629c7
md" ### Randomisation 
The first, and simplest, way to remove selection bias is to randomise individuals to treatment and control groups. Then, 

``` math
E(Y_{i}^{0}|D_{i} = 1) = E(Y_{i}^{0}|D_{i}=0)
```
so we can just compare means.

"

# ╔═╡ afabdddb-c728-4910-a45f-ec1a055f3238
md" Now if we randomise our treatment instead, we get"

# ╔═╡ 77988813-3b5c-4757-bbae-d79a37f6623e
begin

    D_r_dist = Bernoulli(0.5)
	D_r = rand(rng, D_r_dist, 10000)

	Y_caus_r = exp.(β .* X_caus + d .* D_r + error_caus) 

	X_caus_mat_r = [X_caus D_r log.(Y_caus_r)]
    sort!(X_caus_mat_r, dims=1, by = x -> x[1])	
    	Print(mean(X_caus_mat_r[X_caus_mat_r[:,1] .>= 1,3]) - mean(X_caus_mat_r[X_caus_mat_r[:,1] .< 1,3]))

end



# ╔═╡ a72e5d33-0a1f-4a79-b855-79197f4613cb
md" which is a lot better. But, randomisation is not always possible. So we need other tools."

# ╔═╡ 57fa9569-9089-4ef4-bc2b-ecbd8c964a44
md" ### 2) Natural experiments
The second, most common, way to remove selection bias is to exploit some **natural experiment** where some quirk in policy or nature does the randomisation for you. Then, we compare the differences of individuals in one groups in the natural experiment versus the other.

The clearest example is an **arbitrary discontinuity** in policy. For example, imagine that we have an urban renewal program that targets houses in a poorer area based on some geographical boundary. If the houses on each side of the boundary are similar enough that we think their potential outcomes are the same, then we can remove selection bias by comparing individuals on one side of the boundary to the other. We typically do this with a **regression discontinuity design**, or if we have panel data we might use **differences-in-differences**.

In our house price data above, we have an arbitrary discontinuity at $x=1$. So, we can estimate the causal effect by using the OLS regression

```math
y_{i} = \alpha + \beta x_{i} + \delta I(x_{i} < 1) + \epsilon_{i}
```

locally around the discontinuity. Normally, we would also have to estimate the trend in x using some kind of polynomial expansion, but we ignore this here for simplicity.
"

# ╔═╡ ba4ba6a6-d473-4a71-9c91-312b2e75bfa5
begin
	X_rdd_mat = X_caus_mat[(X_caus_mat[:,1] .>= 0.5) .& (X_caus_mat[:,1] .<=1.5),:]

	params_rdd = (X_rdd_mat[:,1:2]' * X_rdd_mat[:,1:2])^-1 * X_rdd_mat[:,1:2]' *      X_rdd_mat[:,3]
    Print("Parameter estimate by RDD: " * string(params_rdd[2]))

	#size(X_rdd_mat)
end
	

# ╔═╡ 2b5ad567-1ce2-4716-a8e8-b22c1f559bf2
begin
scatter(X_caus, log.(Y_caus), label="Data", alpha=0.6)
xlabel!("x")
ylabel!("Log Price")
rdd_fitted = X_caus_mat[:,1:2] * params_rdd
plot!(X_caus_mat[:,1], rdd_fitted, label="RDD fit", lwd=5, color=:red)	
end

# ╔═╡ 37cba339-e070-4e23-ac0b-ac2eeff51279
md"
A less clear example is an **arbitrary change in another variable** that **only affects the treatment status** of some individuals. Then, we can remove selection bias using an **instrument**. An instrument $Z$ is a variable such that

```math
\text{Cov}(Z,Y) \neq 0, \text{ and } Z \perp Y|D.
```

Another way of saying this is that Z affects Y only through altering the treatment assignment D of some units.

Here Z is the variable corresponding to our natural experiment. Then, we can estimate the treatment effect as

```math
\hat{\delta} = \frac{\text{Cov}(Z,Y)}{\text{Cov}(Z,D)}.
```


"

# ╔═╡ 971a83cc-0348-48ce-97c8-23620440e942
md"ERROR IN 2SLS BELOW"

# ╔═╡ 62181f51-71eb-4fe7-94f4-1618b00c362a
begin

	Z_dist = Bernoulli()
	Z = rand(rng, Z_dist, 10000)
	D_iv = zeros(10000)
	D_iv[(Z.==1) .& (X_caus .>= 1)] .= 1

	Y_iv = exp.(bx .* X_caus + d .* D_iv + error_caus)

	fs_iv_mat = [X_caus Z Y_iv]

	fs_params = (fs_iv_mat[:,1:2]' * fs_iv_mat[:,1:2])^-1 * fs_iv_mat[:,1:2]' *      fs_iv_mat[:,3]

	fs_fitted = fs_iv_mat[:,1:2] * fs_params

	ss_iv_mat = [X_caus fs_fitted Y_iv]

	ss_params = (ss_iv_mat[:,1:2]' * ss_iv_mat[:,1:2])^-1 * ss_iv_mat[:,1:2]' *      ss_iv_mat[:,3]

	#Print(ss_params[2])

end

# ╔═╡ e1225785-a96f-4668-b142-e17edc0f20bf
md" ### 3) Structural estimation
The third way to remove any selection bias is to fit a fully specified economic model to the data. The model should include the effect of any relevant mechanisms that generate a bias. To be a bit more precise, we fit some model

```math
Y = f(X,D, \delta, \epsilon)
```

such that

```math
\epsilon \perp D|X.
```

Then, estimating $\hat{\delta}$ from fitting our model is an unbiased and consistent estimator.


An advantage here is that you get a full description of potentially non-linear counterfactuals that can generalise outside of your current sample. A disadvantage is that you need to make stronger assumptions on functional form to get there.

"

# ╔═╡ 9a46c416-6fc1-4bca-a347-f7ff68742ca4
md"## Ordinal data"

# ╔═╡ 2feaea6a-baa5-4742-ac64-9080caee12b8
md" A special type of data we often run into in policy evaluation is data that indicates that an individual has made a certain choice. For example, we might have data on whether a set of individuals purchase a house in a certain neighbourhood." 

# ╔═╡ 3ab2dc19-22f6-430e-b7e8-09d83deec04d
md" The first way we might try to estimate this model is just using least-squares regression. This is called the **linear probability model**, and is useful for instrumental variables regression. But sometimes it is not that useful because our probalem is non-linear...

To fit a non-linear model by maximum-likelihood, the first thing we notice is that by definition any binary (multinomial) variable follows the binomial (multinomial) distribution. So the log-likelihood is
```math
l(y|\beta) = \sum_{i=1}^{N}y_{i}p(y_{i}=1|X, \beta) + (1-y_{i})(1-p(y_{i}=1|X, \beta)).
```

"

# ╔═╡ 348e8426-f74a-43b5-bbb7-384133455a7d
md" ### The additive random utility model"

# ╔═╡ 45b5b2a9-104a-480b-963b-e1b21542b84f
md"From our microeconomics refresher, we know that economists typically think of individuals as making choices to maximise their preferences, which we represent mathematically (under certain assumptions) with a utility function. So, if we observe choices, we think that they are the result of the individual maximising their utility. To be more specific, imagine that utility individuals get from each choice depends on some choice-specific components we can observe, and an additive component that we cannot and we assume is a random variable"

# ╔═╡ 6fb82d2d-8951-485e-9fd6-1a5867949358
md"
```math
\begin{align}
  U_{i1} &= \alpha_{1} + \beta X_{i1} + \epsilon_{i1}\\
U_{i0} &= \alpha_{0} + \beta X_{i0} + \epsilon_{i0}
\end{align}
```
"

# ╔═╡ 316cb637-6f83-4f2f-a354-d75ff59040dd
md" We observe a choice variable 
```math
Y_{i} \in \{0,1\}.
```
From utility maximisation we get that

```math
Y_{i} = 1 \text{ iff } U_{i1} > U_{i0}.
```
i.e
```math
Y_{i} = 1 \text{ iff } \alpha_{1} + \beta X_{i1} + \epsilon_{i1} > \alpha_{0} + \beta X_{i0} + \epsilon_{i0}
```

"

# ╔═╡ 55a75f7e-528c-4aad-a5ab-32ed9d264793
md"So by rearranging, and letting $\tilde{X}_{i}$ denote $X_{i1}- X_{i0}$, we get that
```math
P(Y_{i}=1|X_{i}) = P(\tilde{\epsilon}_{i} > - \tilde{\alpha} - \beta \tilde{X}_{i}|X_{i}).
```
"

# ╔═╡ c71a354c-fc48-40a1-836d-4324791fe499
md" The final thing we need is a model for $\tilde{\epsilon}_{i}$, then we can construct a likelihood using our binomial likelihood from before. Normally we either choose a logistic distribution, giving a logit model, 
```math
p(y_{i}=1|X_{i}) = \frac{e^{\tilde{\alpha} + \beta \tilde{X}_{i}}}{1 + e^{\tilde{\alpha} + \beta \tilde{X}_{i}}}
```

or a normal distribution, giving a probit model

```math
p(y_{i}=1|X_{i}) = \Phi(\tilde{\alpha} + \beta \tilde{X}_{i}).
```
. We choose a logistic distribution specifically because of a result in extreme-value theory. The **Fisher-Tippett-Gnedenko theorem** says that the maximimum of a sequence of i.i.d normal variables converges to a Gumbel distribution. The difference of two Gumbel distributed variables has a logistic distribution."

# ╔═╡ 7e57a34a-880f-4a72-b693-1e7692701dc8
md" For example, here we simulate the utilities above depending, with Gumbel(0,1) additive random utility shocks."


# ╔═╡ 54175a37-1c01-4c9d-a25a-93505b5295ff
begin
	int_slide_0 = @bind alpha_0 Slider(0.1:0.1:10, 2.0, true)
	int_slide_1 = @bind alpha_1 Slider(0.1:0.1:10, 2.0, true)
    choice_slide = @bind beta_choice Slider(0.1:0.1:10, 2.0, true)
	md"alpha 0: $(int_slide_0)\
	alpha 1: $(int_slide_1)\
	beta: $(choice_slide)
	"
end

# ╔═╡ eb3d7ef3-febb-4fc3-a1e5-c40f8a208e29
begin

	price_1_dist = Normal(2,1)
	price_0_dist = Normal(0,1)
	error_dist = Gumbel(0,1)

	p_1 = rand(rng, price_1_dist, 10000)
	p_0 = rand(rng, price_0_dist, 10000)
	e_1 = rand(rng, error_dist, 10000)
	e_0 = rand(rng, error_dist, 10000)

	#histogram(p_1 .- p_0)

	

	u_1 = alpha_1 .+ beta_choice .* p_1 .+ e_1
	u_0 = alpha_0 .+ beta_choice .* p_0 .+ e_0
	diff = u_1 .- u_0

	#histogram(diff)

	choice = zeros(10000)
	choice[diff .> 0] .= 1

	scatter(p_1 .- p_0, choice, label="Data")
	xlabel!("Price difference")

	p_diff = p_1 .- p_0
	param_est = (p_diff' * p_diff)^-1 * p_diff' * choice

	plot_mat = [p_diff choice]
	sort!(plot_mat, dims=1, by = x -> x[1])	
	

	plot!(plot_mat[:,1], param_est .* plot_mat[:,1], label="LPM")
	#Print(param_est)

end

# ╔═╡ 263765d0-5034-4b89-85dc-bb668dfaf540
md" Now lets look what happens when we describe our data using a logit model."

# ╔═╡ 2c0cca95-4d32-4810-89d3-e489823ccfa6
begin

	obj_logit(beta, y, X) = - mean(y .* ( 1 ./(1 .+ exp.(-1 .*( X[:,1] .* beta[1] .+ X[:,2] .* beta[2])))) .+ (1 .- y) .* (1 .- ( 1 ./(1 .+ exp.(-1 .*( X[:,1] .* beta[1] .+ X[:,2] .* beta[2]))))))

	g(x::Vector) = obj_logit(x, plot_mat[:,2],[ones(10000) plot_mat[:,1]]) # fixing objective function

	beta_0 = [0.1,  1.0]

    sol_opt = Optim.optimize(g, beta_0, GradientDescent()) 
	Print(sol_opt.minimizer)


end

# ╔═╡ f52890b5-0e7c-4bed-966e-f41532048583
md"PLOT LOGIT WHEN CORRECT"

# ╔═╡ be1572cc-421d-4d04-92d5-35e62797085c
md" ## Appendix: Fundamental concepts in econometrics"

# ╔═╡ 9628f826-8ed9-4148-bd47-26420c6134b6
md"
The goal of econometrics is to measure things relevant to economic activity or theory. To be more specific, assume we have some true set of variables $\mathbf{Y}, \mathbf{X}$ (**the population**). We are interested the general question of how $\mathbf{X}$ affects $\mathbf{Y}$. 



To answer this question, we must do several things. First, we must assume that there is some **true data generating process** 
```math
y = f(x, \theta)
```

that governs how each element of  $\mathbf{X}$ 'actually' affects each element of $\mathbf{Y}$. Otherwise, our population $\mathbf{Y}$ and $\mathbf{X}$ are not linked at all. Typically, the $\theta$ from this data generating process is what we want to measure. Second, we need to observe some **sample** of our variables $\{y_{i}\}_{i=1}^{N} \subset \mathbf{Y}, \{x_{i}\}_{i=1}^{N} \subset \mathbf{X}$. Once we have both of these things, we need to construct an **estimate** of $\theta$, $\hat{\theta}$ from our sample -- essentially our best guess at what the 'true' $\theta$ is. $\theta$ is called our **estimand** - the thing we are trying to measure.} To do this, we need an **estimator** -- a map from our sample to possible values of our parameters. Often, we do this by assuming that the 'true' data generating process has a certain functional form $y_{i} = g(x_{i}, \theta)$ $\forall i \in \{1, ..., N\}$, and then rearranging this functional form to get our estimate i.e $\hat{\theta} = g^{-1}(\{y_{i}\}_{i=1}^{N}, \{x_{i}\}_{i=1}^{N})$.

But now we run into the **problem of induction** -- how can we say anything about the population quantity $\theta$ -- what we really care about -- from our sample estimate $\hat{\theta}$? Our sample might be weird in some way. Econometricians typically try to solve this problem by **falsification**. The idea of falsification originates with the philosopher Karl Popper. To illustrate it, lets consider an example. Imagine we want to test whether all swans are white. We could construct a huge sample of swans, and see whether each one is white. But we cannot say for sure based on this sample whether all swans are white. The next swan outside of our sample might be black. We can instead try to falsify the claim. If we observe a swan that is not white, we can say that the claim that all swans are white is false for sure, even though we do not observe all the swans. 

Economists treat hypotheses about economic relations the same way. First, we construct a 'null hypothesis' for the value of $\theta$, $\theta = \theta_{0}$. Then, we build a **pivotal quantity**. A pivotal quantity is a function of observations and the unobservable parameters whose asymptotic distribution (distribution as the number of observations gets 'large enough') under the null hypothesis does not depend on the value of the parameter. Then, we compute the observed value based on our parameter and sample. If the value is 'unlikely enough' given the distribution, we say that our null hypothesis is 'falsified'. The standard example of this is a t-statistic, which is asymptotically normal."

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.79"
LaTeXStrings = "~1.3.0"
Optim = "~1.7.4"
Plots = "~1.38.0"
PlutoUI = "~0.7.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "14c3f84a763848906ac681f94cf469a851601d92"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.28"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

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
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "c5b6685d53f933c11404a3ae9822afe30d522494"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.2"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "a7756d098cbabec6b3ac44f369f74915e8cfd70a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.79"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

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

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "9a0472ec2f5409db243160a8b030f94c380167a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.6"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "04ed1f0029b6b3af88343e439b995141cb0d0b8d"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.17.0"

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
git-tree-sha1 = "a69dd6db8a809f78846ff259298678f0d6212180"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.34"

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
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "bcc737c4c3afc86f3bbc55eb1b9fabcee4ff2d81"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "64ef06fa8f814ff0d09ac31454f784c488e22b29"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.2+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

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
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "fd9861adba6b9ae4b42582032d0936d456c8602d"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.6.3"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

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

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

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

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "1903afc76b7d01719d9c30d3c7d501b61db96721"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.4"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6466e524967496866901a78fca3f2e9ea445a559"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.2"

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

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "5b7690dd212e026bbab1860016a6601cb077ab66"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.2"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "513084afca53c9af3491c94224997768b9af37e8"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "18c35ed630d7229c5584b945641a73ca83fb5213"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.2"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

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

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "6954a456979f23d05085727adb17c4551c19ecd1"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.12"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

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
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "e4bdc63f5c6d62e80eb1c0043fcc0360d5950ff7"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.10"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

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
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

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

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

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

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

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
# ╟─561b4e30-2916-47c5-ae90-1c2396343366
# ╠═ff95d6a8-e56b-48c9-9d64-61a134e6effc
# ╠═d4dd4e40-8130-11ed-1d7f-1f21a70ab743
# ╟─66765dfc-7cc3-43af-a282-08dcc08e15a9
# ╟─98e0a048-1a7e-4242-b2a5-982d273d6e87
# ╟─a1595bed-ecf7-4042-bff6-b63fd4fd351f
# ╟─dcb102b5-1cef-48e4-a3b6-53f70a33ecf1
# ╟─7e4e549c-ce8d-47c6-ab82-bf0d570dcacb
# ╟─c824a69f-637a-4b61-acee-e91410ac4f63
# ╟─46ec542c-de39-42cd-929f-dccccc2c462d
# ╟─e5a98d77-a512-4c58-8ca4-4dc384971df2
# ╟─73ee62a1-de4e-40b8-8d62-56bb07e292e8
# ╟─eae896fe-c1c8-4ae2-a896-0641f472d6be
# ╟─a54e2053-fb8f-4c20-acfb-8043aaa3c416
# ╟─69ac43a3-54dc-412b-9595-64abeb80c567
# ╟─a9f50d52-fda7-4bb3-9104-9901eb75b257
# ╟─3b853bc9-e9a5-4b59-ba83-976a3ddd4fd1
# ╟─19f8ca7e-97b1-44e2-87d4-a269d2597675
# ╟─e34fe2b8-fe93-470c-bb40-23777f6d761a
# ╟─64309bbb-3c2e-4e1a-b1de-5b7626cd9b31
# ╟─fab276f4-4067-4c2b-916f-98d5278ffd9d
# ╟─94d0747f-be6f-4055-8def-1914a02a95c5
# ╟─e8612d42-aa11-43fa-ae67-24fca5c21a11
# ╟─8364256f-0a18-4dc1-997c-142a8fc93f34
# ╟─803df173-04c2-491a-8450-361ea126be8e
# ╟─2f0d6238-7e3b-4669-9d7e-72a2994456dc
# ╟─bf4a7cbb-3778-4557-94ec-9fdf382a330d
# ╟─7d1a97f7-d1b0-43b4-8f93-68d89e9b0ce7
# ╟─8f8a6c1d-4fc0-4d76-a9d7-d00782941a49
# ╟─87b02c57-7f4b-413f-a0d3-018f004ae6ad
# ╟─ac263e35-99d6-4491-8bdb-04fd4c94a460
# ╟─37d475f6-9bd1-4a7b-bd5d-5b9a189fdefd
# ╟─35b51e6a-f4e2-4ae1-ba2b-16f2734f556a
# ╟─e4ccd357-535f-4d2a-a832-ab02202f66bc
# ╟─70404453-6444-4ba4-8b8e-f7cb43bd6b70
# ╟─b16801b2-0e2b-44ba-922e-3653a0baef70
# ╟─6f261bf9-dee9-4a0a-91b7-e6b1adf484fd
# ╟─d609e18a-6508-4867-86ab-b45691d92f41
# ╟─57a4e042-8aac-4f5c-a668-8a8060403b51
# ╟─03c57489-3b0e-436d-8e8b-24393c640a2a
# ╟─393f8fa0-a9f5-4dc6-b54f-f460b709ddb1
# ╟─9518b760-2112-437f-bdc7-eee26cb54d8c
# ╟─75a0e831-e61a-4637-ba78-c6b32fe2db96
# ╟─c7d7aeb1-ab48-4985-a7ae-efd6385f5db2
# ╟─cc022b3e-4bd5-4ac9-91fc-c833625fc6d4
# ╟─0fafd79e-fd1f-4639-b9d2-f74ded42b54b
# ╟─20f67f15-ec5b-4f31-ba26-553ff06d56be
# ╟─2cdde6b5-05c9-4511-bbbd-054fd0bb2d9f
# ╟─330a3993-e057-45b2-9836-711032a86db2
# ╟─4e5f42e6-d1d0-4233-8f41-a339db0c0be4
# ╟─235dac45-c8a0-4f32-aa11-f9f6d5b629c7
# ╟─afabdddb-c728-4910-a45f-ec1a055f3238
# ╟─77988813-3b5c-4757-bbae-d79a37f6623e
# ╟─a72e5d33-0a1f-4a79-b855-79197f4613cb
# ╟─57fa9569-9089-4ef4-bc2b-ecbd8c964a44
# ╟─ba4ba6a6-d473-4a71-9c91-312b2e75bfa5
# ╟─2b5ad567-1ce2-4716-a8e8-b22c1f559bf2
# ╟─37cba339-e070-4e23-ac0b-ac2eeff51279
# ╟─971a83cc-0348-48ce-97c8-23620440e942
# ╟─62181f51-71eb-4fe7-94f4-1618b00c362a
# ╟─e1225785-a96f-4668-b142-e17edc0f20bf
# ╟─9a46c416-6fc1-4bca-a347-f7ff68742ca4
# ╟─2feaea6a-baa5-4742-ac64-9080caee12b8
# ╠═3ab2dc19-22f6-430e-b7e8-09d83deec04d
# ╟─348e8426-f74a-43b5-bbb7-384133455a7d
# ╟─45b5b2a9-104a-480b-963b-e1b21542b84f
# ╟─6fb82d2d-8951-485e-9fd6-1a5867949358
# ╟─316cb637-6f83-4f2f-a354-d75ff59040dd
# ╟─55a75f7e-528c-4aad-a5ab-32ed9d264793
# ╟─c71a354c-fc48-40a1-836d-4324791fe499
# ╟─7e57a34a-880f-4a72-b693-1e7692701dc8
# ╟─54175a37-1c01-4c9d-a25a-93505b5295ff
# ╟─eb3d7ef3-febb-4fc3-a1e5-c40f8a208e29
# ╟─263765d0-5034-4b89-85dc-bb668dfaf540
# ╟─2c0cca95-4d32-4810-89d3-e489823ccfa6
# ╟─f52890b5-0e7c-4bed-966e-f41532048583
# ╠═be1572cc-421d-4d04-92d5-35e62797085c
# ╟─9628f826-8ed9-4148-bd47-26420c6134b6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
