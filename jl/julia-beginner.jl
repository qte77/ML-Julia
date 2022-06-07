#https://www.math.purdue.edu/~allen450/ML-with-Julia-Tutorial.html
#https://github.com/tonygallen/JPUG

using Pkg
Pkg.add("Flux")

using Flux
#using Flux.Tracker

f(x)=x^2+1
#get tupel of differential of f at x
df(x)=gradient(f, x)[1] #, nest=true)
ddf(x)=gradient(df, x)[1] #, nest=true)

x=2
print("x=$(x), df: $(df(x)), ddf: $(ddf(x))")

g(x,y,z)=x^2+y^2+z^2
x,y,z=(1,2,3)
gradient(g,x,y,z)

#Linear Regression with Params()
W=rand(5,10) #Matrix m rows * n cols
b=rand(5)
f_hat(x)=W*x+b

function loss(x,y)
    y_hat=f_hat(x) #y pred
    return sum((y-y_hat).^2)
end

x,y=(rand(10),rand(5))
loss(x,y)

Params(W)
