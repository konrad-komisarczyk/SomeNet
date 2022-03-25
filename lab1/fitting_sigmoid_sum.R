sigmoid <- function(x) (1 / (1 + exp(-x)))
squared <- function(x) (90*x^2 -130)
test <- function(x) {
  429 + 90*(10 * sigmoid(0.97*x-1) - 10 * sigmoid(0.97*x+1) + 6 * sigmoid(1.05*x-1/2) - 6 * sigmoid(1.05*x + 1/2)  + 1 * sigmoid(1.0*x-1/4) - 1 * sigmoid(1.0*x + 1/4)) 
}
curve(test, from = -2, to = 2, n = 1000)
curve(squared, from = -2, to = 2, add = T, col="red")

sig2 <- function(x) 4*sigmoid(sigmoid(x)) - 2
curve(sigmoid, from = -8, to = 8, n = 1000)
curve(sig2, from = -8, to = 8, n = 1000, add = T, col="red")

test2 <- function(x) {
  sigmoid((x+0.5)*1000)*80 + sigmoid((x-0.5)*1000)*80 + sigmoid((x-1.5)*1000)*80 - 80
}
curve(test2, from = -1.5, to = 2, n = 1000)

# 90x^2- 130
