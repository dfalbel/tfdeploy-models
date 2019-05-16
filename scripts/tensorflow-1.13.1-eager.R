
reticulate::use_virtualenv("tf-1.13.1")

# Packages ----------------------------------------------------------------

library(tensorflow)
tfe_enable_eager_execution()

mnist <- readRDS("data/mnist.rds")
next_batch <- function() {
  ids <- sample.int(nrow(mnist$train$x), size = 32)
  list(
    x = mnist$train$x[ids,],
    y = mnist$train$y[ids,]
  )
}


W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

optimizer <- tf$train$GradientDescentOptimizer(0.05)

for (i in 1:1000) {
  
  with(tf$GradientTape() %as% tape, {
    
    batches <- next_batch()
    batch_xs <- tf$constant(batches[[1]], dtype = "float")
    batch_ys <- tf$constant(batches[[2]], dtype = "float")
    
    y <- tf$nn$softmax(tf$matmul(tf$constant(batch_xs, dtype = "float"), W) + b)
    loss <- tf$reduce_mean(-tf$reduce_sum(batch_ys * tf$log(y), reduction_indices=1L))
    
  })
  
  grad <- tape$gradient(loss, list(W, b))
  
  optimizer$apply_gradients(
    list(
      list(grad[[1]], W),
      list(grad[[2]], b)
    )
  )
  
}

get_acc <- function(x, y) {
  
  x <- tf$constant(x, dtype = "float")
  y <- tf$constant(y, dtype = "float")
  
  pred <- tf$nn$softmax(tf$matmul(x, W) + b)
  correct_prediction <- tf$equal(tf$argmax(pred, 1L), tf$argmax(y, 1L))
  
  acc <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
  as.numeric(acc)
}

get_acc(mnist$test$x, mnist$test$y)

export_savedmodel(
  sess,
  "models/tf-1.13.1-eager/"
  )




