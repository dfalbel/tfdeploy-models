
reticulate::use_virtualenv("tf-1.12.0")

# Packages ----------------------------------------------------------------

library(tensorflow)

sess <- tf$Session()

mnist <- readRDS("data/mnist.rds")
next_batch <- function() {
  ids <- sample.int(nrow(mnist$train$x), size = 32)
  list(
    x = mnist$train$x[ids,],
    y = mnist$train$y[ids,]
  )
}


x <- tf$placeholder(tf$float32, shape(NULL, 784L))

W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

y <- tf$nn$softmax(tf$matmul(x, W) + b)

y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))

optimizer <- tf$train$GradientDescentOptimizer(0.05)
train_step <- optimizer$minimize(cross_entropy)

init <- tf$global_variables_initializer()

sess$run(init)

for (i in 1:1000) {
  batches <- next_batch()
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

sess$run(accuracy, feed_dict=dict(x = mnist$train$x, y_ = mnist$train$y))

export_savedmodel(
  sess,
  "models/tensorflow-1.12.0/",
  inputs = list(images = x),
  outputs = list(scores = y))




