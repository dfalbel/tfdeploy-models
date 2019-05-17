# Serve a model with tf.1.12.0 and test prediction:

models <- c("models/tensorflow-1.12.0/", "models/tensorflow-1.13.1/")

for (model in models) {
  p <- processx::process$new(
    "Rscript", 
    c(
      "-e", 
      glue::glue("reticulate::use_virtualenv('tf-1.12.0')
                tfdeploy::serve_savedmodel('{model}')")
    ),
    stdout = "|", stderr = "|"
  )
  
  
  Sys.sleep(20)
  instances <- list(instances = list(images = rep(0, 784)))
  
  cont <- httr::POST(
    url = "http://127.0.0.1:8089/serving_default/predict/",
    body = instances,
    httr::content_type_json(),
    encode = "json"
  )
  
  pred <- unlist(httr::content(cont))
  print(pred)
  stopifnot(is.numeric(pred))
  
  swg <- httr::GET("http://127.0.0.1:8089/swagger.json")
  if (swg$status_code == 404)
    stop("Swagger not working.")
  
  
  p$kill()
  
  
  
  while(p$is_alive()) Sys.sleep(1)
}

