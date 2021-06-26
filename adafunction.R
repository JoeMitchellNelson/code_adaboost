myadaboost <- function (niter,depth,rcontrol,train,test) {
  
  # initialize vector for tracking tree weights and df for tracking votes on test df
  tree_wts <- rep(NA,niter)
  votes <- matrix(nrow=nrow(test),ncol=niter) %>% as.data.frame()
  names(votes) <- paste0("m",1:niter)
  
  # start with equal observation weights
  weights_init <- rep(1/nrow(train),nrow(train))
  
  weights0 <- weights_init
  
  for (i in 1:niter) {
    
    stump <- rpart(Outcome ~ .,data=train,weights=weights0
                   ,control=rcontrol
    )
    pred <- predict(stump,train)[,2] %>% round()
    
    # record this tree's votes on the test set
    votes[,i] <- predict(stump,test)[,2] %>% round()
    
    # vectors of correct/incorrect predictions (in training data) will be useful
    correct <- as.numeric(round(pred) == (as.numeric(as.character(train$Outcome))))
    incorrect <- as.numeric(round(pred) != (as.numeric(as.character(train$Outcome))))
    
    # calculate (weighted) total error, aka epsilon
    total_error <- (incorrect %*% weights0) %>% as.numeric()
    
    # adaboost formula for this tree's voting weight, aka alpha
    tree_wt <- 0.5 * log((1-total_error)/total_error)
    
    # record this tree's vote rate
    tree_wts[i] <- tree_wt
    
    # adaboost formula for next tree's weights
    weights_update <- ((weights0 * exp(tree_wt)) * incorrect) + ((weights0 * exp(-1*tree_wt)) * correct)
    
    # force obs weights to add to 1
    weights_update <- weights_update/sum(weights_update)
    
    # and we're ready for the next round
    weights0 <- weights_update
    
    
  }
  
  ensemblepreds <- ((as.matrix(votes) %*% as.matrix(tree_wts))/sum(tree_wts)) %>% round()
  out <- as.numeric(ensemblepreds)
  
  return(out)
  
}