require(pacman)

p_load(fastAdaboost,tidyverse,rpart,rpart.plot,adabag,patchwork)

# pull in Pima diabetes data
diabetes <- read.csv("~/code_adaboost/diabetes.csv")
diabetes$Outcome <- diabetes$Outcome %>% as.factor()

set.seed(666)

# split train/test
t1 <- sample(nrow(diabetes),.7*nrow(diabetes),replace=F)
train <- diabetes[t1,]
test <- diabetes[-t1,]


#### going to use rpart to create the learners

# number of trees
niter <- 100

# initialize vector for tracking tree weights and df for tracking votes on test df
tree_wts <- rep(NA,niter)
votes <- matrix(nrow=nrow(test),ncol=niter) %>% as.data.frame()
names(votes) <- paste0("m",1:niter)

# start with equal observation weights
weights_init <- rep(1/nrow(train),nrow(train))

weights0 <- weights_init

for (i in 1:niter) {
  
  stump <- rpart(Outcome ~ .,data=train,weights=weights0
                # ,control=rpart.control(maxdepth = 1)
                 )
  pred <- predict(stump)[,2] %>% round()

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

ensemblepreds <- (as.matrix(votes) %*% as.matrix(tree_wts))/sum(tree_wts)

# but did i really code up adaboost or did i code up a different boosted ensemble?

####################################################
#### use off-the-shelf packages for comparison  ####
####################################################

# fastAdaboost (lives up to its name)
shelf <- adaboost(Outcome ~ .,data=train,nIter=niter)
shelfpreds <- predict(shelf,test)$prob[,2]

# adabag (somehow slower than my for-loop?)
shelf2 <- boosting(Outcome ~ ., data=train,boos=F,mfinal=niter)
shelfpreds2 <- predict(shelf2,test)$prob[,2]

# want to see my adaboost about as correlated with the canned algorithms as they are with each other
# and that looks solid:

cor(cbind(ensemblepreds,shelfpreds,shelfpreds2)) # class probability
cor(round(cbind(ensemblepreds,shelfpreds,shelfpreds2))) # classification


# next, plot the pairwise combinations of predictions
# plots don't look too bad, at a first pass
# (red compares canned algos to each other)

ggplot() +
  geom_point(aes(x=ensemblepreds,y=shelfpreds),color="purple") +
  geom_abline(slope=1,intercept=0) +
  
ggplot() +
  geom_point(aes(x=ensemblepreds,y=shelfpreds2),color="blue") +
  geom_abline(slope=1,intercept=0) +
  
ggplot() +
  geom_point(aes(x=shelfpreds,y=shelfpreds2),color="red") +
  geom_abline(slope=1,intercept=0)



# but we want the points clustered along the 45-degree line.
# checking univariate ols, we can see that they are NOT
# (want to see slope = 1, intercept= 0)

summary(lm(shelfpreds ~ ensemblepreds))
summary(lm(shelfpreds2 ~ ensemblepreds))

# the two canned algorithms do have that relationship, however.
summary(lm(shelfpreds ~ shelfpreds2))
# I might have an error somewhere, but this is pretty close


y <- as.numeric(as.character(test$Outcome))

# similar performance:
brier_mine <- mean((y - ensemblepreds)^2)
brier_fast <- mean((y - shelfpreds)^2)
brier_adabag <- mean((y- shelfpreds2)^2)

misclass_mine <- mean(y!=round(ensemblepreds))
misclass_fast <- mean(y!=round(shelfpreds))
misclass_adabag <- mean(y!=round(shelfpreds2))

####### compare to logit and rf as a sanity check ########

summary(logitm <- glm(Outcome ~ .,data=train,family="binomial"))
rfm <- randomForest(Outcome ~ .,data=train)

predlogit <- predict(logitm,test) %>% plogis()
predrf <- predict(rfm,test,type="prob")[,2]


# logit and rf outperform all three adaboosts, but not heroically
brier_logit <- mean((as.numeric(as.character(test$Outcome)) - predlogit)^2)
misclass_logit <- mean(y!=round(predlogit))
brier_rf <- mean((as.numeric(as.character(test$Outcome)) - predrf)^2)
misclass_rf <- mean(y!=round(predrf))

# logit and rf substantially different from adaboosts
cor(cbind(ensemblepreds,shelfpreds,shelfpreds2,predlogit,predrf)) # class probability
cor(round(cbind(ensemblepreds,shelfpreds,shelfpreds2,predlogit,predrf))) # classification
