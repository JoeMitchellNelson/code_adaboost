require(pacman)

p_load(tidyverse,rpart,adabag,JOUSBoost,OneR,tictoc)


# import my implementation
source("~/code_adaboost/adafunction.R")


### compare predictions from my implementation to off-the-shelf routines ###

# pull in Pima diabetes data
diabetes <- read.csv("~/code_adaboost/diabetes.csv")
diabetes$Outcome <- diabetes$Outcome %>% as.factor()

set.seed(666)

# split train/test
t1 <- sample(nrow(diabetes),.7*nrow(diabetes),replace=F)
train <- diabetes[t1,]
test <- diabetes[-t1,]


# set common hyperparameters
niter <- 100
depth <- 1
rcontrol <- rpart.control(cp = 0.01, 
                          minsplit = nrow(train),
                          minbucket= 0,
                          maxcompete = 0, maxsurrogate = 0, usesurrogate = 2, xval = 0,
                          surrogatestyle = 0, maxdepth = 1)




# mine
tic()
mypreds <- myadaboost(niter,depth,rcontrol,train,test)
toc()

# JOUSBoost
yJOUS <- ifelse(train$Outcome=="1",1,-1)
yJOUStest <- ifelse(test$Outcome=="1",1,-1)
tic()
shelf <- JOUSBoost::adaboost(y=yJOUS,X=as.matrix(train[,-9]),n_rounds=niter,tree_depth=1,control=rcontrol)
toc()
JOUSpreds <- predict(shelf,test)

# adabag
tic()
shelf2 <- boosting(Outcome ~ ., data=train,boos=F,mfinal=niter,
                   control=rcontrol)
toc()
adabagpreds <- predict(shelf2,test)$prob[,2] %>% round()

# want to see similar confusion matrices

a <- eval_model(JOUSpreds,yJOUStest)$conf_matrix
b <- eval_model(adabagpreds,test)$conf_matrix
c <- eval_model(mypreds,test)$conf_matrix

# side by side confusions
print(cbind(a,"    ",b,"    ",c))

# all three should have roughly similar (and high) pairwise correlations
cor(cbind(mypreds,JOUSpreds,adabagpreds))

#               mypreds JOUSpreds adabagpreds
# mypreds     1.0000000 0.9705337   0.8914174
# JOUSpreds   0.9705337 1.0000000   0.8805814
# adabagpreds 0.8914174 0.8805814   1.0000000

# my output is more highly correlated with both canned algorithms
# than they are with each other
# gonna call that a win

# safe to assume remaining differences are due to 
# differences in splitting function chosen for rpart
# or bootstrapping vs. adjusting sample weights
# (I use the latter)

