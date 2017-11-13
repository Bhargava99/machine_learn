# machine_learn
DEVELOP A CLASSIFICATION MODEL THAT WILL IDENTIFY THE COMPENSATION LEVEL OF AN INDIVIDUAL
comp<-read.csv(file = "D:/ML/Hackathon/Assignment - 3 update/Model_Data.csv",header = TRUE)


# data exploration 
str(comp)

summary(comp)

# splitting the file into train and test data sets.

set.seed(50)
comp_sp=sample.int(n=nrow(comp),size = floor(0.8*nrow(comp)),replace = F)

comp_train=comp[comp_sp,]
write.csv(comp_train,file="D:/ML/Hackathon/Assignment - 3 update/comp_train.csv")
comp_test=comp[-comp_sp,]
write.csv(comp_test,file="D:/ML/Hackathon/Assignment - 3 update/comp_test.csv")



#converting '?' into NA's

library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('rlang') # data manipulation

comp_train1<-as.tibble(fread("D:/ML/Hackathon/Assignment - 3 update/comp_train.csv",na.strings = c("?")))
summary(comp_train1)
str(comp_train1)

comp_train1$workclass<-factor(comp_train1$workclass)
comp_train1$education<-factor(comp_train1$education)
comp_train1$education.num<-factor(comp_train1$education.num)
comp_train1$marital.status<-factor(comp_train1$marital.status)
comp_train1$occupation<-factor(comp_train1$occupation)
comp_train1$relationship<-factor(comp_train1$relationship)
comp_train1$race<-factor(comp_train1$race)
comp_train1$sex<-factor(comp_train1$sex)
comp_train1$native.country<-factor(comp_train1$native.country)
comp_train1$Ethni<-factor(comp_train1$Ethni)
comp_train1$Compen<-factor(comp_train1$Compen)

# data imputation for 'NA' values

Mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

for (var in 1:ncol(comp_train1)) {
  if (class(comp_train1[,var])=="numeric") {
    comp_train1[is.na(comp_train1[,var]),var] <- mean(comp_train1[,var], na.rm = TRUE)
  } else if (class(comp_train1[,var]) %in% c("character", "factor")) {
    comp_train1[is.na(comp_train1[,var]),var] <- Mode(comp_train1[,var], na.rm = TRUE)
  }
}


# data imputation for'NA' values in test data
comp_test1<-as.tibble(fread("D:/ML/Hackathon/Assignment - 3 update/comp_test.csv",na.strings = c("?")))
summary(comp_test1)
str(comp_test1)

comp_test1$workclass<-factor(comp_test1$workclass)
comp_test1$education<-factor(comp_test1$education)
comp_test1$education.num<-factor(comp_test1$education.num)
comp_test1$marital.status<-factor(comp_test1$marital.status)
comp_test1$occupation<-factor(comp_test1$occupation)
comp_test1$relationship<-factor(comp_test1$relationship)
comp_test1$race<-factor(comp_test1$race)
comp_test1$sex<-factor(comp_test1$sex)
comp_test1$native.country<-factor(comp_test1$native.country)
comp_test1$Compen<-factor(comp_test1$Compen)
comp_test1$Ethni<-factor(comp_test1$Ethni)

str(comp_test1)

# data imputation for test data
for (var in 1:ncol(comp_test1)) {
  if (class(comp_test1[,var])=="numeric") {
    comp_test1[is.na(comp_test1[,var]),var] <- mean(comp_test1[,var], na.rm = TRUE)
  } else if (class(comp_test1[,var]) %in% c("character", "factor")) {
    comp_test1[is.na(comp_test1[,var]),var] <- Mode(comp_test1[,var], na.rm = TRUE)
  }
}


str(comp_test1)


# Fitting decision tree model
 # Model1- decision tree

library('randomForest') 
library('rpart')
library('rpart.plot')
library('car')
library('e1071')
library('tree')

model_comp_1=tree(Compen~workclass+education+marital.status+occupation+relationship+race+sex+Ethni,data = comp_train1)
model_pred_comp_1=predict(model_comp_1,comp_test1)

maxidx=function(arr) {
  return(which(arr==max(arr)))
}

idx=apply(model_pred_comp_1,c(1),maxidx)

modelprediction_tit2 = c('<=50K','>50K')[idx]

confmat_tit1=table(modelprediction_tit2,comp_test1$Compen) #confusion matrix

accuracy_tit2=sum(diag(confmat_tit1))/sum(confmat_tit1)

# Model 2 Naive bayes
nai_model_comp=naiveBayes(Compen~workclass+education+marital.status+occupation+relationship+race+sex+Ethni,data = comp_train1)
pred_nai=predict(nai_model_comp,comp_test1[,-16])

confmat_nai=table(pred_nai,comp_test1$Compen)

accu=sum(diag(confmat_nai)/sum(confmat_nai))


# Model 2 KNN
# transofrming all var into numeric
comp_test
comp_train

for(i in 1:ncol(comp_test)){
  comp_test[,i]=as.numeric(comp_test[,i])
}

for(i in 1:ncol(comp_train)){
  comp_train[,i]=as.numeric(comp_train[,i])
}
knn_traindata=comp_train[,c(1:3,5:14)]
knn_testdata=comp_test[,c(1:3,5:14)]
knn_trainlabel=comp_train[,15]
knn_testlabel=comp_test[,15]

library(class)
k=13
knn_pred_label=knn(train=knn_traindata,test=knn_testdata,cl=knn_trainlabel,k)
confmat=table(knn_testlabel,knn_pred_label)
confmat
accuracy=sum(diag(confmat))/sum(confmat)
accuracy

