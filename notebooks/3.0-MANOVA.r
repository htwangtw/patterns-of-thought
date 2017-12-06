library(car)
library(heplots)
library(lsr)
library(MASS)

# df <- read.csv('../data/processed/NYCQ_CCA_score_revision_r2z_4_0.5_0.1.csv', header = TRUE, sep = ',')
df <- read.csv('../data/processed/NYCQ_CCA_score_revision_r2z-clean_4_0.9_0.2.csv', header = TRUE, sep = ',')
# df <- read.csv('../data/processed/NYCQ_CCA_score_revision_zscore-clean_4_0.9_0.3.csv', header = TRUE, sep = ',')
# df <- read.csv('../data/processed/NYCQ_CCA_score_revision_zscore_4_0.9_0.2.csv', header = TRUE, sep = ',')

# set the target
Y <- data.matrix(df[,5:12])
X <-data.matrix(df[,12:ncol(df)])
colnames(Y) <- colnames(df[,5:12])
# reorder
Y <- Y[, c(1, 4, 5, 2, 3, 6, 7, 8)]
# DV-intellegence EV-Yeo 7 MANOVA
Yeo7_m1 <- lm(Y ~ CC_01 + CC_02 + CC_03 + CC_04 , data = df)
# get manova eta square
mod.manova <-  Manova(Yeo7_m1, type = 3, test = "Pillai", p.adjust.methods = "bonferroni")
mod.manova
#print(round(etasq(mod.manova, anova = TRUE), 3))

# univariate results and parameter estimate
for(i in c(1:length(colnames(Y)))){
  print(colnames(Y)[i])
  l <- lm(formula = paste(colnames(Y)[i], " ~ CC_01 + CC_02 + CC_03 + CC_04") , data = df)
  # Univariate
  print(round(etasq(l, type = 3, anova = TRUE), 3))
  
  # Parameter estimate
  paraest <- summary(l, test = "Pillai")
  p.raw<-paraest$coefficients[,4]
  # attatch adjusted p and CI
  paraest$coefficients <- cbind(p.adjust(p.raw, "bonferroni"), confint(l), paraest$coefficients)
  colnames(paraest$coefficients)[1] <- "p.bonferroni"
  # 
  # # Export the beta and p value for plotting
  # cat(paraest$coefficients[,4],
  #     file="../reports/revision/Yeo7_paraest.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest.txt",sep="\t", append = TRUE)
  # cat(paraest$coefficients[,1],
  #     file="../reports/revision/Yeo7_paraest_p.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_p.txt",sep="\t", append = TRUE)
  # 
  # cat(paraest$coefficients[,2:3],
  #     file="../reports/revision/Yeo7_paraest_CI.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_CI.txt",sep="\t", append = TRUE)
  # 
  # cat(paraest$coefficients[,6],
  #     file="../reports/revision/Yeo7_paraest_t.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_t.txt",sep="\t", append = TRUE)
  # 
  # cat(paraest$coefficients[,5],
  #     file="../reports/revision/Yeo7_paraest_err.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_err.txt",sep="\t", append = TRUE)
  paraest$coefficients <- round(paraest$coefficients, digits = 3)
  print(paraest)
}