library(car)
library(heplots)
library(lsr)
library(MASS)

df <- read.csv('../data/processed/NYCQ_CCA_score_revision_yeo7nodes_4_0.8_0.5.csv', header = TRUE, sep = ',')

# set the target
Y <- data.matrix(df[,5:12])
X <-data.matrix(df[,21:ncol(df)])
Y_q <- data.matrix(df[,13:20])

colnames(Y) <- colnames(df[,5:12])
colnames(Y_q) <- colnames(df[,13:20])

# reorder
Y <- Y[, c(1, 4, 5, 2, 3, 6, 7, 8)]
# DV-intellegence EV-Yeo 7 MANOVA
Yeo7_m1 <- lm(Y ~ CC_01 + CC_02 + CC_03 + CC_04 , data = df)
Yeo7_m2 <- lm(Y_q ~ CC_01 + CC_02 + CC_03 + CC_04 , data = df)

# get manova eta square
mod.manova <-  Manova(Yeo7_m1, type = 3, test = "Pillai", p.adjust.methods = "bonferroni")
print(round(etasq(mod.manova, anova = TRUE), 3))

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

  # Export the beta and p value for plotting
  cat(paraest$coefficients[,4],
      file="../reports/revision/Yeo7node_paraest_beta.txt",sep="\t", append = TRUE)
  cat("\n",
      file="../reports/revision/Yeo7node_paraest_beta.txt",sep="\t", append = TRUE)
  cat(paraest$coefficients[,7],
      file="../reports/revision/Yeo7node_paraest_p.txt",sep="\t", append = TRUE)
  cat("\n",
      file="../reports/revision/Yeo7node_paraest_p.txt",sep="\t", append = TRUE)

  # cat(paraest$coefficients[,2:3],
  #     file="../reports/revision/Yeo7_paraest_CI_extensiveclean.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_CI_extensiveclean.txt",sep="\t", append = TRUE)
  # 
  # cat(paraest$coefficients[,6],
  #     file="../reports/revision/Yeo7_paraest_t_extensiveclean.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_t_extensiveclean.txt",sep="\t", append = TRUE)
  # 
  # cat(paraest$coefficients[,5],
  #     file="../reports/revision/Yeo7_paraest_err_extensiveclean.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_err_extensiveclean.txt",sep="\t", append = TRUE)
  paraest$coefficients <- round(paraest$coefficients, digits = 3)
  print(paraest)
}



# questionnaire
mod.manova_q <- Manova(Yeo7_m2, type = 3, test = "Pillai", p.adjust.methods = "bonferroni")
print(round(etasq(mod.manova_q, anova = TRUE), 3))

# univariate results and parameter estimate
for(i in c(1:length(colnames(Y_q)))){
  print(colnames(Y_q)[i])
  l <- lm(formula = paste(colnames(Y_q)[i], " ~ CC_01 + CC_02 + CC_03 + CC_04") , data = df)
  # Univariate
  print(round(etasq(l, type = 3, anova = TRUE), 3))
  
  # Parameter estimate
  paraest <- summary(l, test = "Pillai")
  p.raw<-paraest$coefficients[,4]
  # attatch adjusted p and CI
  paraest$coefficients <- cbind(p.adjust(p.raw, "bonferroni"), confint(l), paraest$coefficients)
  colnames(paraest$coefficients)[1] <- "p.bonferroni"
  
  # Export the beta and p value for plotting
  cat(paraest$coefficients[,4],
      file="../reports/revision/Yeo7node_questionnaires_paraest_beta.txt",sep="\t", append = TRUE)
  cat("\n",
      file="../reports/revision/Yeo7node_questionnaires_paraest_beta.txt",sep="\t", append = TRUE)
  cat(paraest$coefficients[,7],
      file="../reports/revision/Yeo7node_questionnaires_paraest_p.txt",sep="\t", append = TRUE)
  cat("\n",
      file="../reports/revision/Yeo7node_questionnaires_paraest_p.txt",sep="\t", append = TRUE)
  
  # cat(paraest$coefficients[,2:3],
  #     file="../reports/revision/Yeo7_paraest_CI_extensiveclean.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_CI_extensiveclean.txt",sep="\t", append = TRUE)
  # 
  # cat(paraest$coefficients[,6],
  #     file="../reports/revision/Yeo7_paraest_t_extensiveclean.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_t_extensiveclean.txt",sep="\t", append = TRUE)
  # 
  # cat(paraest$coefficients[,5],
  #     file="../reports/revision/Yeo7_paraest_err_extensiveclean.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/revision/Yeo7_paraest_err_extensiveclean.txt",sep="\t", append = TRUE)
  paraest$coefficients <- round(paraest$coefficients, digits = 3)
  print(paraest)
}
