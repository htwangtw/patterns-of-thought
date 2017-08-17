library(car)
library(heplots)
library(lsr)

df <- read.csv('../data/processed/NYCQ_CCA_score_4_0.7_0.3.csv', header = TRUE, sep = ',')
# set the target
Y <- data.matrix(df[,5:12])
X <-data.matrix(df[,12:ncol(df)])
colnames(Y) <- colnames(df[,5:12])
# reorder
Y <- Y[, c(1, 4, 5, 2, 3, 6, 7, 8)]
# reverse cc1 for intepretation
df$CC_01 <- -df$CC_01

# DV-intellegence EV-Yeo 7 MANOVA
Yeo7_m1 <- lm(Y ~ Age + Motion_Jenkinson + CC_01 + 
                CC_02 + CC_03 + CC_04 , data = df)
# get manova eta square
mod.manova <-  Manova(Yeo7_m1, type = 3, test = "Pillai")
etasq(mod.manova, anova = TRUE)
mod.para <- summary(Yeo7_m1, test = "Pillai")

# univariate results and parameter estimate
for(i in c(1:length(colnames(Y)))){
  print(colnames(Y)[i])
  l <- lm(formula = paste(colnames(Y)[i], " ~ Age + Motion_Jenkinson + CC_01 +
            CC_02 + CC_03 + CC_04") , data = df)
  # print(etasq(l, type = 3, anova = TRUE))
  # print(summary(l, test = "Pillai"))
  print(confint(l))
  # cat(summary(l, test = "Pillai")$coefficients[,1],
  #     file="../reports/ver1/paraest.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/ver1/paraest.txt",sep="\t", append = TRUE)
  # cat(summary(l, test = "Pillai")$coefficients[,4],
  #     file="../reports/ver1/paraest_p.txt",sep="\t", append = TRUE)
  # cat("\n",
  #     file="../reports/ver1/paraest_p.txt",sep="\t", append = TRUE)
}

###########################################

require(ggplot2)
scatter_ggplot <- function(data, X, Y, label_x, label_y)
{
  p2<- (
    # scatter plot main data
    # the data need to be in dataframe format. x and y can refer directly to the variables
    ggplot(data, aes(x=X, y=Y)) 
    # dot size, color and shape etc.
    + geom_point(shape=19, size = 1, colour="grey50") 
    # 95% confidence interval and regression line
    + geom_smooth(method=lm, colour="black", size = 1, fullrange = TRUE)
    # title and axis lable
    # + ggtitle("Blob")
    + xlab(label_x)
    + ylab(label_y)
    # limits of the axises
    + xlim(-4, 4) 
    + ylim(-2.5, 2.5)
    
  ) 
  
  # add APA format theme
  p2 + theme_classic() +theme(
    text = element_text(size=18),
    axis.line.x = element_line(colour = "black"),
    axis.line.y = element_line(colour = "black")
  )
}

####################### plots of raw scores ################################

fit_XFAC1 <- lm(CC_01 ~ Age + Motion_Jenkinson + CC_02 + CC_03 + CC_04, data = df)
fit_XFAC2 <- lm(CC_02 ~ Age + Motion_Jenkinson + CC_01 + CC_03 + CC_04, data = df)
fit_XFAC4 <- lm(CC_04 ~ Age + Motion_Jenkinson + CC_01 + CC_02 + CC_03, data = df)

fit_Y_DF_1  <- lm(DF_29 ~ Age + Motion_Jenkinson + CC_02 + CC_03 + CC_04, data = df)

fit_Y_CWI_1 <- lm(DKEFSCWI_40 ~ Age + Motion_Jenkinson + CC_02 + CC_03 + CC_04, data = df)
fit_Y_CWI_2 <- lm(DKEFSCWI_40 ~ Age + Motion_Jenkinson + CC_01 + CC_03 + CC_04, data = df)
fit_Y_CWI_4 <- lm(DKEFSCWI_40 ~ Age + Motion_Jenkinson + CC_01 + CC_02 + CC_03, data = df)

fit_Y_PROV_1 <- lm(PROV_16 ~ Age + Motion_Jenkinson + CC_02 + CC_03 + CC_04, data = df)
fit_Y_PROV_4 <- lm(PROV_16 ~ Age + Motion_Jenkinson + CC_01 + CC_02 + CC_03, data = df)

fit_Y_TMT_1 <- lm(DKEFSTMT_48 ~ Age + Motion_Jenkinson + CC_02 + CC_03 + CC_04, data = df)

fit_Y_VF_1 <- lm(VF_37 ~ Age + Motion_Jenkinson + CC_02 + CC_03 + CC_04, data = df)

fit_Y_WAIS_1 <- lm(INT_17 ~ Age + Motion_Jenkinson + CC_02 + CC_03 + CC_04, data = df)
fit_Y_WAIS_2 <- lm(INT_17 ~ Age + Motion_Jenkinson + CC_01 + CC_03 + CC_04, data = df)
fit_Y_WAIS_4 <- lm(INT_17 ~ Age + Motion_Jenkinson + CC_01 + CC_02 + CC_03, data = df)

fit_Y_WIAT_2 <- lm(WIAT_08 ~ Age + Motion_Jenkinson + CC_01 + CC_03 + CC_04, data = df)
fit_Y_WIAT_4 <- lm(WIAT_08 ~ Age + Motion_Jenkinson + CC_01 + CC_02 + CC_03, data = df)

df_resid<-data.frame(cbind(rstandard(fit_XFAC1), rstandard(fit_Y_DF_1)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 01', 'Spatial construction')
ggsave('./reports/figures/bestmodel_SCCA01_DF_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC1), rstandard(fit_Y_TMT_1)))
p<-scatter_ggplot(df_resid, df_resid$X1, -df_resid$X2, 'SCCA - 01', 'Number-letter switching')
ggsave('./reports/figures/bestmodel_SCCA01_TMT_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC1), rstandard(fit_Y_VF_1)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 01', 'Letter - category fluency')
ggsave('./reports/figures/bestmodel_SCCA01_VF_resid.png', width = 3.5, height = 3.3, dpi = 300)

df_resid<-data.frame(cbind(rstandard(fit_XFAC1), rstandard(fit_Y_CWI_1)))
p<-scatter_ggplot(df_resid, df_resid$X1, -df_resid$X2, 'SCCA - 01', 'Inhibition')
ggsave('./reports/figures/bestmodel_SCCA01_CWI_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC2), rstandard(fit_Y_CWI_2)))
p<-scatter_ggplot(df_resid, df_resid$X1, -df_resid$X2, 'SCCA - 02', 'Inhibition')
ggsave('./reports/figures/bestmodel_SCCA02_CWI_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC4), rstandard(fit_Y_CWI_4)))
p<-scatter_ggplot(df_resid, df_resid$X1, -df_resid$X2, 'SCCA - 04', 'Inhibition')
ggsave('./reports/figures/bestmodel_SCCA04_CWI_resid.png', width = 3.5, height = 3.3, dpi = 300)

df_resid<-data.frame(cbind(rstandard(fit_XFAC1), rstandard(fit_Y_PROV_1)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 01', 'Novel verbal abstractions')
ggsave('./reports/figures/bestmodel_SCCA01_PROV_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC4), rstandard(fit_Y_PROV_4)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 04', 'Novel verbal abstractions')
ggsave('./reports/figures/bestmodel_SCCA04_PROV_resid.png', width = 3.5, height = 3.3, dpi = 300)

df_resid<-data.frame(cbind(rstandard(fit_XFAC1), rstandard(fit_Y_WAIS_1)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 01', 'WAIS')
ggsave('./reports/figures/bestmodel_SCCA01_WAIS_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC2), rstandard(fit_Y_WAIS_2)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 02', 'WAIS')
ggsave('./reports/figures/bestmodel_SCCA02_WAIS_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC4), rstandard(fit_Y_WAIS_4)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 04', 'WAIS')
ggsave('./reports/figures/bestmodel_SCCA04_WAIS_resid.png', width = 3.5, height = 3.3, dpi = 300)

df_resid<-data.frame(cbind(rstandard(fit_XFAC2), rstandard(fit_Y_WIAT_2)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 02', 'WIAT')
ggsave('./reports/figures/bestmodel_SCCA02_WIAT_resid.png', width = 3.5, height = 3.3, dpi = 300)
df_resid<-data.frame(cbind(rstandard(fit_XFAC4), rstandard(fit_Y_WIAT_4)))
p<-scatter_ggplot(df_resid, df_resid$X1, df_resid$X2, 'SCCA - 04', 'WIAT')
ggsave('./reports/figures/bestmodel_SCCA04_WIAT_resid.png', width = 3.5, height = 3.3, dpi = 300)