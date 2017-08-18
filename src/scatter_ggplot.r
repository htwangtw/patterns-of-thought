
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