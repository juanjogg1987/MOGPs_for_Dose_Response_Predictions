#load libraries
library(ggplot2)
library(RColorBrewer)
library(forcats)
library(dplyr)

#read file
KLrelevance_PLX = read.csv ("ANOVA_KL_relevance_PLX.csv", header=T, sep=',', stringsAsFactors = TRUE)

# Create the plot
KLrelevance_PLX %>% 
  mutate(feature_name=fct_rev(fct_reorder(feature_name, KL_relevance, mean)))%>%
  ggplot(aes(x=reorder(feature_name, KL_relevance), y=KL_relevance, fill=dataset))+
  geom_bar(stat = "identity", position = position_dodge(-.7), width=0.7, na.rm = FALSE)+ #position_dodge negative no. in reverse
  scale_fill_manual(values=c('#E8C09C', '#7FAFD2')) + ylab('KL relevance') + theme_classic2()+
  theme(axis.title.y = element_blank(),
        axis.text.y = element_text(size = 8.5, vjust = 0.5, hjust = 0.5),
        axis.title.x = element_text(size = 9),
        plot.margin = unit(c(5.5, 10, 5.5, 0), "pt"),
        legend.position = "none") +
  scale_y_continuous(expand = c(0,0))+
  coord_flip()
