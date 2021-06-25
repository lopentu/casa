library(readr)
library(dplyr)
onto_annot_all <- read_csv("C:/Users/user/Downloads/onto_annot_all.csv")

duplicated_df <- onto_annot_all %>%
  distinct(attribute, candidate, .keep_all = TRUE) %>%
  group_by(candidate) %>% 
  filter(n()>1) %>%
  ungroup() %>%
  arrange(candidate, attribute)
  
write.csv(duplicated_df, 'duplicated_df.csv')
  