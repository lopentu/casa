library(readr)
matched_random <- read_csv("C:/Users/user/Desktop/matched_random.csv")
colnames(matched_random)[3] <- 'score1'
colnames(matched_random)[5] <- 'score2'
colnames(matched_random)[7] <- 'score3'
colnames(matched_random)[9] <- 'score4'
colnames(matched_random)[11] <- 'score5'

new_df <- melt(matched_random, id.vars = "construction")

library(dplyr)

sorted_df <- new_df %>%
  arrange(construction)

write.csv(sorted_df,"C:/Users/user/Desktop/matched.csv", row.names = FALSE)
