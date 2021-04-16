library(dplyr)
library(stringr)

## load all data
annotated_data <- readr::read_csv("data/aspect_tuples_20210406.2.csv")
new_data <- annotated_data %>%
  select(is_context, evaltext) %>%
  filter(evaltext != '#ERROR!')
new_data <- na.omit(new_data)

## load cnstr list
cnstr <- read_csv("data/constructions_0415.csv",
                  col_names = FALSE)
cnstr <- cnstr %>% pull(X1) # df to vector

## match evaltext
regex = paste(cnstr, collapse="|")
new_data$matches = sapply(str_extract_all(new_data$evaltext, regex), 
                          function(x) paste(x, collapse=";"))

sorted_data <- new_data %>%
  arrange(is_context, desc(matches))

write.csv(sorted_data, 'match_data_0415.csv')

## count coverage
new_data_coverage <- sum(new_data$matches != "")/length(new_data$matches)
