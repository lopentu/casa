library(readr)
library(dplyr)
library(stringr)

## load all data
annotated_data <- read_csv("data/aspect_tuples_20210406.2.csv")
new_data <- annotated_data %>%
  select(is_context, evaltext) %>%
  filter(evaltext != '#ERROR!')
new_data <- na.omit(new_data)

## load cnstr list
cnstr <- read_csv("data/constructions_0406.csv",
                  col_names = FALSE)
cnstr <- cnstr %>% pull(X1) # df to vector

## match evaltext
regex = paste(cnstr, collapse="|")
new_data$matches = sapply(str_extract_all(new_data$evaltext, regex), 
                          function(x) paste(x, collapse=";"))

new_data_coverage <- new_data %>%
  group_by(is_context, evaltext) %>%
  arrange(desc(matches))
  
