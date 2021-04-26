new_seeds <- na.omit(data.frame(col = unlist(seeds), row.names = NULL))

write.csv(new_seeds, 'new_seeds.csv', col.names = F)
