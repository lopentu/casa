import numpy as np

# https://github.com/slundberg/shap/blob/9411b68e8057a6c6f3621765b89b24d82bee13d4/shap/plots/_text.py#L169
def process_shap_values(shap_value, group_threshold=1, separator=''):

    tokens = shap_value.data
    values = shap_value.hierarchical_values
    clustering = shap_value.clustering

    # See if we got hierarchical input data. If we did then we need to reprocess the 
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:
        
        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError("The length of the attribution values must match the number of " + \
                             "tokens if shap_values.clustering is None! When passing hierarchical " + \
                             "attributions the clustering is also required.")
        
        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i,0])
            ri = int(clustering[i,1])
            groups.append(groups[li] + groups[ri])
            lower_values[M+i] = lower_values[li] + lower_values[ri] + values[M+i]
            max_values[i+M] = max(abs(values[M+i]) / len(groups[M+i]), max_values[li], max_values[ri])
    
        # compute the upper_values
        upper_values = np.zeros(len(values))
        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i-M,0])
            ri = int(clustering[i-M,1])
            upper_values[i] = value
            value += values[i]
#             lower_credit(upper_values, clustering, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
#             lower_credit(upper_values, clustering, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, clustering, li, value * 0.5)
            lower_credit(upper_values, clustering, ri, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)
        
        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_values = []
        group_sizes = []

        # meta data
        token_id_to_node_id_mapping = np.zeros((M,))
        collapsed_node_ids = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):
            
            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)

                # meta data
                collapsed_node_ids.append(i)
                token_id_to_node_id_mapping[i] = i

            else:

                # compute the dividend at internal nodes
                li = int(clustering[i-M,0])
                ri = int(clustering[i-M,1])
                dv = abs(values[i]) / len(groups[i])
                
                # if the interaction level is too high then just treat this whole group as one token
                if dv > group_threshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(separator.join([tokens[g] for g in groups[li]]) + separator + separator.join([tokens[g] for g in groups[ri]]))
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))

                    # setting collapsed node ids and token id to current node id mapping metadata

                    collapsed_node_ids.append(i)
                    for g in groups[li]:
                        token_id_to_node_id_mapping[g] = i
                    
                    for g in groups[ri]:
                        token_id_to_node_id_mapping[g] = i
                    
                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)
        merge_tokens(new_tokens, new_values, group_sizes, len(group_values) - 1)
        
        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)

        # meta data
        token_id_to_node_id_mapping = np.array(token_id_to_node_id_mapping)
        collapsed_node_ids = np.array(collapsed_node_ids)

        M = len(tokens) 
    else:
        group_sizes = np.ones(M)
        token_id_to_node_id_mapping = np.arange(M)
        collapsed_node_ids = np.arange(M)

    return {"tokens": tokens, "values": values, "group_sizes": group_sizes,
            "upper_values": upper_values, "lower_values": lower_values,
            "group_values": group_values, 
            "max_values": max_values,
            "token_id_to_node_id_mapping": token_id_to_node_id_mapping,
            "collapsed_node_ids": collapsed_node_ids}