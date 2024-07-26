# def low_dimension_attention(query_states, key_states, heavy_budget, recent_budget, penalty):

#     cache_budget = heavy_budget + recent_budget

#     # attn_weights (BS, head, query, keys)
#     dtype_query_states = query_states.dtype
    
#     batch_size = query_states.shape[0]
#     head_num = query_states.shape[1]
#     seq_length = query_states.shape[2]
#     state_dimension = query_states.shape[3]
    
#     history_mask = torch.zeros(batch_size, head_num, seq_length, dtype=dtype_query_states, device=query_states.device)
#     small_dimensions = None
    
#     attn_shape = (batch_size, head_num, seq_length, seq_length)
#     result_attention = torch.zeros(attn_shape, dtype=dtype_query_states, device=query_states.device)

#     for token_index in range(seq_length):
#         if token_index > cache_budget:
#             if small_dimensions is None:
#                 _, small_dimensions = keys[:,:,:token_index-1,:].abs().mean(dim=-2).topk(state_dimension-32, largest=False, dim=-1)
            
#             history = history_mask[:,:,:token_index] + tmp_attn.squeeze(2)
            
#             if recent_budget != 0:
#                 _, unnecessary_tokens = history[:,:,:-recent_budget].topk(1, largest=False, dim=-1)
#             else:
#                 _, unnecessary_tokens = history[:,:,:].topk(1, largest=False, dim=-1)
            
#             batch_indices, head_indices = torch.meshgrid(torch.arange(batch_size), torch.arange(head_num))
#             batch_indices_exp = batch_indices.unsqueeze(-1).expand_as(unnecessary_tokens)
#             head_indices_exp = head_indices.unsqueeze(-1).expand_as(unnecessary_tokens)
            
#             normal = torch.norm(keys[batch_indices_exp, head_indices_exp, unnecessary_tokens], dim=-1)
#             keys[batch_indices_exp, head_indices_exp, unnecessary_tokens, small_dimensions] = 0
#             after_normal = torch.norm(keys[batch_indices_exp, head_indices_exp, unnecessary_tokens], dim=-1)
#             scale = (normal/after_normal).unsqueeze(-1)
#             keys[batch_indices_exp, head_indices_exp, unnecessary_tokens] *= scale
#             history_mask[batch_indices_exp, head_indices_exp, unnecessary_tokens] = torch.inf
            
#         query = query_states[:,:,token_index,:].unsqueeze(2)
#         keys = key_states[:,:,:token_index+1,:]
        
#         tmp_attn = torch.matmul(query, keys.transpose(2,3))/math.sqrt(state_dimension)
#         result_attention[:,:,token_index,:token_index+1] = tmp_attn.squeeze(2)
            
#     return result_attention