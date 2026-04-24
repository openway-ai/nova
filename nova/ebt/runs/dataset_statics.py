import pyarrow.parquet as pq                                                                                                                                      
from nanochat.dataset import list_parquet_files                                                                                                                   
                                            
paths = list_parquet_files()                                                    
print(f'总 parquet 文件数: {len(paths)}')                                       
print(f'训练集文件数: {len(paths) - 1}')                                        
print(f'验证集文件数: 1')                                                                                                                                         
                                                                                
total_rows = 0                                                                  
total_rg = 0                                                                              
for i, p in enumerate(paths):                                                                                                                                     
    pf = pq.ParquetFile(p)                                                      
    nr = pf.metadata.num_rows                                                                                                                                     
    nrg = pf.num_row_groups                                                                                                                                       
    total_rows += nr                                                                                                                                              
    total_rg += nrg                                                             
    if i < 3 or i == len(paths)-1:                                              
        print(f'  shard_{i:05d}: {nr:,} rows, {nrg} row_groups')                
    elif i == 3:                                                                          
        print(f'  ...')                                                                                                                                           
                                                                                
print(f'')                                                                      
print(f'总文档数: {total_rows:,}')                                              
print(f'总 row groups: {total_rg:,}')                                           
print(f'')                                                                                                                                                        
print(f'--- 训练消耗估算 ---')                                                            
tokens_per_step = 8 * 1 * 32 * 1024  # 262144                                             
total_steps = 28000                                                                                                                                               
total_tokens = tokens_per_step * total_steps                                                                                                                                           
print(f'每步 tokens: {tokens_per_step:,}')                                                                                                                        
print(f'总步数: {total_steps:,}')                                                                                                                                                      
print(f'训练消耗 tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)')                                                                                                                  
print(f'考虑 ~35% cropping, 实际读取原始 tokens: ~{total_tokens/0.65/1e9:.2f}B') 