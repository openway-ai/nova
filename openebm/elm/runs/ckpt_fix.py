# import struct, os                                             
                                                                  
# src = '/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-ctx2048-notimeembed-exact-resume-fromstep4312_20260420_200104/s=step=11187-d26-ctx2048-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.7656.ckpt'                                                                                                                                                                                                                                                                 
# dst = src + '.repaired' 
                                                                                                                                                                                                                                                                        
# with open(src, 'rb') as f:                                                                                                                                  
#     data = f.read()                               
                                                                                                                                                            
# LOCAL_SIG = b'PK\x03\x04'                                       
# entries = []                                                    
# pos = 0                                                                                                                            
# while True:                                                     
#     idx = data.find(LOCAL_SIG, pos)                                                                                                                                                                                                                                      
#     if idx == -1:                                                                                                                                                                                                                                                        
#         break                                                                                                                                                                                                                                                            
#     hdr = data[idx:idx+30]                                                                                                                                                                                                                                               
#     if len(hdr) < 30:                                                                                                                                                                                                                                                    
#         break                                                                                                                                                                                                                                                            
#     ver, flags, method, modtime, moddate, crc, comp_sz, uncomp_sz, fname_len, extra_len = struct.unpack('<HHHHHIIIHH', hdr[4:30])                                                                                                              
#     fname = data[idx+30:idx+30+fname_len]                                                                                          
#     data_offset = idx + 30 + fname_len + extra_len              
#     entries.append({                                        
#         'offset': idx,                                          
#         'ver': ver, 'flags': flags, 'method': method,           
#         'modtime': modtime, 'moddate': moddate,
#         'crc': crc, 'comp_sz': comp_sz, 'uncomp_sz': uncomp_sz, 
#         'fname': fname,                                                                                                                                                                                                                                                  
#     })                                                          
#     pos = data_offset + comp_sz                                 
#     if pos <= idx:                      
#         pos = idx + 4              
                                                                
# print(f'Parsed {len(entries)} entries')                         
                                                                
# # Build central directory with ZIP64 extra fields    
# cd_records = bytearray()       
# for e in entries:                                               
#     # ZIP64 extra field: tag(2) + size(2) + uncomp(8) + comp(8) + offset(8)                                                                                                                                                                    
#     zip64_extra = struct.pack('<HH', 0x0001, 24)                
#     zip64_extra += struct.pack('<QQQ', e['uncomp_sz'], e['comp_sz'], e['offset'])                                                                                                                                                              
                                                                
#     rec = struct.pack('<IBBHHHHHIIIHHHHHII',
#         0x02014b50, 45, 0,                                                                                                         
#         45,  # version needed for zip64                         
#         e['flags'], e['method'], e['modtime'], e['moddate'],
#         e['crc'],                                      
#         0xFFFFFFFF,  # placeholder -> zip64                     
#         0xFFFFFFFF,  # placeholder -> zip64                     
#         len(e['fname']),                       
#         len(zip64_extra),                                      
#         0, 0, 0, 0,                                             
#         0xFFFFFFFF,  # offset placeholder -> zip64     
#     )                                                           
#     rec += e['fname'] + zip64_extra    
#     cd_records += rec
                                                                        
# cd_offset = len(data)                                           
# cd_size = len(cd_records)                                                 
# num = len(entries)                               
                                                                
# # 错误: '<IQHHIIQQQQQ' (11个字段)                                                                                                                                                                                                                                                                                                                                                                                         
# # 正确: '<IQHHIIQQQQ'  (10个字段)                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                            
# zip64_eocd = struct.pack('<IQHHIIQQQQ',                                                                                                                                                                                                                                                                                                                                                                                   
#     0x06064b50,      # sig
#     44,              # size of remaining
#     45, 45,          # versions
#     0, 0,            # disk numbers
#     num, num,        # entry counts
#     cd_size,
#     cd_offset,
# )                                                        
                                                                
# # ZIP64 end of central directory locator                                     
# zip64_locator = struct.pack('<IIQI',                                                                                                                                                                                                                                                                          
#     0x07064b50,                                                           
#     0,                                                                                                                                                                                                                                                                                                        
#     cd_offset + cd_size,                                                  
#     1,                                                                                        
# )                                                                            
                                                                            
# # Standard EOCD (with 0xFFFF/0xFFFFFFFF placeholders)                                                                                                                                                                                                                                                                      
# eocd = struct.pack('<IHHHHIIH',                                              
#     0x06054b50,                                                              
#     0, 0,                                                                    
#     min(num, 0xFFFF), min(num, 0xFFFF),                                      
#     min(cd_size, 0xFFFFFFFF),                                                                 
#     0xFFFFFFFF,                                                                                                                                                                                                                                                                                                            
#     0,                                                                       
# )                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                
# with open(dst, 'wb') as f:                                                                    
#     f.write(data)                                                                             
#     f.write(cd_records)                                                                       
#     f.write(zip64_eocd)                                                                       
#     f.write(zip64_locator)                                                                                            
#     f.write(eocd)                                                                                                                                                                                                                                                                                                                                                                               
                                                                                            
# print(f'Written: {dst} ({os.path.getsize(dst)} bytes)')                                                                                                                                                                                                                                                                                                                                         
                                                                                            
# import zipfile                                                                                                        
# try:                                                                                                                  
#     z = zipfile.ZipFile(dst)                                                                                          
#     print(f'Verification OK: {len(z.namelist())} entries')                                                            
#     z.close()                                                                                                         
# except Exception as ex:                                                                                               
#     print(f'Verification failed: {ex}')  


import torch
ckpt = torch.load('/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-ctx2048-notimeembed-exact-resume-fromstep4312_20260420_200104/s=step=11187-d26-ctx2048-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.7656.ckpt.repaired2', map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys()))
print('Step:', ckpt.get('global_step', 'N/A'))


# import zipfile
# z = zipfile.ZipFile('/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-ctx2048-notimeembed-exact-resume-fromstep4312_20260420_200104/s=step=11187-d26-ctx2048-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.7656.ckpt.repaired2')                                                                                                                                                                           
# names = z.namelist()                                  
# print('Total entries:', len(names))
# print('--- First 20 ---')
# for n in names[:20]:
#     info = z.getinfo(n)
#     print('  %12d  %s' % (info.file_size, n))
# print('--- Last 10 ---')
# for n in names[-10:]:
#     info = z.getinfo(n)
#     print('  %12d  %s' % (info.file_size, n))
# for key in ['archive/version', 'archive/data.pkl', 'version', 'data.pkl']:
#     found = 'FOUND' if key in names else 'MISSING'
#     print('%s: %s' % (key, found))
# try:
#     first = z.read(names[0])
#     print('First entry readable: %d bytes' % len(first))
# except Exception as e:
#     print('First entry read FAILED: %s' % e)





# import struct, os, zipfile

# src = '/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-ctx2048-notimeembed-exact-resume-fromstep4312_20260420_200104/s=step=11187-d26-ctx2048-lr0.00025-bs1x32-muon_adamw-valid_loss=valid_loss=2.7656.ckpt'
# dst = src + '.repaired2'

# with open(src, 'rb') as f:
#     data = f.read()

# file_size = len(data)
# LOCAL_SIG = b'PK\x03\x04'

# # Collect all local headers
# headers = []
# pos = 0
# while True:
#     idx = data.find(LOCAL_SIG, pos)
#     if idx == -1:
#         break
#     hdr = data[idx:idx+30]
#     if len(hdr) < 30:
#         break
#     ver, flags, method, modtime, moddate, crc, comp_sz, uncomp_sz, fname_len, extra_len = struct.unpack('<HHHHHIIIHH', hdr[4:30])
#     fname = data[idx+30:idx+30+fname_len]
#     data_start = idx + 30 + fname_len + extra_len
#     headers.append({
#         'offset': idx,
#         'data_start': data_start,
#         'method': method,
#         'modtime': modtime,
#         'moddate': moddate,
#         'crc': crc,
#         'fname': fname,
#         'flags': flags,
#     })
#     pos = idx + 4  # move past sig to find next

# # Compute actual data sizes from gaps between headers
# for i in range(len(headers)):
#     if i + 1 < len(headers):
#         actual_size = headers[i+1]['offset'] - headers[i]['data_start']
#     else:
#         actual_size = file_size - headers[i]['data_start']
#     headers[i]['actual_size'] = actual_size - 16  # strip data descriptor

# print('Parsed %d entries' % len(headers))
# print('First 5:')
# for h in headers[:5]:
#     print('  %s -> %d bytes' % (h['fname'].decode(), h['actual_size']))

# # Rewrite the zip properly
# import io, zlib

# with open(dst, 'wb') as out:
#     cd_entries = []
#     for h in headers:
#         entry_data = data[h['data_start']:h['data_start']+h['actual_size']]
#         crc = zlib.crc32(entry_data) & 0xFFFFFFFF

#         local_offset = out.tell()
#         # Write local header with correct sizes
#         local_hdr = struct.pack('<IHHHHHIIIHH',
#             0x04034b50,
#             45,  # version needed
#             h['flags'] & ~0x08,  # clear data descriptor flag
#             0,   # stored (no compression)
#             h['modtime'], h['moddate'],
#             crc,
#             h['actual_size'],
#             h['actual_size'],
#             len(h['fname']),
#             0,   # no extra
#         )
#         out.write(local_hdr)
#         out.write(h['fname'])
#         out.write(entry_data)

#         cd_entries.append({
#             'fname': h['fname'],
#             'crc': crc,
#             'size': h['actual_size'],
#             'offset': local_offset,
#             'modtime': h['modtime'],
#             'moddate': h['moddate'],
#         })

#     cd_start = out.tell()
#     for e in cd_entries:
#         needs_zip64 = e['size'] >= 0xFFFFFFFF or e['offset'] >= 0xFFFFFFFF
#         if needs_zip64:
#             zip64_extra = struct.pack('<HHQQQ', 0x0001, 24, e['size'], e['size'], e['offset'])
#             sz_field = 0xFFFFFFFF
#             off_field = 0xFFFFFFFF
#         else:
#             zip64_extra = b''
#             sz_field = e['size']
#             off_field = e['offset']

#         cd_rec = struct.pack('<IBBHHHHHIIIHHHHHII',
#             0x02014b50, 45, 0, 45,
#             0, 0,
#             e['modtime'], e['moddate'],
#             e['crc'],
#             min(e['size'], 0xFFFFFFFF),
#             min(e['size'], 0xFFFFFFFF),
#             len(e['fname']),
#             len(zip64_extra),
#             0, 0, 0, 0,
#             off_field,
#         )
#         out.write(cd_rec)
#         out.write(e['fname'])
#         out.write(zip64_extra)

#     cd_end = out.tell()
#     cd_size = cd_end - cd_start
#     num = len(cd_entries)

#     # ZIP64 EOCD
#     out.write(struct.pack('<IQHHIIQQQQ',
#         0x06064b50, 44, 45, 45, 0, 0, num, num, cd_size, cd_start))
#     # ZIP64 locator
#     out.write(struct.pack('<IIQI', 0x07064b50, 0, cd_end, 1))
#     # EOCD
#     out.write(struct.pack('<IHHHHIIH',
#         0x06054b50, 0, 0,
#         min(num, 0xFFFF), min(num, 0xFFFF),
#         min(cd_size, 0xFFFFFFFF), 0xFFFFFFFF, 0))

# print('Written: %s (%d bytes)' % (dst, os.path.getsize(dst)))

# # Verify
# z = zipfile.ZipFile(dst)
# names = z.namelist()
# print('ZIP OK: %d entries' % len(names))
# for key in ['archive/data.pkl', 'archive/byteorder']:
#     if key in names:
#         d = z.read(key)
#         print('%s: %d bytes, content=%s' % (key, len(d), repr(d[:100])))
# z.close()