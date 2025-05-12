import os

directory_path = r'D:\Deaf and Dumb\super'  
count = 102


for filename in os.listdir(directory_path):
    
    old_file = os.path.join(directory_path, filename)
   
   
    if os.path.isdir(old_file):
        continue
   
    new_filename = str(count) + os.path.splitext(filename)[1]  
    count += 1
    
    
    new_file = os.path.join(directory_path, new_filename)
    
    
    os.rename(old_file, new_file)
