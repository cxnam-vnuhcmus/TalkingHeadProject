from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm

gauth = GoogleAuth()    
gauth.CommandLineAuth()      
drive = GoogleDrive(gauth)  

upload_file_list = ['/root/TalkingHeadProject/my_framework/result_A2LM_LMAudioPrev_v2/best_model.pt']

file_name = ['A2LM_LMAudioPrev_NoAttention_v2']
for i, upload_file in tqdm(enumerate(upload_file_list)):
	gfile = drive.CreateFile({'parents': [{'id': '1uNSNvJ5fDzZ5BJ4wDHdrNlDSY3igI3no'}],
    'title': file_name[i]
    })
	# Read file and set it as the content of this instance.
	gfile.SetContentFile(upload_file)
	gfile.Upload() # Upload the file.

# file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1CA_1Kz6-eEpdwBciY1JxAZxHBtB2FgiE')}).GetList()
# for file in file_list:
# 	print('title: %s, id: %s' % (file['title'], file['id']))