from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()    
gauth.CommandLineAuth()      
drive = GoogleDrive(gauth)  

upload_file_list = ['/root/TalkingHead/results/MEAD_A13L74/best_model.pt']
for upload_file in upload_file_list:
	gfile = drive.CreateFile({'parents': [{'id': '1CA_1Kz6-eEpdwBciY1JxAZxHBtB2FgiE'}],
    'title': 'a13l74_256_best_model.pt'
    })
	# Read file and set it as the content of this instance.
	gfile.SetContentFile(upload_file)
	gfile.Upload() # Upload the file.

# file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1CA_1Kz6-eEpdwBciY1JxAZxHBtB2FgiE')}).GetList()
# for file in file_list:
# 	print('title: %s, id: %s' % (file['title'], file['id']))