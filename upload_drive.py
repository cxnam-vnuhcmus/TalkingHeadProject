from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()    
gauth.CommandLineAuth()      
drive = GoogleDrive(gauth)  

upload_file_list = ['/root/TalkingHead/imaginaire/logs/2022_1210_1526_22_ampO1/epoch_00078_iteration_000042000_checkpoint.pt']
for upload_file in upload_file_list:
	gfile = drive.CreateFile({'parents': [{'id': '1uNSNvJ5fDzZ5BJ4wDHdrNlDSY3igI3no'}],
    'title': 'GI2I_e78_i42000.pt'
    })
	# Read file and set it as the content of this instance.
	gfile.SetContentFile(upload_file)
	gfile.Upload() # Upload the file.

# file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1CA_1Kz6-eEpdwBciY1JxAZxHBtB2FgiE')}).GetList()
# for file in file_list:
# 	print('title: %s, id: %s' % (file['title'], file['id']))