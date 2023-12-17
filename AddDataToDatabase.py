import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://app-face-cam-default-rtdb.firebaseio.com/"
})

ref = db.reference('Peoples')

data = {
    "10001":{
        "name":"Victor Fernandes",
        "type":"Local",
        "gender":"Male",
        "last_capture":"2023-12-17 00:00:00"
    },
    "10002":{
        "name":"Jaqueline Fernandes",
        "type":"Local",
        "gender":"Female",
        "last_capture":"2023-12-17 00:00:00"
    }
}

for key, value in data.items():
    ref.child(key).set(value)