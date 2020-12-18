import json, os
from bottle import run, post, request, response
from gtts import gTTS
from playsound import playsound
from time import time

def sayHi(name):
    mess = "Xin chào {0}".format(name)
    output = gTTS(mess, lang="vi", slow=False)

    output.save("output.mp3")
    playsound("output.mp3", True)
    os.remove("output.mp3")

@post('/txt2sp')
def main():
    data = request.body.getvalue().decode('utf-8')
    data = json.loads(data)

    sayHi(data['name'])

    return 'Done'

run(host='localhost', port=8080, debug=True)