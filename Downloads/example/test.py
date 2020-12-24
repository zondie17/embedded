import os
from threading import Thread
import time
import webbrowser

port_number = "8000"

def run_on(port):  
    os.system("python -m http.server " + port)
   
if __name__ == "__main__":
    server = Thread(target=run_on, args=[port_number])
    #run_on(port_number) #Run in main thread
    #server.daemon = True # Do not make us wait for you to exit  
    server.start()
    time.sleep(2) #Wait to start the server first

def test():
    url = "http://localhost:" + port_number
    webbrowser.open(url)
    print(url + " is opened in browser")

test()