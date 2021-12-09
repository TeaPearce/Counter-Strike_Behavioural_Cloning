import os
import time
import struct
import math
import random
import win32api
import win32gui
import win32process
from ctypes  import *
from pymem   import *
import numpy as np
import requests

from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import json

ReadProcessMemory = windll.kernel32.ReadProcessMemory
WriteProcessMemory = windll.kernel32.WriteProcessMemory

# stuff for RAM...
def update_offsets(raw):
    raw = raw.replace('[signatures]','#signatures\n')
    raw = raw.replace('[netvars]','#netvars\n')
    try:
        open('dm_hazedumper_offsets.py','w').write(raw)
        print('updated succesfuly')
    except:
        print('couldnt open offsets.py to preform update')
    return

def getlength(type):
    # integer float char ? require diff lengths
    if type == 'i':
        return 4
    elif type == 'f':
        return 4
    elif type == 'c':
        return 1
    elif type == 'b': #tp added
        return 1 # maybe 4
    elif type == 'h': #tp added
        return 4 

def read_memory(game, address, type):
    buffer = (ctypes.c_byte * getlength(type))()
    bytesRead = ctypes.c_ulonglong(0)
    readlength = getlength(type)
    ReadProcessMemory(game, address, buffer, readlength, byref(bytesRead))
    return struct.unpack(type, buffer)[0]

# stuff for game state integration...

# https://docs.python.org/2/library/basehttpserver.html
# info about HTTPServer, BaseHTTPRequestHandler
class MyServer(HTTPServer):
    def __init__(self, server_address, token, RequestHandler):
        self.auth_token = token
        super(MyServer, self).__init__(server_address, RequestHandler)
        # create all the states of interest here
        self.data_all = None
        # my_dict = {1: 'apple', 2: 'ball'}
        self.round_phase = None
        self.player_status = None

# need this running in the background
class MyRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = self.rfile.read(length).decode('utf-8')

        self.parse_payload(json.loads(body))

        self.send_header('Content-type', 'text/html')
        self.send_response(200)
        self.end_headers()

    def is_payload_authentic(self, payload):
        if 'auth' in payload and 'token' in payload['auth']:
            return payload['auth']['token'] == server.auth_token
        else:
            return False

    def parse_payload(self, payload):
        # Ignore unauthenticated payloads
        if not self.is_payload_authentic(payload):
            return None

        self.server.data_all = payload.copy()


        if False:
            print('\n')
            for key in payload:
                print(key,payload[key])
            time.sleep(2)

        round_phase = self.get_round_phase(payload)

        # could only print when change phase
        if round_phase != self.server.round_phase:
            self.server.round_phase = round_phase

        # get player status - health, armor, kills this round, etc.
        player_status = self.get_player_status(payload)

    def get_round_phase(self, payload):
        if 'round' in payload and 'phase' in payload['round']:
            return payload['round']['phase']
        else:
            return None

    def get_player_status(self, payload):
        if 'player_state' in payload:
            return payload['player_state']
        else:
            return None

    def log_message(self, format, *args):
        """
        Prevents requests from printing into the console
        """
        return

server = MyServer(('localhost', 3000), 'MYTOKENHERE', MyRequestHandler)

