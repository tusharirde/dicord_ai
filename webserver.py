from flask import Flask, render_template, request, jsonify
from threading import Thread
import os
import csv

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("index.html")

def run():
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

def b():
    server = Thread(target=run)
    server.daemon = True
    server.start()