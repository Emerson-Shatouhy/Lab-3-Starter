# Overview 

This directory includes a simple inference server that serves a pretrained image classification model.
The top 1 result, both the prediction label and the probability, will be returned. 


To start the server, in the command line, type in:

```bash 
# you will need to install the required python packages first
# check out poetry if you aren't familiar with it
python image_classification_server.py
```

To test the functionality of the server, you can use `curl` like this:

```bash
curl -X POST -F "file=@test-images/dog.jpg" http://127.0.0.1:5050/predict
```

If the server setup is successful, you should see the output like this:

```bash 
# output
{"predicted_label":"Doberman","score":0.9957420229911804}
```



There is also a simple python client that uses the default camera of the development machine and sends every frame for inference.
Start it in command line with:

```bash 
python client.py
```

