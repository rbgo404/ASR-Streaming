# Streaming Model Template for Whisper-large-v2


## Deploy Whisper-large-v2 using Inferless:

- Deployment of openai/whisper-large-v2 model using [Faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [Whisper_streaming](https://github.com/ufal/whisper_streaming).

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

---
## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the **Create new Runtime** button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Create a Volume in My Volumes 
To access the my volumes window in Inferless, simply navigate to the sidebar and click on the **Create new Volume** button. A pop-up will appear.

Copy the the mount path and keep it handly for step 4 of model import you will need to pass this as ENV variable. "VOLUME_NFS"

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and use the forked repo URL as the **Model URL**.

After the create model step, while setting the configuration for the model make sure to select the appropriate runtime.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/github-custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer <your_api_key>' \
--data '{
  "inputs": [
    {
      "name": "audio_base64",
      "shape": [1],
      "data": [<audio_in_base64>],
      "datatype": "BYTES"
    }
  ]
}'

```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function are `stream_output_handler` and `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input-output-schema](https://docs.inferless.com/model-import/input-output-schema) for more.

```python
def infer(self,inputs,stream_output_handler):
    audio_data = inputs["audio_base64"]
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting `self.online = None`.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
