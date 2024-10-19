


### What are safetensors?

Safetensors are a file format used to store model weights in a secure and efficient manner.
- **Security**: They mitigate risks associated with **executing arbitrary code** during model loading.
- **Efficiency**: Safetensors are designed for faster loading times and **reduced memory usage compared to traditional formats like .pt or .bin**.
- **Compatibility**: They are compatible with various frameworks,

```python
from transformers import AutoModelForCausalLM

# To load a model saved in the Safetensor format, you need to specify the use_safetensors=True parameter when calling the from_pretrained method. 
model = AutoModelForCausalLM.from_pretrained("path_to_model", use_safetensors=True)

#To convert existing models to use Safetensors, save your model weights in this format using the following method
from safetensors.torch import save_file

# Assuming 'model' is your trained Hugging Face model
save_file(model.state_dict(), "model.safetensors")
```
