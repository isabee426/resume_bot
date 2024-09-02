#paste this code to deploy the resumebot


from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("isashap/contexttrained-validationloss-gpt2FINAL")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "isashap/contexttrained-validationloss-gpt2FINAL")
