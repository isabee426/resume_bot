---
library_name: peft
pipeline_tag: text-generation
widget:
- text: 'Job: kitchen Keywords: inventory management Resume Point:'
  example_title: kitchen
- text: 'Job: retail Keywords: customer service Resume Point:'
  example_title: retail
- text: 'Job: laundry Keywords: organization Resume Point:'
  example_title: laundry
tags:
- resume
language:
- en
datasets:
- isashap/resume-dataset-w-context
base_model: gpt2
---
## Training procedure

### Framework versions
-This version only has three jobs, limited keyword options. 

-Check out some of the previous versions to see how far WaldoBot has come.

If you'd like to see the most effective keywords click the dataset underneath the text box.

Tips: 

Sometimes one keyword works best. 

If you want the resume to start with a specific verb, write it and let it autocomplete! (verb in past-tense form) 

If the Waldobot doesnt finish generating a whole sentence, click compute again! 

- PEFT 0.5.0