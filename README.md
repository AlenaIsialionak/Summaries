# Summaries

This repository contains the code for a Streamlight app, that generates summaries of text.  
It supports both English and Russian text. The app allows users to upload text files  and select a summarization model and parameters.

The "facebook/bart-large-cnn"model was selected for English summarization because of its balance of performance, ease of using, and availability. "facebook/bart-large-cnn" offers a good trade-off between accuracy, computational cost. It’s also a widely supported model with extensive documentation.

The "RussianNLP/FRED-T5-Summarizer" model was chosen for Russian text summarization. Finding well-documented and available Russian summarization models can be challenging. This model provided the best performance among the options I evaluated and offered reasonable resource requirements.


## Getting Started

These instructions will guide you on how to set up and run the Summaries application locally.

Getting Started

1. Clone the repository:
$ git clone https://github.com/AlenaIsialionak/Summaries
$ cd Summaries

2. Create a virtual environment and activate it:
$ python -m venv env
$ source env/bin/activate

3. Install the dependencies:
$ pip install -r requirements.txt

4. Run the app locally:
$ streamlit run app.py  