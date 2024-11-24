# SAM-Pipeline
* Customized pipeline that segments Anything in a picture by  utilizing the Meta Research Segment Anything Open Source Research 
* It was a problem to come with an app that can do multiple segmentations by prompt through SAM but my app does it

## Demo
* [Screencast from 2024-06-09 13-34-45.webm](https://github.com/FranklineMisango/SAM-Pipeline/assets/95913228/b4b3a655-1260-4f0c-9ed7-9778ec6fc20e)
* [Video from rangerover](output.avi)

## Installation
* Clone this repo
* Install all the four dependecies on the app.py
* Run pip install torch torchvision on your terminal
* Run pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git on yiur terminal
* To finally run the app, run streamlit run app.py

## Credits 
* ![Arxiv Paper]("https://segment-anything.com/)
* ![Arxix Paper]("https://segment-anything.com/)
* ![Meta SAM Github]("https://github.com/facebookresearch/segment-anything")

## Future Improvement - open to colab
* The app is a bit slow, so I am on pushing it to the AWS EC2 clusters and make it a bit faster for future use cases
