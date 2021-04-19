# Machine Learning Research Topic Extraction

This script can be used to summarize and extract the main topics from a collection of research papers.

Machine learning terms, specified by a given index term text file (default: index-terms.txt), are searched within the collection of research papers, and a word cloud of these terms is created.

The machine learning index terms specified in index-terms.txt were automatically extracted from the following books (and slightly adapted manually):
- Cady, Field. The data science handbook. John Wiley & Sons, 2017.
- EMC Education Services. Data Science and Big Data Analytics: Discovering, Analyzing, Visualizing and Presenting Data. Wiley, 2015.
- Bonaccorso, Giuseppe. Machine learning algorithms. Packt Publishing Ltd, 2017.
- Goodfellow, Ian, Bengio, Yoshua and Courville, Aaron. Deep Learning. MITP, 2018.
- Shanmugamani, Rajalingappaa. Deep Learning for Computer Vision: Expert Techniques to Train Advanced Neural Networks Using TensorFlow and Keras. Packt Publishing, 2018.
- Bhardwaj, Anurag, Di, Wei, and Wei, Jianing. Deep Learning Essentials: Your Hands-On Guide to the Fundamentals of Deep Learning and Neural Network Modeling. Packt Publishing, 2018.

This list can be adapted and extended and a new index-terms.txt can be created.
                 
## Requirements
- Python: 3.8
- WordCloud
- NLTK
- PyPDF2

## Usage
```
python machine_learning_research_topic_extraction.py --pdfs /folder/containing/pdfs
```
