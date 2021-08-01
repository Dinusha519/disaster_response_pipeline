# Disaster Response Application
In a situation of a disaster, disaster response organizations receive millions of messages and they are needed to be directed to different organizations to take care of different types of problems. This web application is built to ease out that process where it will show the classified area when a message is entered.

## Data
The dataset was provided by [Figure8](https://appen.com) with about 26K labelled messages.
```sh
'related', 'request', 'offer', 'aid_related', 
'medical_help', 'medical_products',
'search_and_rescue', 'security', 'military', 
'child_alone', 'water', 'food', 'shelter', 
'clothing', 'money', 'missing_people', 'refugees', 
'death', 'other_aid', 'infrastructure_related', 
'transport', 'buildings', 'electricity', 'tools', 
'hospitals', 'shops', 'aid_centers', 
'other_infrastructure', 'weather_related', 
'floods', 'storm', 'fire', 'earthquake', 'cold', 
'other_weather', 'direct_report'
```
### Sections
The Project is divided in the following Sections:

- Data Processing 
    - ETL Pipeline to extract data from source, clean data and save them in a proper database structure
- Machine Learning Pipeline
    - To train a model able to classify text message in categories
- Web App
    - To show model results in real time

## Getting Started
### Installing Dependencies
Clone this GIT repository and install required packages.
```sh
git clone https://github.com/Dinusha519/disaster_response_pipeline.git
pip install -r requirements.txt
```

### Executing Program:
1. Make the disaster_reponse_pipline folder as the root folder
	|-disaster_reponse_pipline
		|-app
		|-data
		|-images
		|-models
		|-README.md
		|-requirements.txt

2. Run the following commands to set up the database and model:
    - To run the ETL pipeline that cleans and stores the data: 
-       python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
3. Run the ML pipeline that trains and saves the classifier: 
-       python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

4. Run the following command in the app's directory to run the web app. 
-       python run.py

5. To Access the web app navigate to
 http://0.0.0.0:3001/


## Files
1. /data/process_data.py: A data cleaning pipeline that:
- Loads the messages and categories csvs to the environment
- Merges the two datasets and cleans the data
- Stores it in a SQLite database

2. /model/train_classifier.py: A machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model
- Outputs results on the test set
- Exports the final model as a pickle file

3. /app/run.py
- Codes for flask web application

## Screenshots of the application
1. User interface to classify the message
![alt text](https://github.com/Dinusha519/disaster_response_pipeline/blob/main/images/header.png)
2. After clicking on the classify Message, we can see the categories which the message belongs to highlighted
![alt text](https://github.com/Dinusha519/disaster_response_pipeline/blob/main/images/messages.png)
3. The main page shows some graphs about training dataset used to classify the messages
![alt text](https://github.com/Dinusha519/disaster_response_pipeline/blob/main/images/distribution_categories.png)
![alt text](https://github.com/Dinusha519/disaster_response_pipeline/blob/main/images/distribution_genre.png)

