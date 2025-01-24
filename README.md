# Recommendation-System
This Python program implements a simple movie recommendation system using the LightFM library. The model is trained on the MovieLens dataset to suggest movies to users based on their preferences. It was created under the guidance of Siraj Raval.

# Key Features:
## Data Handling:
1. Utilizes the fetch_movielens function from lightfm.datasets to fetch and format the MovieLens dataset.
2. Filters the dataset to include only movies with a minimum rating of 4.0.

## Model Creation and Training:
1. Implements the LightFM recommendation model with the WARP (Weighted Approximate-Rank Pairwise) loss function, optimized for implicit feedback.
2. Trains the model on the MovieLens dataset over 30 epochs using 2 threads for efficiency.

## Recommendation System:
1. Generates personalized movie recommendations for a user based on their interaction history and predicted preferences.
2. Displays movies the user has already liked and suggests new movies they might enjoy.

## Interactive User Input:
Prompts the user to input their user ID to generate recommendations tailored to them.

# Requirements:
## Libraries: 
The program requires the following Python libraries:
1. numpy
2. lightfm
3. scipy (automatically installed with LightFM)

# How It Works:
## Data Loading:
The program fetches the MovieLens dataset and separates it into training and testing sets.

## Model Training:
Trains a WARP-based LightFM model on the training data.

## Recommendation Generation:
For a given user ID, the program:
1. Identifies movies the user has already rated positively.2
2. Predicts new movies they are likely to enjoy.
3. Ranks movies by predicted score and prints the top three known positives and top three recommendations.
