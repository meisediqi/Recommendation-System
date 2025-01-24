import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it
data = fetch_movielens(min_rating = 4.0)

# Print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# Create model
model = LightFM(loss = 'warp')

# Train model
model.fit(data['train'], epochs = 30, num_threads = 2)

def sampleRecommendations(model, data, userIds):

    # Number of users and movies in training data
    nUsers, nItems = data['train'].shape

    # Generate recommendations for each user we input
    for userId in userIds:

        #movies they already like
        knownPositives = data['item_labels'][data['train'].tocsr()[userId].indices]
        user_id_1 = data['train'].tocsr()[userId]
        knownPositives = data['item_labels'][user_id_1.indices]

        # Movies our model predicts they will like
        scores = model.predict(userId, np.arange(nItems))

        # Rank them in order of most liked to least
        topItems = data['item_labels'][np.argsort(-scores)]

        # Print out the results
        print("User %s" % userId)
        print("     Known Positives:")

        for x in knownPositives[:3]:
            print("         %s" % x)
        print("     Recommended:")

        for x in topItems[:3]:
            print("         %s" % x)

user = input("Enter your user ID please: ")
sampleRecommendations(model, data, [int(user)])
