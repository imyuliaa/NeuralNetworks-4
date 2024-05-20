import random
import requests
import csv
import re
from io import StringIO
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from centroid_neural_networks import centroid_neural_net

url1 = "https://web.archive.org/web/20230316224609/https://cs.joensuu.fi/sipu/datasets/unbalance2.txt"
url2 = "https://web.archive.org/web/20230316224609/https://cs.joensuu.fi/sipu/datasets/D31.txt"
url3 = "https://cs.joensuu.fi/sipu/datasets/overlap.txt"

def getData(url):
	response = requests.get(url)
	response.raise_for_status()

	# Replace multiple tabs with a single tab
	text = re.sub(r"\t+", " ", response.text)

	# Use StringIO to convert the string into a file-like object for csv.reader
	csv_file = StringIO(text)

	# Use csv.reader to read the CSV data with two values per line
	reader = csv.reader(csv_file, delimiter=" ")

	# Create a list to store the data
	data = []
	for row in reader:
		values = list(map(float, row[:2]))
		data.append(values)

	# Convert the list to a NumPy array
	return np.array(data)

def getResult(data, num_clusters):
	plt.figure(figsize=(8, 8))
	plt.scatter(data[:, 0], data[:, 1], s=50)
	plt.title('Input Data')

	# POSC-based Clustering Algorithm Demo
	start_time = time.time()
	centroids, labels = centroid_neural_net(data, num_clusters, max_iteration=100)
	end_time = time.time()
	proc_time = end_time - start_time
	print('Speed:', proc_time)

	# Track results
	plt.figure(figsize=(8, 8))
	plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
	plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red')
	plt.title('Clustering Results')
	plt.show()

data1 = getData(url1)
data2 = getData(url2)
data3 = getData(url3)

print("Result for data1:")
getResult(data1, 8)
print("Result for data2:")
getResult(data2, 15)
print("Result for data3:")
getResult(data3, 6)

# 2 Part
print("\nPart2:")

def readData():
    # Read Books.csv
    books_reader = read_csv_file("Books.csv", ['isbn', 'title', 'author', 'year', 'publisher'])
    books_reader = books_reader[(books_reader['year'] >= '1975') & (books_reader['year'] <= '2020')]

    # Read Users.csv
    users_reader = read_csv_file("Users.csv", ['user_id', 'location', 'age'])

    # Read Ratings.csv
    ratings_reader = read_csv_file("Ratings.csv", ['user_id', 'isbn', 'rating'])

    # Merge DataFrames
    df_merged = merge_dataframes(ratings_reader, users_reader, books_reader)

    return df_merged, books_reader

def read_csv_file(file_path, column_names):
    with open(file_path, newline="\n", encoding="latin-1") as file:
        reader = csv.reader(file, delimiter=";", quotechar='"')
        next(reader)  # Skip header
        data = list(reader)

    # Modify column names and remove some columns
    column_indices_to_keep = range(len(column_names))
    new_data = [
        [row[i] for i in column_indices_to_keep] for row in data
    ]

    df = pd.DataFrame(new_data, columns=column_names)
    return df


def merge_dataframes(ratings, users, books):
    df_merged = pd.merge(ratings, users, on='user_id')
    df_merged = pd.merge(df_merged, books, on='isbn')
    return df_merged

def filterUsersBooks(user_min, book_min, df_merged):
    print(f"\nFiltering users with rated {user_min} books at least and books with {book_min} ratings at least")
    user_counts = df_merged['user_id'].value_counts()
    book_counts = df_merged['isbn'].value_counts()

    filtered_users = user_counts[user_counts >= user_min].index
    filtered_books = book_counts[book_counts >= book_min].index

    df_filtered = df_merged[df_merged['user_id'].isin(filtered_users) & df_merged['isbn'].isin(filtered_books)]

    ratings_matrix = df_filtered.pivot_table(index='user_id', columns='isbn', values='rating', fill_value=0, aggfunc=[lambda x: int(x.iloc[0])])
    return ratings_matrix

def CentNNRecomendation(ratings_matrix, books_reader):
    print("CentNN:")
    # Convert ratings_matrix to numpy array
    ratings_matrix_array = np.array(ratings_matrix)

    # Set the number of clusters (adjust as needed)
    num_clusters = 10

    # Apply CentNN algorithm
    w, cluster_indices = centroid_neural_net(ratings_matrix_array, num_clusters)

    # Add cluster information to the DataFrame
    ratings_matrix['cluster'] = cluster_indices
    ratings_matrix['id'] = range(len(cluster_indices))

    random_users = []

    for cluster_id in range(num_clusters):
        users_in_cluster = ratings_matrix[ratings_matrix['cluster'] == cluster_id]
        if len(users_in_cluster) > 0:
            if len(users_in_cluster) == 1:
                random_users.append(users_in_cluster['id'].to_list()[0])
                print(f"\nДля користувача {users_in_cluster.index.to_list()[0]} нема рекомендованих книг.")
            else:
                recommended_books = []
                cluster_books_rating = {}
                for j in range(len(users_in_cluster)):
                    books_dict = dict(users_in_cluster.iloc[j])
                    for k in books_dict.keys():
                        if k[1] != '':
                            value = books_dict[k]
                            if value > 0:
                                if k[1] not in cluster_books_rating:
                                    cluster_books_rating[k[1]] = (value, 1)
                                else:
                                    rating = cluster_books_rating[k[1]]
                                    num = rating[1]
                                    cluster_books_rating[k[1]] = ((rating[0] * num + value) / num + 1, num + 1)
                cluster_books_rating = dict(sorted(cluster_books_rating.items(), key=lambda x: x[1][0], reverse=True))
                id_random_user = int(random.uniform(0, len(users_in_cluster)))
                random_user = users_in_cluster[
                    users_in_cluster.index == users_in_cluster.index.to_list()[id_random_user]]
                random_users.append(random_user['id'].to_list()[0])
                books_read = []
                books_dict = dict(users_in_cluster.iloc[id_random_user])
                for k in books_dict.keys():
                    if k[1] != '':
                        if books_dict[k] > 0:
                            books_read.append(k[1])
                for book in cluster_books_rating.keys():
                    if book not in books_read:
                        recommended_books.append(book)
                        if len(recommended_books) == 10:
                            break
                print(f"\nДля користувача {random_user.index.to_list()[0]} рекомендовані такі книги:")
                for i in range(len(recommended_books)):
                    info = books_reader[books_reader['isbn'] == recommended_books[i]]
                    print(f"{i + 1}. {info['title'].to_list()[0]}, {info['author'].to_list()[0]}, {info['year'].to_list()[0]}, {info['publisher'].to_list()[0]}")
        else:
            print("\nНемає користувачів у кластері ", cluster_id, '.')
    return random_users

def KNNRecomendation(ratings_matrix, books_reader, random_users):
    print("\n\nKNN:")
    # Convert ratings_matrix to numpy array
    ratings_matrix_array = np.array(ratings_matrix)

    # Set the number of neighbors for KNN (adjust as needed)
    num_neighbors = 25

    # Fit KNN model
    knn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
    knn_model.fit(ratings_matrix_array)

    # Find the nearest neighbors for each user in the cluster
    nearest_neighbors = knn_model.kneighbors(ratings_matrix_array, return_distance=False)

    for user in random_users:
        users_in_cluster = ratings_matrix[ratings_matrix['id'].isin(nearest_neighbors[user])]
        recommended_books = []
        cluster_books_rating = {}
        for j in range(len(users_in_cluster)):
            books_dict = dict(users_in_cluster.iloc[j])
            for k in books_dict.keys():
                if k[1] != '':
                    value = books_dict[k]
                    if value > 0:
                        if k[1] not in cluster_books_rating:
                            cluster_books_rating[k[1]] = (value, 1)
                        else:
                            rating = cluster_books_rating[k[1]]
                            num = rating[1]
                            cluster_books_rating[k[1]] = ((rating[0] * num + value) / num + 1, num + 1)
        cluster_books_rating = dict(sorted(cluster_books_rating.items(), key=lambda x: x[1][0], reverse=True))
        random_user = users_in_cluster[users_in_cluster['id'] == user]
        books_read = []
        books_dict = dict(users_in_cluster[users_in_cluster['id'] == user].iloc[0])
        for k in books_dict.keys():
            if k[1] != '':
                if books_dict[k] > 0:
                    books_read.append(k[1])
        for book in cluster_books_rating.keys():
            if book not in books_read:
                recommended_books.append(book)
                if len(recommended_books) == 10:
                    break
        if (len(recommended_books) == 0):
            print(f"\nДля користувача {random_user.index.to_list()[0]} нема рекомендованих книг.")
        else:
            print(f"\nДля користувача {random_user.index.to_list()[0]} рекомендовані такі книги:")
            for i in range(len(recommended_books)):
                info = books_reader[books_reader['isbn'] == recommended_books[i]]
                print(f"{i + 1}. {info['title'].to_list()[0]}, {info['author'].to_list()[0]}, {info['year'].to_list()[0]}, {info['publisher'].to_list()[0]}")

# Main program
df_merged, books_reader = readData()

configs = [
    (40, 300),
    (40, 250),
    (40, 200),
    (15, 100),
    (10, 50)
]

for config in configs:
    user_min, book_min = config
    ratings_matrix = filterUsersBooks(user_min, book_min, df_merged)
    random_users = CentNNRecomendation(ratings_matrix, books_reader)
    KNNRecomendation(ratings_matrix, books_reader, random_users)