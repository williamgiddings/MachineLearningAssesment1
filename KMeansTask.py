import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

centroids = {}
centroid_colours = ['r', 'g', 'm', 'b', 'c']

data = pd.read_csv("Data/Task2 - dataset - dog_breeds.csv")
dog_dataset = pd.DataFrame({
    'height': data['height'].values,
    'leg length': data['leg length'].values,
    'tail length': data['tail length'].values,
    'nose circumference': data['nose circumference'].values,
    'label': np.full(data['height'].shape, -1)})

np.random.seed(1)

def GetDataMinMax(dataset):
    return np.amin(dataset), np.amax(dataset)


def compute_euclidean_distance(vec_1, vec_2):
    distance = np.linalg.norm(np.array(vec_1) - np.array(vec_2))
    return distance


def initialise_centroids(dataset, k):
    centroids = {}
    for centroid in range(k):
        centroids[centroid] = tuple(
            np.random.uniform(GetDataMinMax(dataset[i])[0], GetDataMinMax(dataset[i])[1])
            for i in dataset.keys().values[0:-1]
        )
    return centroids

def update(dataset):
    global centroids

    for centroid in centroids.keys():
        centroids[centroid] = tuple(
            np.mean(dataset[dataset['label'].values == centroid][i])
            for i in dataset.keys().values[0:-1]
        )
    return centroids


def kmeans(dataset, k):
    global centroids

    # Assignment Step
    cluster_assigned = np.full(dataset['height'].values.shape, -1) # initialise the assigned clusters to -1
    for data_index in range(len(dataset['height'].values)): # for each data point
        closest_centroid = -1   # initialise closest centroid as -1 so there is -
        closest_distance = -1   # - no confusion on processed/unprocessed data points
        for centroid_index, centroid in enumerate(centroids.values()): # for each centroid, get distance to data point
            distance = compute_euclidean_distance(
                centroid,
                (dataset['height'].values[data_index],
                 dataset['leg length'].values[data_index],
                 dataset['tail length'].values[data_index],
                 dataset['nose circumference'].values[data_index]
                 ))

            # if our datapoint is unprocessed or the new centroid distance is closes than our current closest
            if distance < closest_distance or closest_centroid == -1:
                closest_centroid = centroid_index
                closest_distance = distance
        # store new label for datapoint
        dataset['label'].values[data_index] = closest_centroid
        cluster_assigned[data_index] = closest_centroid

    # Update Step
    centroids = update(dataset)

    return centroids, cluster_assigned


def plot_dataset(dataset, centroids, x_axis_name, y_axis_name, k):
    plt.figure()
    plt.title('K='+str(k), fontsize=20, loc='right')
    plt.title(x_axis_name+' x '+y_axis_name, fontsize=20)
    axes = plt.gca()
    axes.set_xlim([0, 9])
    axes.set_ylim([0, 9])

    for data_index in range(len(dataset['height'].values)):
        colour = centroid_colours[dataset['label'].values[data_index]]
        plt.plot(
            dataset[x_axis_name].values[data_index],
            dataset[y_axis_name].values[data_index],
            colour + 'x')
    for index, centroid in enumerate(centroids.values()):
        x = [idx for idx, key in enumerate(dataset.keys()) if key == x_axis_name][0] #get index of x axis name
        y = [idx for idx, key in enumerate(dataset.keys()) if key == y_axis_name][0] #get index of y axis name
        plt.plot(centroid[x], centroid[y], centroid_colours[index] + '^')
        plt.text(centroid[x], centroid[y], 'Centroid: ' + str(index+1), horizontalalignment='center', verticalalignment='center', fontsize=11)

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    filename = x_axis_name+' x '+y_axis_name + str(k) + '.jpg'
    plt.savefig(filename)
    os.startfile(filename)


def get_error(dataset, centroids):
    distance = 0
    for centroid_index, centroid in enumerate(centroids.values()):
        centroid_distance = 0
        total_in_cluster = 0
        for data_index, feature in enumerate(dataset[dataset['label'].values == centroid_index]):
            total_in_cluster+=1
            centroid_distance += compute_euclidean_distance(
                centroid,
                (dataset['height'].values[data_index],
                 dataset['leg length'].values[data_index],
                 dataset['tail length'].values[data_index],
                 dataset['nose circumference'].values[data_index]
                 ))
        mean_squared = (centroid_distance / total_in_cluster)**2
        distance += np.sqrt(mean_squared)

    return distance


def plot_objective(iterations, k):
    plt.figure()
    plt.title('K=' + str(k), fontsize=20)
    plt.xlabel('iteration step')
    plt.ylabel('objective function value')
    plt.plot(range(1, len(iterations)+1), np.array(iterations), 'bo-')
    filename = 'iterations' + str(k) + '.jpg'
    plt.savefig(filename)
    os.startfile(filename)


def run(dataset, k):
    global centroids

    centroids = initialise_centroids(dataset, k)
    centroids, cluster_assigned = kmeans(dataset, k)

    iterations = []
    while True:
        old_cluster_assigned = cluster_assigned
        rmse = get_error(dataset, centroids)
        iterations.append(rmse)
        centroids, cluster_assigned = kmeans(dataset, k)
        if np.array_equal(cluster_assigned, old_cluster_assigned):
            break

    plot_dataset(dataset, centroids, 'height', 'leg length', k)
    plot_dataset(dataset, centroids, 'height', 'tail length', k)
    plot_objective(iterations, k)


run(dog_dataset, 3)
run(dog_dataset, 2)
