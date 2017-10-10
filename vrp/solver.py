"""The solver for VRP(Vehicle Routing Problem).
The VRP(or mTSP) is NP-hard problem, therefore this algorithm uses heuristic approach as below:
author: louie
"""

import math
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


NUM_VEHICLES = 25

# Defines the data models.
Warehouse = namedtuple("Warehouse", ['index', 'x', 'y'])
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])
Vehicle = namedtuple("Vehicle", ['index', 'capacity', 'cost', 'x', 'y', 'customers', 'attributes'])


def read_csv_input_data(input_file_csv):
    """
    Reads csv input data file.
    :param input_file_csv:
    :return:
    """
    # Load the data
    locations_df = pd.read_csv(input_file_csv, delimiter=',', header=None, names=['latitude', 'longitude', 'customer'])
    # print(locations_df)

    depot = locations_df.customer == 0
    subset_warehouse = locations_df[depot].dropna()
    subset_customer = locations_df[~depot].dropna()
    print("warehouse: %s" % subset_warehouse)
    print("# of customers: %s" % len(subset_customer))

    warehouses = []
    customers = []
    warehouses.append(Warehouse(int(0), float(subset_warehouse.values[0][0]), float(subset_warehouse.values[0][1])))

    for i in range(0, len(subset_customer)):
        x = subset_customer.values[i][0]
        y = subset_customer.values[i][1]
        customers.append(Customer(int(i), int(1), float(x), float(y)))

    return warehouses, customers


def distance(customer1, customer2):
    """
    Calculates the Euclidean distance between two location coordinates.
    :param customer1:
    :param customer2:
    :return:
    """
    return math.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)


def plot_input_data(warehouses, customers):
    """
    Plots the input data.
    :param warehouses:
    :param customers:
    :return:
    """
    coords_warehouses = np.array([[c.x, c.y] for c in warehouses])
    coords_customers = np.array([[c.x, c.y] for c in customers])

    plt.scatter(coords_customers[:, 0], coords_customers[:, 1], s=60, c='b', label='customer')
    plt.scatter(coords_warehouses[:, 0], coords_warehouses[:, 1], s=120, c='r', label='warehouse')

    plt.legend()
    plt.grid()
    plt.show()


def plot_clusters(warehouses, customers, centroids, clusters, cluster_indexes_to_show):
    """
    Plots the clusters.
    :param warehouses:
    :param customers:
    :param centroids:
    :param clusters:
    :param cluster_indexes_to_show:
    :return:
    """
    coords_warehouses = np.array([[c.x, c.y] for c in warehouses])
    coords_customers = np.array([[c.x, c.y] for c in customers])

    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]

    cmap = plt.cm.get_cmap('Dark2')
    for i in range(len(cluster_labels)):
        if (i in cluster_indexes_to_show) or (cluster_indexes_to_show == []):
            color = cmap(1.0 * cluster_labels[i] / n_clusters)
            label_name = 'cluster' + str(i+1)
            # Plots the customers by each cluster.
            plt.scatter(coords_customers[clusters == i, 0], coords_customers[clusters == i, 1], s=60, c=color,
                        label=label_name)
            # Plots the centroid of each cluster.
            plt.scatter(centroids[i, 0], centroids[i, 1], s=240, c='b', marker='x', linewidths=1)

    # Plots the warehouse.
    plt.scatter(coords_warehouses[:, 0], coords_warehouses[:, 1], s=120, c='r', marker='s', label='warehouse')

    plt.legend()
    plt.grid()
    plt.show()

    return


def plot_assigned_customers(warehouses, vehicles, vehicle_indexes_to_show):
    """
    Plots the assigned customers per vehicle.
    :param warehouses:
    :param vehicles:
    :param vehicle_indexes_to_show:
    :return:
    """
    coords_warehouses = np.array([[c.x, c.y] for c in warehouses])

    cmap = plt.cm.get_cmap('Dark2')
    for i in range(0, len(vehicles)):
        vehicle = vehicles[i]
        if (i in vehicle_indexes_to_show) or (vehicle_indexes_to_show == []):
            color = cmap(1.0 * (i + 1) / len(vehicles))
            label_name = 'vehicle' + str(i+1)
            # Plots the allocated customers by each vehicle.
            coords_customers = np.array([[c.x, c.y] for c in vehicle.customers])
            # print('{0}: {1}'.format(label_name, coords_customers))
            print('{0}'.format(label_name))
            plt.scatter(coords_customers[:, 0], coords_customers[:, 1], s=60, c=color,
                        label=label_name)
            # Plots the centroid of each cluster.
            plt.scatter(vehicle.x, vehicle.y, s=240, c='b', marker='x', linewidths=1)

    # Plots the warehouse.
    plt.scatter(coords_warehouses[:, 0], coords_warehouses[:, 1], s=120, c='r', marker='s', label='warehouse')

    plt.legend()
    plt.grid()
    plt.show()

    return


def detect_outliers(customers, percentile):
    """
    Detects the outliers.
    :param customers:
    :param percentile:
    :return:
    """
    # Find the global one centroid.
    clusters, centroids = cluster_customers(1, customers)
    centroid = Customer(0, 0, centroids[0][0], centroids[0][1])

    # Calculate the Euclidean distance between customer and centroid for all the customers.
    distances = []
    for customer in customers:
        dist = distance(centroid, customer)
        distances.append(dist)

    # Calculate the average distance.
    avg_distance = np.mean(distances)
    threshold_distance = np.percentile(distances, percentile)
    print('average distance from centroid = {0:.5f}'.format(avg_distance))
    print('threshold distance from centroid = {0:.5f}'.format(threshold_distance))

    # Detect the outliers if the Euclidean distance between customer and centroid is greater than average distance.
    inliers = []
    outliers = []
    for i in range(len(distances)):
        if distances[i] > threshold_distance:
            outliers.append(customers[i])
        else:
            inliers.append(customers[i])

    print('outliers: {0} of {1} ({2:.2f})'.format(len(outliers), len(customers), len(outliers)/float(len(customers))))
    return inliers, outliers


def cluster_customers(num_clusters, customers):
    kmeans = KMeans(n_clusters=num_clusters,
                    init='k-means++',   # 'random', 'k-means++'
                    n_init=10,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)

    coords = np.array([[c.x, c.y] for c in customers])
    y_km = kmeans.fit_predict(coords)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    centroids = kmeans.cluster_centers_
    print('clusters: %s' % cluster_labels)
    print('centroid: %s' % centroids)

    return y_km, centroids


def init_vehicles(warehouses, centroids, clusters, customers, max_capacity):
    """
    Initializes and sorts the cluster centroids(i.e. vehicles) by the closest order of
    the distance between the warehouse and centroid.
    :param warehouses:
    :param centroids:
    :param clusters:
    :param customers:
    :param max_capacity:
    :return:
    """

    # Calculate the Euclidean distance between warehouse and each centroid.
    ordered_vehicles = []
    i = 0
    for centroid in centroids:
        # Get the customers in a cluster
        customers_in_cluster = []
        customers_array_in_cluster = np.array(customers)[clusters == i]
        for c in customers_array_in_cluster:
            customers_in_cluster.append(Customer(c[0], c[1], c[2], c[3]))

        dist = distance(warehouses[0], Customer(0, 0, centroid[0], centroid[1]))
        vehicle = Vehicle(i, max_capacity, 0, centroid[0], centroid[1], customers_in_cluster, dist)
        ordered_vehicles.append(vehicle)
        i += 1

    # Sort by distance ascending.
    ordered_vehicles = sorted(ordered_vehicles, key=lambda x: x.attributes)

    # print('ordered vehicles(centroids): %s' % ordered_vehicles)

    return ordered_vehicles


def assign_customers_to_vehicles(customers, vehicles, max_capacity):
    """
    Assigns the customers to vehicles.
    One customer will bd allocated only into one vehicle.
    :param customers:
    :param vehicles:
    :param max_capacity:
    :return:
    """
    vehicles_ = []

    shortage_capacity = len(customers) - len(vehicles) * max_capacity
    if shortage_capacity > 0:
        # Allocate the additional shortage capacity to the first 40% vehicles.
        additional_capacity_vehicle = int(shortage_capacity / (len(vehicles) * 0.4))
        print('shortage capacity: {0}, additional capacity per vehicle: {1}'.format(shortage_capacity,
                                                                                    additional_capacity_vehicle))

    i = 0
    for vehicle in vehicles:
        ordered_customers_tuple = []
        OrderedCustomer = namedtuple("ordered_customer", ['distance', 'data'])

        assigned_customers = []

        customers_in_cluster = vehicle.customers
        remaining_capacity = vehicle.capacity
        if shortage_capacity > 0:
            remaining_capacity += additional_capacity_vehicle
            shortage_capacity -= additional_capacity_vehicle
        # Assign customers in cluster first
        for customer_in_cluster in customers_in_cluster:
            if remaining_capacity == 0:
                break
            for customer in customers:
                if customer.index == customer_in_cluster.index:
                    assigned_customers.append(customer_in_cluster)
                    customers.remove(customer)
                    remaining_capacity -= 1
                    print('[assign(A)-vehicle{0}] remaining customers: {1}, remaining capacity: {2}'
                          .format(int(i+1), len(customers), remaining_capacity))
                    break

            # This occurs error because of customer_in_cluster object is different from the object in customers.
            # customers.remove(customer_in_cluster)
            # That's why removing customer explicitly as below.
            # customers = [x for x in customers if x.index != customer_in_cluster.index]

        # Calculate the Euclidean distance between customer and centroid of cluster(= centroid of vehicle)
        for customer in customers:
            dist = distance(customer, Customer(0, 0, vehicle.x, vehicle.y))
            ordered_customers_tuple.append(OrderedCustomer(dist, customer))

        # Sort by distance ascending.
        ordered_customers_tuple = sorted(ordered_customers_tuple, key=lambda x: x.distance)
        # Assign customers in the remaining by closest distance order
        for j in range(0, remaining_capacity):
            customer = ordered_customers_tuple[j].data
            if j < len(ordered_customers_tuple):
                assigned_customers.append(customer)
                customers.remove(customer)
                remaining_capacity -= 1
                print('[assign(B)-vehicle{0}] remaining customers: {1}, remaining capacity: {2}'
                      .format(int(i + 1), len(customers), remaining_capacity))
                if len(customers) == 0:
                    break

        vehicle_ = Vehicle(i, len(assigned_customers), 0.0, vehicle.x, vehicle.y, assigned_customers, vehicle.attributes)
        print('* vehicle[{0}]: assigned {1} customers'.format(int(i+1), len(assigned_customers)))
        vehicles_.append(vehicle_)
        i += 1
        if len(customers) == 0:
            break

    # Should be zero.
    print('remaining customers = %d' % len(customers))
    # Check that the number of remaining customers is zero.
    assert len(customers) == 0
    return vehicles_


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        index = i - 1
        demand = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        customers.append(Customer(index, demand, x, y))

    # the depot is always the first customer in the input
    depot = customers[0]

    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []

    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    cost = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            cost += distance(depot, vehicle_tour[0])
            for i in range(0, len(vehicle_tour) - 1):
                cost += distance(vehicle_tour[i], vehicle_tour[i + 1])
                cost += distance(vehicle_tour[-1], depot)

    # prepare the solution in the specified output format
    outputData = '%.2f' % cost + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join(
            [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


def solve_vrp(warehouses, customers):
    """
    Solves the vehicle routing problem.
    :param warehouses:
    :param customers:
    :return:
    """
    # 1. EDA for input data.
    plot_input_data(warehouses, customers)
    clusters, centroids = cluster_customers(NUM_VEHICLES, customers)
    plot_clusters(warehouses, customers, centroids, clusters, [])
    # plot_clusters(warehouses, customers, centroids, clusters, [0, 1, 2, 3, 4])
    # plot_clusters(warehouses, customers, centroids, clusters, [5, 6, 7, 8, 9])
    # plot_clusters(warehouses, customers, centroids, clusters, [10, 11, 12, 13, 14])
    # plot_clusters(warehouses, customers, centroids, clusters, [15, 16, 17, 18, 19])
    # plot_clusters(warehouses, customers, centroids, clusters, [20, 21, 22, 23, 24])

    # 2. Detect the outliers.
    # If the distance between global centroid and customer is outside of 85% percentile distance statistice,
    # set as outlier.
    inliers, outliers = detect_outliers(customers, 85)

    # 3. Find the centroids for 25 vehicles only with inliers.
    clusters, centroids = cluster_customers(NUM_VEHICLES, inliers)
    plot_clusters(warehouses, inliers, centroids, clusters, [])

    # 4. Initialize and sort the cluster centroids by the closest order of the distance
    # between the warehouse and centroid.
    # i.e. The sorted cluster centroids are the vehicles to assign the customers.
    # We assume that each vehicle's max capacity is 22 (i.e. capacity = number of customers / number of vehicles)
    max_capacity = len(customers) / NUM_VEHICLES
    print('max capacity = %d' % max_capacity)
    vehicles = init_vehicles(warehouses, centroids, clusters, inliers, max_capacity)

    # 5. Assign all the customers into each cluster centroid(i.e. vehicle) by the order of the centroids.
    # Subject to the constraint of vehicle's capacity.
    vehicles = assign_customers_to_vehicles(customers, vehicles, max_capacity)

    plot_assigned_customers(warehouses, vehicles, [])
    for i in range(0, NUM_VEHICLES):
        if len(vehicles[i].customers) > 0:
            plot_assigned_customers(warehouses, vehicles, [i])


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1].strip()
        warehouses, customers = read_csv_input_data(input_file)
        solve_vrp(warehouses, customers)
    else:
        print('This requires an input file. (eg. python solver.py ../data/locations.csv)')



