"""The solver for VRP(Vehicle Routing Problem).
The VRP(or mTSP) is NP-hard problem, therefore this algorithm uses heuristic approach as below:
author: louie
"""

import math
import itertools
import time
from collections import namedtuple
import numpy as np
import numpy.linalg as LA
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
    locations_df = pd.read_csv(input_file_csv, delimiter=',', header=None, names=['latitude', 'longitude', 'is_customer'])
    # print(locations_df)

    is_warehouse = locations_df.is_customer == 0
    subset_warehouse = locations_df[is_warehouse].dropna()
    subset_customer = locations_df[~is_warehouse].dropna()
    print("warehouse: %s" % subset_warehouse)
    print("# of customers: %s" % len(subset_customer))

    warehouses = []
    customers = []
    warehouses.append(Warehouse(int(0), float(subset_warehouse.values[0][0]), float(subset_warehouse.values[0][1])))

    for i in range(0, len(subset_customer)):
        x = subset_customer.values[i][0]
        y = subset_customer.values[i][1]
        customers.append(Customer(int(i+1), int(1), float(x), float(y)))

    return warehouses, customers


def distance(point1, point2):
    """
    Calculates the Euclidean distance between two location coordinates.
    :param point1:
    :param point2:
    :return:
    """
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def tour_distance(tour, points):
    """
    Calculates the total tour distance between multiple coordinates.
    :param tour:
    :param points:
    :return:
    """
    return sum(distance(points[tour[i - 1]], points[tour[i]]) for i in range(len(tour)))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


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
            print('{0}: {1} customers'.format(label_name, len(vehicle.customers)))
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


def plot_vehicle_tour(vehicle, vehicle_tour):
    """
    Plots the vehicle's tour.
    :param vehicle:
    :param vehicle_tour:
    :return:
    """
    # Plots the warehouse
    plt.scatter(vehicle_tour[0].x, vehicle_tour[0].y, s=240, c='r', marker='s', label='warehouse')

    cmap = plt.cm.get_cmap('Dark2')
    prev_coords = vehicle_tour[0]
    arrow_head_width = 0.0004
    arrow_head_length = 0.0006
    for i in range(1, len(vehicle_tour) - 1):
        color = cmap(1.0 * (i + 1) / (len(vehicle_tour) - 2))
        customer = vehicle_tour[i]
        label_name = 'customer' + str(int(customer.index))
        # Plot the customer
        plt.scatter(customer.x, customer.y, s=240, c=color, label=label_name)

        dx = customer.x - prev_coords.x
        dy = customer.y - prev_coords.y

        if i == 1:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            xaxis_width = (xmax - xmin) / len(plt.xticks())
            yaxis_width = (ymax - ymin) / len(plt.yticks())
            arrow_head_width = yaxis_width / (len(plt.yticks()) * 20)
            arrow_head_length = xaxis_width / (len(plt.xticks()) * 20)
            arrow_width = 0.000001
            print('arrow_head_width:{0}, arrow_head_length:{0}'.format(str(arrow_head_width), str(arrow_head_length)))

        plt.arrow(prev_coords.x, prev_coords.y, dx, dy,
                  head_width=arrow_head_width, head_length=arrow_head_length, width=arrow_width,
                  fc='k', ec='k')
        prev_coords = customer

    dx_home = vehicle_tour[0].x - prev_coords.x
    dy_home = vehicle_tour[0].y - prev_coords.y
    plt.arrow(prev_coords.x, prev_coords.y, dx_home, dy_home,
              head_width=arrow_head_width, head_length=arrow_head_length, width=arrow_width,
              fc='k', ec='k')

    plt.title('Vehicle ' + str(vehicle.index + 1))
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
    """
    Clusters the customers.
    :param num_clusters:
    :param customers:
    :return:
    """
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
    Initializes and sorts the cluster centroids(i.e. vehicles) by the nearest order of
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

    # Sort by distance ascending(i.e. from the nearest from the warehouse)
    # ordered_vehicles = sorted(ordered_vehicles, key=lambda x: x.attributes)
    # Sort by distance descending(i.e. from the farthest from the warehouse)
    ordered_vehicles = sorted(ordered_vehicles, key=lambda x: x.attributes, reverse=True)

    # print('ordered vehicles(centroids): %s' % ordered_vehicles)

    return ordered_vehicles


def assign_customers_to_vehicles(customers, vehicles, max_capacity):
    """
    Assigns the customers to vehicles.
    One customer will be allocated only into one vehicle.
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
        # if shortage_capacity > 0:
        #     remaining_capacity += additional_capacity_vehicle
        #     shortage_capacity -= additional_capacity_vehicle

        # Assign customers in the cluster first
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

        # Calculate the Euclidean distance between customer and centroid of cluster(= centroid of vehicle)
        for customer in customers:
            dist = distance(customer, vehicle)
            ordered_customers_tuple.append(OrderedCustomer(dist, customer))

        # Sort by distance ascending(i.e. the nearest customers from vehicle)
        ordered_customers_tuple = sorted(ordered_customers_tuple, key=lambda x: x.distance)
        # Assign customers in the remaining by nearest distance order
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

    # Assign the remaining customers to the nearest centroid(i.e. vehicle) if the vehicle capacity is available.
    print('number of unassigned customers = %d' % len(customers))
    if len(customers) > 0:
        unassigned_customers = []
        for customer in customers:
            nearest_vehicle = None
            min_distance = np.inf
            for vehicle in vehicles_:
                dist = distance(customer, vehicle)
                if dist <= min_distance and len(vehicle.customers) <= vehicle.capacity + additional_capacity_vehicle:
                    min_distance = dist
                    nearest_vehicle = vehicle
                    # print('min distance: %s' % str(min_distance))

            print('nearest vehicle: %s' % str(nearest_vehicle.index + 1))
            if nearest_vehicle is not None:
                nearest_vehicle.customers.append(customer)
                unassigned_customers.append(customer)

        for customer in unassigned_customers:
            customers.remove(customer)

    # Should be zero
    print('number of remaining customers = %d' % len(customers))
    # Check that the number of remaining customers is zero.
    assert len(customers) == 0
    return vehicles_


def greedy(points):
    """
    Greedy optimization.
    :param points:
    :return:
    """
    point_count = len(points)
    coords = np.array([(point.x, point.y) for point in points])
    tour = [0]
    candidates = set(range(1, point_count))
    while candidates:
        curr_point = tour[-1]
        nearest_neighbor = None
        nearest_dist = np.inf
        for neighbor in candidates:
            if LA.norm(coords[curr_point] - coords[neighbor]) < nearest_dist:
                nearest_neighbor = neighbor
                nearest_dist = LA.norm(coords[curr_point] - coords[neighbor])

        tour.append(nearest_neighbor)
        candidates.remove(nearest_neighbor)
    return tour_distance(tour, points), 0, tour


def swap(tour, dist, start, end, points):
    """
    Swap the points.
    :param tour:
    :param dist:
    :param start:
    :param end:
    :param points:
    :return:
    """
    new_tour = tour[:start] + tour[start:end + 1][::-1] + tour[end + 1:]

    new_distance = dist - \
                   (distance(points[tour[start - 1]], points[tour[start]]) +
                    distance(points[tour[end]], points[tour[(end + 1) % len(tour)]])) + \
                   (distance(points[new_tour[start - 1]], points[new_tour[start]]) +
                    distance(points[new_tour[end]], points[new_tour[(end + 1) % len(tour)]]))
    return new_tour, new_distance


def two_opt(points):
    """
    2-opt optimization.
    :param points:
    :return:
    """
    point_count = len(points)
    best_distance, _, best_tour = greedy(points)
    improved = True
    t = time.clock()
    while improved:
        improved = False
        for start, end in itertools.combinations(range(point_count), 2):
            curr_tour, curr_distance = swap(best_tour, best_distance, start, end, points)
            if curr_distance < best_distance:
                best_tour = curr_tour
                best_distance = curr_distance
                improved = True
                break
        if time.clock() - t >= 4 * 3600 + 59 * 60:
            improved = False
    return tour_distance(best_tour, points), 0, best_tour


def plan_vehicle_routing(warehouse, vehicle):
    """
    Optimizes the vehicle routing.
    :param warehouse:
    :param vehicle:
    :return:
    """
    points = []
    points.append(warehouse)
    for customer in vehicle.customers:
        points.append(customer)

    # Greedy solution (nearest neighbor)
    # Starts from 0, add nearest neighbor to the cycle at each step
    # Generally acceptable, but can be arbitrarily bad
    # best_distance, opt, best_tour = greedy(points)

    # 2-opt solution
    best_distance, opt, best_tour = two_opt(points)
    print('* best distance: {0}'.format(str(best_distance)))
    print('* best tour: {0}'.format(best_tour))

    # Calculate the cost of the solution
    cost = best_distance
    # cost += distance(warehouse, points[best_tour[0]])
    # cost += distance(warehouse, points[best_tour[len(best_tour) - 1]])
    print('* total cost: {0}'.format(str(cost)))

    # Make directed cycle graph starting from the warehouse and returning to the warehouse.
    graph = []
    warehouse_index = len(best_tour)
    index = 0
    lefthand_vertices_of_warehouse = []
    for vertex in best_tour:
        if vertex == 0:  # Start from the warehouse
            warehouse_index = index
            graph.append(vertex)
        elif index > warehouse_index:
            graph.append(vertex)
        else:
            lefthand_vertices_of_warehouse.append(vertex)

        index += 1
    for vertex in lefthand_vertices_of_warehouse:
        graph.append(vertex)

    graph.append(0)  # Edge from the last vertex to the warehouse

    solution = []
    for vertex in graph:
        solution.append(points[vertex])

    return cost, opt, solution


def solve_vrp(warehouses, customers, is_plot):
    """
    Solves the vehicle routing problem.
    :param warehouses:
    :param customers:
    :param is_plot:
    :return:
    """
    # 1. EDA for input data.
    if is_plot is True:
        plot_input_data(warehouses, customers)
    clusters, centroids = cluster_customers(NUM_VEHICLES, customers)
    if is_plot is True:
        plot_clusters(warehouses, customers, centroids, clusters, [])
        # plot_clusters(warehouses, customers, centroids, clusters, [0, 1, 2, 3, 4])
        # plot_clusters(warehouses, customers, centroids, clusters, [5, 6, 7, 8, 9])
        # plot_clusters(warehouses, customers, centroids, clusters, [10, 11, 12, 13, 14])
        # plot_clusters(warehouses, customers, centroids, clusters, [15, 16, 17, 18, 19])
        # plot_clusters(warehouses, customers, centroids, clusters, [20, 21, 22, 23, 24])

    # 2. Detect the outliers.
    # If the distance between global centroid and customer is outside of 85% percentile distance statistice,
    # set as outlier.
    inliers, outliers = detect_outliers(customers, 100)

    # 3. Find the centroids for 25 vehicles only with inliers.
    clusters, centroids = cluster_customers(NUM_VEHICLES, inliers)
    if is_plot is True:
        plot_clusters(warehouses, inliers, centroids, clusters, [])

    # 4. Initialize and sort the cluster centroids by the nearest order of the distance
    # between the warehouse and centroid.
    # i.e. The sorted cluster centroids are the vehicles to assign the customers.
    # We assume that each vehicle's max capacity is 22 (i.e. capacity = number of customers / number of vehicles)
    max_capacity = len(customers) / NUM_VEHICLES
    print('max capacity = %d' % max_capacity)
    vehicles = init_vehicles(warehouses, centroids, clusters, inliers, max_capacity)

    # 5. Assign all the customers into each cluster centroid(i.e. vehicle) by the order of the centroids.
    # Subject to the constraint of vehicle's capacity.
    vehicles = assign_customers_to_vehicles(customers, vehicles, max_capacity)

    if is_plot is True:
        plot_assigned_customers(warehouses, vehicles, [])
        for i in range(0, NUM_VEHICLES):
            if len(vehicles[i].customers) > 0:
                plot_assigned_customers(warehouses, vehicles, [i])

    # 6. Optimize the vehicle routing tour.
    output_data = ''
    total_cost = 0
    for vehicle in vehicles:
        obj, opt, vehicle_tour = plan_vehicle_routing(warehouses[0], vehicle)
        total_cost += obj
        output_data += ' '.join([str(int(vertex.index)) for vertex in vehicle_tour]) + '\n'
        if is_plot is True:
            plot_vehicle_tour(vehicle, vehicle_tour)

    output_data = '%.5f' % total_cost + ' ' + str(0) + '\n' + output_data
    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        input_file = sys.argv[1].strip()
        is_plot = str2bool(sys.argv[2].strip())

        warehouses, customers = read_csv_input_data(input_file)
        output = solve_vrp(warehouses, customers, is_plot)
        print output
    else:
        print('This requires an input file. (eg. python solver.py ../data/locations.csv true)')



