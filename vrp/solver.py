"""The solver for VRP(Vehicle Routing Problem).
The VRP(or mTSP) is NP-hard problem, therefore this algorithm uses heuristic approach as below:
author: louie
"""

import math
from collections import namedtuple
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import pylab
import matplotlib.pyplot as plt

Warehouse = namedtuple("Warehouse", ['index', 'x', 'y'])
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])


def read_csv_input_data(input_file_csv):
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


def plot_input_data(warehouses, customers):
    """
    Plots the input data.
    :param warehouses:
    :param customers:
    :return:
    """

    coords_warehouses = np.array([[c.x, c.y] for c in depots])
    coords_customers = np.array([[c.x, c.y] for c in customers])

    plt.scatter(coords_customers[:, 0], coords_customers[:, 1], s=60, c='b', label='customer')
    plt.scatter(coords_warehouses[:, 0], coords_warehouses[:, 1], s=120, c='r', label='warehouse')

    plt.legend()
    plt.grid()
    plt.show()


def cluster_customers(num_clusters, customers):
    km = KMeans(n_clusters=num_clusters,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)

    coords = np.array([[c.x, c.y] for c in customers])
    y_km = km.fit_predict(coords)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    print('clusters: %s' % cluster_labels)
    # print km.labels_

    return y_km


def distance(customer1, customer2):
    """
    Calculates the Euclidean distance between two location coordinates.
    :param customer1:
    :param customer2:
    :return:
    """
    return math.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)


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


if __name__ == '__main__':
    import sys
    #
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

    if len(sys.argv) > 1:
        input_file = sys.argv[1].strip()
        depots, customers = read_csv_input_data(input_file)

        plot_input_data(depots, customers)
        cluster_customers(25, customers)



