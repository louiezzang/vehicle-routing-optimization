# Capacitated VRP - CVRP

## Problem Statement

The problem is mathematically formulated in the following way: We are given a list of locations *__N = 0 . . . n − 1__* . The location *__0__* is the warehouse location, where all of the vehicles start and end their routes. The remaining locations are customers. Each location is characterized by three values *__⟨d<sub>i</sub>, x<sub>i</sub>, y<sub>i</sub>⟩ i ∈ N__* a demand *__d<sub>i</sub>__* and a point *__x<sub>i</sub>,y<sub>i</sub>__*. The fleet of vehicles *__V = 0...v − 1__* is fixed and each vehicle has a limited capacity *__c__*. All of the demands assigned to a vehicle cannot exceed its capacity *__c__*. For each vehicle *__i ∈ V__*, let *__T<sub>i</sub>__* be the sequence of customer deliveries made by that vehicle and let *__dist(c<sub>1</sub>, c<sub>2</sub>)__* be the Euclidean distance between two customers. Then the vehicle routing problem is formalized as the following optimization problem.

<a href="https://www.codecogs.com/eqnedit.php?latex=minimize:&space;\:&space;\sum_{i\in&space;V}^{}\right{(dist(0,T_{i})&space;&plus;&space;\sum_{<j,k>\in&space;T_{i}}^{}\right{dist(j,k)&plus;dist(T_{i,|T_{i}|-1},0}))}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?minimize:&space;\:&space;\sum_{i\in&space;V}^{}\right{(dist(0,T_{i})&space;&plus;&space;\sum_{<j,k>\in&space;T_{i}}^{}\right{dist(j,k)&plus;dist(T_{i,|T_{i}|-1},0}))}" title="minimize: \: \sum_{i\in V}^{}\right{(dist(0,T_{i}) + \sum_{<j,k>\in T_{i}}^{}\right{dist(j,k)+dist(T_{i,|T_{i}|-1},0}))}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=subject&space;\:&space;to:&space;\sum_{j\in&space;T_{i}}^{}\right{d_{j}&space;\leq&space;c}&space;\:\:\:&space;(i&space;\in&space;V)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?subject&space;\:&space;to:&space;\sum_{j\in&space;T_{i}}^{}\right{d_{j}&space;\leq&space;c}&space;\:\:\:&space;(i&space;\in&space;V)" title="subject \: to: \sum_{j\in T_{i}}^{}\right{d_{j} \leq c} \:\:\: (i \in V)" /></a>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i&space;\in&space;V}\right{(j&space;\in&space;T_{i}})=1&space;\:\:\:&space;(j&space;\in&space;N&space;\:&space;\backslash&space;\:&space;0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i&space;\in&space;V}\right{(j&space;\in&space;T_{i}})=1&space;\:\:\:&space;(j&space;\in&space;N&space;\:&space;\backslash&space;\:&space;0)" title="\sum_{i \in V}\right{(j \in T_{i}})=1 \:\:\: (j \in N \: \backslash \: 0)" /></a>

--------------------------------------------------------------------------------

## Example
Input data: [`./data/location.csv`](./data/locations.csv)
* Format: latitude,longitude,is_customer(0 or 1)
* Number of Locations(*__N__*) = 537
* Number of Vehicles(*__V__*) = 25
* For each location
  * Demand(*__d<sub>i</sub>__*) = 1
* The capacity of the vehicle(*__c__*) = 21 ~ 25 (= 537 / 25)

## Proposed Solution
The vehicle routing problem (VRP) is a NP(Nondeterministic Polymomial-time) hard problem.
This project uses a heurisitic algorithm to solve this problem. The proposed algorithm consists of two phases:
1. Finds the centroids which will be delivered by each vehicle by using k-means clustering algorithm
2. Assigns the customers to each centroid(i.e. vehecle)
3. Optimize the vehicle routing(This is a TSP problem.)

### Plots the input data.
<div style="width:100%;"><center><nobr>
<img src="./images/figure_01_input.png" style="max-width:40%; height:auto;">
 
<table style="vertical-align:middle; text-align:center; border-collapse:collapse;">
<tr>
<td style="border:none"><img src="./images/figure_01_input.png" style="height:auto; max-width:80%;"></td>
<td style="border:none"><img src="images/mono_2_of_4.png" style="height:auto; max-width:80%;"></td>
<tr>
<td style="border:none"><img src="images/mono_3_of_4.png" style="height:auto; max-width:80%;"></td>
<td style="border:none"><img src="images/mono_4_of_4.png" style="height:auto; max-width:80%;"></td>
</tr>
</table>

</nobr></center></div>
