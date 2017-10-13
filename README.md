# Capacitated VRP - CVRP

## Problem Statement

Write an algorithm to solve the vehicle routing problem. The problem is mathematically formulated in the following way: We are given a list of locations *__N = 0 . . . n − 1__* . By convention, location *__0__* is the warehouse location, where all of the vehicles start and end their routes. The remaining locations are customers. Each location is characterized by three values *__⟨d<sub>i</sub>, x<sub>i</sub>, y<sub>i</sub>⟩ i ∈ N__* a demand *__d<sub>i</sub>__* and a point *__x<sub>i</sub>,y<sub>i</sub>__*. The fleet of vehicles *__V = 0...v − 1__* is fixed and each vehicle has a limited capacity *__c__*. All of the demands assigned to a vehicle cannot exceed its capacity *__c__*. For each vehicle *__i ∈ V__*, let *__T<sub>i</sub>__* be the sequence of customer deliveries made by that vehicle and let *__dist(c<sub>1</sub>, c<sub>2</sub>)__* be the Euclidean distance between two customers. Then the vehicle routing problem is formalized as the following optimization problem.

* *__n__* Locations, *__v__* Vehicles
* For each location
  * demand *__di__* and location *__xi,yi__*
* The capacity of the vehicles *__c__*

## Title

### Title
