# Federated Learning on Relational Data & Shallow Learning
Code for spml - report 3 - Federated Leanring Applied on Network Traffic

I implemented a Federated Learning (FL) solution on a network traffic dataset. Instead of a single monitoring point, the idea is to build an Intrusion Detection System (IDS) based on traffic captured at multiple hosts. This scenario is also feasible in the case of encrypted traffic, in which a monitoring device in the network does not have access to the data, but end hosts can decrypt the traffic and thus more information is available. Results show that FL can be used in this case; however, performance is slightly lower than with centralized learning; but even so, the benefits of FL should be considered in many cases, particularly in encrypted network traffic analysis.

# Data
Since the datasets are too large (>2Gb), they can be obtained by contacting me: fares.meghdouri@tuwien.ac.at

# Usage examples

```sh
# federated training (iid) 4 nodes and evaluation
python learn.py --dataset cic2017 --n_users 4 --fed_training --test

# normal training and evaluation
python learn.py --dataset cic2017 --n_users 4 --normal_training --test

```

```sh
# federated training IP aggregation and evaluation
python learn.py --dataset cic2017 --network_users network_users_2017.pkl --fed_training --test
# network_users refer to unsw2015 and network_users_2017 refers to cic2017

```

```sh
# federated training IP aggregation and evaluation. Consider only some nodes not all (usefull if you want to filter out some nodes with single class traffic)
python learn.py --dataset unsw2015 --network_users network_users.pkl --fed_training --test --num_users_first 9 --num_users_last 20
# in this case, we take nodes from 9 to 20 in the network_users.pkl dictionnary

```