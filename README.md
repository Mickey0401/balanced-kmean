What is the development environment?
---
Linux only, preferably CentOS, CoreOS, Ubuntu (in this order)

What is the software development lifecycle you use/we will use?
---
All code will eventually execute inside containerized kubernetes

How you should develop/optimize this code?
---
Create a separate branch  
Name the branch using your name 
Create a python wrapper which takes the sample data file  
Loads it  
Executes the clusterization algorithm  
Outputs the execution time
Outputs which cluster each point is in


What is the business purpose of the project?
---
a) optimize the code to run very quickly  
b) to find at least one performance optimization expert programmer

Do we have to use python laps?
---
No

Do we have to use the latest version of python laps?
---
No

Does it have to stay written in Python?
---
No

What sort of results are we looking for?
---
10,000 lat/lng pairs inserted into balanced clusters in about 10 seconds

Are there restrictions on server hardware, sizes, or costs?
---
No. Any technology can be used to do the clusterization in less than 10 seconds.
If that takes a very expensive GPU or FGPA device, then that's fine.

How will this code eventually be executed/invoked?
---
This code will eventually be invoked inside a python flask worker as part of an asynchronous worker pool.

What is the roadmap / next items?
---
spatial clusterization with obstacles  
polygon clusterization  
gpu/fpga accelerated clusterization  
in-memory inventory management and simulation
