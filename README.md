# FairDiff
Implementation of Fair Diffusion Model for Graphs

Work produced for the Dissertation: "Semi-Synthetic Graph Generation"



| Graph |Nodes| Edges| Avg Deg | Triangles | Largest CC | Rel Edge Dist Entropy | Power Law Exp | Gini Coefficient | IoU*|
| -- |--| --| -- | --| -- | -- | -- | -- |--|
|Cora | 2708 | 5278 | 3.89 | 1630 | 2485 | 0.9552 | 1.9323 | 0.4051| 1|
|1000 sample| 1588 | 4874 | 6.14| 2550 | 1500 | 0.9391 | 1.7116 | 0.4823 | 0.1429 |
|2000 sample| 1923 | 7139 | 7.42 | 4342 | 1865 | 0.9429 | 1.6343 | 0.4847 | 0.1930 | 
|1000 sample (2000 ts) | 1884 | 3219 | 3.42 | 851 | 1573 | 0.9582 | 2.0432 | 0.3968 | 0.4863 |
|2000 sample (2000 ts) | 2214 | 4383 | 3.96 | 1299 | 1971 | 0.9542 | 1.9314 | 0.4186 | 0.5845 |
| -- |--| --| -- | --| -- | -- | -- | -- |--|
|1000 sample  FOCAL | 1626 | 2964 | 3.65 | 828 | 1443 | 0.9489 | 2.022 | 0.4291 | 0.3607 |
|2000 sample FOCAL | 1928 | 4364 | 4.53 | 1629 | 1823 | 0.9456 | 1.8677 | 0.4536 | 0.4230 |
| -- |--| --| -- | --| -- | -- | -- | -- |--|
|1000 sample (1000 ts 50) | 2191 | 4904 | 4.48 | 1545 | 2044 | 0.9487 | 1.8950 | 0.4293 | 0.6975 |
|2000 sample (1000 ts 50)| 2356 | 6032 | 5.12 | 2068 | 2222 | 0.9483 | 1.7741 | 0.4394 | 0.6883 |
|1000 sample (500 ts 50)|1941|5160|5.32|2412|1822|0.9287|1.8165|0.5056|0.4852|
|2000 sample (500 ts 50) |2159|7302|6.76|4663|2057|0.9261|1.7015|0.5233|0.4648|
| -- |--| --| -- | --| -- | -- | -- | -- |--|
|1000 sample (1000 ts 20) |2049|4196|4.1|1362|1934|0.9465|1.9276|0.4381|0.5764|
|2000 sample (1000 ts 20) |2276|5480|4.82|2006|2163|0.9446|1.821|0.4498|0.6170|
|1000 sample (1000 ts 20) FAIR |1427|2517|3.53|643|1304|0.9503|2.045|0.4213|0.3517|
|2000 sample (1000 ts 20) FAIR |1695|3435|4.05|982|1597|0.9474|1.9381|0.4403|0.4151|


## Measures

IDI does it make sense because the nodes are the same
