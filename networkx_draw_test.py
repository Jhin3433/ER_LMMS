# import matplotlib.pyplot as plt
# import networkx as nx
# G = nx.petersen_graph()
# subax1 = plt.subplot(121)
# nx.draw(G, with_labels=True, font_weight='bold')
# subax2 = plt.subplot(122)
# nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
# plt.savefig("graph.png")



# -------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# import networkx as nx

# G = nx.lollipop_graph(4, 6)

# pathlengths = []

# print("source vertex {target:length, }")
# for v in G.nodes():
#     spl = dict(nx.single_source_shortest_path_length(G, v))
#     print(f"{v} {spl} ")
#     for p in spl:
#         pathlengths.append(spl[p])

# print()
# print(f"average shortest path length {sum(pathlengths) / len(pathlengths)}")

# # histogram of path lengths
# dist = {}
# for p in pathlengths:
#     if p in dist:
#         dist[p] += 1
#     else:
#         dist[p] = 1

# print()
# print("length #paths")
# verts = dist.keys()
# for d in sorted(verts):
#     print(f"{d} {dist[d]}")

# print(f"radius: {nx.radius(G)}")
# print(f"diameter: {nx.diameter(G)}")
# print(f"eccentricity: {nx.eccentricity(G)}")
# print(f"center: {nx.center(G)}")
# print(f"periphery: {nx.periphery(G)}")
# print(f"density: {nx.density(G)}")

# pos = nx.spring_layout(G, seed=3068)  # Seed layout for reproducibility
# nx.draw(G, pos=pos, with_labels=True)
# plt.show()
# plt.savefig("graph.png")



import matplotlib.pyplot as plt
import networkx as nx
G = nx.path_graph(4)
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos)
# nx.draw(esg.G)
plt.show()
plt.savefig("graph.png")