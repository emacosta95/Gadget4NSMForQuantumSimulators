import minorminer
import dwave_networkx as dnx
import networkx as nx
from minorminer import verify_embedding

# K4: fully connected, 4 nodes = 6 edges
K4 = nx.complete_graph(4)
print("Problem edges:", list(K4.edges()))

# Hardware graph
hardware = dnx.pegasus_graph(16)

# Find embedding
embedding = minorminer.find_embedding(K4, hardware)
print("\nEmbedding found:")
for logical, chain in embedding.items():
    print(f"  q{logical} → chain {chain}  (length {len(chain)})")

# Verify
try:
    verify_embedding(embedding, K4, hardware)
    print("\n✓ Valid embedding")
except Exception as e:
    print(f"\n✗ Invalid: {e}")
