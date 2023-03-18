import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from graphdatascience import GraphDataScience
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

plt.rcParams['figure.figsize'] = [10, 6]


class NeoGDS:
    def __init__(self):
        host = "bolt://localhost:7687"
        user = "neo4j"
        password = "cs345gnn"
        database = "frauddb"

        self.gds = GraphDataScience(host, auth=(user, password), database=database)
        self.graph = GraphDatabase.driver(host, auth=(user, password), database=database)
        logging.info(f"[NEO] Running on Neo4j GraphDataScience version {self.gds.version()}.")
        try:
            self.gds.run_cypher("CALL gds.graph.drop('fraud')")
            logging.warning("[NEO] Cleaned up (1) old graph projection.")
        except ClientError:
            logging.info("[NEO] Clean start, no graph cleanup necessary.")

    def transactions_by_year(self):
        transactions = self.gds.run_cypher("""
            MATCH p=()-[r:P2P]->()
            RETURN r.transactionDateTime.year AS year, count(*) AS countOfTransactions
            ORDER BY year
        """)
        sns.barplot(x="year", y="countOfTransactions", data=transactions)

    def get_nx_graph(self):
        # Execute the query and get all nodes in the graph
        nodes = set()
        with self.graph.session() as session:
            results = session.run("MATCH (n) RETURN ID(n) AS node LIMIT 1000")
            for record in results:
                nodes.add(record["node"])

        # Execute the query and initialize the adjacency matrix
        with self.graph.session() as session:
            results = session.run("""
            MATCH (n)-[r]->(m)
            RETURN ID(n) AS source, ID(m) AS target
            LIMIT 1000
            """)
            num_nodes = len(nodes)
            adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

            # Update the adjacency matrix for each relationship in the graph
            node_dict = {node: i for i, node in enumerate(nodes)}
            for record in results:
                if record["source"] in node_dict and record["target"] in node_dict:
                    source_idx = node_dict[record["source"]]
                    target_idx = node_dict[record["target"]]
                    adjacency_matrix[source_idx][target_idx] += 1

        # Display the adjacency matrix using a Pandas DataFrame
        df = pd.DataFrame(adjacency_matrix, index=list(nodes), columns=list(nodes))
        return nx.from_pandas_adjacency(df)

    def get_df(self):
        logging.info("[NEO] Retrieving dataframe from Neo4j...")
        df = self.gds.run_cypher("""
            MATCH (u:User)
            CALL { 
              WITH u 
              MATCH (u)-[out:P2P]->() 
              RETURN sum(out.totalAmount) as totalOutgoingAmount, max(out.totalAmount) as maxOutgoingAmount, 
                       avg(out.totalAmount) as avgOutgoingAmount, count(out) as outgoingTransactions
            }
            CALL { 
              WITH u 
              MATCH (u)<-[in:P2P]-() 
              RETURN sum(in.totalAmount) as totalIncomingAmount, max(in.totalAmount) as maxIncomingAmount, 
                       avg(in.totalAmount) as avgIncomingAmount, count(in) as incomingTransactions
            }
            RETURN u.guid AS user_id,
                   u.fraudRisk AS fraudRisk,
                   count{ (u)-[:USED]->() } AS numberOfDevices,
                   count{ (u)-[:HAS_CC]->() } AS numberOfCCs,
                   count{ (u)-[:HAS_IP]->() } AS numberOfIps,
                   coalesce(totalOutgoingAmount, 0) AS totalOutgoingAmount, 
                   coalesce(avgOutgoingAmount, 0) AS avgOutgoingAmount,
                   coalesce(maxOutgoingAmount, 0) AS maxOutgoingAmount,
                   outgoingTransactions,
                   coalesce(totalIncomingAmount, 0) AS totalIncomingAmount,
                   coalesce(avgIncomingAmount, 0) AS avgIncomingAmount,
                   coalesce(maxIncomingAmount, 0) AS maxIncomingAmount,
                   incomingTransactions
        """)

        gds_features = self._compute_graph_features()

        logging.info("[NEO] Merging dataframe with graph features...")
        gds_features['user_id'] = [el.get('guid') for el in gds_features['node_object']]
        df = df.merge(gds_features[['user_id', 'componentSize', 'part_of_community', 'pagerank', 'closeness']],
                      on='user_id')

        logging.info("[NEO] Selecting final features for evaluation...")

        return df[['fraudRisk', 'numberOfDevices', 'numberOfCCs', 'numberOfIps',
                   'totalOutgoingAmount', 'maxOutgoingAmount', 'avgOutgoingAmount',
                   'totalIncomingAmount', 'maxIncomingAmount', 'avgIncomingAmount',
                   'outgoingTransactions', 'incomingTransactions', 'componentSize', 'part_of_community',
                   'pagerank', 'closeness']]
    
    def explore_features(self, df):
        df.head()
        df.describe()
        plt.figure(figsize=(16, 6))

        corr = df.corr()
        sns.heatmap(corr,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    cmap="Blues",
                    annot=True)

    def _compute_graph_features(self):
        # Step 1: get a graph projection where we merge parallel relationships by summing them into one
        G, res = self.gds.graph.project('fraud', ['User', 'Card'],
                                        {'HAS_CC': {
                                            'type': 'HAS_CC'
                                        },
                                            'P2P': {
                                                'type': 'P2P',
                                                'properties': {
                                                    'totalAmount': {
                                                        'aggregation': 'SUM'
                                                    }
                                                }
                                            }})

        # Step 2: Compute graph features from GDS:
        #   1. WCC Algorithm to find "islands" in the network, for users who used the same credit card
        #   2. Compute PageRank centrality score for the P2P transaction network
        #   3. Similar to PageRank, but we use Closeness Centrality
        wcc, page_rank, closeness = self._graph_features(G)

        # * Step 3: Merge WCC & Page Rank & Closeness features
        logging.info("[NEO] Merging computed graph features...")
        node_stream = wcc \
            .merge(page_rank[['nodeId', 'pagerank']], on='nodeId') \
            .merge(closeness[['nodeId', 'closeness']], on='nodeId')

        # * Step 4: Cleanup graph projection
        logging.info(f"[NEO] Dropping graph projection: {G.name()}...")
        G.drop()
        logging.info(f"[NEO] Dropped projection: {G.name()}...")

        # * Step 5: Return merged node stream
        return node_stream

    def _graph_features(self, G):
        return self._wcc(G), self._page_rank(G), self._closeness_centrality(G)

    def _wcc(self, G):
        logging.info("[NEO] Opening WCC stream...")
        node_stream = self.gds.wcc.stream(G, nodeLabels=['User', 'Card'], relationshipTypes=['HAS_CC'])
        # * Eliminate "card" nodes from WCC stream
        logging.info("[NEO] WCC stream open. Removing card nodes...")
        node_stream['node_object'] = self.gds.util.asNodes(node_stream['nodeId'].to_list())
        # * Filter only user nodes
        node_stream = node_stream[[list(x.labels)[0] == 'User' for x in node_stream['node_object']]]
        logging.info("[NEO] Removed Card nodes from WCC stream.")

        # * Compute two features from WCC algorithm: part_of_community and ComponentSize
        # * Get component sizes
        node_stream = node_stream.merge(
            node_stream.groupby('componentId').size().to_frame('componentSize').reset_index(), on="componentId")
        logging.info("[NEO] Computed Feature[componentSize]. Computing next feature...")
        # * Define a feature which indicates if the component has more than 1 member
        node_stream['part_of_community'] = (node_stream['componentSize'] > 1).astype(int)
        logging.info("[NEO] Computed Feature[part_of_community].")
        return node_stream

    def _page_rank(self, G):
        logging.info("[NEO] Opening PageRank stream...")
        page_rank = self.gds.pageRank.stream(G, nodeLabels=['User'],
                                             relationshipTypes=['P2P'], relationshipWeightProperty='totalAmount')
        page_rank['pagerank'] = page_rank['score']
        logging.info("[NEO] Feature[pageRank] computed.")
        return page_rank

    def _closeness_centrality(self, G):
        logging.info("[NEO] Opening Closeness ranking stream...")
        closeness = self.gds.beta.closeness.stream(G, nodeLabels=['User'], relationshipTypes=['P2P'])
        closeness['closeness'] = closeness['score']
        logging.info("[NEO] Feature[closeness] computed.")
        return closeness


def graph_from_cypher(cypher):
    G = nx.MultiDiGraph()

    nodes = list(cypher.graph()._nodes.values())
    for node in nodes:
        G.add_node(node.id, labels=node._labels, properties=node._properties)

    rels = list(cypher.graph()._relationships.values())
    for rel in rels:
        G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

    return G