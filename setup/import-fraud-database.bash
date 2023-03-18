# Run the "ls" command and capture its output in a temporary file
docker exec cs345-neo4j cypher-shell --non-interactive -u neo4j -p cs345gnn 'call dbms.components()' > /tmp/output.txt

# Read the contents of the temporary file into a variable called "cypher_test"
read -r cypher_test < /tmp/output.txt

# Compare the variable "cypher_test" with the string ""
if [ "$cypher_test" == "" ]; then
  echo "Neo4j hasn't started up yet! Please wait until Neo4j starts up before running this script again."
else
  echo "Loading database..."
  docker exec cs345-neo4j neo4j-admin database load --from-path="/setup" frauddb
  docker exec cs345-neo4j cypher-shell --non-interactive -u neo4j -p cs345gnn 'create database frauddb;'
  docker exec cs345-neo4j cypher-shell --non-interactive -u neo4j -p cs345gnn 'start database frauddb;'
  echo "Restarting Neo4j..."
  docker exec cs345-neo4j neo4j-admin server restart
  echo "Starting up Neo4j, reconnect in a few seconds!"
fi