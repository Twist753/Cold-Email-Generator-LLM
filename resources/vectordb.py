import chromadb
import pandas as pd

# Number of portfolios to search for
num = 2
 
class Database:
    def __init__(self, path = "resources\sample_database.csv"):
        self.path = path
        self.df = pd.read_csv(path)
        
        self.client = chromadb.PersistentClient('vectorstore')
        self.collection = self.client.get_or_create_collection(name='portfolios')
        
    
    def load_portfolios(self):
        if not self.collection.count():
            for i, row in self.df.iterrows():
                self.collection.add(
                    documents = [row['Techstack']],
                    metadatas=[{'links' : row['Links']}],
                    ids=[str(i)]
                )
    
    def query_search(self, skills):
        return self.collection.query(
            query_texts=skills,
            n_results=num
        ).get('metadatas', [])
                
            