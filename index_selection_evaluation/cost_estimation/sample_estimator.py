import os
import numpy as np
import copy
import random
import pandas as pd
import psycopg2
import re
import time


def generate_sample_data(db_name):
    
    conn = psycopg2.connect(f"dbname={db_name} user=postgres")

    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cur.fetchall()

    for table in tables:
        table_name = table[0]
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        sample_size = int(count * 0.01)
        cur.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}")
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=[desc[0] for desc in cur.description])
        os.makedirs(f"cost_estimation/samples/{db_name}", exist_ok=True)
        df.to_csv(f"cost_estimation/samples/{db_name}/{table_name}.csv", index=False, header=True)

    conn.close()



class SampleEstimator:
    def __init__(self,db_name) -> None:
        self.db_name = db_name
        self.load_samples()
    
    def load_samples(self):
        dir_path=f'cost_estimation/samples/{self.db_name}'
        if not os.path.exists(dir_path):
            generate_sample_data(self.db_name)

        table_names=os.listdir(dir_path)
        self.table_dirs={}
        for name in table_names:
            df=pd.read_csv(f'cost_estimation/samples/{self.db_name}/{name}',header=0)
            ns=name.split('.')
            assert len(ns)==2,print(name)
            self.table_dirs[ns[0]]=df
    def estimate_selectivity(self,table,predicate):
        df=self.table_dirs[table]
        return len(df.query(predicate))/len(df)