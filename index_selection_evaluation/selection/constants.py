config={}

config["benchmark_name"]='tpch'
config["scale_factor"]=10
config["queries"]=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,21,22]
config['max compress num']=14
config['random_queries']=False
config['threshold']=1e-8
if config['random_queries']:
    config['pre_feature_len']=60
    config['pre_multi_operator_len']=200
    config['pre_multi_index_len']=300
else:
    config['pre_feature_len']=60
    config['pre_multi_operator_len']=200
    config['pre_multi_index_len']=300

config['random_queries_num']=380
config_file = "example_configs/config_tpch.json"


config['times']=30
templete_len=len(config["queries"])*config['times']
if config['random_queries']:
    random_len=config['random_queries_num']
else:
    random_len=0
dir_path=f'./workload_compress/workload/{config["benchmark_name"]}_whatif_scale{config["scale_factor"]}_times{config["times"]}_temlen{templete_len}_randomlen{random_len}_comlen{config["max compress num"]}'
compress_algorithm=['nothing']
config['compress_algorithm']=compress_algorithm
config['dir_path']=dir_path

config['stastics_path']='example_configs/statistics.json'



phy2log_map={'Aggregate':3,'Gather':3, 'Hash Join':2, 'Merge Join':2, 'Nested Loop':6, 'Seq Scan':0,'Bitmap Heap Scan':0,'Subquery Scan':0, 'Sort':1,'Append':4, 'Gather Merge':4, 'Hash':5, 'Limit':4,
'Group':4, 'Unique':4, 'Bitmap Index Scan':0, 'CTE Scan':0, 'SetOp':4, 'WindowAgg':3, 'Materialize':4, 'Index Only Scan':0, 'BitmapAnd':0, 'Result':4, 'Merge Append':1, 'Index Scan':0, 'BitmapOr':0}
indexed_type=['Bitmap Heap Scan','BitmapAnd','Index Scan','Bitmap Index Scan','Index Only Scan','BitmapOr']
can_index_column_tpye=['varchar','char', 'integer', 'decimal']

node_phy_tpye=['Aggregate','Gather', 'Hash Join', 'Merge Join', 'Nested Loop', 'Seq Scan','Bitmap Heap Scan','Subquery Scan', 'Sort','Append', 'Gather Merge', 'Hash', 'Limit', 'Bitmap Index Scan', 'BitmapAnd', 'BitmapOr', 'CTE Scan', 'Group', 'Index Only Scan', 'Index Scan', 'Materialize', 'Merge Append', 'Result', 'SetOp', 'Unique', 'WindowAgg']
node_log_tpye=['Scan', 'Sort', 'Join', 'Aggregate','Other','Hash','Nested Loop','Has Or']

log_type_r=[1, 1, 1, 1, 1, 1, 0, 1]
log_type_b=[0, 0, 0, 0, 0, 0, 0, 0]

key_list=['Group Key','Sort Key','Grouping Sets']
join_key_list=['Hash Cond','Merge Cond','Join Filter']
filter_list=['Filter','Index Cond','Recheck Cond']

column_tpye=['varchar', 'date', 'time', 'char', 'integer', 'decimal']

operator_str=[' <> ',' ~~ ',' ANY (',' >= ',' <= ',' > ',' < ',' = ',' IS ',' IS NOT ',' !~~ '] 
can_index_operator_str=[' ~~ ', ' ANY (', ' >= ', ' <= ', ' > ', ' < ', ' = ', ' IS ', ' IS NOT ']

not_operator_str={' <> ':' = ',' ~~ ':' !~~ ',' ANY (':' !~~ ',' >= ':' < ',' <= ':' > ',' > ':' <= ',' < ':' >= ',' = ':' <> ',' IS ':' IS NOT ',' IS NOT ':' IS ',' !~~ ':' ~~ '}
max_node_len=150
