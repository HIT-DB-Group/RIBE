import psycopg2
from pathlib import Path
from collections import namedtuple
from typing import Dict
import pickle
gettablename = "SELECT tablename FROM pg_tables WHERE tablename NOT LIKE 'pg%' AND tablename NOT LIKE 'sql_%'"
gettableinfo = "SELECT tablename,obj_description(relfilenode,'pg_class') FROM pg_tables a, pg_class b WHERE a.tablename = b.relname AND a.tablename NOT LIKE 'pg%' AND a.tablename NOT LIKE 'sql_%' ORDER BY  a.tablename;"
getcols = "SELECT col_description(a.attrelid, a.attnum) AS comment,format_type(a.atttypid,a.atttypmod) AS type,a.attname AS name,a.attnotnull AS notnull FROM pg_class AS c,pg_attribute AS a WHERE c.relname = \'{}\'  AND a.attrelid = c.oid AND a.attnum>0 "
getcoltype = "SELECT format_type(a.atttypid,a.atttypmod) AS type FROM pg_class AS c, pg_attribute AS a WHERE c.relname = \'{}\' AND a.attname = \'{}\' AND a.attrelid = c.oid  AND a.attnum>0"


class Column:
    def __init__(self, name, dtype, value):
        self.name = name
        self.dtype = dtype
        self.value = value


class Table:
    columns: Dict[str, Column]

    def __init__(self, name):
        self.name = name
        self.columns = dict()


class DataBase:
    tables: Dict[str, Table]

    def __init__(self, name):
        self.name = name
        self.tables = dict()
        self.table_name = []
        self.column_name = []
        self.join_key = dict()
        self.table_column = dict()
        self.join_column = []


class PostgreSQL:
    def __init__(self, DATABASE_URL):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def run(self, sql):
        try:
            self.cur.execute(sql)
            rows = self.cur.fetchall()
            return rows
        except:
            return None

    def dtype(self, table, column):
        return self.run(sql=getcoltype.format(table.lower(), column.lower()))[0][0]

    def parser(self, sql):
        sql = "explain(format json) " + sql
        result = self.run(sql)
        if result == None:
            return None
        else:
            cardinality = result[0][0][0]['Plan']['Plan Rows']
            cost = result[0][0][0]['Plan']['Total Cost']
            return cardinality, cost

    def sample(self, table, attr, percentage=0.1):
        return self.run("select {} from {} TABLESAMPLE system({}) where {} is not null limit 50;".format(attr, table, percentage, attr))


class Settings:

    # data settings
    data_name = "imdb"

    data_root = Path("./data")
    temp_root = data_root / "temp"
    text_root = data_root / "text"
    grammar_root = data_root / "grammar"
    lark_path = grammar_root / "lark.lark"
    basic_sql_lark_path = grammar_root / "base_convert.lark"
    convert_lark_path = grammar_root / "convert_sql.lark"

    predicate_path = text_root / "predicate.pickle"
    origin_sql_path = text_root / f"{data_name}.pickle"
    imdb_info_path = temp_root / "imdb_dataset.pickle"
    tpch_info_path = temp_root / "tpch_dataset.pickle"
    tpcds_info_path = temp_root / "tpcds_dataset.pickle"

    # Regular expressions
    regular_predicate = r"([A-Za-z_]+)\.([A-Za-z_]+)(=|>|<)([0-9.]+)"
    regular_cost = r"cost=([0-9]+\.[0-9]+)\.\.([0-9]+\.[0-9]+)"
    regular_cardinality = r"rows=([0-9]+)"

    # Grammar settings
    PADDING_ACTION = -1
    value_sample_num = 50
    max_expression_num = 10
    max_orderby_num = 3

    Rule = namedtuple("Rule", ['rule', 'parent_rule'])

    """"""""""""""""""""""""""""""""""""""""""""""""

    text_count = 6000  # number of queries
    max_squence_length = 100  # The length of the sql, not the exact length of the sql, but the shorter it is, the shorter the sql usually is.
    # TPC-H
    database_tpch = {
        "database_url": "dbname=your_dbname user=your_user",
        "table_column": {  
            'part': ['p_size', 'p_retailprice'],
            'supplier': ['s_acctbal'],
            'partsupp': ['ps_availqty', 'ps_supplycost'],
            'customer': ['c_acctbal'],
            'orders': ['o_totalprice', 'o_shippriority'],
            'lineitem': ['l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax'],
        },
        'join_key': {
            'part': {
                'partsupp': [('p_partkey', 'ps_partkey')],
            },
            'supplier': {
                'partsupp': [('s_suppkey', 'ps_suppkey')],
                'customer': [('s_nationkey', 'c_nationkey')],
            },
            'partsupp': {
                'lineitem': [('ps_partkey', 'l_partkey'), ('ps_suppkey', 'l_suppkey')],
                'supplier': [('ps_suppkey', 's_suppkey')],
                'part': [('ps_partkey', 'p_partkey')],
            },
            'customer': {
                'orders': [('c_custkey', 'o_custkey')],
                'supplier': [('c_nationkey', 's_nationkey')]
            },

            'lineitem': {
                'partsupp': [('l_partkey', 'ps_partkey'), ('l_suppkey', 'ps_suppkey')],
                'orders': [('l_orderkey', 'o_orderkey')]
            },
            'orders': {
                'lineitem': [('o_orderkey', 'l_orderkey')],
                'customer':[('o_custkey','c_custkey')]
            },
        },
    }
    # TPCDS
    database_tpcds={
    "database_url": "dbname=your_dbname user=your_user",
 'table_column': {'customer_address': ['ca_address_sk', 'ca_gmt_offset'],
  'customer_demographics': ['cd_demo_sk',
   'cd_purchase_estimate',
   'cd_dep_count',
   'cd_dep_employed_count',
   'cd_dep_college_count'],
  'date_dim': ['d_date_sk',
   'd_month_seq',
   'd_week_seq',
   'd_quarter_seq',
   'd_year',
   'd_dow',
   'd_moy',
   'd_dom',
   'd_qoy',
   'd_fy_year',
   'd_fy_quarter_seq',
   'd_fy_week_seq',
   'd_first_dom',
   'd_last_dom',
   'd_same_day_ly',
   'd_same_day_lq'],
  'warehouse': ['w_warehouse_sk', 'w_warehouse_sq_ft', 'w_gmt_offset'],
  'ship_mode': ['sm_ship_mode_sk'],
  'time_dim': ['t_time_sk', 't_time', 't_hour', 't_minute', 't_second'],
  'reason': ['r_reason_sk'],
  'income_band': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'],
  'item': ['i_item_sk',
   'i_current_price',
   'i_wholesale_cost',
   'i_brand_id',
   'i_class_id',
   'i_category_id',
   'i_manufact_id',
   'i_manager_id'],
  'store': ['s_store_sk',
   's_closed_date_sk',
   's_number_employees',
   's_floor_space',
   's_market_id',
   's_division_id',
   's_company_id',
   's_gmt_offset',
   's_tax_precentage'],
  'call_center': ['cc_call_center_sk',
   'cc_closed_date_sk',
   'cc_open_date_sk',
   'cc_employees',
   'cc_sq_ft',
   'cc_mkt_id',
   'cc_division',
   'cc_company',
   'cc_gmt_offset',
   'cc_tax_percentage'],
  'customer': ['c_customer_sk',
   'c_current_cdemo_sk',
   'c_current_hdemo_sk',
   'c_current_addr_sk',
   'c_first_shipto_date_sk',
   'c_first_sales_date_sk',
   'c_birth_day',
   'c_birth_month',
   'c_birth_year',
   'c_last_review_date_sk'],
  'web_site': ['web_site_sk',
   'web_open_date_sk',
   'web_close_date_sk',
   'web_mkt_id',
   'web_company_id',
   'web_gmt_offset',
   'web_tax_percentage'],
  'store_returns': ['sr_returned_date_sk',
   'sr_return_time_sk',
   'sr_item_sk',
   'sr_customer_sk',
   'sr_cdemo_sk',
   'sr_hdemo_sk',
   'sr_addr_sk',
   'sr_store_sk',
   'sr_reason_sk',
   'sr_ticket_number',
   'sr_return_quantity',
   'sr_return_amt',
   'sr_return_tax',
   'sr_return_amt_inc_tax',
   'sr_fee',
   'sr_return_ship_cost',
   'sr_refunded_cash',
   'sr_reversed_charge',
   'sr_store_credit',
   'sr_net_loss'],
  'household_demographics': ['hd_demo_sk',
   'hd_income_band_sk',
   'hd_dep_count',
   'hd_vehicle_count'],
  'web_page': ['wp_web_page_sk',
   'wp_creation_date_sk',
   'wp_access_date_sk',
   'wp_customer_sk',
   'wp_char_count',
   'wp_link_count',
   'wp_image_count',
   'wp_max_ad_count'],
  'promotion': ['p_promo_sk',
   'p_start_date_sk',
   'p_end_date_sk',
   'p_item_sk',
   'p_cost',
   'p_response_target'],
  'catalog_page': ['cp_catalog_page_sk',
   'cp_start_date_sk',
   'cp_end_date_sk',
   'cp_catalog_number',
   'cp_catalog_page_number'],
  'inventory': ['inv_date_sk',
   'inv_item_sk',
   'inv_warehouse_sk',
   'inv_quantity_on_hand'],
  'catalog_returns': ['cr_returned_date_sk',
   'cr_returned_time_sk',
   'cr_item_sk',
   'cr_refunded_customer_sk',
   'cr_refunded_cdemo_sk',
   'cr_refunded_hdemo_sk',
   'cr_refunded_addr_sk',
   'cr_returning_customer_sk',
   'cr_returning_cdemo_sk',
   'cr_returning_hdemo_sk',
   'cr_returning_addr_sk',
   'cr_call_center_sk',
   'cr_catalog_page_sk',
   'cr_ship_mode_sk',
   'cr_warehouse_sk',
   'cr_reason_sk',
   'cr_order_number',
   'cr_return_quantity',
   'cr_return_amount',
   'cr_return_tax',
   'cr_return_amt_inc_tax',
   'cr_fee',
   'cr_return_ship_cost',
   'cr_refunded_cash',
   'cr_reversed_charge',
   'cr_store_credit',
   'cr_net_loss'],
  'web_returns': ['wr_returned_date_sk',
   'wr_returned_time_sk',
   'wr_item_sk',
   'wr_refunded_customer_sk',
   'wr_refunded_cdemo_sk',
   'wr_refunded_hdemo_sk',
   'wr_refunded_addr_sk',
   'wr_returning_customer_sk',
   'wr_returning_cdemo_sk',
   'wr_returning_hdemo_sk',
   'wr_returning_addr_sk',
   'wr_web_page_sk',
   'wr_reason_sk',
   'wr_order_number',
   'wr_return_quantity',
   'wr_return_amt',
   'wr_return_tax',
   'wr_return_amt_inc_tax',
   'wr_fee',
   'wr_return_ship_cost',
   'wr_refunded_cash',
   'wr_reversed_charge',
   'wr_account_credit',
   'wr_net_loss'],
  'web_sales': ['ws_sold_date_sk',
   'ws_sold_time_sk',
   'ws_ship_date_sk',
   'ws_item_sk',
   'ws_bill_customer_sk',
   'ws_bill_cdemo_sk',
   'ws_bill_hdemo_sk',
   'ws_bill_addr_sk',
   'ws_ship_customer_sk',
   'ws_ship_cdemo_sk',
   'ws_ship_hdemo_sk',
   'ws_ship_addr_sk',
   'ws_web_page_sk',
   'ws_web_site_sk',
   'ws_ship_mode_sk',
   'ws_warehouse_sk',
   'ws_promo_sk',
   'ws_order_number',
   'ws_quantity',
   'ws_wholesale_cost',
   'ws_list_price',
   'ws_sales_price',
   'ws_ext_discount_amt',
   'ws_ext_sales_price',
   'ws_ext_wholesale_cost',
   'ws_ext_list_price',
   'ws_ext_tax',
   'ws_coupon_amt',
   'ws_ext_ship_cost',
   'ws_net_paid',
   'ws_net_paid_inc_tax',
   'ws_net_paid_inc_ship',
   'ws_net_paid_inc_ship_tax',
   'ws_net_profit'],
  'catalog_sales': ['cs_sold_date_sk',
   'cs_sold_time_sk',
   'cs_ship_date_sk',
   'cs_bill_customer_sk',
   'cs_bill_cdemo_sk',
   'cs_bill_hdemo_sk',
   'cs_bill_addr_sk',
   'cs_ship_customer_sk',
   'cs_ship_cdemo_sk',
   'cs_ship_hdemo_sk',
   'cs_ship_addr_sk',
   'cs_call_center_sk',
   'cs_catalog_page_sk',
   'cs_ship_mode_sk',
   'cs_warehouse_sk',
   'cs_item_sk',
   'cs_promo_sk',
   'cs_order_number',
   'cs_quantity',
   'cs_wholesale_cost',
   'cs_list_price',
   'cs_sales_price',
   'cs_ext_discount_amt',
   'cs_ext_sales_price',
   'cs_ext_wholesale_cost',
   'cs_ext_list_price',
   'cs_ext_tax',
   'cs_coupon_amt',
   'cs_ext_ship_cost',
   'cs_net_paid',
   'cs_net_paid_inc_tax',
   'cs_net_paid_inc_ship',
   'cs_net_paid_inc_ship_tax',
   'cs_net_profit'],
  'store_sales': ['ss_sold_date_sk',
   'ss_sold_time_sk',
   'ss_item_sk',
   'ss_customer_sk',
   'ss_cdemo_sk',
   'ss_hdemo_sk',
   'ss_addr_sk',
   'ss_store_sk',
   'ss_promo_sk',
   'ss_ticket_number',
   'ss_quantity',
   'ss_wholesale_cost',
   'ss_list_price',
   'ss_sales_price',
   'ss_ext_discount_amt',
   'ss_ext_sales_price',
   'ss_ext_wholesale_cost',
   'ss_ext_list_price',
   'ss_ext_tax',
   'ss_coupon_amt',
   'ss_net_paid',
   'ss_net_paid_inc_tax',
   'ss_net_profit']},
 'join_key': {'customer_address': {'catalog_returns': [('ca_address_sk',
     'cr_returning_addr_sk')],
   'catalog_sales': [('ca_address_sk', 'cs_ship_addr_sk')],
   'customer': [('ca_address_sk', 'c_current_addr_sk')],
   'store_returns': [('ca_address_sk', 'sr_addr_sk')],
   'store_sales': [('ca_address_sk', 'ss_addr_sk')],
   'web_returns': [('ca_address_sk', 'wr_returning_addr_sk')],
   'web_sales': [('ca_address_sk', 'ws_ship_addr_sk')]},
  'customer_demographics': {'catalog_returns': [('cd_demo_sk',
     'cr_returning_cdemo_sk')],
   'catalog_sales': [('cd_demo_sk', 'cs_ship_cdemo_sk')],
   'customer': [('cd_demo_sk', 'c_current_cdemo_sk')],
   'store_returns': [('cd_demo_sk', 'sr_cdemo_sk')],
   'store_sales': [('cd_demo_sk', 'ss_cdemo_sk')],
   'web_returns': [('cd_demo_sk', 'wr_returning_cdemo_sk')],
   'web_sales': [('cd_demo_sk', 'ws_ship_cdemo_sk')]},
  'date_dim': {'call_center': [('d_date_sk', 'cc_open_date_sk')],
   'catalog_page': [('d_date_sk', 'cp_start_date_sk')],
   'catalog_returns': [('d_date_sk', 'cr_returned_date_sk')],
   'catalog_sales': [('d_date_sk', 'cs_sold_date_sk')],
   'customer': [('d_date_sk', 'c_first_shipto_date_sk')],
   'inventory': [('d_date_sk', 'inv_date_sk')],
   'promotion': [('d_date_sk', 'p_start_date_sk')],
   'store': [('d_date_sk', 's_closed_date_sk')],
   'store_returns': [('d_date_sk', 'sr_returned_date_sk')],
   'store_sales': [('d_date_sk', 'ss_sold_date_sk')],
   'web_page': [('d_date_sk', 'wp_creation_date_sk')],
   'web_returns': [('d_date_sk', 'wr_returned_date_sk')],
   'web_sales': [('d_date_sk', 'ws_sold_date_sk')],
   'web_site': [('d_date_sk', 'web_open_date_sk')]},
  'warehouse': {'catalog_returns': [('w_warehouse_sk', 'cr_warehouse_sk')],
   'catalog_sales': [('w_warehouse_sk', 'cs_warehouse_sk')],
   'inventory': [('w_warehouse_sk', 'inv_warehouse_sk')],
   'web_sales': [('w_warehouse_sk', 'ws_warehouse_sk')]},
  'ship_mode': {'catalog_returns': [('sm_ship_mode_sk', 'cr_ship_mode_sk')],
   'catalog_sales': [('sm_ship_mode_sk', 'cs_ship_mode_sk')],
   'web_sales': [('sm_ship_mode_sk', 'ws_ship_mode_sk')]},
  'time_dim': {'catalog_returns': [('t_time_sk', 'cr_returned_time_sk')],
   'catalog_sales': [('t_time_sk', 'cs_sold_time_sk')],
   'store_returns': [('t_time_sk', 'sr_return_time_sk')],
   'store_sales': [('t_time_sk', 'ss_sold_time_sk')],
   'web_returns': [('t_time_sk', 'wr_returned_time_sk')],
   'web_sales': [('t_time_sk', 'ws_sold_time_sk')]},
  'reason': {'catalog_returns': [('r_reason_sk', 'cr_reason_sk')],
   'store_returns': [('r_reason_sk', 'sr_reason_sk')],
   'web_returns': [('r_reason_sk', 'wr_reason_sk')]},
  'income_band': {'household_demographics': [('ib_income_band_sk',
     'hd_income_band_sk')]},
  'item': {'catalog_returns': [('i_item_sk', 'cr_item_sk')],
   'catalog_sales': [('i_item_sk', 'cs_item_sk')],
   'inventory': [('i_item_sk', 'inv_item_sk')],
   'promotion': [('i_item_sk', 'p_item_sk')],
   'store_returns': [('i_item_sk', 'sr_item_sk')],
   'store_sales': [('i_item_sk', 'ss_item_sk')],
   'web_returns': [('i_item_sk', 'wr_item_sk')],
   'web_sales': [('i_item_sk', 'ws_item_sk')]},
  'store': {'date_dim': [('s_closed_date_sk', 'd_date_sk')],
   'store_returns': [('s_store_sk', 'sr_store_sk')],
   'store_sales': [('s_store_sk', 'ss_store_sk')]},
  'call_center': {'date_dim': [('cc_open_date_sk', 'd_date_sk')],
   'catalog_returns': [('cc_call_center_sk', 'cr_call_center_sk')],
   'catalog_sales': [('cc_call_center_sk', 'cs_call_center_sk')]},
  'customer': {'catalog_returns': [('c_customer_sk',
     'cr_returning_customer_sk')],
   'catalog_sales': [('c_customer_sk', 'cs_ship_customer_sk')],
   'customer_address': [('c_current_addr_sk', 'ca_address_sk')],
   'customer_demographics': [('c_current_cdemo_sk', 'cd_demo_sk')],
   'household_demographics': [('c_current_hdemo_sk', 'hd_demo_sk')],
   'date_dim': [('c_first_shipto_date_sk', 'd_date_sk')],
   'store_returns': [('c_customer_sk', 'sr_customer_sk')],
   'store_sales': [('c_customer_sk', 'ss_customer_sk')],
   'web_returns': [('c_customer_sk', 'wr_returning_customer_sk')],
   'web_sales': [('c_customer_sk', 'ws_ship_customer_sk')]},
  'web_site': {'web_sales': [('web_site_sk', 'ws_web_site_sk')],
   'date_dim': [('web_open_date_sk', 'd_date_sk')]},
  'store_returns': {'customer_address': [('sr_addr_sk', 'ca_address_sk')],
   'customer_demographics': [('sr_cdemo_sk', 'cd_demo_sk')],
   'customer': [('sr_customer_sk', 'c_customer_sk')],
   'household_demographics': [('sr_hdemo_sk', 'hd_demo_sk')],
   'item': [('sr_item_sk', 'i_item_sk')],
   'reason': [('sr_reason_sk', 'r_reason_sk')],
   'date_dim': [('sr_returned_date_sk', 'd_date_sk')],
   'time_dim': [('sr_return_time_sk', 't_time_sk')],
   'store': [('sr_store_sk', 's_store_sk')]},
  'household_demographics': {'catalog_returns': [('hd_demo_sk',
     'cr_returning_hdemo_sk')],
   'catalog_sales': [('hd_demo_sk', 'cs_ship_hdemo_sk')],
   'customer': [('hd_demo_sk', 'c_current_hdemo_sk')],
   'income_band': [('hd_income_band_sk', 'ib_income_band_sk')],
   'store_returns': [('hd_demo_sk', 'sr_hdemo_sk')],
   'store_sales': [('hd_demo_sk', 'ss_hdemo_sk')],
   'web_returns': [('hd_demo_sk', 'wr_returning_hdemo_sk')],
   'web_sales': [('hd_demo_sk', 'ws_ship_hdemo_sk')]},
  'web_page': {'date_dim': [('wp_creation_date_sk', 'd_date_sk')],
   'web_returns': [('wp_web_page_sk', 'wr_web_page_sk')],
   'web_sales': [('wp_web_page_sk', 'ws_web_page_sk')]},
  'promotion': {'catalog_sales': [('p_promo_sk', 'cs_promo_sk')],
   'date_dim': [('p_start_date_sk', 'd_date_sk')],
   'item': [('p_item_sk', 'i_item_sk')],
   'store_sales': [('p_promo_sk', 'ss_promo_sk')],
   'web_sales': [('p_promo_sk', 'ws_promo_sk')]},
  'catalog_page': {'date_dim': [('cp_start_date_sk', 'd_date_sk')],
   'catalog_returns': [('cp_catalog_page_sk', 'cr_catalog_page_sk')],
   'catalog_sales': [('cp_catalog_page_sk', 'cs_catalog_page_sk')]},
  'inventory': {'date_dim': [('inv_date_sk', 'd_date_sk')],
   'item': [('inv_item_sk', 'i_item_sk')],
   'warehouse': [('inv_warehouse_sk', 'w_warehouse_sk')]},
  'catalog_returns': {'call_center': [('cr_call_center_sk',
     'cc_call_center_sk')],
   'catalog_page': [('cr_catalog_page_sk', 'cp_catalog_page_sk')],
   'item': [('cr_item_sk', 'i_item_sk')],
   'reason': [('cr_reason_sk', 'r_reason_sk')],
   'customer_address': [('cr_returning_addr_sk', 'ca_address_sk')],
   'customer_demographics': [('cr_returning_cdemo_sk', 'cd_demo_sk')],
   'customer': [('cr_returning_customer_sk', 'c_customer_sk')],
   'household_demographics': [('cr_returning_hdemo_sk', 'hd_demo_sk')],
   'date_dim': [('cr_returned_date_sk', 'd_date_sk')],
   'time_dim': [('cr_returned_time_sk', 't_time_sk')],
   'ship_mode': [('cr_ship_mode_sk', 'sm_ship_mode_sk')],
   'warehouse': [('cr_warehouse_sk', 'w_warehouse_sk')]},
  'web_returns': {'item': [('wr_item_sk', 'i_item_sk')],
   'reason': [('wr_reason_sk', 'r_reason_sk')],
   'customer_address': [('wr_returning_addr_sk', 'ca_address_sk')],
   'customer_demographics': [('wr_returning_cdemo_sk', 'cd_demo_sk')],
   'customer': [('wr_returning_customer_sk', 'c_customer_sk')],
   'household_demographics': [('wr_returning_hdemo_sk', 'hd_demo_sk')],
   'date_dim': [('wr_returned_date_sk', 'd_date_sk')],
   'time_dim': [('wr_returned_time_sk', 't_time_sk')],
   'web_page': [('wr_web_page_sk', 'wp_web_page_sk')]},
  'web_sales': {'customer_address': [('ws_ship_addr_sk', 'ca_address_sk')],
   'customer_demographics': [('ws_ship_cdemo_sk', 'cd_demo_sk')],
   'customer': [('ws_ship_customer_sk', 'c_customer_sk')],
   'household_demographics': [('ws_ship_hdemo_sk', 'hd_demo_sk')],
   'item': [('ws_item_sk', 'i_item_sk')],
   'promotion': [('ws_promo_sk', 'p_promo_sk')],
   'date_dim': [('ws_sold_date_sk', 'd_date_sk')],
   'ship_mode': [('ws_ship_mode_sk', 'sm_ship_mode_sk')],
   'time_dim': [('ws_sold_time_sk', 't_time_sk')],
   'warehouse': [('ws_warehouse_sk', 'w_warehouse_sk')],
   'web_page': [('ws_web_page_sk', 'wp_web_page_sk')],
   'web_site': [('ws_web_site_sk', 'web_site_sk')]},
  'catalog_sales': {'customer_address': [('cs_ship_addr_sk', 'ca_address_sk')],
   'customer_demographics': [('cs_ship_cdemo_sk', 'cd_demo_sk')],
   'customer': [('cs_ship_customer_sk', 'c_customer_sk')],
   'household_demographics': [('cs_ship_hdemo_sk', 'hd_demo_sk')],
   'call_center': [('cs_call_center_sk', 'cc_call_center_sk')],
   'catalog_page': [('cs_catalog_page_sk', 'cp_catalog_page_sk')],
   'item': [('cs_item_sk', 'i_item_sk')],
   'promotion': [('cs_promo_sk', 'p_promo_sk')],
   'date_dim': [('cs_sold_date_sk', 'd_date_sk')],
   'ship_mode': [('cs_ship_mode_sk', 'sm_ship_mode_sk')],
   'time_dim': [('cs_sold_time_sk', 't_time_sk')],
   'warehouse': [('cs_warehouse_sk', 'w_warehouse_sk')]},
  'store_sales': {'customer_address': [('ss_addr_sk', 'ca_address_sk')],
   'customer_demographics': [('ss_cdemo_sk', 'cd_demo_sk')],
   'customer': [('ss_customer_sk', 'c_customer_sk')],
   'household_demographics': [('ss_hdemo_sk', 'hd_demo_sk')],
   'item': [('ss_item_sk', 'i_item_sk')],
   'promotion': [('ss_promo_sk', 'p_promo_sk')],
   'date_dim': [('ss_sold_date_sk', 'd_date_sk')],
   'time_dim': [('ss_sold_time_sk', 't_time_sk')],
   'store': [('ss_store_sk', 's_store_sk')]}}}


    def data_init(database_info, dataname, dump_path):
        postgre = PostgreSQL(database_info['database_url'])
        database = DataBase(name=dataname)
        column_name = set()
        for table, columns in database_info['table_column'].items():
            database.table_name.append(table)
            table_item = Table(name=table)
            for column in columns:
                column_name.add(column)
                coltype = postgre.dtype(table, column)
                value = postgre.sample(table, column)
                value = [val[0] for val in value]
                column_item = Column(name=column, dtype=coltype, value=value)
                table_item.columns[column] = column_item
            database.tables[table] = table_item
        database.join_key = database_info['join_key']
        join_column = set([])
        for t in database.join_key.values():
            for c in t.values():
                for k in c:
                    join_column.add(k[0])
                    join_column.add(k[1])
        database.join_column = list(join_column)
        database.table_column = database_info['table_column']
        database.column_name = list(column_name)
        with dump_path.open("wb") as f:
            pickle.dump(database, f)
        return database

    if tpch_info_path.exists():
        tpch = pickle.load(tpch_info_path.open("rb"))
    else:
        tpch = data_init(database_tpch, "tpch", tpch_info_path)
    
    if tpcds_info_path.exists():
        tpcds = pickle.load(tpcds_info_path.open("rb"))
    else:
        tpcds = data_init(database_tpcds, "tpcds", tpcds_info_path)
    def __init__(self):
        pass
