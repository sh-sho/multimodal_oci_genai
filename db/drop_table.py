import os

import oracledb
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()

UN = os.environ.get("UN")
PW = os.environ.get("PW")
DSN = os.environ.get("DSN")

if __name__ == "__main__":
    print("Start Dropping Table")
    try:
        with oracledb.connect(user=UN, password=PW, dsn=DSN) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS multimodal_contents")
                print("drop table multimodal_contents")
                
                cursor.execute("DROP TABLE IF EXISTS image_contents")
                print("drop table image_contents")
                
                cursor.execute("DROP TABLE IF EXISTS docs_contents")
                print("drop table docs_contents")
                
                cursor.execute("DROP TABLE IF EXISTS uploaded_files")
                print("drop table uploaded_files")
                
            conn.commit()
            print("End Drop Tables")
    except Exception as e:
        print("Error drop_table:", e)