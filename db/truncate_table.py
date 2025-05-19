import os

import oracledb
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
oracledb.init_oracle_client()

UN = os.environ.get("UN")
PW = os.environ.get("PW")
DSN = os.environ.get("DSN")

if __name__ == "__main__":
    print("Start Clearing Table Data")
    try:
        with oracledb.connect(user=UN, password=PW, dsn=DSN) as conn:
            with conn.cursor() as cursor:
                # cursor.execute("TRUNCATE TABLE multimodal_contents")
                # print("cleared table multimodal_contents")
                
                cursor.execute("TRUNCATE TABLE image_contents")
                print("cleared table image_contents")
                
                cursor.execute("TRUNCATE TABLE docs_contents")
                print("cleared table docs_contents")
                
                cursor.execute("TRUNCATE TABLE uploaded_files")
                print("cleared table uploaded_files")
                
            conn.commit()
            print("End Clearing Tables")
    except Exception as e:
        print("Error clearing table data:", e)