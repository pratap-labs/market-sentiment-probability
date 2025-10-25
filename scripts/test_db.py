import psycopg2
import os


# DB_URL = "postgres://postgres:1234_Niftydb@aws-0-eu-north-1.pooler.supabase.com:5432/postgres"

# # Database credentials
# DATABASE_PASSWORD = '1234_Niftydb'
# DATABASE_USER = 'nifty-root'
# DATABASE_HOST = 'nifty-database.c7iioca4yp2m.eu-north-1.rds.amazonaws.com'
# DATABASE_NAME = 'nifty-database'
# DATABASE_PORT = 5432

# try:
#     # Connect to the PostgreSQL database
#     conn = psycopg2.connect(DB_URL)
#     print("✅ Connection successful!")
#     conn.close()
# except Exception as e:
#     print("❌ Connection failed:", e)


import os
from supabase import create_client, Client



url: str = 'https://ehojxadxaqolqqshlzcd.supabase.co'
key: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVob2p4YWR4YXFvbHFxc2hsemNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjExMzIzMDYsImV4cCI6MjA3NjcwODMwNn0.nWeFny73-GybqDWqKVe96j4Ojph-Z8maGfw1sYDtC1M'


try:
    supabase: Client = create_client(url, key)
    print(supabase)
except Exception as e:
    print("❌ Connection failed:", e)
