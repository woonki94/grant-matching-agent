from data.db_conn import SessionLocal
from services.keyword_mining_service import mine_keywords_for_all

if __name__ == "__main__":
    with SessionLocal() as sess:
        report = mine_keywords_for_all(sess, batch_size=50, max_keywords=25)
        print(report)