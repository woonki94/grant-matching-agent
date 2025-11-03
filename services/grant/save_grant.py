#TODO: Make everything absolute path
import sys

from services.grant.call_grant import run_search_pipeline
from dataclasses import asdict
from db.db_conn import SessionLocal
from db.dao.grant import OpportunityDAO, AttachmentDAO


if __name__ == "__main__":
    # Fetch page 1, 5 results
    page_offset = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    page_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    query = sys.argv[3] if len(sys.argv) > 3 else "Robotics"

    response_dto, opps_p, atts_p = run_search_pipeline(page_offset=page_offset, page_size=page_size, q=query)

    print("Search completed.")
    print(f"Found {len(response_dto.data)} opportunities.")

    # Show summarized output
    for o in opps_p:
        print(f"\n{o.opportunity_id}: {o.opportunity_title}")
        print(f"   Agency: {o.agency_name}")
        print(f"   Category: {o.category}")
        print(f"   Status: {o.opportunity_status}")
        print(f"   Post Date: {o.post_date}")

    print(f"\nAttachments fetched: {len(atts_p)} total")
    if atts_p:
        print("Example attachment:", asdict(atts_p[0]))

    with SessionLocal() as sess:
        OpportunityDAO.upsert_many(sess, [asdict(o) for o in opps_p])
        AttachmentDAO.upsert_attachments_batch_with_content(sess,
                                                            [asdict(a) for a in atts_p],
                                                            fetch_content=True)
        sess.commit()
        print("Stored opportunities + attachments in DB.")