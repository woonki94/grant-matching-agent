from __future__ import annotations
from dataclasses import asdict
from typing import List
from flask import Flask, request, jsonify

from test.api_call_test_service import (
    build_search_request,
    search_opportunities,
    enrich_search_with_attachments,  # ← new
)
app = Flask(__name__)

def _csv_arg(name: str) -> List[str] | None:
    raw = request.args.get(name)
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]

@app.route("/portal/search", methods=["GET"])
def portal_search_get():
    try:
        page_offset = int(request.args.get("page_offset", 1))
        page_size = int(request.args.get("page_size", 5))
        order_by = request.args.get("order_by", "post_date")
        sort_direction = request.args.get("sort_direction", "descending")
        statuses = _csv_arg("opportunity_status") or ["forecasted", "posted"]
        q = request.args.get("q")

        if page_offset < 1:
            page_offset = 1
        if page_size < 1 or page_size > 50:
            page_size = 5

        req_dto = build_search_request(
            page_offset=page_offset,
            page_size=page_size,
            order_by=order_by,
            sort_direction=sort_direction,
            statuses=statuses,
            q=q,
        )
        dto = search_opportunities(req_dto)
        dto = enrich_search_with_attachments(dto)


        return jsonify(asdict(dto))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, port=5001)