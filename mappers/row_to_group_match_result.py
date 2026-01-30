from typing import Dict, Any

from dto.group_result_dto import GroupQualityDTO, GroupMatchResultDTO, GroupMatchMetaDTO


def group_match_result_from_row(row: Dict[str, Any]) -> GroupMatchResultDTO:
    q = row["meta"]["quality"]

    quality = GroupQualityDTO(
        cov_app=q["cov_app"],
        cov_res=q["cov_res"],
        cov_total=q["cov_total"],
        breadth_app=q["breadth_app"],
        breadth_res=q["breadth_res"],
        breadth_total=q["breadth_total"],
        critical_hit_app=q["critical_hit_app"],
        critical_hit_res=q["critical_hit_res"],
        critical_hit_total=q["critical_hit_total"],
    )

    meta = GroupMatchMetaDTO(
        algo=row["meta"]["algo"],
        penalty_source=row["meta"]["penalty_source"],
        lambda_grid=row["meta"]["lambda_grid"],
        quality=quality,
    )

    return GroupMatchResultDTO(
        group_id=row["group_id"],
        grant_id=row["grant_id"],
        lambda_=row["lambda"],
        k=row["k"],
        top_n=row["top_n"],
        objective=row["objective"],
        redundancy=row["redundancy"],
        status=row["status"],
        alpha=row["alpha"],
        meta=meta,
    )