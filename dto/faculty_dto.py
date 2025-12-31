from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import re, html

_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = _TAG_RE.sub("", s)
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None

def normalize_newlines(s: Optional[str]) -> Optional[str]:
    """
    Keep newlines (address), normalize line endings, trim trailing spaces.
    """
    if s is None:
        return None
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    out = "\n".join(lines).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out or None

def normalize_list_str(xs: Any) -> Optional[List[str]]:
    if not xs:
        return None
    if isinstance(xs, list):
        out = [str(x).strip() for x in xs if str(x).strip()]
        return out or None
    # tolerate accidental string
    if isinstance(xs, str):
        t = xs.strip()
        return [t] if t else None
    return None


# ============================================================
# Parent persistence DTO (matches FacultyDAO.faculty_row fields)
# ============================================================
@dataclass
class FacultyPersistenceDTO:
    # natural key for upsert
    source_url: str

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    position: Optional[str] = None

    organization: Optional[str] = None
    organizations: Optional[List[str]] = None  # JSON list

    address: Optional[str] = None              # keep newlines
    biography: Optional[str] = None            # text

    research_website_name: Optional[str] = None
    research_website_url: Optional[str] = None

    @staticmethod
    def from_profile_dict(d: Dict[str, Any]) -> "FacultyPersistenceDTO":
        if not d or not d.get("source_url"):
            raise ValueError("profile must include source_url")

        rw = d.get("research_website") or {}
        return FacultyPersistenceDTO(
            source_url=d["source_url"],
            name=d.get("name"),
            email=d.get("email"),
            phone=d.get("phone"),
            position=d.get("position"),
            organization=d.get("organization"),
            organizations=normalize_list_str(d.get("organizations")),
            address=normalize_newlines(d.get("address")),
            # biography is already parsed as text; strip_html is just extra safety
            biography=strip_html(d.get("biography")),
            research_website_name=rw.get("name"),
            research_website_url=rw.get("url"),
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "position": self.position,
            "organization": self.organization,
            "organizations": self.organizations,
            "address": self.address,
            "biography": self.biography,
            "research_website_name": self.research_website_name,
            "research_website_url": self.research_website_url,
        }


# ============================================================
# Child persistence DTOs (match your DAO child row shapes)
# ============================================================
@dataclass
class FacultyDegreePersistenceDTO:
    order_index: int
    degree_text: str

    @staticmethod
    def list_from_profile_dict(d: Dict[str, Any]) -> List["FacultyDegreePersistenceDTO"]:
        out: List[FacultyDegreePersistenceDTO] = []
        for i, t in enumerate(d.get("degrees") or []):
            if t and str(t).strip():
                out.append(FacultyDegreePersistenceDTO(order_index=i, degree_text=str(t).strip()))
        return out

    def to_row(self) -> Dict[str, Any]:
        return {"order_index": self.order_index, "degree_text": self.degree_text}


@dataclass
class FacultyExpertisePersistenceDTO:
    term: str

    @staticmethod
    def list_from_profile_dict(d: Dict[str, Any]) -> List["FacultyExpertisePersistenceDTO"]:
        out: List[FacultyExpertisePersistenceDTO] = []
        for t in (d.get("research_expertise") or []):
            if t and str(t).strip():
                out.append(FacultyExpertisePersistenceDTO(term=str(t).strip()))
        return out

    def to_row(self) -> Dict[str, Any]:
        return {"term": self.term}


@dataclass
class FacultyGroupPersistenceDTO:
    name: Optional[str] = None
    url: Optional[str] = None

    @staticmethod
    def list_from_profile_dict(d: Dict[str, Any]) -> List["FacultyGroupPersistenceDTO"]:
        out: List[FacultyGroupPersistenceDTO] = []
        for g in (d.get("research_groups") or []):
            if not g:
                continue
            nm = (g.get("name") or "").strip() if isinstance(g, dict) else None
            url = (g.get("url") or "").strip() if isinstance(g, dict) else None
            if nm or url:
                out.append(FacultyGroupPersistenceDTO(name=nm or None, url=url or None))
        return out

    def to_row(self) -> Dict[str, Any]:
        return {"name": self.name, "url": self.url}


@dataclass
class FacultyLinkPersistenceDTO:
    name: str
    url: str

    @staticmethod
    def list_from_profile_dict(d: Dict[str, Any]) -> List["FacultyLinkPersistenceDTO"]:
        out: List[FacultyLinkPersistenceDTO] = []
        for lk in (d.get("additional_links") or []):
            if not lk or not isinstance(lk, dict):
                continue
            nm = (lk.get("name") or "").strip()
            url = (lk.get("url") or "").strip()
            if nm and url:
                out.append(FacultyLinkPersistenceDTO(name=nm, url=url))
        return out

    def to_row(self) -> Dict[str, Any]:
        return {"name": self.name, "url": self.url}


# Optional (since parser returns awards). Use if you have/will add a table.
@dataclass
class FacultyAwardPersistenceDTO:
    order_index: int
    award_text: str

    @staticmethod
    def list_from_profile_dict(d: Dict[str, Any]) -> List["FacultyAwardPersistenceDTO"]:
        out: List[FacultyAwardPersistenceDTO] = []
        for i, t in enumerate(d.get("awards") or []):
            if t and str(t).strip():
                out.append(FacultyAwardPersistenceDTO(order_index=i, award_text=str(t).strip()))
        return out

    def to_row(self) -> Dict[str, Any]:
        return {"order_index": self.order_index, "award_text": self.award_text}


@dataclass
class FacultyPublicationPersistenceDTO:
    faculty_id: int
    scholar_author_id: str
    title: str
    year: Optional[int]
    abstract: Optional[str] = None
    openalex_work_id: Optional[str] = None

# ============================================================
# Bundle (opportunity-style “whole response mapper” equivalent)
# ============================================================
@dataclass
class FacultyBundlePersistenceDTO:
    faculty: FacultyPersistenceDTO
    degrees: List[FacultyDegreePersistenceDTO]
    expertise: List[FacultyExpertisePersistenceDTO]
    groups: List[FacultyGroupPersistenceDTO]
    links: List[FacultyLinkPersistenceDTO]
    awards: List[FacultyAwardPersistenceDTO]  # optional downstream

    @staticmethod
    def from_profile_dict(d: Dict[str, Any]) -> "FacultyBundlePersistenceDTO":
        return FacultyBundlePersistenceDTO(
            faculty=FacultyPersistenceDTO.from_profile_dict(d),
            degrees=FacultyDegreePersistenceDTO.list_from_profile_dict(d),
            expertise=FacultyExpertisePersistenceDTO.list_from_profile_dict(d),
            groups=FacultyGroupPersistenceDTO.list_from_profile_dict(d),
            links=FacultyLinkPersistenceDTO.list_from_profile_dict(d),
            awards=FacultyAwardPersistenceDTO.list_from_profile_dict(d),
        )

    def to_dao_args(self, *, include_awards: bool = False) -> Dict[str, Any]:
        """
        Returns kwargs compatible with FacultyDAO.upsert_one_bundle(...)

        If you don't yet have a faculty_awards child table/DAO,
        keep include_awards=False and ignore awards completely.
        """
        args = {
            "faculty_row": self.faculty.to_row(),
            "degrees": [x.to_row() for x in self.degrees],
            "expertise": [x.to_row() for x in self.expertise],
            "groups": [x.to_row() for x in self.groups],
            "links": [x.to_row() for x in self.links],
        }
        if include_awards:
            args["awards"] = [x.to_row() for x in self.awards]
        return args


def build_faculty_bundle(profile: Dict[str, Any]) -> FacultyBundlePersistenceDTO:
    return FacultyBundlePersistenceDTO.from_profile_dict(profile)