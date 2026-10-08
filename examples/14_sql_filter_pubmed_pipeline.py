#!/usr/bin/env python
"""SQLFilter -- pre-filter a corpus before search runs, over real PubMed articles.

Sibling to examples/12_sql_filter_pipeline.py (arXiv metadata) -- same
SQLFilter mechanics (see that file's docstring for the full `where()` /
`where_any()` / `filter_by_roles()` rundown), demonstrated here against a
differently-shaped real dataset: PubMed's MEDLINE citation record is deeply
nested (MedlineCitation -> Article -> Abstract -> AbstractText, a separate
MeshHeadingList, etc.), unlike arXiv's flat one-JSON-object-per-paper schema.

There is no ready-to-use PubMed dataset on HuggingFace's `datasets` library
today -- the official `ncbi/pubmed` listing is a legacy loading-script
dataset, and `datasets` (5.x) no longer supports loading-script datasets at
all ("Dataset scripts are no longer supported"); its auto-converted-Parquet
mirror is empty too (the conversion never ran for it). So this example
fetches directly from NCBI's own E-utilities API instead (esearch for PMIDs,
efetch for the full MEDLINE XML record) -- stdlib `urllib`/`xml.etree` only,
no extra dependency, and it's the same live, canonical, always-current data
`ncbi/pubmed` would have given you anyway.

MeSH headings are the multi-valued attribute here (a paper is indexed under
several MeSH terms at once) -- the same shape as arXiv's `categories`, just
under a different name (`mesh_terms`), which is exactly the point: the same
`where_any()` mechanism applies unchanged across two very differently
shaped real corpora. PubMed has no real access-control data either, so the
role demo repurposes a real field (whether a paper is tagged as a clinical
trial in its own PublicationTypeList) as a stand-in for "who can see this
doc", same as arXiv's "published vs. preprint" repurposing.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from simlar import BM25CIndex
from simlar_engine import SQLFilter

N_ARTICLES = 300
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_BATCH = 150


def _esearch(term: str, retmax: int) -> list[str]:
    params = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"})
    with urllib.request.urlopen(f"{_EUTILS}/esearch.fcgi?{params}", timeout=20) as r:
        return json.load(r)["esearchresult"]["idlist"]


def _efetch(pmids: list[str]) -> ET.Element:
    params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    with urllib.request.urlopen(f"{_EUTILS}/efetch.fcgi?{params}", timeout=30) as r:
        return ET.fromstring(r.read())


def _text(el: ET.Element | None) -> str | None:
    return "".join(el.itertext()).strip() or None if el is not None else None


def _flatten_article(art: ET.Element) -> dict | None:
    """Pull out just what this example needs, defensively -- PubMed records
    genuinely vary in which fields are present, so anything missing what's
    required to index/filter this doc just means skipping it, not crashing."""
    pmid = _text(art.find(".//PMID"))
    title = _text(art.find(".//ArticleTitle"))
    abstract_parts = [_text(a) for a in art.findall(".//Abstract/AbstractText")]
    abstract = " ".join(p for p in abstract_parts if p) or None
    if not pmid or not title or not abstract:
        return None

    journal = _text(art.find(".//Journal/Title")) or "unknown"
    mesh_terms = [
        t for t in (_text(d) for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")) if t
    ]
    pub_types = [t for t in (_text(p) for p in art.findall(".//PublicationTypeList/PublicationType")) if t]
    is_clinical = any("trial" in t.lower() for t in pub_types)

    date_el = art.find(".//DateCompleted")
    if date_el is None:
        date_el = art.find(".//DateRevised")
    if date_el is not None:
        year = _text(date_el.find("Year")) or "0000"
        month = _text(date_el.find("Month")) or "01"
        day = _text(date_el.find("Day")) or "01"
        date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    else:
        date = "0000-01-01"

    return {
        "id": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "date": date,
        "mesh_terms": mesh_terms,
        "is_clinical": is_clinical,
    }


def _load_articles(n: int) -> list[dict]:
    # A dedicated clinical-trial-tagged search merged with a broad one,
    # rather than one plain query -- guarantees a meaningful clinical vs.
    # research split for the access-control demo below (a plain "diabetes"
    # search alone turns up only a couple of clinical trials out of 300).
    clinical_ids = _esearch("diabetes AND clinical trial[pt]", 100)
    broad_ids = _esearch("diabetes", n)
    pmids = list(dict.fromkeys(clinical_ids + broad_ids))[:n]

    articles = []
    for i in range(0, len(pmids), _BATCH):
        root = _efetch(pmids[i : i + _BATCH])
        for art in root.findall(".//PubmedArticle"):
            flat = _flatten_article(art)
            if flat is not None:
                articles.append(flat)
        time.sleep(0.4)  # stay well under NCBI's 3-req/sec guideline
    return articles


def main() -> None:
    print(f"Fetching up to {N_ARTICLES} PubMed articles on 'diabetes' via NCBI E-utilities...")
    articles = _load_articles(N_ARTICLES)
    print(f"Got {len(articles)} articles with a usable title + abstract.\n")

    ids = [a["id"] for a in articles]
    texts = [f"{a['title']}. {a['abstract']}" for a in articles]

    index = BM25CIndex()
    index.add(ids, texts)

    filt = SQLFilter()
    filt.add(
        ids,
        [{"journal": a["journal"], "date": a["date"], "mesh_terms": a["mesh_terms"]} for a in articles],
    )
    for a in articles:
        filt.grant([a["id"]], ["clinical"] if a["is_clinical"] else ["research"])

    top_journal = max({a["journal"] for a in articles}, key=lambda j: sum(a["journal"] == j for a in articles))
    query = "insulin resistance treatment"

    print(f"Most common journal in this sample: {top_journal!r}\n")
    print(f'Query: "{query}"\n')

    print("-- Unrestricted (candidates=None) --")
    unrestricted = index.search(query, k=5, candidates=None)
    _print(unrestricted)

    print(f"\n-- Scalar filter: journal = {top_journal!r} (where) --")
    scalar_candidates = filt.where("journal", "=", top_journal).positions()
    _print_results(index, query, candidates=scalar_candidates)

    # Pick a MeSH term actually shared by one of the unrestricted hits, so
    # the "before vs. after" comparison is grounded in this run's real data.
    mesh_by_id = {a["id"]: a["mesh_terms"] for a in articles}
    date_by_id = {a["id"]: a["date"] for a in articles}
    shared_mesh_term = next((t for r in unrestricted for t in mesh_by_id[r.id]), "Humans")
    print(f"\n-- Multi-valued filter: mesh_terms includes {shared_mesh_term!r} (where_any) --")
    multi_candidates = filt.where_any("mesh_terms", "=", shared_mesh_term).positions()
    _print_results(index, query, candidates=multi_candidates)

    # Median date among this term's own matches, so the date cutoff below
    # is guaranteed to actually split this specific set roughly in half --
    # a fixed hardcoded date might land entirely on one side by chance.
    matching_dates = sorted(date_by_id[ids[p]] for p in multi_candidates)
    cutoff_date = matching_dates[len(matching_dates) // 2]

    print(f"\n-- Chained: date >= {cutoff_date} AND mesh_terms includes {shared_mesh_term!r} --")
    chained_candidates = (
        filt.where("date", ">=", cutoff_date).where_any("mesh_terms", "=", shared_mesh_term).positions()
    )
    _print_results(index, query, candidates=chained_candidates)
    print(f"   ({len(chained_candidates)} of {len(multi_candidates)} {shared_mesh_term!r}-tagged articles pass the date cutoff)")

    print("\n-- Access filter: what a 'clinical' (trial-tagged) role can see --")
    clinical_visible = filt.filter_by_roles(["clinical"])
    _print_results(index, query, candidates=clinical_visible)
    print(f"   ({len(clinical_visible)} of {len(ids)} articles are clinical-trial-tagged)")


def _print_results(index: BM25CIndex, query: str, candidates) -> None:
    _print(index.search(query, k=5, candidates=candidates))


def _print(results) -> None:
    if not results:
        print("  (no matches)")
    for r in results:
        if r.id is not None:
            print(f"  PMID {r.id:10s} score={r.score:.3f}")


if __name__ == "__main__":
    main()
