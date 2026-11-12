"""Tests for A1 atomic evaluation locks, B1 dedup, and selective requeue."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

import aiosqlite

from db.database import (
    JobDatabase,
    clean_job_title,
    generate_all_fingerprints,
    normalize_job_url,
)
from db.models import JobRecord, JobStatus
from scrapers.engine import ScraperEngine


async def test_a1_concurrency_and_notified_guard(db_path: Path) -> None:
    """Verify atomic claiming, stale lock recovery, and single notification."""
    db = JobDatabase(db_path)
    await db.init_db()

    for idx in range(1, 7):
        s_hash, l_hash, tl_hash = generate_all_fingerprints(
            company=f"Firm {idx}",
            title=f"AI Engineer Role {idx}",
            location="Marseille",
        )
        await db.insert_job(
            JobRecord(
                platform="wttj",
                company=f"Firm {idx}",
                title=f"AI Engineer Role {idx}",
                url=f"https://example.com/jobs/{idx}",
                normalized_url=f"https://example.com/jobs/{idx}",
                description="Full job description " * 25,
                content_hash=s_hash,
                loose_hash=l_hash,
                title_loc_hash=tl_hash,
                status=JobStatus.NEW,
            )
        )

    # Réclamation ciblée (ex. mode /indeed) vs lot global concurrent
    batch_targeted, batch_global = await asyncio.gather(
        db.claim_pending_new_jobs(limit=100, job_ids=[5, 6]),
        db.claim_pending_new_jobs(limit=100),
    )
    ids_a = {job.id for job in batch_targeted}
    ids_b = {job.id for job in batch_global}
    assert ids_a.isdisjoint(ids_b), f"Overlap detected: {ids_a & ids_b}"
    assert ids_a | ids_b == {1, 2, 3, 4, 5, 6}

    # Test de libération en cas d'échec
    assert await db.release_claimed_job(1)
    reclaimed = await db.claim_pending_new_jobs(limit=10)
    assert [job.id for job in reclaimed] == [1]

    # Test de reprise d'un verrou expiré (EVALUATING > 10 min)
    stale_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "UPDATE jobs SET updated_at = ? WHERE id = 2;",
            (stale_time,),
        )
        await conn.commit()
    recovered = await db.claim_pending_new_jobs(limit=10)
    assert [job.id for job in recovered] == [2]

    # Verrou atomique d'envoi Telegram
    res1, res2, res3 = await asyncio.gather(
        db.mark_job_notified(3),
        db.mark_job_notified(3),
        db.mark_job_notified(3),
    )
    assert sum([res1, res2, res3]) == 1
    assert await db.clear_job_notified(3)
    assert await db.mark_job_notified(3)


async def test_b1_dedup_and_richer_requeue(db_path: Path) -> None:
    """Verify B1 cross-source dedup, non-false-positive, and richer requeue."""
    db = JobDatabase(db_path)
    await db.init_db()
    engine = ScraperEngine(db=db)

    assert clean_job_title("Ingénieur développeur IA X/ F/H") == (
        "ingenieur developpeur ia"
    )
    assert clean_job_title("Ingénieur IA Expérimenté CDI H/F ALTEN", "ALTEN") == (
        "ingenieur ia experimente"
    )
    assert normalize_job_url("https://www.hellowork.com/fr-fr/emplois/1.html") == (
        normalize_job_url("https://hellowork.com/fr-fr/emplois/1.html")
    )

    # 1. Offre courte rejetée -> doublon nettement plus complet -> ré-évaluation
    s1, l1, tl1 = generate_all_fingerprints(
        "HOUSE OF ABY", "Ingénieur développeur IA X/ F/H", "Marseille"
    )
    saved_aby = await db.insert_job(
        JobRecord(
            platform="linkedin",
            company="HOUSE OF ABY",
            title="Ingénieur développeur IA X/ F/H",
            url="https://linkedin.com/jobs/view/100",
            normalized_url="https://linkedin.com/jobs/view/100",
            description="Ingénieur développeur IA X/ F/H",
            content_hash=s1,
            loose_hash=l1,
            title_loc_hash=tl1,
            score=40,
            status=JobStatus.REJECTED,
        )
    )
    assert saved_aby is not None and saved_aby.id is not None

    # Insertion simulée d'un doublon plus riche via find_existing_duplicate
    rich_desc = "Description complète Python LLM RAG Agentique Marseille " * 60
    s2, l2, tl2 = generate_all_fingerprints(
        "HOUSE OF ABY", "Ingénieur Développeur IA H/F", "Marseille"
    )
    dup, rule = await db.find_existing_duplicate(
        content_hash=s2,
        loose_hash=l2,
        title_loc_hash=tl2,
        company="HOUSE OF ABY",
        title="Ingénieur Développeur IA H/F",
        description=rich_desc,
    )
    assert dup is not None and dup.id == saved_aby.id and rule == "strict"

    _, requeued = await db.merge_duplicate_offer(
        job_id=saved_aby.id,
        platform="france_travail",
        url="https://candidat.francetravail.fr/offres/recherche/detail/8594360",
        description=rich_desc,
        company="HOUSE OF ABY",
    )
    assert requeued is True
    updated_aby = await db.get_job_by_id(saved_aby.id)
    assert updated_aby is not None and updated_aby.status == JobStatus.NEW
    assert len(updated_aby.description) == len(rich_desc.strip())

    # 2. Entreprise différente avec titre générique court -> PAS de faux doublon
    s3, l3, tl3 = generate_all_fingerprints(
        "ABYLSEN", "Ingénieur développeur IA X/F/H (CDI)", "Marseille"
    )
    dup_abylsen, _ = await db.find_existing_duplicate(
        content_hash=s3,
        loose_hash=l3,
        title_loc_hash=tl3,
        company="ABYLSEN",
        title="Ingénieur développeur IA X/F/H (CDI)",
        description="Autre mission totalement différente chez Abylsen " * 40,
    )
    assert dup_abylsen is None

    # 3. Enseigne masquée + enseigne groupe sur titre long -> doublon B1
    cs_desc = (
        "Au sein de la BU Défense & Sécurité de CS GROUP à Aix-en-Provence, "
        "stage Reinforcement Learning et Python avancé. "
    ) * 20
    cs_title = (
        "Stage Ingénieur/e IA – Reinforcement Learning - "
        "Défense & Sécurité - Aix-en-Provence"
    )
    scs, lcs, tlcs = generate_all_fingerprints(
        "CS",
        cs_title,
        "Aix-en-Provence",
    )
    saved_cs = await db.insert_job(
        JobRecord(
            platform="wttj",
            company="CS",
            title=cs_title,
            url="https://welcometothejungle.com/fr/companies/cs/jobs/1",
            normalized_url="https://welcometothejungle.com/fr/companies/cs/jobs/1",
            description=cs_desc,
            content_hash=scs,
            loose_hash=lcs,
            title_loc_hash=tlcs,
            score=85,
            status=JobStatus.PENDING_REVIEW,
        )
    )
    assert saved_cs is not None and saved_cs.id is not None

    ft_title = (
        "Stage Ingénieur/e IA - Reinforcement Learning - "
        "Défense & Sécurité - Aix-en-Provence H/F"
    )
    sft, lft, tlft = generate_all_fingerprints(
        "Enseigne non disponible",
        ft_title,
        "13 - Aix-en-Provence",
    )
    dup_ft, rule_ft = await db.find_existing_duplicate(
        content_hash=sft,
        loose_hash=lft,
        title_loc_hash=tlft,
        company="Enseigne non disponible",
        title=ft_title,
        description=cs_desc[:1200],
    )
    assert dup_ft is not None and dup_ft.id == saved_cs.id
    assert rule_ft == "title_loc_anonymous"

    # Version stub Indeed Sopra Steria -> détectée comme doublon via title_loc
    sind, lind, tlind = generate_all_fingerprints(
        "Sopra Steria",
        cs_title,
        "13290 Aix-en-Provence",
    )
    dup_ind, rule_ind = await db.find_existing_duplicate(
        content_hash=sind,
        loose_hash=lind,
        title_loc_hash=tlind,
        company="Sopra Steria",
        title=cs_title,
        description=(
            "Stage Ingénieur/e IA – Reinforcement Learning - "
            "Défense & Sécurité"
        ),
    )
    assert dup_ind is not None and dup_ind.id == saved_cs.id
    assert rule_ind == "title_loc"
    del engine


async def main() -> None:
    """Execute test suite inside temporary directories."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        await test_a1_concurrency_and_notified_guard(tmp_dir / "a1.db")
        await test_b1_dedup_and_richer_requeue(tmp_dir / "b1.db")
    print("ALL A1 + B1 TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
