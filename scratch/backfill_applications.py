"""Backfill script to generate evaluation and tailored applications for pending jobs."""

import asyncio
from config.settings import get_settings
from core.analyzer import JobAnalyzer
from core.generator import (
    JobGenerator,
    TailoredApplication,
    export_application_markdown,
)
from db.database import JobDatabase
from db.models import JobStatus


async def backfill() -> None:
    """Backfill evaluation and application payloads for pending review jobs."""
    settings = get_settings()
    db = JobDatabase(settings.database_path)
    await db.init_db()

    analyzer = JobAnalyzer(
        api_key=settings.gemini_api_key.get_secret_value(),
        model_name=settings.gemini_model,
        profile_path=settings.candidate_profile_path,
    )
    generator = JobGenerator(
        api_key=settings.gemini_api_key.get_secret_value(),
        model_name=settings.gemini_model,
        profile_path=settings.candidate_profile_path,
    )

    pending_jobs = await db.get_jobs_by_status(
        JobStatus.PENDING_REVIEW, limit=100
    )
    print(f"Trouvé {len(pending_jobs)} offre(s) PENDING_REVIEW.")

    for idx, job in enumerate(pending_jobs, start=1):
        if job.id is None:
            continue

        payload = dict(job.raw_payload or {})
        eval_data = payload.get("evaluation")
        app_data = payload.get("application")

        # 1. Évaluation Gemini si manquante
        if not eval_data:
            print(
                f"[{idx}/{len(pending_jobs)}] Analyse Gemini pour #{job.id} "
                f"({job.company} - {job.title[:30]})..."
            )
            eval_res = await analyzer.analyze_job(
                title=job.title,
                company=job.company,
                description=job.description,
                platform=job.platform,
            )
            eval_data = {
                "score": eval_res.score,
                "reasoning": eval_res.reasoning,
                "key_matches": eval_res.key_matches,
                "missing_skills": eval_res.missing_skills,
                "contract_type": eval_res.contract_type,
            }
            payload["evaluation"] = eval_data

        # 2. Génération lettre de motivation et pitch si manquants
        if not app_data:
            print(
                f"[{idx}/{len(pending_jobs)}] Génération pitch/lettre pour "
                f"#{job.id} ({job.company})..."
            )
            app_doc = await generator.generate_application(
                title=job.title,
                company=job.company,
                description=job.description,
                key_matches=eval_data.get("key_matches"),
                missing_skills=eval_data.get("missing_skills"),
                platform=job.platform,
            )
            app_data = {
                "language": app_doc.language,
                "subject_line": app_doc.subject_line,
                "quick_pitch": app_doc.quick_pitch,
                "full_cover_letter": app_doc.full_cover_letter,
            }
            payload["application"] = app_data
        else:
            app_doc = TailoredApplication(
                language=str(app_data.get("language") or "fr"),
                subject_line=str(
                    app_data.get("subject_line")
                    or f"Candidature - {job.title}"
                ),
                quick_pitch=str(app_data.get("quick_pitch") or ""),
                full_cover_letter=str(app_data.get("full_cover_letter") or ""),
            )

        # 3. Sauvegarde dans SQLite
        await db.update_job_payload(job.id, payload)

        # 4. Export Markdown dans Applications/
        md_file = export_application_markdown(
            job_id=job.id,
            company=job.company,
            title=job.title,
            url=job.url,
            score=job.score or eval_data.get("score"),
            contract_type=job.contract_type or eval_data.get("contract_type"),
            application=app_doc,
            key_matches=eval_data.get("key_matches"),
            missing_skills=eval_data.get("missing_skills"),
            reasoning=eval_data.get("reasoning"),
        )
        print(f"  ✓ #{job.id} sauvegardé -> {md_file}")

    print("\n✅ Rétro-alimentation (backfill) terminée avec succès !")


if __name__ == "__main__":
    asyncio.run(backfill())
