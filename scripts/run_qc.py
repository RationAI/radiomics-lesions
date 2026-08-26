from kube_jobs import storage, submit_job


submit_job(
    job_name="project_name-qc",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/project_name workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.qc +data=<dataset>",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
