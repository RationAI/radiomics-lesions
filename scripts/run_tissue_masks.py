from kube_jobs import storage, submit_job


submit_job(
    job_name="project_name-tissue-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",  # approximately 4GiB per process
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/project_name workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.tissue_masks +data=<dataset>",
    ],
    storage=[storage.secure.DATA],
)
