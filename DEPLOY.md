# Deploying DocQA

DocQA needs two things: the container in this repo and a PostgreSQL 15+
database where the `vector` extension can be created. The app creates its own
tables on startup.

## Option A: Google Cloud Run + Neon (free tiers)

1. Create a free Postgres database at [neon.tech](https://neon.tech). pgvector
   is available there by default. Copy the connection string, which looks like
   `postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require`.
2. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), then:

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

   gcloud run deploy docqa \
     --source . \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --set-env-vars DOCQA_DATABASE_URL='postgresql://...',DOCQA_LLM_API_KEY='sk-...'
   ```

   Cloud Run builds the Dockerfile and prints a public `https://docqa-xxxx.run.app` URL.
   For anything beyond a demo, store the API key in Secret Manager and use
   `--set-secrets` instead of `--set-env-vars`.

To use Cloud SQL for PostgreSQL instead of Neon, enable the extension with
`CREATE EXTENSION vector;` as the `postgres` user once, then pass the Cloud SQL
connection string.

## Option B: Render

1. Create a **PostgreSQL** instance on Render and copy its *Internal Database URL*.
2. Create a **Web Service** from this GitHub repo with runtime **Docker**.
3. Add the environment variables `DOCQA_DATABASE_URL` (the URL from step 1)
   and, optionally, `DOCQA_LLM_API_KEY`.

## Notes

- Uploaded files are not kept; only the extracted text chunks and embeddings
  are stored in Postgres, so the container itself is stateless.
- A public instance with no authentication lets anyone upload documents. For a
  public demo, set `DOCQA_MAX_UPLOAD_MB` low and keep an eye on the database size.
