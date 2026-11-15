# Deploy CKDPredict on Hugging Face Spaces (free, Docker)

1. Create a free account at https://huggingface.co and click **New → Space**.
2. Name: `ckdpredict` · SDK: **Docker** (Blank) · Hardware: **CPU basic (free)** · Visibility: **Public** (or Private) → Create Space.
3. **Settings → Variables and secrets → New secret** (add both):
   - `APP_PASSWORD` = the password for your demo (protects the patient-style data)
   - `SECRET_KEY`  = any long random text
4. Upload the project files to the Space (**Files → Add file → Upload files**): drag the *contents* of the `CKDPredict` folder
   (Dockerfile, README.md, app.py, requirements.txt, ml/, templates/, static/, data/, models/, sample_reports/ ...) so that
   `Dockerfile` and `README.md` sit at the top level of the Space. Commit.
   - or with git: `git clone https://huggingface.co/spaces/<you>/ckdpredict`, copy the files in, `git add . && git commit -m "deploy" && git push`
5. The Space builds (5-10 min the first time: installs packages + trains the model). When it says **Running** your link is
   `https://<you>-ckdpredict.hf.space` - open it on any phone.

Notes
- Free Spaces sleep after a while without visitors; the first open then takes ~1 min. Open it once before your review.
- Storage is temporary: patient history is lost when the Space restarts. Use dummy patients only.
- Scanned-image OCR works here because the Dockerfile installs Tesseract.
- Backup for demo day (no account needed): run the app on your laptop and `cloudflared tunnel --url http://localhost:5000`.
