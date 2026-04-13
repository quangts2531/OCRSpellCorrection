# Deployment Guide — Flask OCR Application

> **Prerequisites:** Ubuntu/Debian VPS with Docker, Docker Compose, and Nginx installed.

---

## Phase 1 — Build & Start the Docker Container

```bash
# Clone or upload the project to your VPS, then:
cd /path/to/ocr

# Build the image and start in detached mode
docker compose up -d --build
```

Verify the container is running:

```bash
docker compose ps
```

> **Note:** First startup will be slow — the YOLO model (~250 MB) downloads from
> HuggingFace Hub. Subsequent restarts use the cached volume (`hf-cache`).

---

## Phase 2 — Configure Nginx Reverse Proxy

```bash
# 1. Copy the config to sites-available
sudo cp nginx/ocr-app.conf /etc/nginx/sites-available/ocr-app.conf

# 2. Enable the site (symlink into sites-enabled)
sudo ln -sf /etc/nginx/sites-available/ocr-app.conf /etc/nginx/sites-enabled/ocr-app.conf

# 3. (Optional) Remove default site to avoid conflicts
sudo rm -f /etc/nginx/sites-enabled/default

# 4. Test Nginx configuration for syntax errors
sudo nginx -t

# 5. Reload Nginx to apply changes
sudo systemctl reload nginx
```

---

## Phase 3 — Verify the Deployment

```bash
# Quick smoke test (from the VPS itself)
curl -s -o /dev/null -w "%{http_code}" http://localhost
# Expected: 200
```

Then open `http://<YOUR_VPS_IP>` in a browser and upload a test image.

---

## Useful Commands

| Action | Command |
|---|---|
| View logs | `docker compose logs -f` |
| Restart container | `docker compose restart` |
| Rebuild after code changes | `docker compose up -d --build` |
| Stop everything | `docker compose down` |
| Check Nginx error log | `sudo tail -f /var/log/nginx/error.log` |

---

## (Optional) Enable HTTPS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

Certbot will automatically modify the Nginx config to redirect HTTP → HTTPS.
