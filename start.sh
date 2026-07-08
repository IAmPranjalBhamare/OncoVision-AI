#!/bin/sh
exec gunicorn --workers=1 --threads=2 --worker-class=gthread --bind=0.0.0.0:${PORT:-10000} app:app
