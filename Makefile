.PHONY: dev docker-dev docker-down

# Local Development (Runs both frontend & backend from frontend workspace)
dev:
	npm --prefix frontend run dev:all

# Docker Deployment (Single Port 3000)
docker-dev:
	docker compose up --build

docker-down:
	docker compose down
