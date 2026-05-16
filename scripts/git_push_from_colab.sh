#!/usr/bin/env bash
set -euo pipefail

: "${GIT_NAME:?Defina GIT_NAME. Exemplo: export GIT_NAME='Seu Nome'}"
: "${GIT_EMAIL:?Defina GIT_EMAIL. Exemplo: export GIT_EMAIL='seu-email@gmail.com'}"
: "${GITHUB_USERNAME:?Defina GITHUB_USERNAME.}"
: "${GITHUB_REPO:?Defina GITHUB_REPO.}"
: "${GITHUB_TOKEN:?Defina GITHUB_TOKEN usando um token do GitHub. Não commite esse token.}"

git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

git remote set-url origin "https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git"

git add .
git commit -m "${COMMIT_MESSAGE:-Update project and trained model artifacts}" || echo "Nada novo para commitar."
git push origin "${GIT_BRANCH:-main}"
