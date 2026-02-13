#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Jeffrey Liao,jliao23\nVincent Chau,vkchau\nCalvin Tsai,ctsai7" > submit/team.txt

# train model
# uv run python src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
uv run python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
# cp Dockerfile submit/Dockerfile

# submit source code
# cp -r src submit/src

# submit checkpoints
# cp -r work submit/work

# submit uv dependencies
# cp pyproject.toml uv.lock .python-version submit/

# submit all files except blacklist
rsync -a \
	--exclude=".git/" \
	--exclude="*.md" \
    --exclude="*.pyc" \
	--exclude=".ruff_cache/" \
	--exclude=".venv/" \
	--exclude="data/" \
	--exclude="example/" \
	--exclude="grader/" \
	--exclude="work/" \
	--exclude="wandb/" \
	--exclude=".dockerignore" \
	--exclude=".gitignore" \
	--exclude=".pre-commit-config.yaml" \
	--exclude="submit.sh" \
	--exclude="submit/" \
	--exclude="submit.zip" \
    # --exclude=".env" \
	./ submit/

# submit selected work files (edit this list each time)
WORK_FILES=(
	# e.g. "work/char6.binary"
	# e.g. "work/vocab.json"
    "work/15k_char6_000122.binary"
    "work/15k_vocab.json"
)
mkdir -p submit/work
for wf in "${WORK_FILES[@]}"; do
	cp -r "$wf" submit/work/
done

# make zip file
zip -r submit.zip submit

uv run python grader/grade.py submit/pred.txt example/answer.txt --verbose
