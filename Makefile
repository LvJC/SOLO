PY_SOURCE_FILES=tools/

format:
	autoflake --in-place --remove-all-unused-imports --recursive ${PY_SOURCE_FILES}
	isort --profile=black ${PY_SOURCE_FILES}
	black ${PY_SOURCE_FILES}
 
lint:
	export LC_ALL=C.UTF-8
	export LC_CTYPE=en_US.UTF-8
	isort --check --diff ${PY_SOURCE_FILES}
	black --check --diff ${PY_SOURCE_FILES} \
		--line-length 88
	flake8 ${PY_SOURCE_FILES} \
		--count \
		--show-source \
		--statistics \
		--max-line-length=88 \
		--ignore=E741,E501,E266,E203,E722,E402,F841,F401,E302,E305,W503,W291,W293