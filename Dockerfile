FROM python:3.8
WORKDIR /image-search

copy hello.py /image-search/
RUN python -m pip install requests
CMD ["python","/image-search/hello.py"]