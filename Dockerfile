FROM tensorflow/tensorflow:2.16.1
WORKDIR /image-search

# 将当前目录下的所有文件复制到工作目录中
COPY . /image-search/

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","/image-search/tests/image_to_net_util_test.py"]