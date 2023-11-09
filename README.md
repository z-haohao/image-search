images-search 项目处理商品图片，将图片向量化存储到milvus数据库中，提供给外部检索图片

版本: 
python 3.8.x 

依赖:
pip install -r requirements.txt

任务流程：
kafka -> tensorflow[resnet50] -> milvus

kafka: 接收图片地址, 去minio/图片资源服务器拉取图片
tensorflow: 图片向量化
milvus: 向量存储


