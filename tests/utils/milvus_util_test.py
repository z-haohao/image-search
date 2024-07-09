from pymilvus import (
    MilvusClient,
)



client = MilvusClient(user='bdc', password='um5R3UYhgbv6L4Ri', uri='http://10.251.37.251:19531',
                      db_name='bdc_image_vectors'   )
import csv

with open(file='picture_id.csv',encoding='utf-8',mode='r') as file:

    file_csv = csv.reader(file)
    for row in file_csv:
        # print(row[0])
        ids = row[0]
        print(f'有数据-{ids}, 执行删除')
        client.delete(
            collection_name="bi_mdm_product_ecommerce_picture",
            ids=ids
        )
        # print(f'执行查询-{ids}')
        # res = client.get(
        #     collection_name="bi_mdm_product_ecommerce_picture",
        #     ids=ids
        # )
        # print(res)
        #
        # if res is not None and len(res) > 0:
        #
        #     print(f'有数据-{ids}, 执行删除')
        #     # 执行删除
        #     client.delete(
        #         collection_name="bi_mdm_product_ecommerce_picture",
        #         ids=ids
        #     )



# res = client.get(
#     collection_name="bi_mdm_product_ecommerce_img",
#     ids=420741
# )
# print(res)
