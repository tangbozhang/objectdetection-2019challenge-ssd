# coding=utf-8
"""
提交结果
"""
import httplib
import json
import sys
import urllib

import os
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient

API_HOST = "192.168.0.179"
API_PORT = 8012
BOS_ACL_URL = "/studio/match/bosacl"
SUBMIT_URL = "/studio/match/submit"
CHECK_TOKEN_URL = "/studio/match/checktoken"


def bos_acl():
    """
    返回临时token
    :return:
    """
    conn = httplib.HTTPConnection(host=API_HOST, port=API_PORT, strict=False, timeout=300)
    conn.request(method='GET', url=BOS_ACL_URL, body=None)
    rs = conn.getresponse()
    status_code = rs.status
    result = rs.read()
    return status_code, result


def submit(file_key, submit_token, file_name):
    """
    提交结果
    :param file_key:
    :param submit_token:
    :param file_name:
    :return:
    """
    project_id = os.environ.get('PROJECT_ID')
    if project_id is None:
        project_id = 0
    params = urllib.urlencode({"fileKey": file_key,
                               "token": submit_token,
                               "fileOriginName": file_name,
                               "projectId": project_id})
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    conn = httplib.HTTPConnection(host=API_HOST, port=API_PORT, strict=False, timeout=300)
    conn.request(method='POST', url=SUBMIT_URL, body=params, headers=headers)
    rs = conn.getresponse()
    status_code = rs.status
    result = rs.read()
    return status_code, result


def check_token(submit_token):
    """
    检测token是否有效
    :param submit_token: 
    :return: 
    """
    params = urllib.urlencode({"token": submit_token})
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    conn = httplib.HTTPConnection(host=API_HOST, port=API_PORT, strict=False, timeout=300)
    conn.request(method='POST', url=CHECK_TOKEN_URL, body=params, headers=headers)
    rs = conn.getresponse()
    status_code = rs.status
    result = rs.read()
    return status_code, result


def main():
    """
    main函数
    :return: 
    """
    if len(sys.argv) != 3:
        print "参数个数不正确，请执行 sh test.sh 文件路径 Token"
        exit(0)
    path = sys.argv[1]
    token = sys.argv[2]
    if not os.path.exists(path):
        print "文件:" + path + " 不存在，请确认文件路径参数是否正确"
        exit(0)
    code, acl = bos_acl()
    if code != 200:
        print "服务异常，请稍候再试"
        exit(0)
    acl = json.loads(acl)
    if acl['errorCode'] != 0:
        print "服务异常，请稍候再试"
        exit(0)
    ak = unicode2str(acl['result']['accessKeyId'])
    sk = unicode2str(acl['result']['secretAccessKey'])
    bos_token = unicode2str(acl['result']['sessionToken'])
    config = BceClientConfiguration(credentials=BceCredentials(ak, sk),
                                    endpoint=unicode2str("http://bj.bcebos.com"),
                                    security_token=bos_token)
    bos_client = BosClient(config)
    bucket_name = unicode2str(acl['result']['bucketName'])
    object_key = unicode2str(acl['result']['fileKey'])
    token_status_code, token_data = check_token(token)
    if token_status_code != 200:
        print "服务异常，请稍候再试"
        exit(0)
    token_data = json.loads(token_data)
    if token_data['errorCode'] != 0:
        print token_data['errorMsg']
        exit(0)
    print "开始提交"
    upload_id = bos_client.initiate_multipart_upload(bucket_name, object_key).upload_id
    left_size = os.path.getsize(path)
    # left_size用于设置分块开始位置
    # 设置分块的开始偏移位置
    offset = 0
    part_number = 1
    part_list = []
    total_size = left_size
    while left_size > 0:
        # 设置每块为5MB
        part_size = 10 * 1024 * 1024
        if left_size < part_size:
            part_size = left_size
        print total_size - left_size, "/", total_size
        response = bos_client.upload_part_from_file(
            bucket_name, object_key, upload_id, part_number, part_size, path, offset)

        left_size -= part_size
        offset += part_size
        part_list.append({
            "partNumber": part_number,
            "eTag": response.metadata.etag
        })
        part_number += 1
    print total_size, "/", total_size
    bos_client.complete_multipart_upload(bucket_name, object_key, upload_id, part_list)
    file_name = os.path.basename(path)
    status_code, data = submit(object_key, token, file_name)
    if status_code != 200:
        print "服务异常，请稍候再试"
    data = json.loads(data)
    if data['errorCode'] == 0:
        print data['result']
    else:
        print data['errorMsg']


def unicode2str(u_code):
    """
    unicode转str
    :param u_code: 
    :return: 
    """
    if isinstance(u_code, unicode):
        return u_code.encode('unicode-escape').decode('string_escape')
    return u_code


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("提交结果失败，请稍后重试，如果多次重试仍然失败，请重新下载脚本")
