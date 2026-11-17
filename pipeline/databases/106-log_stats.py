#!/usr/bin/env python3
"""Statistiques sur les logs Nginx, avec le top 10 des IPs"""
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    nginx_collection = client.logs.nginx

    total = nginx_collection.count_documents({})
    print("{} logs".format(total))

    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = nginx_collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    status_check = nginx_collection.count_documents(
        {"method": "GET", "path": "/status"}
    )
    print("{} status check".format(status_check))

    group_step = {
        "$group": {
            "_id": "$ip",
            "count": {"$sum": 1}
        }
    }
    sort_step = {"$sort": {"count": -1}}
    limit_step = {"$limit": 10}
    pipeline = [group_step, sort_step, limit_step]

    top_ips = nginx_collection.aggregate(pipeline)

    print("IPs:")
    for ip in top_ips:
        print("\t{}: {}".format(ip.get("_id"), ip.get("count")))
