#!/usr/bin/env python3
"""Changes the topics of school documents in a MongoDB collection"""


def update_topics(mongo_collection, name, topics):
    """Sets topics on every document whose name matches name"""
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})
